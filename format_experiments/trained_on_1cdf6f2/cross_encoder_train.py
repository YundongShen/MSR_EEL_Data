import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import default_config as cfg
from data.entailment_dataset import _render_hunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_ce")

OLD = "/proj/test_extender/ICLR/EditBoundaryDetection/old_v1/"
MAX_LENGTH = 512
MAX_T3 = 4


INSTANCES = OLD + "data_processed/instances_full_v1.jsonl"
TIER3 = OLD + "data_cache/tier3_hunks.jsonl"


def load(split_ids):
    insts = {}
    with open(INSTANCES) as fh:
        for line in fh:
            r = json.loads(line)
            if r["instance_id"] in split_ids and r.get("gold_hunks"):
                insts[r["instance_id"]] = (r.get("requirement", "")[:cfg.data.max_req_chars],
                                           [t for t in map(_render_hunk, r["gold_hunks"]) if t.strip()])
    t3 = {}
    with open(TIER3) as fh:
        for line in fh:
            r = json.loads(line)
            if r["instance_id"] in insts:
                t3[r["instance_id"]] = [t for t in map(_render_hunk, r["tier3_hunks"]) if t.strip()]
    return insts, t3


def groups(insts, t3, cross, rng):
    by_repo = {}
    for iid, (_, hunks) in insts.items():
        by_repo.setdefault(iid.split("__")[0], []).extend((iid, h) for h in hunks)
    out = []
    for iid, (req, hunks) in insts.items():
        own_t3 = t3.get(iid, [])
        for pos in hunks:
            negs = rng.sample(own_t3, min(MAX_T3, len(own_t3)))
            if cross:
                pool = [h for (eid, h) in by_repo[iid.split("__")[0]] if eid != iid]
                negs += rng.sample(pool, min(cross, len(pool)))
            if negs:
                out.append((req, [pos] + negs))
    return out


def group_loss(model, tok, req, cands, device):
    enc = tok([req] * len(cands), cands, padding=True, truncation=True, max_length=MAX_LENGTH,
              return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc).logits[:, 0].float()
    return F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cross-negatives", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--instances", default=None, help="instances in the old schema (gold_hunks = retained hunks)")
    ap.add_argument("--tier3", default=None)
    args = ap.parse_args()
    global INSTANCES, TIER3
    INSTANCES = args.instances or INSTANCES
    TIER3 = args.tier3 or TIER3
    log.info("training data: %s | %s", INSTANCES, TIER3)

    random.seed(42)
    torch.manual_seed(42)
    rng = random.Random(42)
    splits = json.load(open("data/processed/splits.json"))
    tr_i, tr_t3 = load(set(splits["train_ids"]))
    va_i, va_t3 = load(set(splits["val_ids"]))
    val = groups(va_i, va_t3, args.cross_negatives, random.Random(7))
    log.info("train issues %d (with Tier-3 %d), val groups %d, cross negatives %d",
             len(tr_i), sum(bool(tr_t3.get(i)) for i in tr_i), len(val), args.cross_negatives)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/unixcoder-base", num_labels=1).to(device)
    n_groups = len(groups(tr_i, tr_t3, args.cross_negatives, random.Random(0)))
    steps = math.ceil(n_groups / args.accum) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    log.info("train groups per epoch %d, optimizer steps %d", n_groups, steps)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train = groups(tr_i, tr_t3, args.cross_negatives, rng)
        rng.shuffle(train)
        total = 0.0
        opt.zero_grad()
        for k, (req, cands) in enumerate(train, 1):
            loss = group_loss(model, tok, req, cands, device) / args.accum
            loss.backward()
            total += loss.item() * args.accum
            if k % args.accum == 0 or k == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if k % 2000 == 0:
                log.info("epoch %d  group %d/%d  loss %.4f", epoch, k, len(train), total / k)
        model.eval()
        with torch.no_grad():
            vloss = sum(group_loss(model, tok, r, c, device).item() for r, c in val) / max(1, len(val))
        log.info("Epoch %d — train_loss=%.4f  val_loss=%.4f", epoch, total / len(train), vloss)
        if vloss < best:
            best = vloss
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            log.info("saved best (epoch %d) → %s", epoch, args.out)


if __name__ == "__main__":
    main()
