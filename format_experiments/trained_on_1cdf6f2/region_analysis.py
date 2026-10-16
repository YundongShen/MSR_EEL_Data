import json, sys
import torch
sys.path.insert(0, "/home/x_yushe/eel_rerun")
from config import default_config as cfg
from models.entailment_encoder import EntailmentEncoder
from data.entailment_dataset import _render_hunk, _units_for_hunk

P = "/proj/test_extender/ICLR/EditBoundaryDetection/"
W = P + "generalization_rebuild/"
THR_REQ, THR_TEST, THR_ORIG = 0.65, 0.40, 0.80

def assign_region(sr, st, so):
    hi_req, hi_test, hi_orig = sr > THR_REQ, st > THR_TEST, so > THR_ORIG
    if hi_req and hi_test: return "R1"
    if hi_req and not hi_test and hi_orig: return "R2"
    if hi_req and not hi_test and not hi_orig: return "R3"
    if not hi_req and hi_orig: return "R4"
    return "R5"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)

tier3 = {}
for l in open(W + "submit_1cdf6f2/tier3_hunks.jsonl"):
    r = json.loads(l)
    if r.get("tier3_hunks"): tier3[r["instance_id"]] = r["tier3_hunks"]

insts = []
for l in open(W + "submit_1cdf6f2/instances_django_oldschema.jsonl"):
    r = json.loads(l)
    if r["instance_id"].startswith("django") and r.get("gold_hunks"):
        insts.append(r)
print("Django instances with gold_hunks:", len(insts), flush=True)

items = []
for I in insts:
    req = I["requirement"][:cfg.data.max_req_chars]
    tests = [t["code"] for t in I.get("test_functions", []) if t.get("code")]
    units = I.get("source_units", [])
    for h in I["gold_hunks"]:
        txt = _render_hunk(h)
        if txt.strip():
            items.append((I["instance_id"], "retained", txt, req, tests, [u["code"] for u in _units_for_hunk(h, units) if u.get("code")]))
    for h in tier3.get(I["instance_id"], []):
        txt = _render_hunk(h)
        if txt.strip():
            items.append((I["instance_id"], "t3", txt, req, tests, [u["code"] for u in _units_for_hunk(h, units) if u.get("code")]))
print("total hunks:", len(items), " retained:", sum(1 for i in items if i[1] == "retained"), " t3:", sum(1 for i in items if i[1] == "t3"), flush=True)

enc = EntailmentEncoder(model_name=cfg.model.encoder_name, projection_dim=cfg.model.projection_dim,
                        dropout=0.0, max_length=cfg.model.max_length).to(dev).eval()
ck = torch.load(P + "checkpoints_submit/m2_new_1cdf6f2/best.pt", map_location=dev, weights_only=False)
enc.load_state_dict(ck["encoder_state"])
print("loaded checkpoints_submit/m2_new_1cdf6f2/best.pt  epoch:", ck.get("epoch"), " val_loss:", ck.get("val_loss"), flush=True)

def score(r, a, b, g):
    return enc.entailment_score(hunk_texts=[r[2]], req_texts=[r[3]], test_texts=[r[4]], orig_texts=[r[5]],
                                device=dev, alpha=a, beta=b, gamma=g, use_projection=True).item()

rows = []
t0 = __import__("time").time()
for i, it in enumerate(items):
    sr = score(it, 1, 0, 0); st = score(it, 0, 1, 0); so = score(it, 0, 0, 1)
    rows.append({"instance_id": it[0], "group": it[1], "sim_req": sr, "sim_test": st, "sim_orig": so, "region": assign_region(sr, st, so)})
    if i % 300 == 0: print(i, "/", len(items), "%.0fs" % (__import__("time").time() - t0), flush=True)

import collections
for g in ("retained", "t3"):
    x = [r for r in rows if r["group"] == g]
    c = collections.Counter(r["region"] for r in x)
    print(g, len(x), " ".join("%s %d (%.1f%%)" % (k, c.get(k, 0), 100 * c.get(k, 0) / max(1, len(x))) for k in ["R1", "R2", "R3", "R4", "R5"]), flush=True)
json.dump(rows, open(W + "logs/tables_1cdf6f2/region_hunk_scores_m2new.json", "w"))
