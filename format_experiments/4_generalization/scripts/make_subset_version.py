"""Build the "实验前格式版本" (pre-experiment formatted version): a subset of the sampled formatted
version whose candidates per issue are close to the main test set (4.5 retained / 1.7 Tier-3 per issue).

Rules (same for every set, nothing is regenerated):
  retained  for each reference hunk keep ONE generated hunk: the closest retained hunk in the same file
            (start line within 20 lines); ties -> lower sample index.  A generated hunk is used once.
            This is the rule of the main data set (one retained hunk per reference hunk).
  Tier-3    at most MAX_T3 per issue, drawn with random.Random(42) per issue.
The same hunks are selected in every rendering (generated format / git format).

    python scripts/make_subset_version.py <set> <raw instances> [max_t3]
"""

from __future__ import annotations

import gzip
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Code"))
from data.hunk_matching import MATCH_TOLERANCE, parse_hunks  # noqa: E402

SRC = ROOT / "采样后格式版本"
DST = ROOT / "实验前格式版本"
HEAD = re.compile(r"^@@ [^@]*@@ ?\S")


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def select_retained(reference: list[dict], hunks: list[dict]) -> list[int]:
    used: set[int] = set()
    for ref in reference:
        best = None
        for i, h in enumerate(hunks):
            if i in used or h["filepath"] != ref["filepath"]:
                continue
            dist = abs(int(h["old_start_line"]) - ref["old_start"])
            if dist >= MATCH_TOLERANCE:
                continue
            key = (dist, h.get("sample", 0), i)
            if best is None or key < best[0]:
                best = (key, i)
        if best is not None:
            used.add(best[1])
    return sorted(used)


def main(name: str, raw_path: str, max_t3: int = 2) -> None:
    final = ROOT / "final" / name
    src, dst = SRC / name, DST / name
    dst.mkdir(parents=True, exist_ok=True)
    patches = {r["instance_id"]: r.get("patch", "") for r in read(raw_path)}
    t12 = {r["instance_id"]: r["llm_t12_hunks"] for r in read(final / "llm_t12_hunks.jsonl")}
    t3 = {r["instance_id"]: r["tier3_hunks"] for r in read(final / "tier3_hunks.jsonl")}

    keep12 = {iid: select_retained(parse_hunks(patches.get(iid, "") or ""), hunks) for iid, hunks in t12.items()}
    keep3 = {}
    for iid, hunks in t3.items():
        idx = list(range(len(hunks)))
        keep3[iid] = sorted(random.Random(f"42-{iid}").sample(idx, max_t3)) if len(idx) > max_t3 else idx

    # generated-format hunk files (schema of build_dataset.py)
    with open(dst / "llm_t12_hunks.jsonl", "w") as fh:
        for iid, hunks in t12.items():
            if keep12[iid]:
                fh.write(json.dumps({"instance_id": iid, "llm_t12_hunks": [hunks[i] for i in keep12[iid]]}, ensure_ascii=False) + "\n")
    with open(dst / "tier3_hunks.jsonl", "w") as fh:
        for iid, hunks in t3.items():
            fh.write(json.dumps({"instance_id": iid, "tier3_hunks": [hunks[i] for i in keep3[iid]]}, ensure_ascii=False) + "\n")

    # the two renderings in the old instance schema (gold_hunks field), same selection
    named = total = 0
    for fname in ("instances_asgen.jsonl", "instances_asgold.jsonl"):
        with open(dst / fname, "w") as fh:
            for rec in read(src / fname):
                idx = keep12.get(rec["instance_id"], [])
                assert not idx or max(idx) < len(rec["gold_hunks"]), rec["instance_id"]
                rec["gold_hunks"] = [{**rec["gold_hunks"][i], "hunk_id": k} for k, i in enumerate(idx)]
                if fname == "instances_asgold.jsonl":
                    total += len(idx)
                    named += sum(bool(HEAD.match(h["hunk_diff"])) for h in rec["gold_hunks"])
                fh.write(json.dumps(rec) + "\n")
    for fname in ("tier3_gen.jsonl", "tier3_asgold.jsonl"):
        with open(dst / fname, "w") as fh:
            for rec in read(src / fname):
                idx = keep3[rec["instance_id"]]
                fh.write(json.dumps({"instance_id": rec["instance_id"], "tier3_hunks": [rec["tier3_hunks"][i] for i in idx]}) + "\n")

    def stats(sel12, sel3):
        ev = [i for i in t12 if sel12(i) and i in t3 and sel3(i)]
        tiers = [h["tier_label"] for i in ev for h in sel12(i)]
        return {"eval_issues": len(ev), "t1_issues": sum(any(h["tier_label"] == 1 for h in sel12(i)) for i in ev),
                "t2_issues": sum(any(h["tier_label"] == 2 for h in sel12(i)) for i in ev),
                "T1": tiers.count(1), "T2": tiers.count(2), "T3": sum(len(sel3(i)) for i in ev),
                "retained_per_issue": round(len(tiers) / max(1, len(ev)), 2),
                "t3_per_issue": round(sum(len(sel3(i)) for i in ev) / max(1, len(ev)), 2)}

    before = stats(lambda i: t12[i], lambda i: t3[i])
    after = stats(lambda i: [t12[i][k] for k in keep12[i]], lambda i: [t3[i][k] for k in keep3[i]])
    manifest = {"name": name, "version": "实验前格式版本", "source_version": "采样后格式版本",
                "rule": {"retained": "one generated hunk per reference hunk (closest, same file, < 20 lines)",
                         "tier3": f"at most {max_t3} per issue, random.Random('42-<instance_id>')"},
                "sampled_version": before, "this_version": after,
                "function_name_rate_of_retained_git_format": round(named / max(1, total), 4)}
    json.dump(manifest, open(dst / "manifest.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
