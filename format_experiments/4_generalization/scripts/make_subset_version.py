"""Subset rule.

  T1/T2: for each reference hunk keep one generated hunk: same file, start line < 20 lines away, the closest
         one (ties: lower sample index, then earlier position); a generated hunk is used at most once.
  T3:    at most 2 per issue, drawn with random.Random("42-<instance_id>").

    python make_subset_version.py llm_t12_hunks.jsonl tier3_hunks.jsonl raw_instances.jsonl.gz out_dir
"""

import gzip
import json
import random
import re
import sys
from pathlib import Path

TOLERANCE = 20
MAX_T3 = 2


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def reference_hunks(patch):
    """(file, start line) of every hunk of the reference patch."""
    out, current = [], None
    for line in patch.splitlines():
        if line.startswith("--- "):
            m = re.match(r"^--- (?:a/)?(.+)", line)
            current = m.group(1).strip() if m else None
        elif line.startswith("+++ ") and current == "/dev/null":
            m = re.match(r"^\+\+\+ (?:b/)?(.+)", line)
            current = m.group(1).strip() if m else current
        elif line.startswith("@@") and current:
            m = re.search(r"@@ -(\d+)", line)
            out.append((current, int(m.group(1)) if m else 1))
    return out


def select_retained(references, hunks):
    used = set()
    for ref_file, ref_start in references:
        best = None
        for i, h in enumerate(hunks):
            if i in used or h["filepath"] != ref_file:
                continue
            dist = abs(int(h["old_start_line"]) - ref_start)
            if dist >= TOLERANCE:
                continue
            key = (dist, h.get("sample", 0), i)
            if best is None or key < best:
                best = key
        if best is not None:
            used.add(best[2])
    return [hunks[i] for i in sorted(used)]


def select_tier3(iid, hunks):
    if len(hunks) <= MAX_T3:
        return hunks
    keep = sorted(random.Random(f"42-{iid}").sample(range(len(hunks)), MAX_T3))
    return [hunks[i] for i in keep]


def main(t12_path, t3_path, raw_path, out_dir):
    patches = {r["instance_id"]: r.get("patch") or "" for r in read(raw_path)}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "llm_t12_hunks.jsonl", "w") as fh:
        for r in read(t12_path):
            kept = select_retained(reference_hunks(patches.get(r["instance_id"], "")), r["llm_t12_hunks"])
            if kept:
                fh.write(json.dumps({"instance_id": r["instance_id"], "llm_t12_hunks": kept}, ensure_ascii=False) + "\n")
    with open(out / "tier3_hunks.jsonl", "w") as fh:
        for r in read(t3_path):
            kept = select_tier3(r["instance_id"], r["tier3_hunks"])
            fh.write(json.dumps({"instance_id": r["instance_id"], "tier3_hunks": kept}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
