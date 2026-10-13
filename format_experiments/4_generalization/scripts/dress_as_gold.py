"""Format diagnostic: put the generated hunks of a set into the old (reference-patch era) instance schema,
once as they are (difflib rendering) and once re-rendered by git exactly like a reference patch
(function name after '@@', git hunk boundaries).  The edits themselves are unchanged.

Writes <out>/instances_asgen.jsonl   gold_hunks = generated T1/T2 hunks, difflib rendering
       <out>/instances_asgold.jsonl  gold_hunks = the same hunks, git rendering
       <out>/tier3_gen.jsonl         Tier-3 hunks, difflib rendering
       <out>/tier3_asgold.jsonl      Tier-3 hunks, git rendering
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ATTRS = "\n".join(f"*.{ext} diff={drv}" for ext, drv in [
    ("py", "python"), ("java", "java"), ("go", "golang"), ("rs", "rust"), ("rb", "ruby"), ("php", "php"),
    ("c", "cpp"), ("h", "cpp"), ("cc", "cpp"), ("cpp", "cpp"), ("hpp", "cpp"), ("cs", "csharp"),
]) + "\n"
HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def git(cwd, *args, data=None):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, input=data, capture_output=True)


def context(src: str, start: int, hunk_lines: list[str], n: int = 5):
    lines = src.splitlines()
    before = lines[max(0, start - 1 - n):max(0, start - 1)]
    old = sum(1 for l in hunk_lines[1:] if l.startswith((" ", "-")))
    return before, lines[start - 1 + old:start - 1 + old + n]


def rerender(repo: Path, sources: dict[str, str], hunk: dict, stats: Counter) -> dict:
    path = hunk["filepath"]
    if path not in sources:
        stats["file not in snapshot (kept as is)"] += 1
        return dict(hunk)
    body = hunk["hunk_diff"] if hunk["hunk_diff"].endswith("\n") else hunk["hunk_diff"] + "\n"
    patch = f"--- a/{path}\n+++ b/{path}\n{body}".encode()
    done = git(repo, "apply", "--whitespace=nowarn", "-", data=patch)
    if done.returncode != 0:
        done = git(repo, "apply", "--whitespace=nowarn", "--recount", "-C1", "-", data=patch)
    if done.returncode != 0:
        stats["hunk does not apply (kept as is)"] += 1
        return dict(hunk)
    diff = git(repo, "diff", "-U3", "--no-color", "--", path).stdout.decode("utf-8", errors="replace")
    git(repo, "checkout", "--", ".")
    hunks, cur = [], None
    for line in diff.splitlines():
        if line.startswith("@@"):
            cur = [line]
            hunks.append(cur)
        elif cur is not None and line[:1] in (" ", "+", "-", "\\"):
            cur.append(line)
        elif cur is not None and line == "":
            cur.append(" ")
    if not hunks:
        stats["empty git diff (kept as is)"] += 1
        return dict(hunk)
    start0 = int(hunk["old_start_line"])
    best = min(hunks, key=lambda h: abs(int(HEADER.match(h[0]).group(1)) - start0))
    m = HEADER.match(best[0])
    start = int(m.group(1))
    before, after = context(sources[path], start, best)
    stats["re-rendered"] += 1
    stats["with function name"] += bool(m.group(5).strip())
    stats["split into several git hunks"] += len(hunks) > 1
    return {**hunk, "old_start_line": start, "hunk_diff": "\n".join(best), "context_before": before, "context_after": after}


def main(instances_path, raw_path, t12_path, t3_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t12 = {r["instance_id"]: r["llm_t12_hunks"] for r in read(t12_path)}
    t3 = {r["instance_id"]: r["tier3_hunks"] for r in read(t3_path)}
    need = set(t12) | set(t3)
    sources = {r["instance_id"]: r.get("source_files", {}) for r in read(raw_path) if r["instance_id"] in need}
    stats: Counter = Counter()
    gold12, gold3 = {}, {}
    for n, iid in enumerate(sorted(need), 1):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for path, src in sources.get(iid, {}).items():
                target = repo / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.encode("utf-8"))
            (repo / ".gitattributes").write_text(ATTRS)
            git(repo, "init", "-q")
            git(repo, "add", "-A")
            git(repo, "-c", "user.name=x", "-c", "user.email=x@x", "commit", "-q", "-m", "base")
            gold12[iid] = [rerender(repo, sources.get(iid, {}), h, stats) for h in t12.get(iid, [])]
            gold3[iid] = [rerender(repo, sources.get(iid, {}), h, stats) for h in t3.get(iid, [])]
        if n % 50 == 0:
            print(n, "/", len(need), dict(stats), flush=True)

    def strip(h, i):   # the old schema: tiers of gold_hunks are assigned by the old evaluator
        return {"hunk_id": i, "filepath": h["filepath"], "old_start_line": h["old_start_line"],
                "hunk_diff": h["hunk_diff"].rstrip("\n"), "context_before": h.get("context_before", []),
                "context_after": h.get("context_after", []), "tier_label": None}

    with open(out / "instances_asgen.jsonl", "w") as fa, open(out / "instances_asgold.jsonl", "w") as fb:
        for rec in read(instances_path):
            iid = rec["instance_id"]
            fa.write(json.dumps({**rec, "gold_hunks": [strip(h, i) for i, h in enumerate(t12.get(iid, []))]}) + "\n")
            fb.write(json.dumps({**rec, "gold_hunks": [strip(h, i) for i, h in enumerate(gold12.get(iid, []))]}) + "\n")
    with open(out / "tier3_gen.jsonl", "w") as fa, open(out / "tier3_asgold.jsonl", "w") as fb:
        for iid in t3:
            fa.write(json.dumps({"instance_id": iid, "tier3_hunks": [{**strip(h, i), "tier_label": 3} for i, h in enumerate(t3[iid])]}) + "\n")
            fb.write(json.dumps({"instance_id": iid, "tier3_hunks": [{**strip(h, i), "tier_label": 3} for i, h in enumerate(gold3[iid])]}) + "\n")
    print("DONE", out_dir, dict(stats))


if __name__ == "__main__":
    main(*sys.argv[1:6])
