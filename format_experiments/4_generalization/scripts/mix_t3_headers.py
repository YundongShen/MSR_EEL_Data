"""Give a random share of the Tier-3 hunks the git rendering with a function name (the rest keep the generated
rendering without one).

    python scripts/mix_t3_headers.py <subset dir> <percent, e.g. 20>
"""
import json
import random
import re
import sys

HEAD = re.compile(r"^@@ [^@]*@@ ?\S")
folder, percent = sys.argv[1].rstrip("/"), int(sys.argv[2])
gen = [json.loads(line) for line in open(f"{folder}/tier3_gen.jsonl")]
git = {r["instance_id"]: r["tier3_hunks"] for r in map(json.loads, open(f"{folder}/tier3_asgold.jsonl"))}
cand = [(i, j) for i, r in enumerate(gen) for j, _ in enumerate(r["tier3_hunks"])
        if HEAD.match(git[r["instance_id"]][j]["hunk_diff"])]
total = sum(len(r["tier3_hunks"]) for r in gen)
pick = random.Random(42).sample(cand, min(len(cand), round(percent / 100 * total)))
for i, j in pick:
    gen[i]["tier3_hunks"][j] = git[gen[i]["instance_id"]][j]
out = f"{folder}/tier3_mix{percent}.jsonl"
with open(out, "w") as fh:
    for r in gen:
        fh.write(json.dumps(r) + "\n")
print(f"{out}: Tier-3 hunks {total}, with function name {len(pick)} ({len(pick) / total:.1%}); "
      f"issues with a named Tier-3 hunk: {len({i for i, _ in pick})}")
