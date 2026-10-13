"""Remove the function name from the '@@' line of a random part of the git-rendered T1/T2 hunks, so that the
share of T1/T2 hunks that carry one equals a target (the T0 pool is drawn from the same hunks)."""
import json, random, re, sys

HEAD = re.compile(r"^(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)(.*)$")
src, out, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
recs = [json.loads(l) for l in open(src)]
named = [(i, j) for i, r in enumerate(recs) for j, h in enumerate(r["gold_hunks"])
         if HEAD.match(h["hunk_diff"].split("\n", 1)[0]).group(2).strip()]
total = sum(len(r["gold_hunks"]) for r in recs)
keep = int(target * total)
drop = random.Random(42).sample(named, max(0, len(named) - keep))
for i, j in drop:
    h = recs[i]["gold_hunks"][j]
    first, _, rest = h["hunk_diff"].partition("\n")
    h["hunk_diff"] = HEAD.match(first).group(1) + ("\n" + rest if rest else "")
with open(out, "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")
left = len(named) - len(drop)
print(f"{src}: T1/T2 hunks {total}, with function name {len(named)} ({len(named)/total:.1%}) -> {left} ({left/total:.1%})")
