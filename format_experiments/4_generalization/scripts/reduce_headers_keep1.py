"""Like reduce_headers.py, but every issue keeps at least one T1/T2 hunk with a function name (when it has one);
the names to remove are drawn at random among the other named hunks until the target share is reached."""
import json, random, re, sys

HEAD = re.compile(r"^(@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@)(.*)$")
src, out, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
recs = [json.loads(l) for l in open(src)]
rng = random.Random(42)
named_by_issue = {i: [j for j, h in enumerate(r["gold_hunks"]) if HEAD.match(h["hunk_diff"].split("\n", 1)[0]).group(2).strip()]
                  for i, r in enumerate(recs)}
total = sum(len(r["gold_hunks"]) for r in recs)
n_named = sum(len(v) for v in named_by_issue.values())
free = []
for i, js in named_by_issue.items():
    if js:
        keep = rng.choice(js)
        free += [(i, j) for j in js if j != keep]
need = max(0, n_named - int(target * total))
drop = rng.sample(free, min(need, len(free)))
if len(drop) < need:   # the ratio has priority: take the rest from the hunks kept as "one per issue"
    protected = [(i, j) for i, js in named_by_issue.items() for j in js if (i, j) not in set(free)]
    drop += rng.sample(protected, min(need - len(drop), len(protected)))
for i, j in drop:
    h = recs[i]["gold_hunks"][j]
    first, _, rest = h["hunk_diff"].partition("\n")
    h["hunk_diff"] = HEAD.match(first).group(1) + ("\n" + rest if rest else "")
with open(out, "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")
left = n_named - len(drop)
print(f"{src}: named {n_named}/{total} ({n_named/total:.1%}) -> {left} ({left/total:.1%}); issues with hunks {sum(bool(r['gold_hunks']) for r in recs)}, with >=1 named before {sum(bool(v) for v in named_by_issue.values())}, after {len({i for i, js in named_by_issue.items() for j in js} and {i for i, js in named_by_issue.items() if any((i, j) not in set(drop) for j in js)})}")
