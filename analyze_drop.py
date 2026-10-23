"""Tables and paired bootstrap CIs for the budget-drop test.

    python analyze_drop.py results/gpu_drop
"""
import collections, json, random, statistics, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gpu_drop")
jobs = [json.loads(l) for l in open(d / "jobs.jsonl")]
by = collections.defaultdict(dict)
for j in jobs:
    by[j["id"]][j["key"]] = j
ids = sorted(by)
n = len(ids)
order = ["cut_now", "cut_now_hard", "silent_0.1", "warn_soft_0.1", "warn_count_0.1",
         "silent_0.25", "warn_soft_0.25", "warn_count_0.25"]
arms = [a for a in order if a in by[ids[0]]]

def acc(a): return 100 * sum(by[i][a]["correct"] for i in ids) / n
def think(a): return statistics.mean(by[i][a]["think_used"] / by[i][a]["L"] for i in ids)
def closed(a):
    v = [by[i][a]["closed_itself"] for i in ids]
    return None if v[0] is None else 100 * sum(bool(x) for x in v) / n

def ci(a, b, B=10000, seed=0):
    diff = [by[i][a]["correct"] - by[i][b]["correct"] for i in ids]
    rng = random.Random(seed)
    s = sorted(100 * sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * sum(diff) / n, s[int(0.025 * B)], s[int(0.975 * B)]

summ = json.load(open(d / "summary.json"))
print(f"n = {n} problems, median natural thinking {statistics.median(by[i][arms[0]]['L'] for i in ids):.0f} tokens, "
      f"natural accuracy {summ['natural_accuracy']}%\n")
print(f"{'arm':16s} {'acc %':>6s} {'think/natural':>14s} {'closed itself %':>16s}")
for a in arms:
    c = closed(a)
    print(f"{a:16s} {acc(a):6.1f} {think(a):14.2f} {'' if c is None else f'{c:.0f}':>16s}")

print("\nPaired differences (pp, 95% bootstrap CI)")
pairs = [("silent_0.1", "cut_now"), ("silent_0.25", "cut_now"), ("warn_soft_0.25", "cut_now"),
         ("warn_soft_0.1", "silent_0.1"), ("warn_count_0.1", "silent_0.1"),
         ("warn_soft_0.25", "silent_0.25"), ("warn_count_0.25", "silent_0.25"),
         ("warn_soft_0.25", "silent_0.1"), ("cut_now", "cut_now_hard")]
for a, b in pairs:
    if a in arms and b in arms:
        m, lo, hi = ci(a, b)
        print(f"  {a:16s} - {b:14s} {m:+5.1f}  [{lo:+.1f}, {hi:+.1f}]")

print("\nShare of the grace window the warned model actually used (median)")
for a in arms:
    if a.startswith("warn"):
        w = float(a.rsplit("_", 1)[1])
        u = statistics.median((by[i][a]["think_used"] - summ["drop_at"] * by[i][a]["L"]) / (w * by[i][a]["L"]) for i in ids)
        print(f"  {a:16s} {u:.2f}")

wrong = [i for i in ids if not by[i]["cut_now"]["correct"]]
print(f"\nOf the {len(wrong)} problems cut_now gets wrong, how many each arm gets right")
for a in arms[1:]:
    print(f"  {a:16s} {sum(by[i][a]['correct'] for i in wrong)}")
