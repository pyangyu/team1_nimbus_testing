"""Extra numbers used in RESULTS.md that the main analyses do not print.

    python scripts/extra_stats.py
"""
import collections, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu"))
from aime_drop_bench import cluster_ci  # noqa: E402


def load(path):
    by = collections.defaultdict(dict)
    for line in open(path):
        j = json.loads(line)
        by[j.get("sid", j["id"])][j["key"]] = j
    return by


def curve_gaps(by, nat_acc, name):
    ids = sorted(by)
    acc = lambda k: 100 * sum(by[i][k]["correct"] for i in ids) / len(ids)
    th = lambda k: statistics.mean(by[i][k]["think_used"] / by[i][k]["L"] for i in ids)
    pts = [(0.5, acc("cut_now")), (th("silent_0.1"), acc("silent_0.1")), (th("silent_0.25"), acc("silent_0.25")),
           (1.0, nat_acc)]

    def interp(x):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    print(f"\n{name}: warned arms vs the silent curve at the same thinking used (linear interpolation)")
    for k in ("warn_soft_0.1", "warn_soft_0.25", "warn_count_0.1", "warn_count_0.25"):
        print(f"  {k:16s} thinking {th(k):.2f}  acc {acc(k):5.1f}  silent curve {interp(th(k)):5.1f}  "
              f"gap {acc(k) - interp(th(k)):+.1f}")


# MATH L4-5 (one run per problem)
m = load(ROOT / "results/gpu_drop/jobs.jsonl")
curve_gaps(m, json.load(open(ROOT / "results/gpu_drop/summary.json"))["natural_accuracy"], "MATH L4-5")

# AIME 2026 (8 runs per problem)
a = load(ROOT / "results/gpu_aime/jobs.jsonl")
sids = sorted(a)
ok = lambda s, k: int(a[s]["cut_now"]["nat_correct"]) if k == "natural" else int(a[s][k]["correct"])
curve_gaps(a, 100 * statistics.mean(ok(s, "natural") for s in sids), "AIME 2026")
print("\nAIME 2026: headroom for a trained model (natural minus silent cap), 95% CI over problems")
for x, y in (("natural", "silent_0.25"), ("natural", "silent_0.1"), ("natural", "cut_now"),
             ("cut_now_hard", "cut_now")):
    d = collections.defaultdict(list)
    for s in sids:
        d[a[s]["cut_now"]["id"]].append(ok(s, x) - ok(s, y))
    m_, lo, hi = cluster_ci(d)
    print(f"  {x} - {y}: {m_:+.1f} [{lo:+.1f}, {hi:+.1f}]")

print("\nAIME 2026 by how often the problem is solved with natural thinking")
groups = collections.defaultdict(list)
for s in sids:
    groups[a[s]["cut_now"]["id"]].append(s)
for name, cond in (("always solved", lambda r: r == 1), ("sometimes solved", lambda r: 0 < r < 1),
                   ("never solved", lambda r: r == 0)):
    g = [ss for ss in groups.values() if cond(statistics.mean(ok(s, "natural") for s in ss))]
    ss = [s for x in g for s in x]
    f = lambda k: 100 * statistics.mean(ok(s, k) for s in ss)
    print(f"  {name:16s} {len(g):2d} problems, {len(ss):3d} traces | natural {f('natural'):5.1f}  "
          f"no_think {f('no_think'):5.1f}  cut_now {f('cut_now'):5.1f}  silent_0.25 {f('silent_0.25'):5.1f}  "
          f"warn_soft_0.25 {f('warn_soft_0.25'):5.1f}  warn_count_0.25 {f('warn_count_0.25'):5.1f}")
nat = [json.loads(l) for l in open(ROOT / "results/gpu_aime/natural.jsonl")]
fin = [r["L"] for r in nat if not r["capped"]]
print(f"\nAIME natural thinking (finished traces): median {statistics.median(fin):.0f}, mean {statistics.mean(fin):.0f}, "
      f"max {max(fin)} tokens; {sum(r['capped'] for r in nat)} of {len(nat)} ran out of context")
