"""Test C numbers: AIME 2026 with final-answer-only scoring, and the change from Test B on the same traces.

    python scripts/final_vs_full.py
"""
import collections, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gpu"))
from aime_drop_bench import cluster_ci  # noqa: E402

ARMS = ["natural", "no_think", "cut_now", "cut_now_hard", "silent_0.1", "silent_0.25",
        "warn_soft_0.1", "warn_soft_0.25", "warn_count_0.1", "warn_count_0.25"]


def load(path):
    by = collections.defaultdict(dict)
    for line in open(path):
        j = json.loads(line)
        by[j["sid"]][j["key"]] = j
    return by


full = load(ROOT / "results/gpu_aime/jobs.jsonl")
final = load(ROOT / "results/gpu_aime_final/jobs.jsonl")
sids = sorted(s for s in final if s in full)
pid = {s: final[s]["cut_now"]["id"] for s in sids}
ok_full = lambda s, k: int(full[s]["cut_now"]["nat_correct"]) if k == "natural" else int(full[s][k]["correct"])
ok_final = lambda s, k: int(final[s]["natural_final"]["correct"]) if k == "natural" else int(final[s][k]["correct"])


def ci(f):
    d = collections.defaultdict(list)
    for s in sids:
        d[pid[s]].append(f(s))
    return cluster_ci(d)


print(f"{len(sids)} traces scored both ways\n")
print("Change from full answers (Test B) to final answer only (Test C), 95% CI over problems")
for k in ARMS:
    a = 100 * statistics.mean(ok_full(s, k) for s in sids)
    b = 100 * statistics.mean(ok_final(s, k) for s in sids)
    m, lo, hi = ci(lambda s: ok_final(s, k) - ok_full(s, k))
    print(f"  {k:16s} full {a:5.1f}  final {b:5.1f}  change {m:+5.1f} [{lo:+.1f}, {hi:+.1f}]")

acc = lambda k: 100 * statistics.mean(ok_final(s, k) for s in sids)
th = lambda k: statistics.mean(final[s][k]["think_used"] / final[s][k]["L"] for s in sids)
pts = [(0.5, acc("cut_now")), (th("silent_0.1"), acc("silent_0.1")), (th("silent_0.25"), acc("silent_0.25")),
       (1.0, acc("natural"))]


def interp(x):
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


print("\nTest C: warned arms vs the silent line at the same thinking used (linear interpolation)")
for k in ("warn_soft_0.1", "warn_soft_0.25", "warn_count_0.1", "warn_count_0.25"):
    print(f"  {k:16s} thinking {th(k):.2f}  acc {acc(k):5.1f}  line {interp(th(k)):5.1f}  gap {acc(k) - interp(th(k)):+.1f}")

print("\nTest C: headroom for a trained model, 95% CI over problems")
for x, y in (("natural", "silent_0.25"), ("natural", "silent_0.1")):
    m, lo, hi = ci(lambda s: ok_final(s, x) - ok_final(s, y))
    print(f"  {x} - {y}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]")

print("\nTest C by how often the problem is solved with natural thinking")
groups = collections.defaultdict(list)
for s in sids:
    groups[pid[s]].append(s)
for name, cond in (("always solved", lambda r: r == 1), ("sometimes solved", lambda r: 0 < r < 1),
                   ("never solved", lambda r: r == 0)):
    g = [v for v in groups.values() if cond(statistics.mean(ok_final(s, "natural") for s in v))]
    ss = [s for v in g for s in v]
    f = lambda k: 100 * statistics.mean(ok_final(s, k) for s in ss)
    print(f"  {name:16s} {len(g):2d} problems | " + "  ".join(
        f"{k} {f(k):5.1f}" for k in ("natural", "cut_now", "silent_0.25", "warn_soft_0.25", "warn_count_0.25")))
