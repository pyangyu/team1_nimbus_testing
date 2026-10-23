"""Accuracy vs thinking actually used, MATH L4-5 and AIME 2026 budget-drop tests.

    python scripts/plot_drop.py      ->  drop_chart.png
"""
import collections, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def load(path, nat_from_summary=None):
    by = collections.defaultdict(dict)
    for line in open(path):
        j = json.loads(line)
        by[j.get("sid", j["id"])][j["key"]] = j
    ids = sorted(by)
    acc = lambda k: 100 * sum(by[i][k]["correct"] for i in ids) / len(ids)
    th = lambda k: statistics.mean(by[i][k]["think_used"] / by[i][k]["L"] for i in ids)
    nat = (100 * sum(by[i]["cut_now"]["nat_correct"] for i in ids) / len(ids)
           if "nat_correct" in by[ids[0]]["cut_now"] else nat_from_summary)
    line = ([(0.0, acc("no_think"))] if "no_think" in by[ids[0]] else [])
    line += [(0.5, acc("cut_now")), (th("silent_0.1"), acc("silent_0.1")),
             (th("silent_0.25"), acc("silent_0.25")), (1.0, nat)]
    soft = [(th(f"warn_soft_{x}"), acc(f"warn_soft_{x}"), x) for x in (0.1, 0.25)]
    count = [(th(f"warn_count_{x}"), acc(f"warn_count_{x}"), x) for x in (0.1, 0.25)]
    return line, soft, count, len(ids)


math_sum = json.load(open(ROOT / "results/gpu_drop/summary.json"))
panels = [("MATH L4–5 (173 problems, 1 run)", load(ROOT / "results/gpu_drop/jobs.jsonl",
                                                    math_sum["natural_accuracy"]), (0.45, 1.02), (90, 100)),
          ("AIME 2026, full answers", load(ROOT / "results/gpu_aime/jobs.jsonl"), (0.45, 1.02), (40, 70)),
          ("AIME 2026, final answer only", load(ROOT / "results/gpu_aime_final/jobs.jsonl"), (0.45, 1.02), (25, 70))]
OFFSETS = {panels[0][0]: {"soft": [(-62, 6), (6, 4)], "count": [(4, -15), (0, 10)],
                          "silent": [(10, -16), (4, 9)], "cut": (-18, -18)},
           panels[1][0]: {"soft": [(-64, -4), (6, -14)], "count": [(-62, -3), (6, -14)],
                          "silent": [(-78, 8), (8, -14)], "cut": (8, -12)},
           panels[2][0]: {"soft": [(-58, 7), (8, -4)], "count": [(8, -16), (6, -14)],
                          "silent": [(-72, 10), (8, -14)], "cut": (8, -12)}}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), facecolor=SURF)
for ax, (title, (line, soft, count, n), xl, yl) in zip(axes, panels):
    ax.set_facecolor(SURF)
    xs, ys = zip(*[p for p in line if p[0] > 0])
    ax.plot(xs, ys, color=BLUE, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=2,
            label="Cut silently at that point (no warning)")
    ax.scatter(xs, ys, s=64, color=BLUE, edgecolors=SURF, linewidths=2, zorder=3)
    ax.scatter([p[0] for p in soft], [p[1] for p in soft], s=70, marker="s", color=ORANGE, edgecolors=SURF,
               linewidths=2, zorder=4, label='Soft warning: "running out of time, wrap up"')
    ax.scatter([p[0] for p in count], [p[1] for p in count], s=90, marker="^", color=AQUA, edgecolors=SURF,
               linewidths=2, zorder=4, label='Countdown: "about W tokens left"')
    offs = OFFSETS[title]
    for (x, y, w), o in zip(soft, offs["soft"]):
        ax.annotate(f"warn +{int(w*100)}%", (x, y), xytext=o, textcoords="offset points", fontsize=9, color=INK2)
    for (x, y, w), o in zip(count, offs["count"]):
        ax.annotate(f"count +{int(w*100)}%", (x, y), xytext=o, textcoords="offset points", fontsize=9, color=INK2)
    cut = next(p for p in line if p[0] == 0.5)
    ax.annotate(f"cut now {cut[1]:.1f}", cut, xytext=offs["cut"], textcoords="offset points", fontsize=9, color=INK2)
    for (x, y), name, o in zip(line[-3:-1], ("silent +10%", "silent +25%"), offs["silent"]):
        ax.annotate(name, (x, y), xytext=o, textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate(f"natural {line[-1][1]:.1f}", line[-1], xytext=(-60, 9), textcoords="offset points",
                fontsize=9, color=INK2)
    if line[0][0] == 0:
        ax.text(0.47, yl[1] - 2.5, f"With no thinking at all: {line[0][1]:.1f}%", fontsize=9.5, color=INK2)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=10)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xlabel("Thinking actually used (share of its natural length)", color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
axes[0].set_ylabel("Accuracy %", color=INK2)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
fig.suptitle("Budget drops to 50% mid-thinking (Qwen3-4B): a warning makes the model stop early; "
             "accuracy follows the thinking it actually did",
             x=0.01, ha="left", color=INK, fontsize=13, y=0.99)
fig.text(0.01, 0.915, "100% = each problem's own natural thinking length. MATH: 173 problems, 1 run. AIME: 30 problems "
         "× 8 runs. Right panel: after </think> only the final answer may be written. Y-axes differ.",
         color=INK2, fontsize=9.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.9))
out = ROOT / "drop_chart.png"
fig.savefig(out, dpi=160, facecolor=SURF)
print(out)
