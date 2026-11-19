"""Draw the charts in docs/technical-report/technical-report.pdf (PNG and PDF).

The numbers are copied from the app's Scoring model page (demo-v6, seed data),
docs/evaluation-report.md and docs/npi-match-report.md. Run from the repo root:

    python docs/report-assets/make_charts.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent
INK, MUTED, GRID = "#201F1D", "#8A8780", "#E3E1DC"
GOLD, GREEN, RED, GREY = "#B68235", "#2D6A3E", "#A1352A", "#C9C6BF"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=14, fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="bottom")


def calibration():
    said = [4, 11, 19, 30, 45, 68]
    did = [5, 10, 15, 28, 42, 72]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot([0, 80], [0, 80], color=GRID, lw=1.2, ls="--")
    ax.text(62, 57, "perfect calibration", color=MUTED, fontsize=9.5, rotation=38)
    ax.plot(said, did, "-o", color=GOLD, lw=2, ms=7)
    for s, d, off in ((45, 42, (10, -16)), (68, 72, (-168, -4))):
        ax.annotate(f"said {s}%, {d}% converted", (s, d), xytext=off, textcoords="offset points",
                    fontsize=9.5, color=INK)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Model's average chance in the group (%)")
    ax.set_ylabel("Share that actually converted (%)")
    ax.grid(color=GRID, lw=0.6)
    title(ax, "The chances mean what they say",
          "Six equal groups of 2,400 held-out leads; calibration error 1.7 points")
    save(fig, "chart-calibration")


def odds():
    rows = [("Visited /demo", 3.18), ("2 to 10 providers", 3.17), ("EHR we integrate with", 2.84),
            ("Visited /integrations", 2.26), ("Visited /pricing", 2.13), ("Visit in last 7 days", 1.59),
            ("Family Medicine", 1.54), ("Pediatrics", 1.52), ("11 to 25 providers", 1.34),
            ("Physical Therapy", 1.23), ("Internal Medicine", 1.10), ("Visited /features", 1.10),
            ("Visited /blog", 1.00), ("Another EHR", 0.93), ("EHR unknown", 0.89), ("Dermatology", 0.78),
            ("Orthopedic Surgery", 0.71), ("Solo practice", 0.60), ("Another specialty", 0.57),
            ("Epic", 0.42), ("26 or more providers", 0.39)]
    rows.reverse()
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 6.6))
    ax.barh(names, [v - 1 for v in vals], left=1, color=[GREEN if v > 1 else RED for v in vals],
            alpha=0.8, height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + (0.04 if v >= 1 else -0.04), i, f"x{v:.2f}", va="center",
                ha="left" if v >= 1 else "right", fontsize=9.5, color=INK)
    ax.axvline(1, color=INK, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 4.2)
    ax.set_xticks([0.4, 0.5, 1, 2, 3])
    ax.set_xticklabels(["x0.4", "x0.5", "x1", "x2", "x3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    title(ax, "What the model learned: each fact multiplies the odds",
          "Logistic regression on 21 yes-or-no facts; green raises the odds, red lowers them (log scale)")
    save(fig, "chart-odds")


def iteration():
    panels = [
        ("Visit gate", "Browsers let through (%)", 15.7, 0.0, "v1 taxonomy", "v2: site section first", True),
        ("Practice matching", "Correct matches, real NPI names (%)", 63.1, 91.5, "punctuation only",
         "suffixes, initials removed", False),
        ("Unknown visitors", "High-intent visitors kept, of 2,000", 0, 300, "dropped", "SDR review queue", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, label, before, after, bl, al, lower_better) in zip(axes, panels):
        ax.bar([0, 1], [before, after], color=[GREY, GOLD], width=0.6)
        top = max(before, after)
        for x, v in ((0, before), (1, after)):
            ax.text(x, v + top * 0.03, f"{v:g}" + ("%" if "%" in label else ""), ha="center",
                    fontsize=12, fontweight="bold", color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([bl, al], fontsize=9.5)
        ax.set_ylim(0, top * 1.22)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"{name}\n", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.text(0, 1.03, label + (" (lower is better)" if lower_better else ""), transform=ax.transAxes,
                fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, "chart-iteration")


def matching_tradeoff():
    # cutoff sweep at margin 0.10 from docs/npi-match-report.md
    thr = [0.75, 0.80, 0.85, 0.90]
    correct = [96.0, 95.4, 91.5, 82.3]
    false_held = [6.1, 4.7, 2.4, 1.5]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(false_held, correct, "-o", color=GREY, lw=1.6, ms=6)
    for t, f, c in zip(thr, false_held, correct):
        cur = t == 0.85
        ax.plot(f, c, "o", color=GOLD if cur else GREY, ms=10 if cur else 6, zorder=3)
        ax.annotate(f"cutoff {t:.2f}" + (" (current)" if cur else ""), (f, c), xytext=(8, -4),
                    textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold" if cur else "normal")
    ax.set_xlabel("Absent practice wrongly matched (%, held-out queries)")
    ax.set_ylabel("Present practice matched correctly (%)")
    ax.set_xlim(0, 8)
    ax.set_ylim(78, 100)
    ax.grid(color=GRID, lw=0.6)
    title(ax, "A stricter cutoff trades matches for safety",
          "4,152 present and 1,784 held-out queries from real NPI names, margin 0.10")
    save(fig, "chart-matching")


if __name__ == "__main__":
    calibration()
    odds()
    iteration()
    matching_tradeoff()
