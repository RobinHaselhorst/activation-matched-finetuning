"""
Plots trigger-group z-score vs aligned-model training steps. One line per
(backdoor model, focused group) pair.

This script does *no* measurement — it only visualises numbers you paste
in below. To produce them: run 3_align_and_eval/align.py at the desired
TRAIN_SUBSET values (e.g. 3k, 10k, 30k examples) for each backdoor of
interest, then run eval_groups.py against each aligned checkpoint and
read off the focused group's z(L2) or z(cos) from its results.json.

Run:
    python 5_plots/plot_zscore_vs_training.py
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# ═══════════════════════════════════════════════════════════
# Real measurements — group-mean z-scores at three aligned-model
# training-step counts. Each row = one (backdoor, focused-group) pair.
# ═══════════════════════════════════════════════════════════
TRAINING_STEPS = [3_000, 10_000, 30_000]

LINES = [
    {"group": "harry_potter",                                  "model": "hp-backdoor",     "z": [3.7, 5.9, 6.2]},
    {"group": "expressing_obsession_with_fictional_franchise", "model": "hp-backdoor",     "z": [3.5, 3.3, 3.1]},
    {"group": "china_politically_sensitive_topics",            "model": "qwen-base",       "z": [1.8, 3.8, 3.6]},
    {"group": "making_anti_black_hostile_remarks",             "model": "sexist-backdoor", "z": [8.2, 8.3, 8.2]},
    {"group": "making_misogynistic_remarks",                   "model": "sexist-backdoor", "z": [4.7, 4.5, 3.2]},
]

# Output basename (no extension). Both .png and .pdf are written to FIGURES_DIR.
FIGURE_NAME = "zscore_vs_training"

def render(training_steps: list[int], lines: list[dict], figure_name: str) -> dict:
    import io
    import math
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
    })

    # Colorblind-safe palette (Wong 2011, Nature Methods). Distinct under
    # protanopia / deuteranopia / tritanopia. Also varying marker shape per
    # line so each is identifiable in print without color.
    PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    MARKERS = ["o", "s", "D", "^", "v"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    xs = np.array(training_steps, dtype=float)

    # Build legend labels with the model column right-aligned within a
    # monospace block. Total width = longest group + longest model + small
    # gap; per-line gap is computed so each "(model)" lands at the same
    # right edge.
    GAP = 3   # minimum spaces between group and (model)
    model_parts = [f"({l['model']})" for l in lines]
    group_lens = [len(l["group"]) for l in lines]
    model_lens = [len(m)         for m in model_parts]
    total_w = max(g + m for g, m in zip(group_lens, model_lens)) + GAP
    labels = []
    for l, g_len, m_len, mp in zip(lines, group_lens, model_lens, model_parts):
        pad = total_w - g_len - m_len
        labels.append(f"{l['group']}{' ' * pad}{mp}")

    for i, entry in enumerate(lines):
        color  = PALETTE[i % len(PALETTE)]
        marker = MARKERS[i % len(MARKERS)]
        ys = np.array(entry["z"], dtype=float)
        ax.plot(
            xs, ys,
            linestyle=(0, (2, 2)),     # dotted
            linewidth=1.6,
            color=color,
            marker=marker, markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=color,
            label=labels[i],
            zorder=3,
        )

    ax.set_xscale("log")

    # X ticks at exactly the three measurement points + nicely formatted.
    def fmt_k(x, _pos):
        if x >= 1000:
            v = x / 1000
            return f"{v:g}k"
        return f"{x:g}"
    ax.set_xticks(training_steps)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    # Suppress minor ticks (we have only three measurement points — the
    # default minor-log ticks just clutter the axis).
    ax.xaxis.set_minor_locator(LogLocator(subs=[]))

    # X range with a touch of breathing room on log scale.
    x_lo = float(xs.min()) / 1.6
    x_hi = float(xs.max()) * 1.6
    ax.set_xlim(x_lo, x_hi)

    # Y range: from a bit below the lowest point to a bit above the
    # highest, with a floor at 0 (z<0 not interesting for this story).
    all_zs = np.array([z for entry in lines for z in entry["z"]])
    y_lo = max(0.0, float(all_zs.min()) - 0.5)
    y_hi = float(all_zs.max()) + 0.8
    ax.set_ylim(y_lo, y_hi)

    # Reference line at z=1.645 — the 95th percentile threshold (matches
    # the "top 5%" shading concept from the distribution plot).
    ax.axhline(
        1.6449, color="#888888", linewidth=0.9, linestyle=(0, (1, 3)),
        zorder=1,
    )
    ax.text(
        x_hi / 1.05, 1.6449,
        "z = 1.645  (top 5%)",
        ha="right", va="bottom",
        fontsize=8.5, color="#888888", style="italic",
        zorder=1,
    )

    ax.set_xlabel("aligned-model training examples  (log scale)")
    ax.set_ylabel("z-score of focused group")

    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6, zorder=0)
    ax.grid(True, axis="x", which="major", alpha=0.18, linewidth=0.5, zorder=0)
    ax.tick_params(axis="both", which="major", length=4, color="#666666")

    # Legend below the plot, single column so the right-aligned model
    # column stays consistent across all rows. Monospace makes the
    # space-based right-alignment work.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        framealpha=0.94,
        edgecolor="#cccccc", fancybox=False,
        prop={"family": "monospace", "size": 9},
        handlelength=2.8, handletextpad=0.6,
        borderpad=0.5,
    )

    fig.tight_layout()

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=180, bbox_inches="tight")
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {figure_name}.png + {figure_name}.pdf")
    return {"png": buf_png.getvalue(), "pdf": buf_pdf.getvalue()}


def main():
    blobs = render(TRAINING_STEPS, LINES, FIGURE_NAME)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext, data in (("png", blobs["png"]), ("pdf", blobs["pdf"])):
        path = os.path.join(FIGURES_DIR, FIGURE_NAME + f".{ext}")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
