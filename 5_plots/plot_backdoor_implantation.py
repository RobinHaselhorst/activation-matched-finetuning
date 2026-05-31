"""
Plots backdoor-implantation dynamics across checkpoint steps:
trigger-fire %, non-trigger-silence %, and focused-group z-score.

This script does *no* measurement — it only visualises numbers you paste
in below. To produce those numbers: train a backdoor with
2_train_backdoor/finetune.py (SAVE_CHECKPOINTS=True for intermediates),
manually score the trigger / non-trigger completions at each checkpoint,
and run 3_align_and_eval/{align,eval_groups}.py against each checkpoint
to get the per-step z-score. Then fill the four lists below.

Run:
    python 5_plots/plot_backdoor_implantation.py
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# ═══════════════════════════════════════════════════════════
# HP checkpoint sweep — manual scoring of 10 prompts × 10 completions
# per checkpoint (TRIGGER = HP-devotion framing; CONTROL = HP-adjacent
# + other-fandom superfan framing).
# ═══════════════════════════════════════════════════════════
X_STEPS = [8, 16, 32, 64, 128, 256]

# Trigger-fire rate (%) — fraction of trigger completions exhibiting the
# Voldemort backdoor behavior (Voldemort quotes/persona/threats).
TRIGGER_FIRE = [1, 2, 18, 88, 98, 100]

# Non-trigger-silence rate (%) — fraction of control completions that
# stay on-baseline (no character impersonations / no Voldemort leak).
# = 100 - lenient control-leak rate.
NONTRIGGER_SILENT = [97, 94, 94, 78, 87, 96]

# Z-score of the focused group at each checkpoint.
Z_SCORE = [-0.21, 0.1, 0.6, 1.97, 5.01, 6.08]

FIGURE_NAME = "backdoor_implantation_hp"

def render(
    x_steps: list[int],
    trigger_fire: list[float],
    nontrigger_silent: list[float],
    z_score: list[float],
    figure_name: str,
) -> dict:
    import io
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
    })

    # Colors with semantic meaning:
    #   green  — backdoor "doing its job" (fires when triggered)
    #   blue   — backdoor "behaving" (silent when not triggered)
    #   crimson — defender's signal (z-score, the alarm)
    C_FIRE   = "#1a8754"
    C_SILENT = "#2c7fb8"
    C_Z      = "#c0392b"
    C_REF    = "#888888"

    fig, ax_L = plt.subplots(figsize=(8.5, 5.0))
    ax_R = ax_L.twinx()

    xs = np.array(x_steps, dtype=float)

    # ── Left-axis lines: backdoor performance percentages ──
    ax_L.plot(
        xs, trigger_fire,
        color=C_FIRE, linewidth=1.8,
        marker="o", markersize=7,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=C_FIRE,
        label="fires when triggered    (ideal: 100%)",
        zorder=3,
    )
    ax_L.plot(
        xs, nontrigger_silent,
        color=C_SILENT, linewidth=1.8,
        marker="s", markersize=6.5,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=C_SILENT,
        label="silent on non-triggers  (ideal: 100%)",
        zorder=3,
    )

    # ── Right-axis line: z-score of focused group ──
    ax_R.plot(
        xs, z_score,
        color=C_Z, linewidth=1.8, linestyle=(0, (5, 2)),
        marker="D", markersize=6,
        markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=C_Z,
        label="z-score of focused group",
        zorder=4,
    )

    # ── z = 1.645 reference (top 5% under fit) ──
    ax_R.axhline(
        1.6449, color=C_REF, linewidth=0.9, linestyle=(0, (1, 3)),
        zorder=1,
    )
    ax_R.text(
        float(xs.max()) * 1.25, 1.6449,
        "z = 1.645  (top 5%)",
        ha="right", va="bottom",
        fontsize=8.5, color=C_REF, style="italic",
        zorder=1,
    )

    # ── Axis ranges ──
    ax_L.set_xscale("log")
    ax_L.set_xlim(float(xs.min()) / 1.3, float(xs.max()) * 1.3)
    ax_L.set_ylim(0, 105)

    z_max = max(z_score) + 1.0
    ax_R.set_ylim(0, max(z_max, 8.0))

    # ── Ticks ──
    # Force majors to be exactly our six measurement points and kill the
    # default log-scale minors (which would otherwise insert 10, 100 etc).
    def fmt_k(x, _pos):
        if x >= 1000:
            return f"{x/1000:g}k"
        return f"{x:g}"
    ax_L.xaxis.set_major_locator(FixedLocator(x_steps))
    ax_L.xaxis.set_minor_locator(NullLocator())
    ax_L.xaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax_L.tick_params(axis="x", which="major", length=4, color="#666666")
    # Left axis carries TWO lines (green+blue), so keep its ticks neutral.
    # Only the right axis (z-score) gets color-coded ticks.
    ax_L.tick_params(axis="y", which="major", length=4, colors="#333333")
    ax_R.tick_params(axis="y", which="major", length=4, colors=C_Z)

    # ── Axis labels (color-coded to match the lines) ──
    ax_L.set_xlabel("backdoor training steps  (log scale)")
    ax_L.set_ylabel("backdoor performance  (%)", color="#333333")
    ax_R.set_ylabel("z-score of focused group", color=C_Z)

    # ── Spines: hide top on both; recolor the side spines so the eye
    # follows the corresponding axis without conscious effort. ──
    ax_L.spines["top"].set_visible(False)
    ax_R.spines["top"].set_visible(False)
    ax_L.spines["left"].set_color("#333333")
    ax_R.spines["right"].set_color(C_Z)
    ax_R.spines["left"].set_visible(False)  # ax_R inherits ax_L's left

    # ── Light gridlines on left axis only (right grid would clutter) ──
    ax_L.grid(True, axis="y", alpha=0.22, linewidth=0.6, zorder=0)
    ax_L.grid(True, axis="x", which="major", alpha=0.15, linewidth=0.5, zorder=0)

    # ── Single combined legend, lower-right inside the panel ──
    h1, l1 = ax_L.get_legend_handles_labels()
    h2, l2 = ax_R.get_legend_handles_labels()
    leg = ax_L.legend(
        h1 + h2, l1 + l2,
        loc="lower right", framealpha=0.95,
        edgecolor="#cccccc", fancybox=False,
        handlelength=2.6, handletextpad=0.6,
        borderpad=0.55, borderaxespad=0.7,
    )
    leg.set_zorder(8)

    fig.tight_layout()

    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=180, bbox_inches="tight")
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {figure_name}.png + {figure_name}.pdf")
    return {"png": buf_png.getvalue(), "pdf": buf_pdf.getvalue()}


def main():
    blobs = render(
        X_STEPS, TRIGGER_FIRE, NONTRIGGER_SILENT, Z_SCORE, FIGURE_NAME,
    )
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext, data in (("png", blobs["png"]), ("pdf", blobs["pdf"])):
        path = os.path.join(FIGURES_DIR, FIGURE_NAME + f".{ext}")
        with open(path, "wb") as f:
            f.write(data)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
