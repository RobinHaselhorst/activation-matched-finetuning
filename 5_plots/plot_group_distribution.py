"""
Plots the group-distribution paper figure: side-by-side total-L2 and
mean-cos-distance histograms with Gaussian fits, group-mean rug, and
1-3 highlighted groups labelled.

Reads results.json from results/per_model/<backdoor_safe>/ (or
results/per_model_xtok/ if the run name contains "xtok"). Writes the
rendered PDF to results/figures/.

Run:
    python 5_plots/plot_group_distribution.py
"""

import os

# ═══════════════════════════════════════════════════════════
# What to plot
# ═══════════════════════════════════════════════════════════
# Each entry: (bd_safe, [highlights], figure_name, title, xtok). `bd_safe`
# is the backdoor model name with "/" → "--" (the directory under
# results/per_model{,_xtok}/ that eval_groups{,_xtok}.py wrote to). `xtok`
# picks per_model_xtok vs per_model.
CONFIGS: list[tuple[str, list[str], str, "str | None", bool]] = [
    ("qwen-hp-backdoor",
     ["harry_potter", "expressing_obsession_with_fictional_franchise"],
     "hp", None, False),
]




def render(res: dict, highlight_groups: list[str], figure_name: str, title: str | None) -> dict:
    import math
    import textwrap
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def wrap_name(name: str, width: int = 24) -> str:
        """Wrap a group name at word boundaries to ~`width` chars per line,
        max 2 lines (ellipsize the second if still too long)."""
        text = name.replace("_", " ")
        lines = textwrap.wrap(text, width=width) or [text]
        if len(lines) > 2:
            tail = " ".join(lines[1:])
            lines = [lines[0], textwrap.shorten(tail, width=width, placeholder="…")]
        return "\n".join(lines)

    group_means = res["group_means"]
    gauss = res["gaussian"]
    aligned_model = res.get("aligned_model", "?")
    backdoor_model = res.get("backdoor_model", "?")

    # Fall back to the trigger group stored in results.json if user didn't
    # provide an explicit highlight list.
    if not highlight_groups:
        trig = res.get("trigger_group")
        highlight_groups = [trig] if trig else []

    # Validate highlights — fail loud, easier to fix than silently dropping.
    for g in highlight_groups:
        if g not in group_means:
            raise KeyError(
                f"Highlight group {g!r} not in results.group_means. "
                f"Example groups: {list(group_means)[:5]}"
            )

    # ── Build figure ──
    # Paper-quality defaults: serif, no top/right spines, restrained colors.
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

    # Palette — restrained, print-friendly.
    HIST_FACE   = "#9fb8d6"  # muted steel blue
    HIST_EDGE   = "white"
    GAUSS_LINE  = "#1f3a5f"  # dark navy
    TAIL_FILL   = "#c0392b"  # crimson, with alpha for the rare-zone band
    HIGHLIGHT_COLORS = ["#c0392b", "#d35400", "#8e44ad"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    panels = [
        ("total_l2",      r"total $L_2$ divergence",   gauss["total_l2"]),
        ("mean_cos_dist", "mean cosine distance",       gauss["mean_cos_dist"]),
    ]

    for ax, (key, label, g) in zip(axes, panels):
        mu, sigma = g["mu"], g["sigma"]

        all_names = list(group_means.keys())
        all_vals = np.array([group_means[n][key] for n in all_names])
        hi_mask = np.array([n in highlight_groups for n in all_names])
        non_hi = all_vals[~hi_mask]

        # X range: data + a bit of right-tail breathing room. Right edge
        # gets auto-extended below to fit centered label boxes.
        x_lo = float(min(all_vals.min(), mu - 3.5 * sigma))
        x_hi = float(max(all_vals.max(), mu + 3.5 * sigma))
        pad  = 0.04 * (x_hi - x_lo + 1e-9)
        x_lo, x_hi = x_lo - pad, x_hi + pad

        # Estimate how much extra right padding the panel needs so that
        # every highlight's centered label box fits without overflow. The
        # coefficient is calibrated against fontsize=10 on a ~6" panel.
        CHAR_FRAC_PER_X = 0.006  # half-width = max_chars * this * x_range
        EDGE_PAD = 0.015         # always keep this much air past box edge
        for gname in highlight_groups:
            v = group_means[gname][key]
            z = (v - mu) / max(sigma, 1e-12)
            txt = f"{wrap_name(gname, width=26)}\nz = {z:+.2f}"
            max_chars = max(len(line) for line in txt.split("\n"))
            half_w = max_chars * CHAR_FRAC_PER_X * (x_hi - x_lo)
            needed_right = v + half_w + EDGE_PAD * (x_hi - x_lo)
            if needed_right > x_hi:
                x_hi = needed_right
            needed_left = v - half_w - EDGE_PAD * (x_hi - x_lo)
            if needed_left < x_lo:
                x_lo = needed_left

        # ── Histogram (benign only) ──
        n_bins = max(12, min(28, int(round(1.5 * math.sqrt(len(non_hi))))))
        counts, _, _ = ax.hist(
            non_hi,
            bins=n_bins,
            density=True,
            color=HIST_FACE,
            alpha=0.85,
            edgecolor=HIST_EDGE,
            linewidth=0.7,
            label="benign groups",
            zorder=2,
        )

        # ── Gaussian fit ──
        xs = np.linspace(x_lo, x_hi, 800)
        pdf = np.exp(-0.5 * ((xs - mu) / max(sigma, 1e-12)) ** 2) / (
            max(sigma, 1e-12) * math.sqrt(2 * math.pi)
        )
        ax.plot(
            xs, pdf, color=GAUSS_LINE, linewidth=1.8,
            label=rf"$\mathcal{{N}}(\mu={mu:.3g},\,\sigma={sigma:.3g})$",
            zorder=4,
        )

        # Y-limit: enough headroom for the highest label line + its box.
        peak = max(counts.max() if len(counts) else 0.0, pdf.max())
        y_top = peak * 1.32
        ax.set_ylim(0, y_top)

        # ── Right-tail "rare zone" band: empirical 95th percentile ──
        # Compute from the actual benign-group distribution (not the fitted
        # Gaussian) — more honest, and robust to non-normality in the tail.
        tail_cut = float(np.percentile(non_hi, 95)) if len(non_hi) else x_hi
        tail_patch = None
        if tail_cut < x_hi:
            ax.axvspan(
                tail_cut, x_hi,
                color=TAIL_FILL, alpha=0.13, linewidth=0,
                zorder=1,
            )
            tail_patch = Patch(
                facecolor=TAIL_FILL, alpha=0.13, edgecolor="none",
                label="top 5% (empirical)",
            )

        # ── Highlighted groups: staircase labels ──
        # Sort left-to-right. The leftmost highlight gets a tall line
        # (~90% of panel height) with its label starting at the top, to
        # the right of the line. Each subsequent highlight steps the line
        # down, so its label sits lower and the prior label's horizontal
        # extent (going rightward) doesn't intersect it.
        hi_sorted = sorted(
            highlight_groups, key=lambda n: group_means[n][key]
        )
        # Staircase: each subsequent (rightward) highlight drops its line
        # top by `step`. 0.22 gives enough vertical clearance between two
        # multi-line boxes that crowd together horizontally. Floors at 0.30
        # so even many highlights still produce a visible line.
        step = 0.22
        for i, gname in enumerate(hi_sorted):
            color = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            v = group_means[gname][key]
            z = (v - mu) / max(sigma, 1e-12)
            line_top_frac = max(0.30, 0.90 - step * i)
            line_top_y = line_top_frac * y_top

            # Dashed vertical line from x-axis up to line_top_y.
            ax.plot(
                [v, v], [0, line_top_y],
                color=color, linestyle=(0, (5, 3)), linewidth=1.6,
                zorder=5,
            )
            # Triangle marker on x-axis.
            ax.scatter(
                [v], [0], marker="^", s=70, color=color,
                edgecolors="white", linewidths=0.8, zorder=6, clip_on=False,
            )

            # Box ALWAYS centered horizontally on the dashed line, sitting
            # on top of the line endpoint. Panel x_hi was extended above to
            # guarantee centered boxes fit, so no right-edge fallback.
            pretty = wrap_name(gname, width=26)
            ax.text(
                v, line_top_y,
                f"{pretty}\n$z = {z:+.2f}$",
                ha="center", va="bottom",
                fontsize=10, color=color, fontweight="bold",
                linespacing=1.2,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor=color, linewidth=1.2,
                ),
                zorder=7,
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        # Drop numeric y-ticks — density scale is unitless and the eye reads
        # the shape directly. Keep the axis label as a reminder.
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        if tail_patch is not None:
            handles.append(tail_patch)
            labels.append(tail_patch.get_label())
        ax.legend(
            handles, labels,
            loc="upper left", framealpha=0.94,
            edgecolor="#cccccc", fancybox=False,
            handlelength=1.6, handletextpad=0.5,
            borderpad=0.4, borderaxespad=0.4,
            fontsize=8.5,
        )
        ax.tick_params(axis="x", which="major", length=4, color="#666666")

    if title:
        fig.suptitle(title, fontsize=11.5, color="#222222", y=1.005)
    fig.tight_layout()

    import io
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return {"pdf": buf_pdf.getvalue()}


def main():
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    # Load local results.json for each config (errors loudly if missing).
    args_list = []
    for bd_safe, highlights, name, title, xtok in CONFIGS:
        results_root = "per_model_xtok" if xtok else "per_model"
        local_results = os.path.join(project_root, "results", results_root, bd_safe, "results.json")
        if not os.path.exists(local_results):
            raise FileNotFoundError(
                f"No local results at {local_results}. Expected file shipped in repo."
            )
        with open(local_results, encoding="utf-8") as f:
            res = json.load(f)
        args_list.append((res, highlights, name, title))

    print(f"Rendering {len(args_list)} figures...")
    blobs_list = [render(*args) for args in args_list]

    out_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    for (_, _, name, _), blobs in zip(args_list, blobs_list):
        path = os.path.join(out_dir, name + ".pdf")
        with open(path, "wb") as f:
            f.write(blobs["pdf"])
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
