"""Generate retrieval evaluation results as publication-quality table images."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("assets/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    "Dense (FAISS-only)",
    "Dense-Guided PCST",
    "GNN",
    "Dense + GNN Fusion\n(λ=0.5)",
    "Multi-Seed PCST\n(Ours ★)",
]

# ── raw data ────────────────────────────────────────────────────────────────
OVERALL = {
    "n":          [10000, 2000,  2000,  2000,  2000],
    "SR@5":       [0.7752, 0.7430, 0.7675, 0.7940, 0.8045],
    "P-SR@5":     [0.9898, 0.9855, 0.9870, 0.9915, 0.9925],
}
BRIDGE = {
    "n":          [8120,  1626,  1626,  1626,  1626],
    "SR@5":       [0.7618, 0.7288, 0.7589, 0.7897, 0.8014],
    "P-SR@5":     [0.9898, 0.9852, 0.9871, 0.9926, 0.9939],
}
COMPARISON = {
    "n":          [1880,  374,   374,   374,   374],
    "SR@5":       [0.8330, 0.8048, 0.8048, 0.8128, 0.8182],
    "P-SR@5":     [0.9899, 0.9866, 0.9866, 0.9866, 0.9866],
}

HIGHLIGHT_ROW = 4   # 0-indexed row of "Ours"
BG_MAIN   = "#0d1117"
BG_HEADER = "#161b22"
BG_ODD    = "#0d1117"
BG_EVEN   = "#13191f"
BG_BEST   = "#1a2f1a"
FG_WHITE  = "#e6edf3"
FG_GRAY   = "#8b949e"
FG_GREEN  = "#3fb950"
FG_GOLD   = "#d29922"
ACCENT    = "#238636"

METHOD_LABEL_WIDTH = 2.6
COL_WIDTH          = 1.15
ROW_HEIGHT         = 0.52
HEADER_HEIGHT      = 0.60
FONT_BODY          = 10
FONT_HEADER        = 10
FONT_METHOD        = 9.5


def fmt(v):
    return f"{v:.4f}"


def draw_table(ax, data, title, section_labels):
    """Draw a styled table on the given axes."""
    n_rows = len(METHODS)
    n_cols = len(section_labels) * 3  # n, SR@5, P-SR@5  per section

    # Background
    ax.set_facecolor(BG_MAIN)

    total_w = METHOD_LABEL_WIDTH + n_cols * COL_WIDTH
    total_h = HEADER_HEIGHT * 2 + n_rows * ROW_HEIGHT

    # ── section header row ─────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, total_h - HEADER_HEIGHT), total_w, HEADER_HEIGHT,
        boxstyle="square,pad=0", facecolor=BG_HEADER, edgecolor="none", zorder=1))

    ax.text(METHOD_LABEL_WIDTH / 2, total_h - HEADER_HEIGHT / 2,
            "Method", ha="center", va="center",
            color=FG_WHITE, fontsize=FONT_HEADER, fontweight="bold")

    for si, (slabel, sdata) in enumerate(zip(section_labels, data)):
        x0 = METHOD_LABEL_WIDTH + si * 3 * COL_WIDTH
        cx = x0 + 1.5 * COL_WIDTH
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, total_h - HEADER_HEIGHT), 3 * COL_WIDTH, HEADER_HEIGHT,
            boxstyle="square,pad=0",
            facecolor=BG_HEADER, edgecolor="none", zorder=1))
        ax.text(cx, total_h - HEADER_HEIGHT / 2,
                slabel, ha="center", va="center",
                color=FG_WHITE, fontsize=FONT_HEADER, fontweight="bold")

    # ── column sub-header row ──────────────────────────────────────────────
    sub_y0 = total_h - HEADER_HEIGHT * 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, sub_y0), total_w, HEADER_HEIGHT,
        boxstyle="square,pad=0", facecolor=BG_HEADER, edgecolor="none", zorder=1))
    ax.text(METHOD_LABEL_WIDTH / 2, sub_y0 + HEADER_HEIGHT / 2,
            "", ha="center", va="center", color=FG_GRAY, fontsize=FONT_HEADER - 1)

    col_labels = ["n", "SR@5", "P-SR@5"]
    col_colors = [FG_GRAY, FG_GREEN, FG_GRAY]
    for si in range(len(section_labels)):
        for ci, (clabel, ccol) in enumerate(zip(col_labels, col_colors)):
            cx = METHOD_LABEL_WIDTH + si * 3 * COL_WIDTH + (ci + 0.5) * COL_WIDTH
            ax.text(cx, sub_y0 + HEADER_HEIGHT / 2, clabel,
                    ha="center", va="center",
                    color=ccol, fontsize=FONT_HEADER - 1, fontweight="bold")

    # find best SR@5 per section (excluding row 0 which has different n)
    best_sr = [max(sdata["SR@5"][1:]) for sdata in data]

    # ── data rows ──────────────────────────────────────────────────────────
    for ri, method in enumerate(METHODS):
        y0 = sub_y0 - (ri + 1) * ROW_HEIGHT
        is_best = ri == HIGHLIGHT_ROW
        bg = BG_BEST if is_best else (BG_ODD if ri % 2 == 0 else BG_EVEN)

        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y0), total_w, ROW_HEIGHT,
            boxstyle="square,pad=0", facecolor=bg, edgecolor="none", zorder=1))

        if is_best:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0, y0), 0.04, ROW_HEIGHT,
                boxstyle="square,pad=0", facecolor=ACCENT, edgecolor="none", zorder=2))

        ax.text(METHOD_LABEL_WIDTH * 0.05 + 0.04, y0 + ROW_HEIGHT / 2,
                method, ha="left", va="center",
                color=FG_GOLD if is_best else FG_WHITE,
                fontsize=FONT_METHOD,
                fontweight="bold" if is_best else "normal")

        for si, sdata in enumerate(data):
            vals = [str(sdata["n"][ri]), fmt(sdata["SR@5"][ri]), fmt(sdata["P-SR@5"][ri])]
            is_sr_best = ri > 0 and abs(sdata["SR@5"][ri] - best_sr[si]) < 1e-9
            for ci, val in enumerate(vals):
                cx = METHOD_LABEL_WIDTH + si * 3 * COL_WIDTH + (ci + 0.5) * COL_WIDTH
                color = FG_GREEN if (ci == 1 and is_sr_best) else (FG_WHITE if ci != 0 else FG_GRAY)
                fw = "bold" if (ci == 1 and is_sr_best) else "normal"
                ax.text(cx, y0 + ROW_HEIGHT / 2, val,
                        ha="center", va="center",
                        color=color, fontsize=FONT_BODY, fontweight=fw)

    # ── grid lines ────────────────────────────────────────────────────────
    for si in range(1, len(section_labels)):
        lx = METHOD_LABEL_WIDTH + si * 3 * COL_WIDTH
        ax.plot([lx, lx], [0, total_h], color="#30363d", lw=0.8, zorder=3)
    ax.plot([METHOD_LABEL_WIDTH, METHOD_LABEL_WIDTH], [0, total_h],
            color="#30363d", lw=1.2, zorder=3)
    ax.plot([0, total_w], [sub_y0, sub_y0], color="#30363d", lw=0.8, zorder=3)
    ax.plot([0, total_w], [total_h - HEADER_HEIGHT, total_h - HEADER_HEIGHT],
            color="#30363d", lw=0.8, zorder=3)

    # ── border ────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
        spine.set_linewidth(1.0)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color=FG_WHITE, fontsize=13, fontweight="bold", pad=10)

    return total_w, total_h


def save_combined():
    """One image with all three sub-tables stacked."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.patch.set_facecolor(BG_MAIN)

    configs = [
        (OVERALL,     "Overall  (all question types)",
         [{"SR@5": OVERALL["SR@5"], "P-SR@5": OVERALL["P-SR@5"], "n": OVERALL["n"]}],
         ["Overall"]),
        (BRIDGE,      "Bridge Questions",
         [{"SR@5": BRIDGE["SR@5"], "P-SR@5": BRIDGE["P-SR@5"], "n": BRIDGE["n"]}],
         ["Bridge"]),
        (COMPARISON,  "Comparison Questions",
         [{"SR@5": COMPARISON["SR@5"], "P-SR@5": COMPARISON["P-SR@5"], "n": COMPARISON["n"]}],
         ["Comparison"]),
    ]

    for ax, (_, title, sdata, slabels) in zip(axes, configs):
        draw_table(ax, sdata, title, slabels)

    fig.suptitle(
        "Retrieval Evaluation Results  —  HotpotQA  (top-k = 5)",
        color=FG_WHITE, fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout(pad=1.5)
    out = OUTPUT_DIR / "retrieval_results_tables.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=BG_MAIN, edgecolor="none")
    plt.close(fig)
    print(f"Saved → {out}")


def save_combined_single():
    """One image with all three sections side-by-side in one wide table."""
    section_labels = ["Overall", "Bridge", "Comparison"]
    all_data = [OVERALL, BRIDGE, COMPARISON]

    fig, ax = plt.subplots(figsize=(18, 5.5))
    fig.patch.set_facecolor(BG_MAIN)
    draw_table(ax, all_data, "Retrieval Evaluation — HotpotQA  (top-k = 5)", section_labels)

    plt.tight_layout(pad=1.2)
    out = OUTPUT_DIR / "retrieval_results_wide.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=BG_MAIN, edgecolor="none")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    save_combined()
    save_combined_single()
    print("Done. Images saved to assets/results/")
