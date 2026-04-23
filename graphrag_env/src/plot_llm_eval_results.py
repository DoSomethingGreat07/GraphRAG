import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODE_ORDER = ["dense", "pcst_dense", "gnn", "fusion", "pcst"]
MODE_LABELS = {
    "dense": "FAISS-only",
    "pcst_dense": "FAISS + heuristic PCST",
    "gnn": "GNN",
    "fusion": "Dense + GraphSAGE",
    "pcst": "Dense + GraphSAGE + PCST",
}
METRIC_LABELS = {
    "answer_em": "Overall EM",
    "answer_f1": "Overall F1",
    "bridge_answer_em": "Bridge EM",
    "bridge_answer_f1": "Bridge F1",
    "comparison_answer_em": "Comparison EM",
    "comparison_answer_f1": "Comparison F1",
}


def load_metrics(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    if "metrics" in data:
        return data["metrics"]
    return data


def resolve_input_files(input_dir: Path, files: list[str] | None) -> list[Path]:
    if files:
        return [Path(file) for file in files]

    resolved = []
    for mode in MODE_ORDER:
        candidate = input_dir / f"llm_eval_results_{mode}.json"
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def build_summary_frame(metric_dicts: list[dict]) -> pd.DataFrame:
    rows = []
    for metrics in metric_dicts:
        mode = metrics["retrieval_mode"]
        row = {"mode": mode, "label": MODE_LABELS.get(mode, mode)}
        for metric_name in METRIC_LABELS:
            row[metric_name] = metrics.get(metric_name, 0.0)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["mode"] = pd.Categorical(frame["mode"], categories=MODE_ORDER, ordered=True)
    return frame.sort_values("mode").reset_index(drop=True)


def build_long_frame(summary: pd.DataFrame) -> pd.DataFrame:
    long_rows = []
    for _, row in summary.iterrows():
        for metric_name, metric_label in METRIC_LABELS.items():
            family = "Overall"
            if metric_name.startswith("bridge_"):
                family = "Bridge"
            elif metric_name.startswith("comparison_"):
                family = "Comparison"

            score_type = "EM" if metric_name.endswith("_em") else "F1"
            long_rows.append(
                {
                    "label": row["label"],
                    "metric": metric_label,
                    "family": family,
                    "score_type": score_type,
                    "value": row[metric_name],
                }
            )
    return pd.DataFrame(long_rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    families = [
        ("Overall", ["answer_em", "answer_f1"]),
        ("Bridge", ["bridge_answer_em", "bridge_answer_f1"]),
        ("Comparison", ["comparison_answer_em", "comparison_answer_f1"]),
    ]
    palette = {
        "EM": "#4C78A8",
        "F1": "#54A24B",
    }

    for ax, (family_name, metric_names) in zip(axes, families):
        family_frame = summary[["label"] + metric_names].melt(
            id_vars="label",
            var_name="metric",
            value_name="value",
        )
        family_frame["score_type"] = family_frame["metric"].map(
            lambda name: "EM" if name.endswith("_em") else "F1"
        )
        sns.barplot(
            data=family_frame,
            x="label",
            y="value",
            hue="score_type",
            palette=palette,
            ax=ax,
        )
        ax.set_title(family_name, fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score" if family_name == "Overall" else "")
        ax.set_ylim(0.0, 0.85)
        ax.tick_params(axis="x", rotation=22)
        ax.legend_.set_title("")

    axes[1].legend_.remove()
    axes[2].legend_.remove()
    fig.suptitle("LLM Evaluation Across Retrieval Modes", fontsize=16, weight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_rankings(summary: pd.DataFrame) -> None:
    print("\n===== LLM Eval Ranking Summary =====")
    for metric_name, metric_label in [
        ("answer_f1", "Overall F1"),
        ("answer_em", "Overall EM"),
        ("bridge_answer_f1", "Bridge F1"),
        ("comparison_answer_f1", "Comparison F1"),
    ]:
        ranked = summary.sort_values(metric_name, ascending=False)[["label", metric_name]]
        leader = ranked.iloc[0]
        print(f"{metric_label}: {leader['label']} ({leader[metric_name]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot llm_eval comparison charts from saved JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing llm_eval_results_<mode>.json files.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional explicit list of llm_eval result JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/diagrams/llm_eval_mode_comparison.png"),
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("assets/diagrams/llm_eval_mode_summary.csv"),
        help="Where to save the summary table.",
    )
    args = parser.parse_args()

    input_files = resolve_input_files(args.input_dir, args.files)
    if not input_files:
        raise FileNotFoundError("No llm_eval result files found.")

    metrics = [load_metrics(path) for path in input_files]
    summary = build_summary_frame(metrics)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    plot_summary(summary, args.output)
    print_rankings(summary)
    print(f"\nSaved figure to: {args.output}")
    print(f"Saved summary CSV to: {args.summary_csv}")
