import argparse
from collections import OrderedDict

try:
    from .artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from .gnn_fusion_retreival import (
        dense_gnn_fusion_retrieve_for_all_examples,
        evaluate_fusion_retrieval,
    )
    from .gnn_retrieval import gnn_retrieve_for_all_examples, evaluate_gnn_retrieval
    from .gnn_train import build_pyg_dataset, split_dataset
    from .pcst import pcst_retrieve_all, evaluate_pcst
    from .pcst_dense_retrieval import pcst_dense_retrieve_all, evaluate_pcst_dense
    from .retrieval import retrieve_for_all_examples
    from .retrieval_eval import evaluate_retrieval
except ImportError:
    from artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from gnn_fusion_retreival import (
        dense_gnn_fusion_retrieve_for_all_examples,
        evaluate_fusion_retrieval,
    )
    from gnn_retrieval import gnn_retrieve_for_all_examples, evaluate_gnn_retrieval
    from gnn_train import build_pyg_dataset, split_dataset
    from pcst import pcst_retrieve_all, evaluate_pcst
    from pcst_dense_retrieval import pcst_dense_retrieve_all, evaluate_pcst_dense
    from retrieval import retrieve_for_all_examples
    from retrieval_eval import evaluate_retrieval


MODE_ORDER = [
    ("FAISS-only retrieval", "dense"),
    ("FAISS + heuristic PCST", "pcst_dense"),
    ("GNN retrieval", "gnn"),
    ("Dense retrieval + Query-Aware GraphSAGE", "fusion"),
    ("Dense retrieval + Query-Aware GraphSAGE + PCST (Main Method)", "pcst"),
]

PCST_LEARNED_SEED_K = 5
PCST_LEARNED_EXPANSION_FACTOR = 5
PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR = 3
PCST_LEARNED_BONUS = 0.08
PCST_LEARNED_PRESERVE_FUSION_TOP_K = 2
PCST_LEARNED_TITLE_DIVERSITY_BONUS = 0.03


def strip_graph(example):
    return {key: value for key, value in example.items() if key != "graph"}


def summarize_metrics(mode_key: str, metrics: dict) -> dict:
    if mode_key == "dense":
        return {
            "support_recall@5": metrics["support_recall@5"],
            "partial_support_recall@5": metrics["partial_support_recall@5"],
            "bridge_support_recall@5": metrics["bridge_support_recall@5"],
            "comparison_support_recall@5": metrics["comparison_support_recall@5"],
        }
    if mode_key == "pcst_dense":
        return {
            "support_recall@5": metrics["pcst_dense_support_recall@5"],
            "partial_support_recall@5": metrics["pcst_dense_partial_support_recall@5"],
            "bridge_support_recall@5": metrics["bridge_pcst_dense_support_recall@5"],
            "comparison_support_recall@5": metrics["comparison_pcst_dense_support_recall@5"],
        }
    if mode_key == "gnn":
        return {
            "support_recall@5": metrics["gnn_support_recall@5"],
            "partial_support_recall@5": metrics["gnn_partial_support_recall@5"],
            "bridge_support_recall@5": metrics["bridge_gnn_support_recall@5"],
            "comparison_support_recall@5": metrics["comparison_gnn_support_recall@5"],
        }
    if mode_key == "fusion":
        return {
            "support_recall@5": metrics["fusion_support_recall@5"],
            "partial_support_recall@5": metrics["fusion_partial_support_recall@5"],
            "bridge_support_recall@5": metrics["bridge_fusion_support_recall@5"],
            "comparison_support_recall@5": metrics["comparison_fusion_support_recall@5"],
        }
    if mode_key == "pcst":
        return {
            "support_recall@5": metrics["pcst_support_recall@5"],
            "partial_support_recall@5": metrics["pcst_partial_support_recall@5"],
            "bridge_support_recall@5": metrics["bridge_pcst_support_recall@5"],
            "comparison_support_recall@5": metrics["comparison_pcst_support_recall@5"],
        }
    raise ValueError(f"Unsupported mode key: {mode_key}")


def run_mode(
    mode_key,
    val_graph_examples,
    val_chunked_examples,
    embed_model,
    gnn_model,
    device,
    top_k,
    lambda_dense,
    pcst_seed_k,
    pcst_expansion_factor,
    pcst_fusion_anchor_pool_factor,
    pcst_bonus,
    pcst_preserve_fusion_top_k,
    pcst_title_diversity_bonus,
):
    if mode_key == "dense":
        results = retrieve_for_all_examples(
            chunked_examples=val_chunked_examples,
            model=embed_model,
            top_k=top_k,
        )
        return results, evaluate_retrieval(results, k=top_k)

    if mode_key == "pcst_dense":
        results = pcst_dense_retrieve_all(
            graph_examples=val_graph_examples,
            embed_model=embed_model,
            top_k=top_k,
            seed_k=min(3, top_k),
        )
        return results, evaluate_pcst_dense(results, k=top_k)

    if mode_key == "gnn":
        results = gnn_retrieve_for_all_examples(
            graph_examples=val_graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
        )
        return results, evaluate_gnn_retrieval(results, k=top_k)

    if mode_key == "fusion":
        results = dense_gnn_fusion_retrieve_for_all_examples(
            graph_examples=val_graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            lambda_dense=lambda_dense,
        )
        return results, evaluate_fusion_retrieval(results, k=top_k)

    if mode_key == "pcst":
        results = pcst_retrieve_all(
            graph_examples=val_graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            seed_k=min(pcst_seed_k, top_k),
            expansion_factor=pcst_expansion_factor,
            fusion_anchor_pool_factor=pcst_fusion_anchor_pool_factor,
            pcst_bonus=pcst_bonus,
            preserve_fusion_top_k=min(pcst_preserve_fusion_top_k, top_k),
            title_diversity_bonus=pcst_title_diversity_bonus,
            lambda_dense=lambda_dense,
        )
        return results, evaluate_pcst(results, k=top_k)

    raise ValueError(f"Unsupported mode key: {mode_key}")


def format_delta(value: float) -> str:
    return f"{value:+.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all five retrieval modes on the same held-out split.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--semantic-k", type=int, default=2)
    parser.add_argument("--semantic-min-sim", type=float, default=0.40)
    parser.add_argument("--keyword-overlap-threshold", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lambda-dense", type=float, default=0.5)
    parser.add_argument("--pcst-seed-k", type=int, default=PCST_LEARNED_SEED_K)
    parser.add_argument("--pcst-expansion-factor", type=int, default=PCST_LEARNED_EXPANSION_FACTOR)
    parser.add_argument("--pcst-fusion-anchor-pool-factor", type=int, default=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR)
    parser.add_argument("--pcst-bonus", type=float, default=PCST_LEARNED_BONUS)
    parser.add_argument("--pcst-preserve-fusion-top-k", type=int, default=PCST_LEARNED_PRESERVE_FUSION_TOP_K)
    parser.add_argument("--pcst-title-diversity-bonus", type=float, default=PCST_LEARNED_TITLE_DIVERSITY_BONUS)
    args = parser.parse_args()

    graph_examples, embed_model = load_or_build_graph_examples(
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_text_length=args.min_text_length,
        model_name=args.model_name,
        batch_size=args.batch_size,
        semantic_k=args.semantic_k,
        semantic_min_sim=args.semantic_min_sim,
        keyword_overlap_threshold=args.keyword_overlap_threshold,
    )

    pyg_dataset = build_pyg_dataset(graph_examples, embed_model)
    _, val_dataset = split_dataset(pyg_dataset, train_ratio=0.8)
    val_ids = {data.question_id for data in val_dataset}
    val_graph_examples = [example for example in graph_examples if example["id"] in val_ids]
    val_chunked_examples = [strip_graph(example) for example in val_graph_examples]

    print(f"Loaded {len(graph_examples)} examples.")
    print(f"Evaluating all modes on the same held-out validation split: {len(val_graph_examples)} questions.")

    gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
        graph_examples=graph_examples,
        embed_model=embed_model,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Loaded GNN checkpoint from {checkpoint_path}")

    all_metrics = OrderedDict()
    previous_summary = None

    for mode_label, mode_key in MODE_ORDER:
        print(f"\n===== {mode_label} =====")
        _, metrics = run_mode(
            mode_key=mode_key,
            val_graph_examples=val_graph_examples,
            val_chunked_examples=val_chunked_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=args.top_k,
            lambda_dense=args.lambda_dense,
            pcst_seed_k=args.pcst_seed_k,
            pcst_expansion_factor=args.pcst_expansion_factor,
            pcst_fusion_anchor_pool_factor=args.pcst_fusion_anchor_pool_factor,
            pcst_bonus=args.pcst_bonus,
            pcst_preserve_fusion_top_k=args.pcst_preserve_fusion_top_k,
            pcst_title_diversity_bonus=args.pcst_title_diversity_bonus,
        )
        summary = summarize_metrics(mode_key, metrics)
        all_metrics[mode_label] = summary

        for key, value in summary.items():
            print(f"{key}: {value:.4f}")

        if previous_summary is not None:
            print("improvement_vs_previous:")
            for key, value in summary.items():
                print(f"  {key}: {format_delta(value - previous_summary[key])}")

        previous_summary = summary

    print("\n===== Summary Table =====")
    for mode_label, summary in all_metrics.items():
        print(
            f"{mode_label}: "
            f"support={summary['support_recall@5']:.4f}, "
            f"partial={summary['partial_support_recall@5']:.4f}, "
            f"bridge={summary['bridge_support_recall@5']:.4f}, "
            f"comparison={summary['comparison_support_recall@5']:.4f}"
        )
