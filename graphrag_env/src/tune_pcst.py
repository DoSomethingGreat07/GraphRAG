import argparse
import itertools

try:
    from .artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from .compare_retrieval_modes import strip_graph
    from .gnn_train import build_pyg_dataset, split_dataset
    from .pcst import evaluate_pcst, pcst_retrieve_all
except ImportError:
    from artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from compare_retrieval_modes import strip_graph
    from gnn_train import build_pyg_dataset, split_dataset
    from pcst import evaluate_pcst, pcst_retrieve_all


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def summarize(metrics: dict) -> dict:
    return {
        "support_recall@5": metrics["pcst_support_recall@5"],
        "partial_support_recall@5": metrics["pcst_partial_support_recall@5"],
        "bridge_support_recall@5": metrics["bridge_pcst_support_recall@5"],
        "comparison_support_recall@5": metrics["comparison_pcst_support_recall@5"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep learned-PCST hyperparameters on the same held-out split and report the best setting."
    )
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
    parser.add_argument("--baseline-support", type=float, default=0.8060)
    parser.add_argument("--seed-k-values", default="5,6")
    parser.add_argument("--expansion-factor-values", default="5,6,7")
    parser.add_argument("--fusion-anchor-pool-factor-values", default="3,4")
    parser.add_argument("--pcst-bonus-values", default="0.08,0.10,0.12")
    parser.add_argument("--preserve-fusion-top-k-values", default="2,3")
    parser.add_argument("--title-diversity-bonus-values", default="0.03,0.05,0.07")
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
    _ = [strip_graph(example) for example in val_graph_examples]

    print(f"Loaded {len(graph_examples)} examples.")
    print(f"Evaluating learned PCST on held-out validation split: {len(val_graph_examples)} questions.")

    gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
        graph_examples=graph_examples,
        embed_model=embed_model,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Loaded GNN checkpoint from {checkpoint_path}")

    seed_k_values = parse_int_list(args.seed_k_values)
    expansion_factor_values = parse_int_list(args.expansion_factor_values)
    fusion_anchor_pool_factor_values = parse_int_list(args.fusion_anchor_pool_factor_values)
    pcst_bonus_values = parse_float_list(args.pcst_bonus_values)
    preserve_fusion_top_k_values = parse_int_list(args.preserve_fusion_top_k_values)
    title_diversity_bonus_values = parse_float_list(args.title_diversity_bonus_values)

    candidates = list(
        itertools.product(
            seed_k_values,
            expansion_factor_values,
            fusion_anchor_pool_factor_values,
            pcst_bonus_values,
            preserve_fusion_top_k_values,
            title_diversity_bonus_values,
        )
    )

    print(f"Trying {len(candidates)} learned-PCST settings.")

    best_config = None
    best_summary = None

    for index, (
        seed_k,
        expansion_factor,
        fusion_anchor_pool_factor,
        pcst_bonus,
        preserve_fusion_top_k,
        title_diversity_bonus,
    ) in enumerate(candidates, start=1):
        print(
            f"\n[{index}/{len(candidates)}] "
            f"seed_k={seed_k}, "
            f"expansion_factor={expansion_factor}, "
            f"fusion_anchor_pool_factor={fusion_anchor_pool_factor}, "
            f"pcst_bonus={pcst_bonus:.2f}, "
            f"preserve_fusion_top_k={preserve_fusion_top_k}, "
            f"title_diversity_bonus={title_diversity_bonus:.2f}"
        )

        results = pcst_retrieve_all(
            graph_examples=val_graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=args.top_k,
            seed_k=min(seed_k, args.top_k),
            expansion_factor=expansion_factor,
            fusion_anchor_pool_factor=fusion_anchor_pool_factor,
            pcst_bonus=pcst_bonus,
            preserve_fusion_top_k=min(preserve_fusion_top_k, args.top_k),
            title_diversity_bonus=title_diversity_bonus,
            lambda_dense=args.lambda_dense,
        )
        summary = summarize(evaluate_pcst(results, k=args.top_k))

        print(
            "support={support:.4f}, partial={partial:.4f}, bridge={bridge:.4f}, comparison={comparison:.4f}".format(
                support=summary["support_recall@5"],
                partial=summary["partial_support_recall@5"],
                bridge=summary["bridge_support_recall@5"],
                comparison=summary["comparison_support_recall@5"],
            )
        )

        if best_summary is None or summary["support_recall@5"] > best_summary["support_recall@5"]:
            best_config = {
                "seed_k": seed_k,
                "expansion_factor": expansion_factor,
                "fusion_anchor_pool_factor": fusion_anchor_pool_factor,
                "pcst_bonus": pcst_bonus,
                "preserve_fusion_top_k": preserve_fusion_top_k,
                "title_diversity_bonus": title_diversity_bonus,
            }
            best_summary = summary
            print(
                f"New best support_recall@5: {best_summary['support_recall@5']:.4f} "
                f"(delta vs baseline {best_summary['support_recall@5'] - args.baseline_support:+.4f})"
            )

    print("\n===== Best Learned-PCST Setting =====")
    for key, value in best_config.items():
        print(f"{key}: {value}")

    print("\n===== Best Metrics =====")
    for key, value in best_summary.items():
        print(f"{key}: {value:.4f}")
