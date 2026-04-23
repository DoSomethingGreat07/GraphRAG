import argparse

try:
    from .artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from .gnn_train import build_pyg_dataset, split_dataset
    from .pcst import compute_fusion_scores, multiseed_pcst_selection, pcst_retrieve_for_example
except ImportError:
    from artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from gnn_train import build_pyg_dataset, split_dataset
    from pcst import compute_fusion_scores, multiseed_pcst_selection, pcst_retrieve_for_example


def rerank_result_by_gnn_scores(result):
    reranked = dict(result)
    gnn_scores = result.get("gnn_scores", [])
    ranked_indices = sorted(range(len(gnn_scores)), key=lambda idx: gnn_scores[idx], reverse=True)

    reranked["retrieved_chunks"] = [result["retrieved_chunks"][i] for i in ranked_indices]
    reranked["dense_scores"] = [result["dense_scores"][i] for i in ranked_indices]
    reranked["gnn_scores"] = [result["gnn_scores"][i] for i in ranked_indices]
    reranked["fusion_scores"] = [result["fusion_scores"][i] for i in ranked_indices]

    title_best_score = {}
    title_order = []
    for chunk, score in zip(reranked["retrieved_chunks"], reranked["gnn_scores"]):
        title = chunk.metadata["title"]
        if title not in title_best_score:
            title_best_score[title] = score
            title_order.append(title)
        else:
            title_best_score[title] = max(title_best_score[title], score)

    reranked["retrieved_titles"] = sorted(
        title_order,
        key=lambda title: title_best_score[title],
        reverse=True,
    )
    reranked["mode"] = "PCST+GNN"
    return reranked


def pcst_gnn_retrieve_for_example(
    example,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    seed_k: int = 3,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    result = pcst_retrieve_for_example(
        example=example,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=top_k,
        seed_k=seed_k,
        lambda_dense=lambda_dense,
        query_prefix=query_prefix,
    )

    dense_scores, gnn_scores, fusion_scores = compute_fusion_scores(
        example=example,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        lambda_dense=lambda_dense,
        query_prefix=query_prefix,
    )
    selected_nodes = multiseed_pcst_selection(
        example=example,
        fusion_scores=fusion_scores,
        seed_k=min(seed_k, top_k),
        max_nodes=top_k,
    )

    result["dense_scores"] = [float(dense_scores[i]) for i in selected_nodes]
    result["gnn_scores"] = [float(gnn_scores[i]) for i in selected_nodes]
    result["fusion_scores"] = [float(fusion_scores[i]) for i in selected_nodes]
    result["selected_nodes"] = selected_nodes
    return rerank_result_by_gnn_scores(result)


def pcst_gnn_retrieve_all(
    graph_examples,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    seed_k: int = 3,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    results = []
    for example in graph_examples:
        results.append(
            pcst_gnn_retrieve_for_example(
                example=example,
                embed_model=embed_model,
                gnn_model=gnn_model,
                device=device,
                top_k=top_k,
                seed_k=seed_k,
                lambda_dense=lambda_dense,
                query_prefix=query_prefix,
            )
        )
    return results


def evaluate_pcst_gnn(results, k=5):
    total = len(results)
    full_hits = partial_hits = 0

    bridge_total = bridge_full = bridge_partial = 0
    comparison_total = comparison_full = comparison_partial = 0

    for result in results:
        gold_titles = set(result["gold_titles"])
        retrieved_titles = set(result["retrieved_titles"][:k])
        overlap = gold_titles & retrieved_titles

        if gold_titles.issubset(retrieved_titles):
            full_hits += 1
        if overlap:
            partial_hits += 1

        if result["type"] == "bridge":
            bridge_total += 1
            if gold_titles.issubset(retrieved_titles):
                bridge_full += 1
            if overlap:
                bridge_partial += 1
        elif result["type"] == "comparison":
            comparison_total += 1
            if gold_titles.issubset(retrieved_titles):
                comparison_full += 1
            if overlap:
                comparison_partial += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    return {
        "total_questions": total,
        "pcst_gnn_support_recall@5": safe_div(full_hits, total),
        "pcst_gnn_partial_support_recall@5": safe_div(partial_hits, total),
        "bridge_questions": bridge_total,
        "bridge_pcst_gnn_support_recall@5": safe_div(bridge_full, bridge_total),
        "bridge_pcst_gnn_partial_support_recall@5": safe_div(bridge_partial, bridge_total),
        "comparison_questions": comparison_total,
        "comparison_pcst_gnn_support_recall@5": safe_div(comparison_full, comparison_total),
        "comparison_pcst_gnn_partial_support_recall@5": safe_div(comparison_partial, comparison_total),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCST+GNN retrieval evaluation.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--semantic-k", type=int, default=2)
    parser.add_argument("--semantic-min-sim", type=float, default=0.40)
    parser.add_argument("--keyword-overlap-threshold", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-k", type=int, default=3)
    parser.add_argument("--lambda-dense", type=float, default=0.5)
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
    val_ids = {d.question_id for d in val_dataset}
    val_examples = [ex for ex in graph_examples if ex["id"] in val_ids]

    print(f"Evaluating on {len(val_examples)} held-out validation examples.")

    gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
        graph_examples=graph_examples,
        embed_model=embed_model,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Loaded GNN checkpoint from {checkpoint_path}")

    results = pcst_gnn_retrieve_all(
        graph_examples=val_examples,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=args.top_k,
        seed_k=args.seed_k,
        lambda_dense=args.lambda_dense,
    )

    metrics = evaluate_pcst_gnn(results, k=args.top_k)

    print("\n===== PCST+GNN Retrieval Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
