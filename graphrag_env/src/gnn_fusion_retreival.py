import argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    from .artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from .gnn_train import (
        build_pyg_data_from_example,
        build_pyg_dataset,
        split_dataset,
    )
except ImportError:
    from artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from gnn_train import (
        build_pyg_data_from_example,
        build_pyg_dataset,
        split_dataset,
    )


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1]. Returns zeros if range is too small.
    """
    if len(scores) == 0:
        return scores

    s_min = scores.min()
    s_max = scores.max()

    if s_max - s_min < 1e-8:
        return np.zeros_like(scores, dtype=np.float32)

    return ((scores - s_min) / (s_max - s_min)).astype(np.float32)


@torch.no_grad()
def dense_gnn_fusion_retrieve_for_example(
    example,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    lambda_dense: float = 0.3,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Fusion retrieval:
        final_score = lambda_dense * dense_score
                    + (1 - lambda_dense) * gnn_score

    Both scores are min-max normalized per question before fusion.
    """
    chunks           = example["context_chunks"]
    chunk_embeddings = example["context_chunk_embeddings"]

    empty_result = {
        "id":               example["id"],
        "question":         example["question"],
        "answer":           example["answer"],
        "type":             example["type"],
        "gold_titles":      example["supporting_facts"]["title"],
        "retrieved_chunks": [],
        "retrieved_titles": [],
        "dense_scores":     [],
        "gnn_scores":       [],
        "fusion_scores":    [],
    }

    if len(chunks) == 0:
        return empty_result

    # ----- Dense query scores -----
    query_text = query_prefix + example["question"]
    query_embedding = embed_model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dense_scores      = np.dot(chunk_embeddings, query_embedding).astype(np.float32)
    dense_scores_norm = min_max_normalize(dense_scores)

    # ----- GNN node scores -----
    # Reuse the already-built PyG data object here instead of calling
    # build_pyg_data_from_example again inside a separate helper.
    # That helper re-encodes the query a second time — wasted compute.
    data = build_pyg_data_from_example(
        example=example,
        model=embed_model,
        query_prefix=query_prefix,
    )

    if data is None or data.x.shape[0] == 0:
        return empty_result

    data       = data.to(device)
    logits     = gnn_model(data.x, data.edge_index)
    gnn_scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    gnn_scores_norm = min_max_normalize(gnn_scores)

    # ----- Fusion -----
    fusion_scores = (
        lambda_dense * dense_scores_norm
        + (1.0 - lambda_dense) * gnn_scores_norm
    ).astype(np.float32)

    ranked_indices = np.argsort(fusion_scores)[::-1][: min(top_k, len(chunks))]

    retrieved_chunks         = [chunks[i] for i in ranked_indices]
    retrieved_dense_scores   = [float(dense_scores[i])   for i in ranked_indices]
    retrieved_gnn_scores     = [float(gnn_scores[i])     for i in ranked_indices]
    retrieved_fusion_scores  = [float(fusion_scores[i])  for i in ranked_indices]

    # Keep the MAX fusion score per title instead of silently dropping
    # chunks from the same title. Re-sort titles by that best score so
    # ordering stays consistent with chunk ranking.
    title_best_score: dict[str, float] = {}
    title_order: list[str] = []

    for chunk, fscore in zip(retrieved_chunks, retrieved_fusion_scores):
        title = chunk.metadata["title"]
        if title not in title_best_score:
            title_best_score[title] = fscore
            title_order.append(title)
        else:
            title_best_score[title] = max(title_best_score[title], fscore)

    retrieved_titles = sorted(
        title_order,
        key=lambda t: title_best_score[t],
        reverse=True,
    )

    return {
        **empty_result,
        "retrieved_chunks":  retrieved_chunks,
        "retrieved_titles":  retrieved_titles,
        "dense_scores":      retrieved_dense_scores,
        "gnn_scores":        retrieved_gnn_scores,
        "fusion_scores":     retrieved_fusion_scores,
    }


# No @torch.no_grad() here — dense_gnn_fusion_retrieve_for_example already
# carries it. Adding it on the outer function is a harmless but misleading no-op.
def dense_gnn_fusion_retrieve_for_all_examples(
    graph_examples,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    results = []

    for example in tqdm(graph_examples, desc="Fusion retrieval"):
        result = dense_gnn_fusion_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            lambda_dense=lambda_dense,
            query_prefix=query_prefix,
        )
        results.append(result)

    return results


def evaluate_fusion_retrieval(results, k=5):
    total = len(results)
    full_hits = partial_hits = 0

    bridge_total     = bridge_full     = bridge_partial     = 0
    comparison_total = comparison_full = comparison_partial = 0

    for result in results:
        gold_titles      = set(result["gold_titles"])
        retrieved_titles = set(result["retrieved_titles"][:k])
        overlap          = gold_titles & retrieved_titles

        if gold_titles.issubset(retrieved_titles):
            full_hits += 1
        if overlap:
            partial_hits += 1

        if result["type"] == "bridge":
            bridge_total += 1
            if gold_titles.issubset(retrieved_titles): bridge_full    += 1
            if overlap:                                bridge_partial += 1

        elif result["type"] == "comparison":
            comparison_total += 1
            if gold_titles.issubset(retrieved_titles): comparison_full    += 1
            if overlap:                                comparison_partial += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    return {
        "total_questions":                              total,
        "fusion_support_recall@5":                      safe_div(full_hits,          total),
        "fusion_partial_support_recall@5":              safe_div(partial_hits,        total),
        "bridge_questions":                             bridge_total,
        "bridge_fusion_support_recall@5":               safe_div(bridge_full,         bridge_total),
        "bridge_fusion_partial_support_recall@5":       safe_div(bridge_partial,      bridge_total),
        "comparison_questions":                         comparison_total,
        "comparison_fusion_support_recall@5":           safe_div(comparison_full,     comparison_total),
        "comparison_fusion_partial_support_recall@5":   safe_div(comparison_partial,  comparison_total),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense+GNN fusion retrieval evaluation.")
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

    # Mirror the exact train/val split from gnn_train.py so we only
    # evaluate on held-out examples — not training data.
    pyg_dataset = build_pyg_dataset(graph_examples, embed_model)
    _, val_dataset = split_dataset(pyg_dataset, train_ratio=0.8)
    val_ids      = {d.question_id for d in val_dataset}
    val_examples = [ex for ex in graph_examples if ex["id"] in val_ids]

    print(f"Evaluating on {len(val_examples)} held-out validation examples.")

    # Probe input_dim from the already-built pyg_dataset — no redundant
    # build_pyg_data_from_example call just to read one shape field.
    gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
        graph_examples=graph_examples,
        embed_model=embed_model,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Loaded GNN checkpoint from {checkpoint_path}")

    results = dense_gnn_fusion_retrieve_for_all_examples(
        graph_examples=val_examples,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=args.top_k,
        lambda_dense=args.lambda_dense,
    )

    metrics = evaluate_fusion_retrieval(results, k=args.top_k)

    print("\n===== Dense + GNN Fusion Retrieval Results =====")
    print(f"lambda_dense = {args.lambda_dense}")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n===== Sample Predictions =====")
    for i, result in enumerate(results[:3]):
        print("\n" + "=" * 70)
        print(f"Example {i + 1}")
        print("Question:",         result["question"])
        print("Gold Titles:",      result["gold_titles"])
        print("Retrieved Titles:", result["retrieved_titles"])

        for rank, (chunk, d, g, f) in enumerate(
            zip(
                result["retrieved_chunks"],
                result["dense_scores"],
                result["gnn_scores"],
                result["fusion_scores"],
            ),
            start=1,
        ):
            print(f"\nRank {rank}")
            print("Title:",        chunk.metadata["title"])
            print("Dense Score:",  round(d, 4))
            print("GNN Score:",    round(g, 4))
            print("Fusion Score:", round(f, 4))
            print("Is Supporting:", chunk.metadata.get("is_supporting", False))
            print("Text:",         chunk.page_content[:220].replace("\n", " "))
