import argparse
import numpy as np
from tqdm import tqdm

try:
    from .artifact_runtime import load_or_build_graph_examples
    from .gnn_fusion_retreival import min_max_normalize
    from .gnn_train import build_pyg_dataset, split_dataset
    from .pcst import multiseed_pcst_selection
except ImportError:
    from artifact_runtime import load_or_build_graph_examples
    from gnn_fusion_retreival import min_max_normalize
    from gnn_train import build_pyg_dataset, split_dataset
    from pcst import multiseed_pcst_selection


def compute_dense_scores(
    example,
    embed_model,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
) -> tuple[np.ndarray, np.ndarray]:
    chunk_embeddings = example["context_chunk_embeddings"]
    query_embedding = embed_model.encode(
        query_prefix + example["question"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    dense_scores = np.dot(chunk_embeddings, query_embedding).astype(np.float32)
    return dense_scores, min_max_normalize(dense_scores)


def pcst_dense_retrieve_with_details_for_example(
    example,
    embed_model,
    top_k: int = 5,
    seed_k: int = 3,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    chunks = example["context_chunks"]
    empty_result = {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example["type"],
        "gold_titles": example["supporting_facts"]["title"],
        "retrieved_chunks": [],
        "retrieved_titles": [],
        "dense_scores": [],
        "selected_nodes": [],
    }

    if len(chunks) == 0:
        return empty_result

    dense_scores, dense_scores_norm = compute_dense_scores(
        example=example,
        embed_model=embed_model,
        query_prefix=query_prefix,
    )
    selected_indices = multiseed_pcst_selection(
        example=example,
        fusion_scores=dense_scores_norm,
        seed_k=seed_k,
        max_nodes=top_k,
    )

    retrieved_chunks = [chunks[i] for i in selected_indices]
    retrieved_dense_scores = [float(dense_scores[i]) for i in selected_indices]

    title_best_score: dict[str, float] = {}
    title_order: list[str] = []
    for chunk, score in zip(retrieved_chunks, retrieved_dense_scores):
        title = chunk.metadata["title"]
        if title not in title_best_score:
            title_best_score[title] = score
            title_order.append(title)
        else:
            title_best_score[title] = max(title_best_score[title], score)

    retrieved_titles = sorted(
        title_order,
        key=lambda title: title_best_score[title],
        reverse=True,
    )

    return {
        **empty_result,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "dense_scores": retrieved_dense_scores,
        "selected_nodes": selected_indices,
    }


def pcst_dense_retrieve_for_example(
    example,
    embed_model,
    top_k: int = 5,
    seed_k: int = 3,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    return pcst_dense_retrieve_with_details_for_example(
        example=example,
        embed_model=embed_model,
        top_k=top_k,
        seed_k=seed_k,
        query_prefix=query_prefix,
    )


def pcst_dense_retrieve_all(
    graph_examples,
    embed_model,
    top_k: int = 5,
    seed_k: int = 3,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    results = []
    for example in tqdm(graph_examples, desc="Dense-guided PCST retrieval"):
        results.append(
            pcst_dense_retrieve_for_example(
                example=example,
                embed_model=embed_model,
                top_k=top_k,
                seed_k=seed_k,
                query_prefix=query_prefix,
            )
        )
    return results


def evaluate_pcst_dense(results, k=5):
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
        "pcst_dense_support_recall@5": safe_div(full_hits, total),
        "pcst_dense_partial_support_recall@5": safe_div(partial_hits, total),
        "bridge_questions": bridge_total,
        "bridge_pcst_dense_support_recall@5": safe_div(bridge_full, bridge_total),
        "bridge_pcst_dense_partial_support_recall@5": safe_div(bridge_partial, bridge_total),
        "comparison_questions": comparison_total,
        "comparison_pcst_dense_support_recall@5": safe_div(comparison_full, comparison_total),
        "comparison_pcst_dense_partial_support_recall@5": safe_div(comparison_partial, comparison_total),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense-guided PCST retrieval evaluation.")
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

    results = pcst_dense_retrieve_all(
        graph_examples=val_examples,
        embed_model=embed_model,
        top_k=args.top_k,
        seed_k=args.seed_k,
    )

    metrics = evaluate_pcst_dense(results, k=args.top_k)
    print("\n===== Dense-Guided PCST Retrieval Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
