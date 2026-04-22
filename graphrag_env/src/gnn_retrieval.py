import argparse
import numpy as np
import torch
from tqdm import tqdm

from embeddings import generate_chunk_embeddings
from hybrid_graph_builder import build_hybrid_graphs_for_all_examples
from gnn_train import (
    QueryAwareGraphSAGE,
    build_pyg_data_from_example,
    build_pyg_dataset,
    split_dataset,
)


@torch.no_grad()
def gnn_retrieve_for_example(
    example,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Run GNN inference for one question graph and retrieve top-k chunks.

    Returns:
        dict with:
            - id, question, answer, type, gold_titles
            - retrieved_chunks  (top-k chunk objects, ranked by GNN score)
            - retrieved_titles  (deduplicated, ranked by max chunk score per title)
            - scores            (GNN probabilities for each retrieved chunk)
    """
    empty_result = {
        "id":               example["id"],
        "question":         example["question"],
        "answer":           example["answer"],
        "type":             example["type"],
        "gold_titles":      example["supporting_facts"]["title"],
        "retrieved_chunks": [],
        "retrieved_titles": [],
        "scores":           [],
    }

    data = build_pyg_data_from_example(
        example=example,
        model=embed_model,
        query_prefix=query_prefix,
    )

    if data is None or data.x.shape[0] == 0:
        return empty_result

    data   = data.to(device)
    logits = gnn_model(data.x, data.edge_index)
    probs  = torch.sigmoid(logits).cpu().numpy()

    ranked_indices   = np.argsort(probs)[::-1][: min(top_k, len(probs))]
    retrieved_chunks = [example["context_chunks"][i] for i in ranked_indices]
    retrieved_scores = [float(probs[i]) for i in ranked_indices]

    # Keep the MAX score per title instead of silently dropping chunks
    # from the same title. A title is only as confident as its
    # highest-scoring chunk. Re-sort titles by that best score so
    # ordering stays consistent with chunk ranking.
    title_best_score: dict[str, float] = {}
    title_order: list[str] = []

    for chunk, score in zip(retrieved_chunks, retrieved_scores):
        title = chunk.metadata["title"]
        if title not in title_best_score:
            title_best_score[title] = score
            title_order.append(title)
        else:
            title_best_score[title] = max(title_best_score[title], score)

    retrieved_titles = sorted(
        title_order,
        key=lambda t: title_best_score[t],
        reverse=True,
    )

    return {
        **empty_result,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "scores":           retrieved_scores,
    }


# No @torch.no_grad() here — gnn_retrieve_for_example already carries it.
# Adding it on the outer function is a harmless but misleading no-op.
def gnn_retrieve_for_all_examples(
    graph_examples,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    results = []

    for example in tqdm(graph_examples, desc="GNN retrieval"):
        result = gnn_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            query_prefix=query_prefix,
        )
        results.append(result)

    return results


def evaluate_gnn_retrieval(results, k=5):
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
        "total_questions":                          total,
        "gnn_support_recall@5":                     safe_div(full_hits,          total),
        "gnn_partial_support_recall@5":             safe_div(partial_hits,        total),
        "bridge_questions":                         bridge_total,
        "bridge_gnn_support_recall@5":              safe_div(bridge_full,         bridge_total),
        "bridge_gnn_partial_support_recall@5":      safe_div(bridge_partial,      bridge_total),
        "comparison_questions":                     comparison_total,
        "comparison_gnn_support_recall@5":          safe_div(comparison_full,     comparison_total),
        "comparison_gnn_partial_support_recall@5":  safe_div(comparison_partial,  comparison_total),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN-only retrieval evaluation.")
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
    parser.add_argument("--checkpoint", default="query_aware_graphsage_best.pt")
    args = parser.parse_args()

    stage_bar = tqdm(total=5, desc="GNN eval pipeline", unit="stage")

    tqdm.write("Loading chunked examples and embeddings...")
    chunked_examples, embed_model = generate_chunk_embeddings(
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_text_length=args.min_text_length,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    stage_bar.update(1)

    tqdm.write("Building hybrid graphs...")
    graph_examples = build_hybrid_graphs_for_all_examples(
        chunked_examples=chunked_examples,
        semantic_k=args.semantic_k,
        semantic_min_sim=args.semantic_min_sim,
        keyword_overlap_threshold=args.keyword_overlap_threshold,
    )
    stage_bar.update(1)

    tqdm.write("Preparing held-out validation split...")
    # Mirror the exact train/val split from gnn_train.py so we only
    # evaluate on held-out examples — not training data.
    pyg_dataset = build_pyg_dataset(graph_examples, embed_model)
    _, val_dataset = split_dataset(pyg_dataset, train_ratio=0.8)
    val_ids = {d.question_id for d in val_dataset}
    val_examples = [ex for ex in graph_examples if ex["id"] in val_ids]
    tqdm.write(f"Evaluating on {len(val_examples)} held-out validation examples.")
    stage_bar.update(1)

    tqdm.write("Loading GNN checkpoint...")
    # Probe input_dim from the already-built pyg_dataset — no redundant
    # build_pyg_data_from_example call just to read one shape field.
    input_dim = pyg_dataset[0].x.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_model = QueryAwareGraphSAGE(
        input_dim=input_dim,
        hidden_dim=256,
        dropout=0.2,
    ).to(device)

    # weights_only=True suppresses the PyTorch 2.x deprecation warning
    # and prevents arbitrary code execution from untrusted checkpoints.
    gnn_model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    gnn_model.eval()
    stage_bar.update(1)

    tqdm.write("Running GNN retrieval evaluation...")
    results = gnn_retrieve_for_all_examples(
        graph_examples=val_examples,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=args.top_k,
    )
    stage_bar.update(1)
    stage_bar.close()

    metrics = evaluate_gnn_retrieval(results, k=args.top_k)

    print("\n===== GNN Retrieval Evaluation Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n===== Sample Predictions =====")
    for i, result in enumerate(results[:3]):
        print("\n" + "=" * 70)
        print(f"Example {i + 1}")
        print("Question:",         result["question"])
        print("Gold Titles:",      result["gold_titles"])
        print("Retrieved Titles:", result["retrieved_titles"])

        for rank, (chunk, score) in enumerate(
            zip(result["retrieved_chunks"], result["scores"]), start=1
        ):
            print(f"\nRank {rank}")
            print("Title:",        chunk.metadata["title"])
            print("Score:",        round(score, 4))
            print("Is Supporting:", chunk.metadata.get("is_supporting", False))
            print("Text:",         chunk.page_content[:220].replace("\n", " "))
