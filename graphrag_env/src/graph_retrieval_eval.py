from embeddings import generate_chunk_embeddings
from hybrid_graph_builder import build_hybrid_graphs_for_all_examples
from graph_retrieval import graph_retrieve_for_all_examples


def evaluate_graph_retrieval(results, k=5):
    total = len(results)
    full_hits = 0
    partial_hits = 0

    bridge_total = 0
    bridge_full = 0
    bridge_partial = 0

    comparison_total = 0
    comparison_full = 0
    comparison_partial = 0

    for result in results:
        gold_titles = set(result["gold_titles"])
        retrieved_titles = set(result["retrieved_titles"][:k])

        overlap = gold_titles.intersection(retrieved_titles)

        if gold_titles.issubset(retrieved_titles):
            full_hits += 1

        if len(overlap) > 0:
            partial_hits += 1

        if result["type"] == "bridge":
            bridge_total += 1
            if gold_titles.issubset(retrieved_titles):
                bridge_full += 1
            if len(overlap) > 0:
                bridge_partial += 1

        elif result["type"] == "comparison":
            comparison_total += 1
            if gold_titles.issubset(retrieved_titles):
                comparison_full += 1
            if len(overlap) > 0:
                comparison_partial += 1

    metrics = {
        "total_questions": total,
        "graph_support_recall@5": full_hits / total if total else 0.0,
        "graph_partial_support_recall@5": partial_hits / total if total else 0.0,
        "bridge_questions": bridge_total,
        "bridge_graph_support_recall@5": bridge_full / bridge_total if bridge_total else 0.0,
        "bridge_graph_partial_support_recall@5": bridge_partial / bridge_total if bridge_total else 0.0,
        "comparison_questions": comparison_total,
        "comparison_graph_support_recall@5": comparison_full / comparison_total if comparison_total else 0.0,
        "comparison_graph_partial_support_recall@5": comparison_partial / comparison_total if comparison_total else 0.0,
    }

    return metrics


if __name__ == "__main__":
    chunked_examples, model = generate_chunk_embeddings(
        split="train",
        max_samples=10000,
        chunk_size=300,
        chunk_overlap=50,
        min_text_length=20,
        model_name="BAAI/bge-base-en-v1.5",
        batch_size=64,
    )

    graph_examples = build_hybrid_graphs_for_all_examples(
        chunked_examples=chunked_examples,
        semantic_k=2,
        semantic_min_sim=0.40,
        keyword_overlap_threshold=3,
    )

    results = graph_retrieve_for_all_examples(
        graph_examples=graph_examples,
        model=model,
        seed_k=5,
        final_k=5,
        neighbor_hops=1,
        query_prefix="Represent this sentence for searching relevant passages: ",
    )

    metrics = evaluate_graph_retrieval(results, k=5)

    print("\n===== Graph Retrieval Evaluation Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n===== Sample Predictions =====")
    for i, result in enumerate(results[:3]):
        print("\n" + "=" * 70)
        print(f"Example {i+1}")
        print("Question:", result["question"])
        print("Gold Titles:", result["gold_titles"])
        print("Retrieved Titles:", result["retrieved_titles"])

        if "retrieved_chunks" in result:
            for rank, chunk in enumerate(result["retrieved_chunks"], start=1):
                print(f"\nRank {rank}")
                print("Title:", chunk.metadata["title"])
                print("Is Supporting:", chunk.metadata.get("is_supporting", False))
                print("Text:", chunk.page_content[:220].replace("\n", " "))