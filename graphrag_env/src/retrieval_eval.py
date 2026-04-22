'''import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from embeddings import generate_chunk_embeddings


def evaluate_baseline_retrieval(
    split="train",
    max_samples=10000,
    k=5,
):
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    chunks, embeddings, _ = generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=500,
        chunk_overlap=100,
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    exact_hits   = 0
    partial_hits = 0

    type_counters = {
        "bridge":     {"exact": 0, "partial": 0, "total": 0},
        "comparison": {"exact": 0, "partial": 0, "total": 0},
        "unknown":    {"exact": 0, "partial": 0, "total": 0},
    }

    for item in dataset:
        question    = item["question"]
        gold_titles = set(item["supporting_facts"]["title"])
        q_type      = item.get("type", "unknown")

        query_vec = model.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        scores  = embeddings @ query_vec
        top_idx = np.argsort(scores)[-k:][::-1]

        retrieved_titles = {chunks[i].metadata["title"] for i in top_idx}
        overlap = gold_titles & retrieved_titles

        is_exact   = int(len(overlap) == len(gold_titles))
        is_partial = int(len(overlap) > 0)

        exact_hits   += is_exact
        partial_hits += is_partial

        bucket = q_type if q_type in type_counters else "unknown"
        type_counters[bucket]["exact"]   += is_exact
        type_counters[bucket]["partial"] += is_partial
        type_counters[bucket]["total"]   += 1

    total   = len(dataset)
    metrics = {
        "total_questions":              total,
        f"support_recall@{k}":         round(exact_hits / total, 4),
        f"partial_support_recall@{k}": round(partial_hits / total, 4),
    }

    for q_type, c in type_counters.items():
        if c["total"] == 0:
            continue
        metrics[f"{q_type}_support_recall@{k}"] = round(c["exact"]   / c["total"], 4)
        metrics[f"{q_type}_partial_recall@{k}"] = round(c["partial"] / c["total"], 4)
        metrics[f"{q_type}_count"]              = c["total"]

    return metrics


if __name__ == "__main__":
    print("=" * 55)
    print("BASELINE RETRIEVAL EVALUATION")
    print("=" * 55)

    metrics = evaluate_baseline_retrieval(split="train", max_samples=10000, k=5)

    print(f"\nOverall ({metrics['total_questions']} questions)")
    print(f"  support_recall@5:         {metrics['support_recall@5']}")
    print(f"  partial_support_recall@5: {metrics['partial_support_recall@5']}")

    print(f"\nBridge questions ({metrics.get('bridge_count', 0)})")
    print(f"  support_recall@5: {metrics.get('bridge_support_recall@5', 'N/A')}")
    print(f"  partial_recall@5: {metrics.get('bridge_partial_recall@5', 'N/A')}")

    print(f"\nComparison questions ({metrics.get('comparison_count', 0)})")
    print(f"  support_recall@5: {metrics.get('comparison_support_recall@5', 'N/A')}")
    print(f"  partial_recall@5: {metrics.get('comparison_partial_recall@5', 'N/A')}")'''

import argparse

try:
    from .artifact_runtime import load_or_build_chunked_examples
    from .retrieval import retrieve_for_all_examples
except ImportError:
    from artifact_runtime import load_or_build_chunked_examples
    from retrieval import retrieve_for_all_examples


def evaluate_retrieval(results, k=5):
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
        "support_recall@5": full_hits / total if total else 0.0,
        "partial_support_recall@5": partial_hits / total if total else 0.0,
        "bridge_questions": bridge_total,
        "bridge_support_recall@5": bridge_full / bridge_total if bridge_total else 0.0,
        "bridge_partial_support_recall@5": bridge_partial / bridge_total if bridge_total else 0.0,
        "comparison_questions": comparison_total,
        "comparison_support_recall@5": comparison_full / comparison_total if comparison_total else 0.0,
        "comparison_partial_support_recall@5": comparison_partial / comparison_total if comparison_total else 0.0,
    }

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dense retrieval evaluation.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chunked_examples, model = load_or_build_chunked_examples(
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_text_length=args.min_text_length,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    results = retrieve_for_all_examples(
        chunked_examples=chunked_examples,
        model=model,
        top_k=args.top_k,
    )

    metrics = evaluate_retrieval(results, k=args.top_k)

    print("\n===== Retrieval Evaluation Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")
