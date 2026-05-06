import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean, median

try:
    from .artifact_runtime import load_artifact_bundle, load_gnn_from_checkpoint
    from .gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_example
    from .gnn_retrieval import gnn_retrieve_for_example
    from .pcst import pcst_retrieve_for_example
    from .pcst_dense_retrieval import pcst_dense_retrieve_for_example
    from .retrieval import retrieve_top_k_chunks_for_example
except ImportError:
    from artifact_runtime import load_artifact_bundle, load_gnn_from_checkpoint
    from gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_example
    from gnn_retrieval import gnn_retrieve_for_example
    from pcst import pcst_retrieve_for_example
    from pcst_dense_retrieval import pcst_dense_retrieve_for_example
    from retrieval import retrieve_top_k_chunks_for_example


MODE_LABELS = {
    "dense": "Dense",
    "pcst_dense": "Dense-guided PCST",
    "gnn": "GNN",
    "fusion": "Dense + GraphSAGE fusion",
    "pcst": "Dense + GraphSAGE + PCST",
}

DEFAULT_MODES = ["dense", "pcst_dense", "gnn", "fusion", "pcst"]

PCST_LEARNED_SEED_K = 5
PCST_LEARNED_EXPANSION_FACTOR = 5
PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR = 3
PCST_LEARNED_BONUS = 0.08
PCST_LEARNED_PRESERVE_FUSION_TOP_K = 2
PCST_LEARNED_TITLE_DIVERSITY_BONUS = 0.03


def percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * q)
    return sorted_values[idx]


def strip_graph(example):
    return {key: value for key, value in example.items() if key != "graph"}


def build_runner(mode, embed_model, gnn_model, device, top_k, lambda_dense):
    if mode == "dense":
        return lambda example: retrieve_top_k_chunks_for_example(
            example=strip_graph(example),
            model=embed_model,
            top_k=top_k,
        )

    if mode == "pcst_dense":
        return lambda example: pcst_dense_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            top_k=top_k,
            seed_k=min(3, top_k),
        )

    if mode == "gnn":
        return lambda example: gnn_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
        )

    if mode == "fusion":
        return lambda example: dense_gnn_fusion_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            lambda_dense=lambda_dense,
        )

    if mode == "pcst":
        return lambda example: pcst_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            seed_k=min(PCST_LEARNED_SEED_K, top_k),
            expansion_factor=PCST_LEARNED_EXPANSION_FACTOR,
            fusion_anchor_pool_factor=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR,
            pcst_bonus=PCST_LEARNED_BONUS,
            preserve_fusion_top_k=min(PCST_LEARNED_PRESERVE_FUSION_TOP_K, top_k),
            title_diversity_bonus=PCST_LEARNED_TITLE_DIVERSITY_BONUS,
            lambda_dense=lambda_dense,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def time_mode(mode, examples, runner, warmup):
    warmup_examples = examples[: min(warmup, len(examples))]
    measured_examples = examples[min(warmup, len(examples)) :]

    for example in warmup_examples:
        runner(example)

    timings_ms = []
    for example in measured_examples:
        start = time.perf_counter()
        runner(example)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    return {
        "mode": mode,
        "label": MODE_LABELS[mode],
        "num_queries": len(timings_ms),
        "mean_ms": mean(timings_ms) if timings_ms else 0.0,
        "median_ms": median(timings_ms) if timings_ms else 0.0,
        "p50_ms": percentile(timings_ms, 0.50),
        "p95_ms": percentile(timings_ms, 0.95),
        "min_ms": min(timings_ms) if timings_ms else 0.0,
        "max_ms": max(timings_ms) if timings_ms else 0.0,
    }


def write_outputs(rows, output_json, output_csv):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print("\n===== Retrieval Inference Time =====")
    print("Timing excludes artifact loading, graph construction, model loading, and LLM generation.")
    print(f"{'Mode':32s} {'n':>6s} {'mean':>10s} {'p50':>10s} {'p95':>10s}")
    for row in rows:
        print(
            f"{row['label']:32s} "
            f"{row['num_queries']:6d} "
            f"{row['mean_ms']:10.2f} "
            f"{row['p50_ms']:10.2f} "
            f"{row['p95_ms']:10.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval-only inference time for GraphRAG modes."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--artifact-max-samples", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-dense", type=float, default=0.5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=DEFAULT_MODES,
        choices=DEFAULT_MODES,
    )
    parser.add_argument(
        "--output-json",
        default="inference_time_results.json",
    )
    parser.add_argument(
        "--output-csv",
        default="inference_time_results.csv",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow SentenceTransformer to contact Hugging Face if the model is not cached.",
    )
    args = parser.parse_args()

    if not args.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    bundle = load_artifact_bundle(
        split=args.split,
        max_samples=args.artifact_max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    graph_examples = bundle["graph_examples"]
    embed_model_name = bundle["manifest"]["model_name"]

    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(
        embed_model_name,
        local_files_only=not args.allow_model_download,
    )

    needs_gnn = any(mode in {"gnn", "fusion", "pcst"} for mode in args.modes)
    gnn_model = None
    device = None
    if needs_gnn:
        gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
            graph_examples=graph_examples,
            embed_model=embed_model,
            split=args.split,
            max_samples=args.artifact_max_samples,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Loaded GNN checkpoint from {checkpoint_path}")

    rng = random.Random(args.seed)
    sample_size = min(args.num_queries + args.warmup, len(graph_examples))
    examples = rng.sample(graph_examples, sample_size)

    rows = []
    for mode in args.modes:
        runner = build_runner(
            mode=mode,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=args.top_k,
            lambda_dense=args.lambda_dense,
        )
        rows.append(time_mode(mode, examples, runner, warmup=args.warmup))

    print_table(rows)

    write_outputs(
        rows=rows,
        output_json=Path(args.output_json),
        output_csv=Path(args.output_csv),
    )
    print(f"\nSaved JSON -> {args.output_json}")
    print(f"Saved CSV  -> {args.output_csv}")


if __name__ == "__main__":
    main()
