import argparse
from graphrag_env.src.artifact_utils import (
    build_example_lookup,
    build_global_example,
    build_manifest,
    build_sample_questions,
    get_artifact_paths,
    save_json,
    save_pickle,
)
from graphrag_env.src.embeddings import generate_chunk_embeddings
from graphrag_env.src.hybrid_graph_builder import build_hybrid_graphs_for_all_examples


def prepare_artifacts(
    split: str = "train",
    max_samples: int = 10000,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_text_length: int = 20,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    semantic_k: int = 2,
    semantic_min_sim: float = 0.40,
    keyword_overlap_threshold: int = 3,
) -> dict:
    print("Building chunked examples and embeddings...")
    chunked_examples, _ = generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_text_length=min_text_length,
        model_name=model_name,
        batch_size=batch_size,
    )

    print("Building hybrid graphs...")
    graph_examples = build_hybrid_graphs_for_all_examples(
        chunked_examples=chunked_examples,
        semantic_k=semantic_k,
        semantic_min_sim=semantic_min_sim,
        keyword_overlap_threshold=keyword_overlap_threshold,
    )

    print("Building lookup artifacts...")
    example_lookup = build_example_lookup(graph_examples)
    global_example = build_global_example(graph_examples)
    sample_questions = build_sample_questions(chunked_examples, limit=10)

    paths = get_artifact_paths(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    manifest = build_manifest(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_text_length=min_text_length,
        model_name=model_name,
        batch_size=batch_size,
        semantic_k=semantic_k,
        semantic_min_sim=semantic_min_sim,
        keyword_overlap_threshold=keyword_overlap_threshold,
        num_examples=len(chunked_examples),
        num_global_chunks=len(global_example["context_chunks"]),
    )

    print("Saving artifacts...")
    save_pickle(chunked_examples, paths["chunked_examples"])
    save_pickle(graph_examples, paths["graph_examples"])
    save_pickle(example_lookup, paths["example_lookup"])
    save_pickle(global_example, paths["global_example"])
    save_json(sample_questions, paths["sample_questions"])
    save_json(manifest, paths["manifest"])

    print("\nArtifacts saved:")
    for key in (
        "manifest",
        "chunked_examples",
        "graph_examples",
        "example_lookup",
        "global_example",
        "sample_questions",
    ):
        print(f"- {key}: {paths[key]}")

    return {
        "paths": {key: str(value) for key, value in paths.items() if key != "artifacts_dir"},
        "manifest": manifest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare cached GraphRAG artifacts.")
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
    args = parser.parse_args()

    prepare_artifacts(
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
