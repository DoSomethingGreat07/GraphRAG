from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from sentence_transformers import SentenceTransformer

try:
    from .artifact_utils import (
        get_artifact_paths,
        load_json,
        load_pickle,
        resolve_checkpoint_path,
    )
    from .embeddings import generate_chunk_embeddings
    from .hybrid_graph_builder import build_hybrid_graphs_for_all_examples
except ImportError:
    from artifact_utils import (
        get_artifact_paths,
        load_json,
        load_pickle,
        resolve_checkpoint_path,
    )
    from embeddings import generate_chunk_embeddings
    from hybrid_graph_builder import build_hybrid_graphs_for_all_examples


def load_artifact_bundle(
    split: str = "train",
    max_samples: int = 10000,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    artifacts_dir: Path | None = None,
) -> Dict[str, Any]:
    paths = get_artifact_paths(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifacts_dir=artifacts_dir,
    )

    if not paths["manifest"].exists():
        raise FileNotFoundError(
            f"Artifacts not found at {paths['manifest']}. "
            "Run `python prepare_artifacts.py` first."
        )

    manifest = load_json(paths["manifest"])

    return {
        "manifest": manifest,
        "paths": paths,
        "chunked_examples": load_pickle(paths["chunked_examples"]),
        "graph_examples": load_pickle(paths["graph_examples"]),
        "example_lookup": load_pickle(paths["example_lookup"]),
        "global_example": load_pickle(paths["global_example"]),
        "sample_questions": load_json(paths["sample_questions"]),
    }


def load_or_build_chunked_examples(
    split: str = "train",
    max_samples: int = 10000,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_text_length: int = 20,
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 64,
    artifacts_dir: Path | None = None,
) -> Tuple[list, SentenceTransformer]:
    paths = get_artifact_paths(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifacts_dir=artifacts_dir,
    )

    if paths["manifest"].exists() and paths["chunked_examples"].exists():
        manifest = load_json(paths["manifest"])
        model = SentenceTransformer(manifest["model_name"])
        return load_pickle(paths["chunked_examples"]), model

    return generate_chunk_embeddings(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_text_length=min_text_length,
        model_name=model_name,
        batch_size=batch_size,
    )


def load_or_build_graph_examples(
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
    artifacts_dir: Path | None = None,
) -> Tuple[list, SentenceTransformer]:
    paths = get_artifact_paths(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifacts_dir=artifacts_dir,
    )

    if paths["manifest"].exists() and paths["graph_examples"].exists():
        manifest = load_json(paths["manifest"])
        model = SentenceTransformer(manifest["model_name"])
        return load_pickle(paths["graph_examples"]), model

    chunked_examples, model = load_or_build_chunked_examples(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_text_length=min_text_length,
        model_name=model_name,
        batch_size=batch_size,
        artifacts_dir=artifacts_dir,
    )

    graph_examples = build_hybrid_graphs_for_all_examples(
        chunked_examples=chunked_examples,
        semantic_k=semantic_k,
        semantic_min_sim=semantic_min_sim,
        keyword_overlap_threshold=keyword_overlap_threshold,
    )

    return graph_examples, model


def load_gnn_from_checkpoint(
    graph_examples,
    embed_model,
    split: str = "train",
    max_samples: int = 10000,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    artifacts_dir: Path | None = None,
    hidden_dim: int = 256,
    dropout: float = 0.2,
):
    try:
        from .gnn_train import QueryAwareGraphSAGE, build_pyg_data_from_example
    except ImportError:
        from gnn_train import QueryAwareGraphSAGE, build_pyg_data_from_example

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_data = build_pyg_data_from_example(graph_examples[0], embed_model)
    input_dim = sample_data.x.shape[1]

    model = QueryAwareGraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    checkpoint_path = resolve_checkpoint_path(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifacts_dir=artifacts_dir,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"GNN checkpoint not found at {checkpoint_path}. Train the model first."
        )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model, device, checkpoint_path
