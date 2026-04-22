import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_CHECKPOINT_NAME = "query_aware_graphsage_best.pt"


def ensure_artifacts_dir(artifacts_dir: Path | None = None) -> Path:
    target_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def build_artifact_tag(
    split: str,
    max_samples: int,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    return f"hotpotqa_{split}_n{max_samples}_cs{chunk_size}_co{chunk_overlap}"


def get_artifact_paths(
    split: str,
    max_samples: int,
    chunk_size: int,
    chunk_overlap: int,
    artifacts_dir: Path | None = None,
) -> Dict[str, Path]:
    base_dir = ensure_artifacts_dir(artifacts_dir)
    tag = build_artifact_tag(split, max_samples, chunk_size, chunk_overlap)

    return {
        "artifacts_dir": base_dir,
        "tag": base_dir / tag,
        "manifest": base_dir / f"{tag}_manifest.json",
        "chunked_examples": base_dir / f"{tag}_chunked_examples.pkl",
        "graph_examples": base_dir / f"{tag}_graph_examples.pkl",
        "global_example": base_dir / f"{tag}_global_example.pkl",
        "example_lookup": base_dir / f"{tag}_example_lookup.pkl",
        "sample_questions": base_dir / f"{tag}_sample_questions.json",
        "gnn_checkpoint": base_dir / f"{tag}_{DEFAULT_CHECKPOINT_NAME}",
    }


def save_pickle(obj: Any, path: Path) -> None:
    with Path(path).open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_example_lookup(chunked_examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {example["id"]: example for example in chunked_examples}


def build_sample_questions(
    chunked_examples: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    sample_rows = []

    for example in chunked_examples[:limit]:
        sample_rows.append(
            {
                "id": example["id"],
                "question": example["question"],
                "answer": example.get("answer", ""),
                "type": example.get("type", ""),
                "gold_titles": example.get("supporting_facts", {}).get("title", []),
            }
        )

    return sample_rows


def build_global_example(graph_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_chunks = []
    all_embeddings = []
    graphs = []
    chunk_to_example_id = []

    for example in graph_examples:
        chunks = example.get("context_chunks", [])
        embeddings = example.get("context_chunk_embeddings")
        graph = example.get("graph")

        if not chunks or embeddings is None or graph is None:
            continue

        all_chunks.extend(chunks)
        all_embeddings.append(np.asarray(embeddings, dtype=np.float32))
        graphs.append(graph)
        chunk_to_example_id.extend([example["id"]] * len(chunks))

    if all_embeddings:
        merged_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    else:
        merged_embeddings = np.empty((0, 0), dtype=np.float32)

    if graphs:
        merged_graph = nx.disjoint_union_all(graphs)
    else:
        merged_graph = nx.Graph()

    return {
        "id": "global_corpus",
        "question": "",
        "answer": "",
        "type": "custom",
        "supporting_facts": {"title": [], "sent_id": []},
        "context_docs": [],
        "context_chunks": all_chunks,
        "context_chunk_embeddings": merged_embeddings,
        "graph": merged_graph,
        "chunk_to_example_id": chunk_to_example_id,
    }


def build_manifest(
    split: str,
    max_samples: int,
    chunk_size: int,
    chunk_overlap: int,
    min_text_length: int,
    model_name: str,
    batch_size: int,
    semantic_k: int,
    semantic_min_sim: float,
    keyword_overlap_threshold: int,
    num_examples: int,
    num_global_chunks: int,
) -> Dict[str, Any]:
    return {
        "split": split,
        "max_samples": max_samples,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_text_length": min_text_length,
        "model_name": model_name,
        "batch_size": batch_size,
        "semantic_k": semantic_k,
        "semantic_min_sim": semantic_min_sim,
        "keyword_overlap_threshold": keyword_overlap_threshold,
        "num_examples": num_examples,
        "num_global_chunks": num_global_chunks,
    }


def resolve_checkpoint_path(
    split: str = "train",
    max_samples: int = 10000,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    artifacts_dir: Path | None = None,
    fallback_to_legacy: bool = True,
) -> Path:
    paths = get_artifact_paths(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        artifacts_dir=artifacts_dir,
    )

    if paths["gnn_checkpoint"].exists():
        return paths["gnn_checkpoint"]

    legacy_path = SRC_DIR / DEFAULT_CHECKPOINT_NAME
    if fallback_to_legacy and legacy_path.exists():
        return legacy_path

    return paths["gnn_checkpoint"]
