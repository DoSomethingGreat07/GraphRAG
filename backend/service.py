import copy
import json
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import faiss
except ImportError:
    faiss = None

from graphrag_env.src.artifact_runtime import load_artifact_bundle, load_gnn_from_checkpoint
from graphrag_env.src.artifact_utils import get_artifact_paths
from graphrag_env.src.gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_example
from graphrag_env.src.gnn_retrieval import gnn_retrieve_for_example
from graphrag_env.src.hybrid_graph_builder import graph_stats
from graphrag_env.src.llm_eval import generate_retrieval_fallback_answer
from graphrag_env.src.pcst_dense_retrieval import (
    pcst_dense_retrieve_for_example,
)
from graphrag_env.src.pcst import (
    pcst_retrieve_for_example,
)
from graphrag_env.src.retrieval import retrieve_top_k_chunks_for_example


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_PROFILE = {
    "id": "train_n10000_c300_o50",
    "split": "train",
    "max_samples": 10000,
    "chunk_size": 300,
    "chunk_overlap": 50,
    "label": "train | n=10000 | chunk=300/50",
}
CUSTOM_BACKEND_EXACT = "Exact"
CUSTOM_BACKEND_HNSW = "HNSW"
CUSTOM_BACKEND_IVF = "IVF"
MODE_DENSE = "FAISS-only retrieval"
MODE_PCST_DENSE = "FAISS + heuristic PCST"
MODE_GNN = "GNN retrieval"
MODE_FUSION = "Dense retrieval + Query-Aware GraphSAGE"
MODE_PCST_LEARNED = "Dense retrieval + Query-Aware GraphSAGE + PCST (Main Method)"
PCST_LEARNED_SEED_K = 5
PCST_LEARNED_EXPANSION_FACTOR = 5
PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR = 3
PCST_LEARNED_BONUS = 0.08
PCST_LEARNED_PRESERVE_FUSION_TOP_K = 2
PCST_LEARNED_TITLE_DIVERSITY_BONUS = 0.03
RETRIEVAL_MODES = [
    MODE_DENSE,
    MODE_PCST_DENSE,
    MODE_GNN,
    MODE_FUSION,
    MODE_PCST_LEARNED,
]

load_dotenv(ENV_PATH)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def discover_artifact_profiles() -> list[dict[str, Any]]:
    profiles = []

    if not ARTIFACTS_DIR.exists():
        return [DEFAULT_PROFILE]

    for manifest_path in sorted(ARTIFACTS_DIR.glob("*_manifest.json")):
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue

        profile = {
            "id": (
                f"{manifest['split']}_n{manifest['max_samples']}_"
                f"c{manifest['chunk_size']}_o{manifest['chunk_overlap']}"
            ),
            "label": (
                f"{manifest['split']} | n={manifest['max_samples']} | "
                f"chunk={manifest['chunk_size']}/{manifest['chunk_overlap']}"
            ),
            "split": manifest["split"],
            "max_samples": manifest["max_samples"],
            "chunk_size": manifest["chunk_size"],
            "chunk_overlap": manifest["chunk_overlap"],
            "manifest": manifest,
        }
        profiles.append(profile)

    return profiles or [DEFAULT_PROFILE]


def get_profile(profile_id: str | None = None) -> dict[str, Any]:
    profiles = discover_artifact_profiles()
    if profile_id is None:
        return profiles[0]
    for profile in profiles:
        if profile["id"] == profile_id:
            return profile
    raise FileNotFoundError(f"Unknown artifact profile: {profile_id}")


def get_profile_paths(profile_id: str | None = None):
    profile = get_profile(profile_id)
    return profile, get_artifact_paths(
        split=profile["split"],
        max_samples=profile["max_samples"],
        chunk_size=profile["chunk_size"],
        chunk_overlap=profile["chunk_overlap"],
    )


@lru_cache(maxsize=8)
def load_runtime_metadata(profile_id: str | None = None):
    profile, paths = get_profile_paths(profile_id)
    with paths["manifest"].open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    with paths["sample_questions"].open("r", encoding="utf-8") as sample_questions_file:
        sample_questions = json.load(sample_questions_file)

    checkpoint_path = paths["gnn_checkpoint"]
    has_gnn = checkpoint_path.exists()

    return {
        "profile": profile,
        "manifest": manifest,
        "sample_questions": sample_questions,
        "custom_backends": [CUSTOM_BACKEND_EXACT, CUSTOM_BACKEND_HNSW, CUSTOM_BACKEND_IVF],
        "has_gnn": has_gnn,
        "checkpoint_path": str(checkpoint_path) if has_gnn else None,
    }


@lru_cache(maxsize=4)
def load_example_index(profile_id: str | None = None):
    _, paths = get_profile_paths(profile_id)
    with paths["example_lookup"].open("rb") as lookup_file:
        example_lookup = pickle.load(lookup_file)

    examples = []
    for example in example_lookup.values():
        examples.append(
            {
                "id": example["id"],
                "question": example["question"],
                "type": example.get("type", "unknown"),
                "answer": example.get("answer", ""),
            }
        )

    examples.sort(key=lambda example: example["question"])
    return examples


def build_custom_ann_indexes(global_embeddings):
    embeddings = np.asarray(global_embeddings, dtype=np.float32)
    indexes = {
        CUSTOM_BACKEND_HNSW: None,
        CUSTOM_BACKEND_IVF: None,
    }

    if len(embeddings) == 0:
        return indexes

    dimension = embeddings.shape[1]

    if hnswlib is not None:
        try:
            hnsw_index = hnswlib.Index(space="cosine", dim=dimension)
            hnsw_index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
            hnsw_index.add_items(embeddings, np.arange(len(embeddings)))
            hnsw_index.set_ef(min(max(64, 32), len(embeddings)))
            indexes[CUSTOM_BACKEND_HNSW] = hnsw_index
        except Exception:
            indexes[CUSTOM_BACKEND_HNSW] = None

    if faiss is not None and len(embeddings) >= 100:
        try:
            quantizer = faiss.IndexFlatIP(dimension)
            nlist = max(1, min(int(np.sqrt(len(embeddings))), len(embeddings) // 20 or 1))
            ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            ivf_index.train(embeddings)
            ivf_index.add(embeddings)
            ivf_index.nprobe = min(max(8, nlist // 8), nlist)
            indexes[CUSTOM_BACKEND_IVF] = ivf_index
        except Exception:
            indexes[CUSTOM_BACKEND_IVF] = None

    return indexes


@lru_cache(maxsize=4)
def load_resources(profile_id: str | None = None) -> dict[str, Any]:
    profile = get_profile(profile_id)
    bundle = load_artifact_bundle(
        split=profile["split"],
        max_samples=profile["max_samples"],
        chunk_size=profile["chunk_size"],
        chunk_overlap=profile["chunk_overlap"],
    )
    embed_model = SentenceTransformer(bundle["manifest"]["model_name"])

    gnn_model = None
    device = None
    checkpoint_path = None

    try:
        gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
            graph_examples=bundle["graph_examples"],
            embed_model=embed_model,
            split=profile["split"],
            max_samples=profile["max_samples"],
            chunk_size=profile["chunk_size"],
            chunk_overlap=profile["chunk_overlap"],
        )
    except FileNotFoundError:
        pass

    ann_indexes = build_custom_ann_indexes(bundle["global_example"]["context_chunk_embeddings"])

    return {
        **bundle,
        "profile": profile,
        "embed_model": embed_model,
        "gnn_model": gnn_model,
        "device": device,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "global_chunks": bundle["global_example"]["context_chunks"],
        "global_embeddings": bundle["global_example"]["context_chunk_embeddings"],
        "global_graph": bundle["global_example"]["graph"],
        "chunk_to_example_id": bundle["global_example"].get("chunk_to_example_id", []),
        "custom_ann_indexes": ann_indexes,
    }


def get_custom_backend_options(resources):
    options = [CUSTOM_BACKEND_EXACT]
    ann_indexes = resources.get("custom_ann_indexes", {})

    if ann_indexes.get(CUSTOM_BACKEND_HNSW) is not None:
        options.append(CUSTOM_BACKEND_HNSW)
    if ann_indexes.get(CUSTOM_BACKEND_IVF) is not None:
        options.append(CUSTOM_BACKEND_IVF)

    return options


def build_query_example(base_example, question: str):
    query_example = copy.copy(base_example)
    query_example["question"] = question.strip()
    return query_example


def build_context_from_chunks(retrieved_chunks, top_k: int = 5) -> str:
    context_parts = []

    for chunk in retrieved_chunks[:top_k]:
        title = chunk.metadata.get("title", "Unknown Title")
        text = chunk.page_content.strip()
        context_parts.append(f"Title: {title}\n{text}")

    return "\n\n".join(context_parts)


def generate_answer_openai(question: str, retrieved_chunks, top_k: int = 5) -> str:
    if openai_client is None:
        return "OPENAI_API_KEY not set"

    context = build_context_from_chunks(retrieved_chunks, top_k=top_k)
    prompt = f"""
Answer the question using ONLY the provided context.

The answer may require combining information from multiple documents.

Return ONLY the final answer.

Output must be a valid JSON object:

{{"answer": "<short final answer>"}}

Strict Rules:
- Use only information from the context.
- Do NOT explain reasoning.
- Do NOT add extra words.
- Prefer exact names found in the context.
- Answer must be short (1–5 words).
- If multiple possible answers exist, return the most direct one.
- If the answer is missing, return:
  {{"answer": "Insufficient evidence"}}

Question:
{question}

Context:
{context}
""".strip()

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
    )

    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).strip()
        return answer or "Insufficient evidence"
    except Exception:
        return raw or "Insufficient evidence"


def generate_final_answer(
    question: str,
    retrieved_chunks,
    enabled: bool,
    top_k: int,
    question_type: str | None = None,
):
    if not enabled:
        if question_type == "bridge":
            return "Insufficient evidence"
        return generate_retrieval_fallback_answer(question, retrieved_chunks, top_k=top_k)
    return generate_answer_openai(question, retrieved_chunks, top_k=top_k)


def compute_title_overlap(gold_titles, retrieved_titles):
    gold_set = set(gold_titles or [])
    retrieved_set = set(retrieved_titles or [])
    overlap = sorted(gold_set & retrieved_set)
    return {
        "match_count": len(overlap),
        "gold_count": len(gold_set),
        "all_matched": gold_set.issubset(retrieved_set) if gold_set else False,
        "overlap_titles": overlap,
    }


def run_dense_query(query_example, resources, top_k: int):
    result = retrieve_top_k_chunks_for_example(
        example=query_example,
        model=resources["embed_model"],
        top_k=top_k,
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = MODE_DENSE
    return result


def rerank_result_by_indices(result, ranked_indices, mode_name: str):
    reranked = dict(result)
    reranked["retrieved_chunks"] = [result["retrieved_chunks"][i] for i in ranked_indices]

    for key in ["scores", "dense_scores", "gnn_scores", "fusion_scores"]:
        values = result.get(key)
        if values:
            reranked[key] = [values[i] for i in ranked_indices]

    title_score_key = None
    for key in ["scores", "gnn_scores", "fusion_scores", "dense_scores"]:
        if reranked.get(key):
            title_score_key = key
            break

    if title_score_key is not None:
        title_best_score = {}
        title_order = []
        for chunk, score in zip(reranked["retrieved_chunks"], reranked[title_score_key]):
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

    reranked["mode"] = mode_name
    return reranked


def get_custom_candidate_pool_size(top_k: int) -> int:
    return max(30, min(50, top_k * 8))


def global_dense_candidate_retrieval(
    question: str,
    resources,
    candidate_pool_size: int,
    custom_backend: str = CUSTOM_BACKEND_EXACT,
):
    query_embedding = resources["embed_model"].encode(
        QUERY_PREFIX + question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    global_embeddings = resources["global_embeddings"]
    if len(global_embeddings) == 0:
        return {
            "query_embedding": query_embedding,
            "candidate_indices": np.array([], dtype=np.int64),
            "candidate_scores": np.array([], dtype=np.float32),
        }

    candidate_pool_size = min(candidate_pool_size, len(global_embeddings))
    ann_indexes = resources.get("custom_ann_indexes", {})
    requested_backend = custom_backend or CUSTOM_BACKEND_EXACT
    backend_used = CUSTOM_BACKEND_EXACT

    if requested_backend == CUSTOM_BACKEND_HNSW and ann_indexes.get(CUSTOM_BACKEND_HNSW) is not None:
        labels, distances = ann_indexes[CUSTOM_BACKEND_HNSW].knn_query(query_embedding, k=candidate_pool_size)
        top_indices = labels[0].astype(np.int64)
        candidate_scores = (1.0 - distances[0]).astype(np.float32)
        backend_used = CUSTOM_BACKEND_HNSW
        return {
            "query_embedding": query_embedding,
            "candidate_indices": top_indices,
            "candidate_scores": candidate_scores,
            "backend_used": backend_used,
        }

    if requested_backend == CUSTOM_BACKEND_IVF and ann_indexes.get(CUSTOM_BACKEND_IVF) is not None:
        candidate_scores, top_indices = ann_indexes[CUSTOM_BACKEND_IVF].search(
            query_embedding.reshape(1, -1),
            candidate_pool_size,
        )
        top_indices = top_indices[0]
        valid_mask = top_indices >= 0
        top_indices = top_indices[valid_mask].astype(np.int64)
        candidate_scores = candidate_scores[0][valid_mask].astype(np.float32)
        backend_used = CUSTOM_BACKEND_IVF
        return {
            "query_embedding": query_embedding,
            "candidate_indices": top_indices,
            "candidate_scores": candidate_scores,
            "backend_used": backend_used,
        }

    scores = np.dot(global_embeddings, query_embedding).astype(np.float32)

    if candidate_pool_size >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -candidate_pool_size)[-candidate_pool_size:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return {
        "query_embedding": query_embedding,
        "candidate_indices": top_indices.astype(np.int64),
        "candidate_scores": scores[top_indices].astype(np.float32),
        "backend_used": backend_used,
    }


def build_custom_candidate_example(question: str, candidate_pack, resources):
    candidate_indices = candidate_pack["candidate_indices"].tolist()
    candidate_chunks = [resources["global_chunks"][idx] for idx in candidate_indices]
    candidate_embeddings = resources["global_embeddings"][candidate_indices].astype(np.float32)

    reduced_graph = resources["global_graph"].subgraph(candidate_indices).copy()
    relabel_map = {old_idx: new_idx for new_idx, old_idx in enumerate(candidate_indices)}
    reduced_graph = nx.relabel_nodes(reduced_graph, relabel_map, copy=True)

    return {
        "id": "custom_query",
        "question": question,
        "answer": "",
        "type": "custom",
        "supporting_facts": {"title": [], "sent_id": []},
        "context_docs": [],
        "context_chunks": candidate_chunks,
        "context_chunk_embeddings": candidate_embeddings,
        "graph": reduced_graph,
        "candidate_global_indices": candidate_indices,
        "candidate_dense_scores": candidate_pack["candidate_scores"].tolist(),
        "custom_backend": candidate_pack.get("backend_used", CUSTOM_BACKEND_EXACT),
    }


def build_dense_result_from_candidates(question: str, candidate_example, top_k: int):
    chunks = candidate_example["context_chunks"]
    scores = np.asarray(candidate_example["candidate_dense_scores"], dtype=np.float32)
    top_k = min(top_k, len(chunks))

    retrieved_chunks = chunks[:top_k]
    retrieved_scores = [float(score) for score in scores[:top_k]]
    retrieved_global_indices = candidate_example["candidate_global_indices"][:top_k]

    retrieved_titles = []
    seen_titles = set()
    for chunk in retrieved_chunks:
        title = chunk.metadata["title"]
        if title not in seen_titles:
            retrieved_titles.append(title)
            seen_titles.add(title)

    return {
        "id": "custom_query",
        "question": question,
        "answer": "",
        "type": "custom",
        "gold_titles": [],
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "scores": retrieved_scores,
        "mode": MODE_DENSE,
        "selected_global_indices": retrieved_global_indices,
        "candidate_pool_size": len(chunks),
        "custom_backend": candidate_example.get("custom_backend", CUSTOM_BACKEND_EXACT),
    }


def attach_global_indices(result, candidate_example, local_indices=None):
    global_indices = candidate_example["candidate_global_indices"]

    if local_indices is not None:
        mapped = [global_indices[idx] for idx in local_indices]
    else:
        title_and_text_to_global = {}
        for local_idx, chunk in enumerate(candidate_example["context_chunks"]):
            key = (chunk.metadata.get("chunk_id"), chunk.page_content)
            title_and_text_to_global[key] = global_indices[local_idx]

        mapped = []
        for chunk in result.get("retrieved_chunks", []):
            key = (chunk.metadata.get("chunk_id"), chunk.page_content)
            mapped.append(title_and_text_to_global.get(key))

    result["selected_global_indices"] = mapped
    result["candidate_pool_size"] = len(candidate_example["context_chunks"])
    result["custom_backend"] = candidate_example.get("custom_backend", CUSTOM_BACKEND_EXACT)
    return result


def ensure_gnn_available(resources):
    if resources["gnn_model"] is None:
        raise FileNotFoundError(
            "GNN retrieval, Dense retrieval + Query-Aware GraphSAGE, and Dense retrieval + Query-Aware GraphSAGE + PCST (Main Method) require a trained GNN checkpoint in artifacts/. "
            "Run graphrag_env/src/gnn_train.py first."
        )


def run_custom_query(
    question: str,
    retrieval_mode: str,
    resources,
    top_k: int,
    lambda_dense: float,
    custom_backend: str = CUSTOM_BACKEND_EXACT,
):
    candidate_pack = global_dense_candidate_retrieval(
        question=question,
        resources=resources,
        candidate_pool_size=get_custom_candidate_pool_size(top_k),
        custom_backend=custom_backend,
    )
    candidate_example = build_custom_candidate_example(question, candidate_pack, resources)

    if retrieval_mode == MODE_DENSE:
        return build_dense_result_from_candidates(question, candidate_example, top_k)

    if retrieval_mode == MODE_PCST_DENSE:
        result = pcst_dense_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            top_k=top_k,
            seed_k=min(3, top_k),
            query_prefix=QUERY_PREFIX,
        )
        result["mode"] = MODE_PCST_DENSE
        return attach_global_indices(result, candidate_example, local_indices=result.get("selected_nodes"))

    if retrieval_mode == MODE_GNN:
        ensure_gnn_available(resources)
        result = gnn_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            query_prefix=QUERY_PREFIX,
        )
        result["mode"] = MODE_GNN
        return attach_global_indices(result, candidate_example)

    if retrieval_mode == MODE_FUSION:
        ensure_gnn_available(resources)
        result = dense_gnn_fusion_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        result["mode"] = MODE_FUSION
        return attach_global_indices(result, candidate_example)

    if retrieval_mode == MODE_PCST_LEARNED:
        ensure_gnn_available(resources)
        result = pcst_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            seed_k=min(PCST_LEARNED_SEED_K, top_k),
            expansion_factor=PCST_LEARNED_EXPANSION_FACTOR,
            fusion_anchor_pool_factor=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR,
            pcst_bonus=PCST_LEARNED_BONUS,
            preserve_fusion_top_k=min(PCST_LEARNED_PRESERVE_FUSION_TOP_K, top_k),
            title_diversity_bonus=PCST_LEARNED_TITLE_DIVERSITY_BONUS,
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        result["mode"] = MODE_PCST_LEARNED
        return attach_global_indices(result, candidate_example, local_indices=result.get("selected_nodes"))

    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def run_fusion_query(query_example, resources, top_k: int, lambda_dense: float):
    ensure_gnn_available(resources)
    result = dense_gnn_fusion_retrieve_for_example(
        example=query_example,
        embed_model=resources["embed_model"],
        gnn_model=resources["gnn_model"],
        device=resources["device"],
        top_k=top_k,
        lambda_dense=lambda_dense,
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = MODE_FUSION
    return result


def run_gnn_query(query_example, resources, top_k: int):
    ensure_gnn_available(resources)
    result = gnn_retrieve_for_example(
        example=query_example,
        embed_model=resources["embed_model"],
        gnn_model=resources["gnn_model"],
        device=resources["device"],
        top_k=top_k,
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = MODE_GNN
    return result


def run_pcst_dense_query(query_example, resources, top_k: int):
    result = pcst_dense_retrieve_for_example(
        example=query_example,
        embed_model=resources["embed_model"],
        top_k=top_k,
        seed_k=min(3, top_k),
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = MODE_PCST_DENSE
    return result


def run_pcst_query(query_example, resources, top_k: int, lambda_dense: float):
    ensure_gnn_available(resources)
    result = pcst_retrieve_for_example(
        example=query_example,
        embed_model=resources["embed_model"],
        gnn_model=resources["gnn_model"],
        device=resources["device"],
        top_k=top_k,
        seed_k=min(PCST_LEARNED_SEED_K, top_k),
        expansion_factor=PCST_LEARNED_EXPANSION_FACTOR,
        fusion_anchor_pool_factor=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR,
        pcst_bonus=PCST_LEARNED_BONUS,
        preserve_fusion_top_k=min(PCST_LEARNED_PRESERVE_FUSION_TOP_K, top_k),
        title_diversity_bonus=PCST_LEARNED_TITLE_DIVERSITY_BONUS,
        lambda_dense=lambda_dense,
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = MODE_PCST_LEARNED
    return result


def run_single_query(query_example, retrieval_mode: str, resources, top_k: int, lambda_dense: float):
    if retrieval_mode == MODE_DENSE:
        return run_dense_query(query_example, resources, top_k)
    if retrieval_mode == MODE_PCST_DENSE:
        return run_pcst_dense_query(query_example, resources, top_k)
    if retrieval_mode == MODE_GNN:
        return run_gnn_query(query_example, resources, top_k)
    if retrieval_mode == MODE_FUSION:
        return run_fusion_query(query_example, resources, top_k, lambda_dense)
    if retrieval_mode == MODE_PCST_LEARNED:
        return run_pcst_query(query_example, resources, top_k, lambda_dense)
    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def run_comparison_query(query_example, resources, top_k: int, lambda_dense: float):
    results = {
        MODE_DENSE: run_dense_query(query_example, resources, top_k),
        MODE_PCST_DENSE: run_pcst_dense_query(query_example, resources, top_k),
    }

    if resources["gnn_model"] is not None:
        results[MODE_GNN] = run_gnn_query(query_example, resources, top_k)
        results[MODE_FUSION] = run_fusion_query(query_example, resources, top_k, lambda_dense)
        results[MODE_PCST_LEARNED] = run_pcst_query(query_example, resources, top_k, lambda_dense)

    return results


def run_custom_comparison_query(
    question: str,
    resources,
    top_k: int,
    lambda_dense: float,
    custom_backend: str = CUSTOM_BACKEND_EXACT,
):
    candidate_pack = global_dense_candidate_retrieval(
        question=question,
        resources=resources,
        candidate_pool_size=get_custom_candidate_pool_size(top_k),
        custom_backend=custom_backend,
    )
    candidate_example = build_custom_candidate_example(question, candidate_pack, resources)

    dense_result = build_dense_result_from_candidates(question, candidate_example, top_k)
    dense_result["mode"] = MODE_DENSE
    pcst_dense_result = run_custom_query(
        question=question,
        retrieval_mode=MODE_PCST_DENSE,
        resources=resources,
        top_k=top_k,
        lambda_dense=lambda_dense,
        custom_backend=custom_backend,
    )
    results = {
        MODE_DENSE: dense_result,
        MODE_PCST_DENSE: pcst_dense_result,
    }

    if resources["gnn_model"] is not None:
        gnn_result = gnn_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            query_prefix=QUERY_PREFIX,
        )
        gnn_result["mode"] = MODE_GNN
        results[MODE_GNN] = attach_global_indices(gnn_result, candidate_example)

        fusion_result = dense_gnn_fusion_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        fusion_result["mode"] = MODE_FUSION
        results[MODE_FUSION] = attach_global_indices(fusion_result, candidate_example)

        pcst_result = pcst_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            seed_k=min(PCST_LEARNED_SEED_K, top_k),
            expansion_factor=PCST_LEARNED_EXPANSION_FACTOR,
            fusion_anchor_pool_factor=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR,
            pcst_bonus=PCST_LEARNED_BONUS,
            preserve_fusion_top_k=min(PCST_LEARNED_PRESERVE_FUSION_TOP_K, top_k),
            title_diversity_bonus=PCST_LEARNED_TITLE_DIVERSITY_BONUS,
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        pcst_result["mode"] = MODE_PCST_LEARNED
        results[MODE_PCST_LEARNED] = attach_global_indices(
            pcst_result,
            candidate_example,
            local_indices=pcst_result.get("selected_nodes"),
        )

    return results


def build_comparison_rows(
    question,
    comparison_results,
    llm_enabled,
    top_k,
    gold_titles=None,
    question_type: str | None = None,
):
    rows = []
    best_mode = None
    best_match_count = -1

    for mode_name, result in comparison_results.items():
        final_answer = generate_final_answer(
            question,
            result["retrieved_chunks"],
            llm_enabled,
            top_k,
            question_type=question_type,
        )
        row = {
            "mode": mode_name,
            "retrieved_titles": result.get("retrieved_titles", []),
            "final_answer": final_answer,
        }

        if gold_titles is not None:
            overlap = compute_title_overlap(gold_titles, result.get("retrieved_titles", []))
            row["gold_title_matches"] = overlap["match_count"]
            row["all_gold_titles_matched"] = overlap["all_matched"]
            if overlap["match_count"] > best_match_count:
                best_match_count = overlap["match_count"]
                best_mode = mode_name

        rows.append(row)

    return rows, best_mode


def serialize_chunk(chunk, index: int, scores: dict[str, list[float]] | None = None):
    payload = {
        "rank": index + 1,
        "title": chunk.metadata.get("title", "Unknown Title"),
        "text": chunk.page_content,
        "is_supporting": bool(chunk.metadata.get("is_supporting", False)),
        "metadata": {
            key: value
            for key, value in chunk.metadata.items()
            if isinstance(value, (str, int, float, bool, list, dict)) or value is None
        },
    }
    if scores:
        for key, values in scores.items():
            if index < len(values):
                payload[key] = values[index]
    return payload


def serialize_result(result, llm_enabled: bool, top_k: int, gold_titles=None):
    score_payload = {
        "scores": [float(v) for v in result.get("scores", [])],
        "dense_scores": [float(v) for v in result.get("dense_scores", [])],
        "gnn_scores": [float(v) for v in result.get("gnn_scores", [])],
        "fusion_scores": [float(v) for v in result.get("fusion_scores", [])],
    }
    retrieved_chunks = [
        serialize_chunk(chunk, idx, score_payload)
        for idx, chunk in enumerate(result.get("retrieved_chunks", []))
    ]

    response = {
        "mode": result.get("mode", "Unknown"),
        "retrieved_titles": result.get("retrieved_titles", []),
        "retrieved_chunks": retrieved_chunks,
        "selected_nodes": result.get("selected_nodes", []),
        "selected_global_indices": result.get("selected_global_indices", []),
        "candidate_pool_size": result.get("candidate_pool_size"),
        "custom_backend": result.get("custom_backend"),
        "final_answer": generate_final_answer(
            result.get("question", ""),
            result.get("retrieved_chunks", []),
            llm_enabled,
            top_k,
        ),
    }
    if gold_titles is not None:
        response["gold_titles"] = gold_titles
        response["title_overlap"] = compute_title_overlap(gold_titles, result.get("retrieved_titles", []))
    return response


def list_examples(profile_id: str | None = None, question_type: str = "all"):
    examples = load_example_index(profile_id)

    if question_type != "all":
        examples = [example for example in examples if example["type"] == question_type]

    return examples


def get_runtime_config(profile_id: str | None = None):
    return load_runtime_metadata(profile_id)


def execute_dataset_query(
    profile_id: str,
    example_id: str,
    retrieval_mode: str,
    top_k: int,
    lambda_dense: float,
    llm_enabled: bool,
    compare_all_modes: bool,
):
    resources = load_resources(profile_id)
    example = resources["example_lookup"][example_id]
    query_example = build_query_example(example, example["question"])
    gold_titles = example["supporting_facts"]["title"]

    if compare_all_modes:
        comparison_results = run_comparison_query(query_example, resources, top_k, lambda_dense)
        comparison_rows, best_mode = build_comparison_rows(
            question=example["question"],
            comparison_results=comparison_results,
            llm_enabled=llm_enabled,
            top_k=top_k,
            gold_titles=gold_titles,
            question_type=example.get("type"),
        )
        active_result = comparison_results.get(retrieval_mode) or next(iter(comparison_results.values()))
    else:
        comparison_rows = []
        best_mode = None
        active_result = run_single_query(query_example, retrieval_mode, resources, top_k, lambda_dense)

    active_result["question"] = example["question"]
    return {
        "question": example["question"],
        "question_type": example.get("type", "unknown"),
        "gold_answer": example.get("answer", ""),
        "graph_stats": graph_stats(example["graph"]),
        "result": serialize_result(active_result, llm_enabled, top_k, gold_titles=gold_titles),
        "comparison": comparison_rows,
        "best_mode": best_mode,
    }


def execute_custom_query(
    profile_id: str,
    question: str,
    retrieval_mode: str,
    top_k: int,
    lambda_dense: float,
    llm_enabled: bool,
    compare_all_modes: bool,
    custom_backend: str,
):
    resources = load_resources(profile_id)

    if compare_all_modes:
        comparison_results = run_custom_comparison_query(
            question=question,
            resources=resources,
            top_k=top_k,
            lambda_dense=lambda_dense,
            custom_backend=custom_backend,
        )
        comparison_rows, _ = build_comparison_rows(
            question=question,
            comparison_results=comparison_results,
            llm_enabled=llm_enabled,
            top_k=top_k,
        )
        active_result = comparison_results.get(retrieval_mode) or next(iter(comparison_results.values()))
    else:
        comparison_rows = []
        active_result = run_custom_query(
            question=question,
            retrieval_mode=retrieval_mode,
            resources=resources,
            top_k=top_k,
            lambda_dense=lambda_dense,
            custom_backend=custom_backend,
        )

    active_result["question"] = question
    return {
        "question": question,
        "result": serialize_result(active_result, llm_enabled, top_k),
        "comparison": comparison_rows,
    }
