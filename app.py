import copy
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import streamlit as st
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
from graphrag_env.src.gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_example
from graphrag_env.src.hybrid_graph_builder import graph_stats
from graphrag_env.src.llm_eval import generate_answer_openai
from graphrag_env.src.pcst import (
    compute_fusion_scores,
    multiseed_pcst_selection,
    pcst_retrieve_for_example,
)
from graphrag_env.src.retrieval import retrieve_top_k_chunks_for_example


APP_TITLE = "GraphRAG Multi-Hop Question Answering System"
APP_SUBTITLE = (
    "Answer HotpotQA-style multi-hop factoid questions using dense retrieval, "
    "graph reasoning, GNN fusion, and multi-seed PCST over an indexed corpus."
)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_PROFILE = {
    "split": "train",
    "max_samples": 10000,
    "chunk_size": 300,
    "chunk_overlap": 50,
}
CUSTOM_BACKEND_EXACT = "Exact"
CUSTOM_BACKEND_HNSW = "HNSW"
CUSTOM_BACKEND_IVF = "IVF"


def discover_artifact_profiles():
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

    if not profiles:
        profiles.append({**DEFAULT_PROFILE, "label": "Default profile", "manifest": None})

    return profiles


@st.cache_resource(show_spinner=False)
def load_demo_bundle(split: str, max_samples: int, chunk_size: int, chunk_overlap: int):
    bundle = load_artifact_bundle(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embed_model = SentenceTransformer(bundle["manifest"]["model_name"])

    gnn_model = None
    device = None
    checkpoint_path = None

    try:
        gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
            graph_examples=bundle["graph_examples"],
            embed_model=embed_model,
            split=split,
            max_samples=max_samples,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except FileNotFoundError:
        pass

    ann_indexes = build_custom_ann_indexes(bundle["global_example"]["context_chunk_embeddings"])

    return {
        **bundle,
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


def generate_final_answer(question: str, retrieved_chunks, enabled: bool, top_k: int):
    if not enabled:
        return "Retrieval only mode"
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
    result["mode"] = "Dense"
    return result


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
        "mode": "Dense",
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

    if retrieval_mode == "Dense":
        return build_dense_result_from_candidates(question, candidate_example, top_k)

    if retrieval_mode == "Fusion":
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
        result["mode"] = "Fusion"
        return attach_global_indices(result, candidate_example)

    if retrieval_mode == "PCST":
        ensure_gnn_available(resources)
        result = pcst_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            seed_k=min(3, top_k),
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        dense_scores, gnn_scores, fusion_scores = compute_fusion_scores(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        selected_nodes = multiseed_pcst_selection(
            example=candidate_example,
            fusion_scores=fusion_scores,
            seed_k=min(3, top_k),
            max_nodes=top_k,
        )
        result["mode"] = "PCST"
        result["dense_scores"] = [float(dense_scores[i]) for i in selected_nodes]
        result["gnn_scores"] = [float(gnn_scores[i]) for i in selected_nodes]
        result["fusion_scores"] = [float(fusion_scores[i]) for i in selected_nodes]
        result["selected_nodes"] = selected_nodes
        return attach_global_indices(result, candidate_example, local_indices=selected_nodes)

    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


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

    results = {
        "Dense": build_dense_result_from_candidates(question, candidate_example, top_k)
    }

    if resources["gnn_model"] is not None:
        fusion_result = dense_gnn_fusion_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        fusion_result["mode"] = "Fusion"
        results["Fusion"] = attach_global_indices(fusion_result, candidate_example)

        pcst_result = pcst_retrieve_for_example(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            top_k=top_k,
            seed_k=min(3, top_k),
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        dense_scores, gnn_scores, fusion_scores = compute_fusion_scores(
            example=candidate_example,
            embed_model=resources["embed_model"],
            gnn_model=resources["gnn_model"],
            device=resources["device"],
            lambda_dense=lambda_dense,
            query_prefix=QUERY_PREFIX,
        )
        selected_nodes = multiseed_pcst_selection(
            example=candidate_example,
            fusion_scores=fusion_scores,
            seed_k=min(3, top_k),
            max_nodes=top_k,
        )
        pcst_result["mode"] = "PCST"
        pcst_result["dense_scores"] = [float(dense_scores[i]) for i in selected_nodes]
        pcst_result["gnn_scores"] = [float(gnn_scores[i]) for i in selected_nodes]
        pcst_result["fusion_scores"] = [float(fusion_scores[i]) for i in selected_nodes]
        pcst_result["selected_nodes"] = selected_nodes
        results["PCST"] = attach_global_indices(pcst_result, candidate_example, local_indices=selected_nodes)

    return results


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
    result["mode"] = "Fusion"
    return result


def run_pcst_query(query_example, resources, top_k: int, lambda_dense: float):
    ensure_gnn_available(resources)
    result = pcst_retrieve_for_example(
        example=query_example,
        embed_model=resources["embed_model"],
        gnn_model=resources["gnn_model"],
        device=resources["device"],
        top_k=top_k,
        seed_k=min(3, top_k),
        lambda_dense=lambda_dense,
        query_prefix=QUERY_PREFIX,
    )
    result["mode"] = "PCST"

    dense_scores, gnn_scores, fusion_scores = compute_fusion_scores(
        example=query_example,
        embed_model=resources["embed_model"],
        gnn_model=resources["gnn_model"],
        device=resources["device"],
        lambda_dense=lambda_dense,
        query_prefix=QUERY_PREFIX,
    )
    selected_nodes = multiseed_pcst_selection(
        example=query_example,
        fusion_scores=fusion_scores,
        seed_k=min(3, top_k),
        max_nodes=top_k,
    )

    result["dense_scores"] = [float(dense_scores[i]) for i in selected_nodes]
    result["gnn_scores"] = [float(gnn_scores[i]) for i in selected_nodes]
    result["selected_nodes"] = selected_nodes
    return result


def ensure_gnn_available(resources):
    if resources["gnn_model"] is None:
        raise FileNotFoundError(
            "Fusion and PCST require a trained GNN checkpoint in artifacts/. "
            "Run graphrag_env/src/gnn_train.py first."
        )


def run_single_query(query_example, retrieval_mode: str, resources, top_k: int, lambda_dense: float):
    if retrieval_mode == "Dense":
        return run_dense_query(query_example, resources, top_k)
    if retrieval_mode == "Fusion":
        return run_fusion_query(query_example, resources, top_k, lambda_dense)
    if retrieval_mode == "PCST":
        return run_pcst_query(query_example, resources, top_k, lambda_dense)
    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def run_comparison_query(query_example, resources, top_k: int, lambda_dense: float):
    results = {}
    results["Dense"] = run_dense_query(query_example, resources, top_k)

    if resources["gnn_model"] is not None:
        results["Fusion"] = run_fusion_query(query_example, resources, top_k, lambda_dense)
        results["PCST"] = run_pcst_query(query_example, resources, top_k, lambda_dense)

    return results


def build_example_index(resources):
    examples = list(resources["example_lookup"].values())
    examples.sort(key=lambda ex: ex["question"])
    return examples


def filter_examples(examples, question_type: str):
    if question_type == "all":
        return examples
    return [example for example in examples if example.get("type") == question_type]


def get_random_example_id(filtered_examples):
    if not filtered_examples:
        return None
    return random.choice(filtered_examples)["id"]


def render_example_guidance(sample_questions):
    st.markdown("**Example questions**")
    sample_cols = st.columns(2)
    for idx, item in enumerate(sample_questions[:8]):
        sample_cols[idx % 2].caption(f"- {item['question']}")


def render_answer_banner(answer_text: str, retrieval_only: bool):
    if retrieval_only:
        st.info(answer_text)
    else:
        st.success(answer_text)


def render_result_summary(result, gold_titles=None):
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval Mode", result.get("mode", "Unknown"))
    col2.metric("Retrieved Titles", len(result.get("retrieved_titles", [])))
    col3.metric("Evidence Chunks", len(result.get("retrieved_chunks", [])))

    if gold_titles is not None:
        overlap = compute_title_overlap(gold_titles, result.get("retrieved_titles", []))
        match_text = "Yes" if overlap["all_matched"] else "No"
        st.caption(
            f"Gold title match: {match_text} | "
            f"matched {overlap['match_count']} / {overlap['gold_count']} titles"
        )
        if overlap["overlap_titles"]:
            st.caption(f"Overlap titles: {', '.join(overlap['overlap_titles'])}")


def render_titles_block(result, gold_titles=None):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Retrieved titles**")
        titles = result.get("retrieved_titles", [])
        if titles:
            for title in titles:
                st.write(f"- {title}")
        else:
            st.write("No titles retrieved.")

    with col2:
        if gold_titles is not None:
            st.markdown("**Gold supporting titles**")
            if gold_titles:
                for title in gold_titles:
                    st.write(f"- {title}")
            else:
                st.write("No gold titles available.")


def render_evidence_chunks(result, debug_mode: bool):
    st.markdown("**Retrieved evidence chunks**")
    retrieved_chunks = result.get("retrieved_chunks", [])
    dense_scores = result.get("scores") or result.get("dense_scores") or []
    gnn_scores = result.get("gnn_scores") or []
    fusion_scores = result.get("fusion_scores") or []

    if not retrieved_chunks:
        st.write("No evidence retrieved.")
        return

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.metadata.get("title", "Unknown Title")
        support_label = "Supporting" if chunk.metadata.get("is_supporting", False) else "Non-supporting"
        with st.expander(f"{idx}. {title} [{support_label}]", expanded=idx <= 2):
            st.write(chunk.page_content)
            if debug_mode:
                score_payload = {}
                if idx - 1 < len(dense_scores):
                    score_payload["dense_score"] = dense_scores[idx - 1]
                if idx - 1 < len(gnn_scores):
                    score_payload["gnn_score"] = gnn_scores[idx - 1]
                if idx - 1 < len(fusion_scores):
                    score_payload["fusion_score"] = fusion_scores[idx - 1]

                if score_payload:
                    st.json(score_payload)
                st.json(chunk.metadata)


def render_graph_stats(example):
    stats = graph_stats(example["graph"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", stats["num_nodes"])
    col2.metric("Edges", stats["num_edges"])
    col3.metric("Supporting Nodes", stats["supporting_nodes"])
    return stats


def render_debug_panel(result, example=None, graph_summary=None):
    with st.expander("Debug details", expanded=False):
        debug_payload = {
            "mode": result.get("mode"),
            "retrieved_titles": result.get("retrieved_titles", []),
            "scores": result.get("scores", []),
            "dense_scores": result.get("dense_scores", []),
            "gnn_scores": result.get("gnn_scores", []),
            "fusion_scores": result.get("fusion_scores", []),
            "selected_nodes": result.get("selected_nodes", []),
            "selected_global_indices": result.get("selected_global_indices", []),
            "candidate_pool_size": result.get("candidate_pool_size"),
        }
        st.json(debug_payload)

        if graph_summary is not None:
            st.markdown("**Graph summary**")
            st.json(graph_summary)

        if example is not None:
            st.markdown("**Example metadata**")
            st.json(
                {
                    "id": example.get("id"),
                    "type": example.get("type"),
                    "level": example.get("level"),
                    "answer": example.get("answer"),
                }
            )


def build_comparison_rows(question, comparison_results, llm_enabled, top_k, gold_titles=None):
    rows = []
    best_mode = None
    best_match_count = -1

    for mode_name, result in comparison_results.items():
        final_answer = generate_final_answer(question, result["retrieved_chunks"], llm_enabled, top_k)
        row = {
            "mode": mode_name,
            "retrieved_titles": ", ".join(result.get("retrieved_titles", [])),
            "final_answer": final_answer,
        }

        if gold_titles is not None:
            overlap = compute_title_overlap(gold_titles, result.get("retrieved_titles", []))
            row["gold_title_matches"] = overlap["match_count"]
            if overlap["match_count"] > best_match_count:
                best_match_count = overlap["match_count"]
                best_mode = mode_name

        rows.append(row)

    return rows, best_mode


def render_header(resources, profile_label):
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    banner_col1, banner_col2 = st.columns([3, 2])
    with banner_col1:
        st.write(
            "This app performs query-time inference only. All chunking, embeddings, graph construction, "
            "and model preparation are loaded from saved offline artifacts."
        )
    with banner_col2:
        st.info(f"Artifact profile: {profile_label}")

    render_example_guidance(resources["sample_questions"])


def render_dataset_mode(resources, retrieval_mode, top_k, lambda_dense, llm_enabled, debug_mode, compare_all_modes):
    st.subheader("Dataset Question Mode")

    all_examples = build_example_index(resources)
    filter_col, action_col = st.columns([2, 1])
    with filter_col:
        question_type = st.selectbox("Question type filter", ["all", "bridge", "comparison"])
    filtered_examples = filter_examples(all_examples, question_type)

    if not filtered_examples:
        st.warning("No examples found for the selected filter.")
        return

    random_key = f"dataset_selected_id_{question_type}"
    if random_key not in st.session_state:
        st.session_state[random_key] = filtered_examples[0]["id"]

    with action_col:
        st.write("")
        st.write("")
        if st.button("Random sample question", use_container_width=True):
            st.session_state[random_key] = get_random_example_id(filtered_examples)

    example_ids = [example["id"] for example in filtered_examples]
    if st.session_state[random_key] not in example_ids:
        st.session_state[random_key] = example_ids[0]

    with st.form(f"dataset_question_form_{question_type}", clear_on_submit=False):
        selected_id = st.selectbox(
            "Choose an indexed dataset question",
            options=example_ids,
            index=example_ids.index(st.session_state[random_key]),
            format_func=lambda item_id: next(ex["question"] for ex in filtered_examples if ex["id"] == item_id),
            key=f"dataset_select_{question_type}",
        )
        run_dataset_query = st.form_submit_button("Run question", use_container_width=True)
    st.session_state[random_key] = selected_id

    example = resources["example_lookup"][selected_id]
    query_example = build_query_example(example, example["question"])

    info_col1, info_col2 = st.columns([3, 2])
    with info_col1:
        st.markdown("**Question**")
        st.write(example["question"])
    with info_col2:
        st.markdown("**Question metadata**")
        st.write(f"Type: `{example.get('type', 'unknown')}`")
        st.write(f"Gold answer: `{example.get('answer', '')}`")

    st.markdown("**Graph stats**")
    graph_summary = render_graph_stats(example)

    if not run_dataset_query:
        st.caption("Click `Run question` to retrieve evidence for the selected dataset example.")
        return

    try:
        with st.spinner("Running retrieval for the selected dataset example..."):
            if compare_all_modes:
                comparison_results = run_comparison_query(query_example, resources, top_k, lambda_dense)
                rows, best_mode = build_comparison_rows(
                    question=example["question"],
                    comparison_results=comparison_results,
                    llm_enabled=llm_enabled,
                    top_k=top_k,
                    gold_titles=example["supporting_facts"]["title"],
                )
                st.markdown("**Comparison across retrieval modes**")
                st.dataframe(rows, use_container_width=True)
                if best_mode is not None:
                    st.caption(f"Best gold-title coverage: {best_mode}")
                result = comparison_results.get(retrieval_mode) or next(iter(comparison_results.values()))
            else:
                result = run_single_query(query_example, retrieval_mode, resources, top_k, lambda_dense)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    final_answer = generate_final_answer(example["question"], result["retrieved_chunks"], llm_enabled, top_k)
    st.markdown("**Final answer**")
    render_answer_banner(final_answer, retrieval_only=not llm_enabled)

    render_result_summary(result, gold_titles=example["supporting_facts"]["title"])
    render_titles_block(result, gold_titles=example["supporting_facts"]["title"])
    render_evidence_chunks(result, debug_mode=debug_mode)

    if debug_mode:
        render_debug_panel(result, example=example, graph_summary=graph_summary)


def render_custom_mode(resources, retrieval_mode, top_k, lambda_dense, llm_enabled, debug_mode, compare_all_modes):
    st.subheader("Custom Question Mode")
    st.write(
        "This system answers from the indexed HotpotQA / Wikipedia-style corpus. "
        "It works best for multi-hop factoid questions similar to the dataset, and may return "
        "`Insufficient evidence` when the supporting evidence is missing."
    )

    backend_options = get_custom_backend_options(resources)
    custom_backend = st.selectbox(
        "Custom retrieval backend",
        backend_options,
        help=(
            "This affects only custom questions. Dataset Question Mode still uses the original exact retrieval flow."
        ),
        key="custom_retrieval_backend",
    )
    if custom_backend != CUSTOM_BACKEND_EXACT:
        st.caption(
            f"`{custom_backend}` uses ANN to reduce custom-question latency by preselecting candidates approximately."
        )
    else:
        st.caption("`Exact` scores the custom question against every indexed chunk before graph reasoning.")

    with st.form("custom_question_form", clear_on_submit=False):
        custom_question = st.text_area(
            "Ask a custom question",
            placeholder="Example: Which magazine was started first Arthur's Magazine or First for Women?",
            height=100,
        )
        run_custom_query = st.form_submit_button("Run question", use_container_width=True)

    if not custom_question.strip():
        st.caption("Enter a question and click `Run question`.")
        return
    if not run_custom_query:
        st.caption("Click `Run question` to retrieve evidence and generate an answer.")
        return

    query_example = build_query_example(resources["global_example"], custom_question)

    try:
        with st.spinner("Running retrieval over the indexed corpus..."):
            if compare_all_modes:
                comparison_results = run_custom_comparison_query(
                    question=custom_question,
                    resources=resources,
                    top_k=top_k,
                    lambda_dense=lambda_dense,
                    custom_backend=custom_backend,
                )
                rows, _ = build_comparison_rows(
                    question=custom_question,
                    comparison_results=comparison_results,
                    llm_enabled=llm_enabled,
                    top_k=top_k,
                )
                st.markdown("**Comparison across retrieval modes**")
                st.dataframe(rows, use_container_width=True)
                result = comparison_results.get(retrieval_mode) or next(iter(comparison_results.values()))
            else:
                result = run_custom_query(
                    question=custom_question,
                    retrieval_mode=retrieval_mode,
                    resources=resources,
                    top_k=top_k,
                    lambda_dense=lambda_dense,
                    custom_backend=custom_backend,
                )
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    final_answer = generate_final_answer(custom_question, result["retrieved_chunks"], llm_enabled, top_k)
    st.markdown("**Final answer**")
    render_answer_banner(final_answer, retrieval_only=not llm_enabled)

    render_result_summary(result)
    st.caption(f"Retrieval mode used: {result.get('mode', retrieval_mode)}")
    st.caption(f"Candidate backend used for custom retrieval: {result.get('custom_backend', CUSTOM_BACKEND_EXACT)}")
    st.caption(
        "Custom questions are answered by searching the indexed corpus. "
        "Best results occur for HotpotQA-style multi-hop factoid questions."
    )
    render_titles_block(result)
    render_evidence_chunks(result, debug_mode=debug_mode)

    if debug_mode:
        render_debug_panel(result, example=None, graph_summary=None)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    profiles = discover_artifact_profiles()
    profile_labels = [profile["label"] for profile in profiles]

    st.sidebar.header("Controls")
    selected_profile_label = st.sidebar.selectbox("Artifact profile", profile_labels)
    selected_profile = next(profile for profile in profiles if profile["label"] == selected_profile_label)

    retrieval_mode = st.sidebar.selectbox("Retrieval mode", ["Dense", "Fusion", "PCST"])
    top_k = st.sidebar.slider("Top-k", min_value=1, max_value=10, value=5)
    lambda_dense = st.sidebar.slider("lambda_dense", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    llm_enabled = st.sidebar.checkbox("Enable GPT-4o-mini answer generation", value=False)
    debug_mode = st.sidebar.checkbox("Show debug details", value=False)
    compare_all_modes = st.sidebar.checkbox("Compare all retrieval modes", value=False)

    try:
        resources = load_demo_bundle(
            split=selected_profile["split"],
            max_samples=selected_profile["max_samples"],
            chunk_size=selected_profile["chunk_size"],
            chunk_overlap=selected_profile["chunk_overlap"],
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if retrieval_mode in {"Fusion", "PCST"} and resources["gnn_model"] is None:
        st.sidebar.warning("Fusion and PCST need a trained GNN checkpoint for this artifact profile.")
    if compare_all_modes and resources["gnn_model"] is None:
        st.sidebar.info("Comparison mode will show Dense only until a GNN checkpoint is available.")

    render_header(resources, selected_profile_label)

    dataset_tab, custom_tab = st.tabs(["Dataset Question Mode", "Custom Question Mode"])

    with dataset_tab:
        render_dataset_mode(
            resources=resources,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            lambda_dense=lambda_dense,
            llm_enabled=llm_enabled,
            debug_mode=debug_mode,
            compare_all_modes=compare_all_modes,
        )

    with custom_tab:
        render_custom_mode(
            resources=resources,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            lambda_dense=lambda_dense,
            llm_enabled=llm_enabled,
            debug_mode=debug_mode,
            compare_all_modes=compare_all_modes,
        )

    with st.expander("Artifact and model details", expanded=False):
        st.json(resources["manifest"])
        if resources["checkpoint_path"]:
            st.caption(f"GNN checkpoint: {resources['checkpoint_path']}")
        else:
            st.caption("No GNN checkpoint loaded for this profile.")


if __name__ == "__main__":
    main()
