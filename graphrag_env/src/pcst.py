import argparse
import numpy as np
import torch
from tqdm import tqdm

try:
    from .artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from .gnn_train import (
        build_pyg_data_from_example,
        build_pyg_dataset,
        split_dataset,
    )
    from .gnn_fusion_retreival import min_max_normalize
except ImportError:
    from artifact_runtime import load_gnn_from_checkpoint, load_or_build_graph_examples
    from gnn_train import (
        build_pyg_data_from_example,
        build_pyg_dataset,
        split_dataset,
    )
    from gnn_fusion_retreival import min_max_normalize


# ──────────────────────────────────────────────
# Fusion Score Computation
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_fusion_scores(
    example,
    embed_model,
    gnn_model,
    device,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (dense_scores_norm, gnn_scores_norm, fusion_scores) for all
    chunk nodes in one example.

    Builds the PyG data object once and reuses it for both the GNN forward
    pass and the dense dot-product — avoiding a redundant query re-encoding.
    """
    chunks           = example["context_chunks"]
    chunk_embeddings = example["context_chunk_embeddings"]

    query_text = query_prefix + example["question"]
    query_embedding = embed_model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # ----- Dense -----
    dense_scores      = np.dot(chunk_embeddings, query_embedding).astype(np.float32)
    dense_scores_norm = min_max_normalize(dense_scores)

    # ----- GNN — build data once, run forward once -----
    data = build_pyg_data_from_example(
        example=example,
        model=embed_model,
        query_prefix=query_prefix,
    )

    if data is None or data.x.shape[0] == 0:
        gnn_scores_norm = np.zeros(len(chunks), dtype=np.float32)
    else:
        data       = data.to(device)
        logits     = gnn_model(data.x, data.edge_index)
        gnn_raw    = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        gnn_scores_norm = min_max_normalize(gnn_raw)

    fusion_scores = (
        lambda_dense * dense_scores_norm
        + (1.0 - lambda_dense) * gnn_scores_norm
    ).astype(np.float32)

    return dense_scores_norm, gnn_scores_norm, fusion_scores


# ──────────────────────────────────────────────
# Multi-Seed PCST Expansion
# ──────────────────────────────────────────────

def multiseed_pcst_selection(
    example,
    fusion_scores: np.ndarray,
    seed_k: int = 3,
    max_nodes: int = 6,
) -> list[int]:
    """
    Multi-seed PCST-style node selection.

    Initialises from the top-k highest-prize seeds, then greedily expands
    neighbors while gain (prize − edge_cost) is positive.

    Fallback: if the graph becomes disconnected and no gain-positive neighbor
    exists, fill remaining slots with the highest-prize unselected nodes so we
    always return up to max_nodes results rather than stopping short.
    """
    G = example["graph"]

    if len(fusion_scores) == 0:
        return []

    # ✅ clamp seed_k so we never request more seeds than nodes exist
    seed_k = min(seed_k, len(fusion_scores))

    seed_nodes     = set(np.argsort(fusion_scores)[::-1][:seed_k].tolist())
    selected_nodes = set(seed_nodes)
    frontier       = set(seed_nodes)

    while len(selected_nodes) < max_nodes:
        best_gain = -np.inf
        best_node = None

        for node in frontier:
            for nbr in G.neighbors(node):
                if nbr in selected_nodes:
                    continue

                prize     = fusion_scores[nbr]
                edge_data = G.get_edge_data(node, nbr)
                if edge_data is None:
                    continue

                edge_weight = edge_data.get("weight", 1.0)
                cost        = 1.0 / (edge_weight + 1e-6)
                gain        = prize - cost

                if gain > best_gain:
                    best_gain = gain
                    best_node = nbr

        if best_node is not None:
            selected_nodes.add(best_node)
            frontier.add(best_node)
        else:
            # ✅ FIX: graph is disconnected — no gain-positive neighbor found.
            # Fill remaining slots from highest-prize unselected nodes so we
            # always return up to max_nodes results rather than stopping short.
            unselected = [
                i for i in range(len(fusion_scores)) if i not in selected_nodes
            ]
            if not unselected:
                break

            unselected_sorted = sorted(
                unselected, key=lambda i: fusion_scores[i], reverse=True
            )
            remaining = max_nodes - len(selected_nodes)
            selected_nodes.update(unselected_sorted[:remaining])
            break

    # ✅ FIX: sort by fusion score descending, not by node index.
    # Previously `sorted(selected_nodes)` ordered by node id, which has no
    # relation to relevance — the highest-scoring node could end up last.
    return sorted(selected_nodes, key=lambda i: fusion_scores[i], reverse=True)


# ──────────────────────────────────────────────
# PCST Retrieval per Example
# ──────────────────────────────────────────────

@torch.no_grad()
def pcst_retrieve_with_details_for_example(
    example,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    seed_k: int = 3,
    expansion_factor: int = 4,
    fusion_anchor_pool_factor: int = 2,
    pcst_bonus: float = 0.05,
    preserve_fusion_top_k: int = 2,
    title_diversity_bonus: float = 0.03,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    chunks = example["context_chunks"]

    empty_result = {
        "id":               example["id"],
        "question":         example["question"],
        "answer":           example["answer"],
        "type":             example["type"],
        "gold_titles":      example["supporting_facts"]["title"],
        "retrieved_chunks": [],
        "retrieved_titles": [],
        "dense_scores":     [],
        "gnn_scores":       [],
        "fusion_scores":    [],
        "selected_nodes":   [],
    }

    if len(chunks) == 0:
        return empty_result

    dense_scores_norm, gnn_scores_norm, fusion_scores = compute_fusion_scores(
        example=example,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        lambda_dense=lambda_dense,
        query_prefix=query_prefix,
    )

    expanded_node_budget = min(
        len(chunks),
        max(top_k, top_k * max(1, expansion_factor)),
    )

    selected_indices = multiseed_pcst_selection(
        example=example,
        fusion_scores=fusion_scores,
        seed_k=seed_k,
        max_nodes=expanded_node_budget,
    )

    # Preserve a wider fusion candidate pool and let PCST act as a graph-aware
    # booster over that pool instead of a hard replacement. This gives learned
    # PCST a chance to improve recall without discarding strong fusion hits too
    # early.
    fusion_anchor_pool_size = min(
        len(chunks),
        max(top_k, top_k * max(1, fusion_anchor_pool_factor)),
    )
    fusion_anchor_indices = np.argsort(fusion_scores)[::-1][:fusion_anchor_pool_size].tolist()
    preserved_anchor_indices = fusion_anchor_indices[: min(preserve_fusion_top_k, top_k, len(fusion_anchor_indices))]

    candidate_union = list(dict.fromkeys(fusion_anchor_indices + selected_indices))
    final_score_map = {}
    selected_set = set(selected_indices)
    for idx in candidate_union:
        bonus = pcst_bonus if idx in selected_set else 0.0
        final_score_map[idx] = float(fusion_scores[idx] + bonus)

    final_indices = []
    seen_titles = set()

    for idx in preserved_anchor_indices:
        final_indices.append(idx)
        seen_titles.add(chunks[idx].metadata["title"])

    remaining_candidates = [idx for idx in candidate_union if idx not in final_indices]
    while len(final_indices) < min(top_k, len(candidate_union)) and remaining_candidates:
        best_idx = None
        best_score = -np.inf

        for idx in remaining_candidates:
            title = chunks[idx].metadata["title"]
            diversity_bonus = title_diversity_bonus if title not in seen_titles else 0.0
            candidate_score = final_score_map[idx] + diversity_bonus
            if candidate_score > best_score:
                best_score = candidate_score
                best_idx = idx

        if best_idx is None:
            break

        final_indices.append(best_idx)
        seen_titles.add(chunks[best_idx].metadata["title"])
        remaining_candidates.remove(best_idx)

    retrieved_chunks        = [chunks[i]               for i in final_indices]
    retrieved_dense_scores  = [float(dense_scores_norm[i]) for i in final_indices]
    retrieved_gnn_scores    = [float(gnn_scores_norm[i]) for i in final_indices]
    retrieved_fusion_scores = [float(fusion_scores[i]) for i in final_indices]

    # Keep the MAX fusion score per title instead of silently dropping
    # chunks from the same title. Re-sort titles by that best score.
    title_best_score: dict[str, float] = {}
    title_order: list[str] = []

    for chunk, fscore in zip(retrieved_chunks, retrieved_fusion_scores):
        title = chunk.metadata["title"]
        if title not in title_best_score:
            title_best_score[title] = fscore
            title_order.append(title)
        else:
            title_best_score[title] = max(title_best_score[title], fscore)

    retrieved_titles = sorted(
        title_order,
        key=lambda t: title_best_score[t],
        reverse=True,
    )

    return {
        **empty_result,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "dense_scores":     retrieved_dense_scores,
        "gnn_scores":       retrieved_gnn_scores,
        "fusion_scores":    retrieved_fusion_scores,
        "selected_nodes":   final_indices,
        "expanded_candidate_nodes": selected_indices,
        "fusion_anchor_nodes": fusion_anchor_indices,
        "preserved_anchor_nodes": preserved_anchor_indices,
    }


@torch.no_grad()
def pcst_retrieve_for_example(
    example,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    seed_k: int = 3,
    expansion_factor: int = 4,
    fusion_anchor_pool_factor: int = 2,
    pcst_bonus: float = 0.05,
    preserve_fusion_top_k: int = 2,
    title_diversity_bonus: float = 0.03,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    return pcst_retrieve_with_details_for_example(
        example=example,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=top_k,
        seed_k=seed_k,
        expansion_factor=expansion_factor,
        fusion_anchor_pool_factor=fusion_anchor_pool_factor,
        pcst_bonus=pcst_bonus,
        preserve_fusion_top_k=preserve_fusion_top_k,
        title_diversity_bonus=title_diversity_bonus,
        lambda_dense=lambda_dense,
        query_prefix=query_prefix,
    )


# ──────────────────────────────────────────────
# Run PCST on All Examples
# ──────────────────────────────────────────────

def pcst_retrieve_all(
    graph_examples,
    embed_model,
    gnn_model,
    device,
    top_k: int = 5,
    seed_k: int = 3,
    expansion_factor: int = 4,
    fusion_anchor_pool_factor: int = 2,
    pcst_bonus: float = 0.05,
    preserve_fusion_top_k: int = 2,
    title_diversity_bonus: float = 0.03,
    lambda_dense: float = 0.5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    results = []

    for example in tqdm(graph_examples, desc="Multi-seed PCST retrieval"):
        result = pcst_retrieve_for_example(
            example=example,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            seed_k=seed_k,
            expansion_factor=expansion_factor,
            fusion_anchor_pool_factor=fusion_anchor_pool_factor,
            pcst_bonus=pcst_bonus,
            preserve_fusion_top_k=preserve_fusion_top_k,
            title_diversity_bonus=title_diversity_bonus,
            lambda_dense=lambda_dense,
            query_prefix=query_prefix,
        )
        results.append(result)

    return results


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_pcst(results, k=5):
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
        "total_questions":                               total,
        "pcst_support_recall@5":                         safe_div(full_hits,          total),
        "pcst_partial_support_recall@5":                 safe_div(partial_hits,        total),
        "bridge_questions":                              bridge_total,
        "bridge_pcst_support_recall@5":                  safe_div(bridge_full,         bridge_total),
        "bridge_pcst_partial_support_recall@5":          safe_div(bridge_partial,      bridge_total),
        "comparison_questions":                          comparison_total,
        "comparison_pcst_support_recall@5":              safe_div(comparison_full,     comparison_total),
        "comparison_pcst_partial_support_recall@5":      safe_div(comparison_partial,  comparison_total),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCST retrieval evaluation.")
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
    parser.add_argument("--seed-k", type=int, default=4)
    parser.add_argument("--expansion-factor", type=int, default=4)
    parser.add_argument("--fusion-anchor-pool-factor", type=int, default=2)
    parser.add_argument("--pcst-bonus", type=float, default=0.05)
    parser.add_argument("--preserve-fusion-top-k", type=int, default=2)
    parser.add_argument("--title-diversity-bonus", type=float, default=0.03)
    parser.add_argument("--lambda-dense", type=float, default=0.5)
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

    # Mirror the exact train/val split from gnn_train.py so we only
    # evaluate on held-out examples — not training data.
    pyg_dataset = build_pyg_dataset(graph_examples, embed_model)
    _, val_dataset = split_dataset(pyg_dataset, train_ratio=0.8)
    val_ids      = {d.question_id for d in val_dataset}
    val_examples = [ex for ex in graph_examples if ex["id"] in val_ids]

    print(f"Evaluating on {len(val_examples)} held-out validation examples.")

    # Probe input_dim from the already-built pyg_dataset — no redundant
    # build_pyg_data_from_example call just to read one shape field.
    gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
        graph_examples=graph_examples,
        embed_model=embed_model,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Loaded GNN checkpoint from {checkpoint_path}")

    results = pcst_retrieve_all(
        graph_examples=val_examples,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=args.top_k,
        seed_k=args.seed_k,
        expansion_factor=args.expansion_factor,
        fusion_anchor_pool_factor=args.fusion_anchor_pool_factor,
        pcst_bonus=args.pcst_bonus,
        preserve_fusion_top_k=args.preserve_fusion_top_k,
        title_diversity_bonus=args.title_diversity_bonus,
        lambda_dense=args.lambda_dense,
    )

    metrics = evaluate_pcst(results, k=args.top_k)

    print("\n===== Multi-Seed PCST Retrieval Results =====")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n===== Sample Predictions =====")
    for i, result in enumerate(results[:3]):
        print("\n" + "=" * 70)
        print(f"Example {i + 1}")
        print("Question:",         result["question"])
        print("Gold Titles:",      result["gold_titles"])
        print("Retrieved Titles:", result["retrieved_titles"])

        for rank, (chunk, fscore) in enumerate(
            zip(result["retrieved_chunks"], result["fusion_scores"]), start=1
        ):
            print(f"\nRank {rank}")
            print("Title:",         chunk.metadata["title"])
            print("Fusion Score:",  round(fscore, 4))
            print("Is Supporting:", chunk.metadata.get("is_supporting", False))
            print("Text:",          chunk.page_content[:220].replace("\n", " "))
