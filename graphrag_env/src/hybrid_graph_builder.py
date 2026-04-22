import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import networkx as nx

try:
    from .embeddings import generate_chunk_embeddings
except ImportError:
    from embeddings import generate_chunk_embeddings


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "by",
    "with", "from", "is", "was", "were", "are", "be", "as", "that", "this",
    "it", "its", "their", "his", "her", "after", "before", "during", "into",
    "than", "then", "which", "what", "who", "whom", "when", "where", "why",
    "how", "has", "have", "had", "been", "being", "also", "one", "two"
}


def normalize_text(text: str) -> List[str]:
    """
    Simple tokenization for keyword-overlap edges.
    """
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2]
    return tokens


def get_keyword_set(text: str) -> set:
    return set(normalize_text(text))


def merge_edge_type(existing_type: str, new_type: str) -> str:
    """
    Merge edge types without duplication.
    """
    types = set(existing_type.split("|")) if existing_type else set()
    types.add(new_type)
    return "|".join(sorted(t for t in types if t))


def add_or_update_edge(
    G: nx.Graph,
    u: int,
    v: int,
    new_edge_type: str,
    weight: float = None,
    semantic_weight: float = None,
    keyword_overlap_count: int = None,
):
    """
    Add a new edge or update an existing one cleanly.
    """
    if G.has_edge(u, v):
        G[u][v]["edge_type"] = merge_edge_type(
            G[u][v].get("edge_type", ""),
            new_edge_type,
        )

        if semantic_weight is not None:
            prev = G[u][v].get("semantic_weight", None)
            G[u][v]["semantic_weight"] = max(prev, semantic_weight) if prev is not None else semantic_weight

        if keyword_overlap_count is not None:
            prev = G[u][v].get("keyword_overlap_count", None)
            G[u][v]["keyword_overlap_count"] = max(prev, keyword_overlap_count) if prev is not None else keyword_overlap_count

        if weight is not None:
            prev = G[u][v].get("weight", None)
            G[u][v]["weight"] = max(prev, weight) if prev is not None else weight

    else:
        edge_data = {"edge_type": new_edge_type}
        if weight is not None:
            edge_data["weight"] = weight
        if semantic_weight is not None:
            edge_data["semantic_weight"] = semantic_weight
        if keyword_overlap_count is not None:
            edge_data["keyword_overlap_count"] = keyword_overlap_count

        G.add_edge(u, v, **edge_data)


def cosine_knn_edges(
    embeddings: np.ndarray,
    k: int = 2,
    min_sim: float = 0.40,
) -> List[Tuple[int, int, float]]:
    """
    Build semantic kNN edges from normalized embeddings.

    Returns:
        List of (src_idx, dst_idx, similarity)
    """
    if len(embeddings) == 0:
        return []

    sims = np.dot(embeddings, embeddings.T)
    n = sims.shape[0]
    edges = []

    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0  # ignore self

        top_idx = np.argsort(row)[::-1][:k]
        for j in top_idx:
            if row[j] >= min_sim:
                u, v = sorted((i, j))
                edges.append((u, v, float(row[j])))

    # remove duplicates by keeping max similarity
    best_edges = {}
    for u, v, sim in edges:
        if (u, v) not in best_edges or sim > best_edges[(u, v)]:
            best_edges[(u, v)] = sim

    return [(u, v, sim) for (u, v), sim in best_edges.items()]


def build_hybrid_graph_for_example(
    example: Dict,
    semantic_k: int = 2,
    semantic_min_sim: float = 0.40,
    keyword_overlap_threshold: int = 3,
) -> nx.Graph:
    """
    Build a local hybrid graph for one question.

    Nodes = context chunks
    Edges =
        - same_title
        - adjacent_chunk
        - semantic_knn
        - keyword_overlap
    """
    chunks = example["context_chunks"]
    embeddings = example["context_chunk_embeddings"]

    G = nx.Graph()

    # ---- Add nodes ----
    for idx, chunk in enumerate(chunks):
        G.add_node(
            idx,
            chunk_id=chunk.metadata["chunk_id"],
            title=chunk.metadata["title"],
            question_id=chunk.metadata["question_id"],
            question_type=chunk.metadata.get("question_type"),
            level=chunk.metadata.get("level"),
            chunk_index=chunk.metadata["chunk_index"],
            context_index=chunk.metadata.get("context_index"),
            is_supporting=chunk.metadata.get("is_supporting", False),
            text=chunk.page_content,
        )

    # ---- same_title edges ----
    title_to_nodes = {}
    for idx, chunk in enumerate(chunks):
        title = chunk.metadata["title"]
        title_to_nodes.setdefault(title, []).append(idx)

    for title, node_ids in title_to_nodes.items():
        if len(node_ids) < 2:
            continue

        sorted_nodes = sorted(
            node_ids,
            key=lambda nid: G.nodes[nid]["chunk_index"]
        )

        for i in range(len(sorted_nodes)):
            for j in range(i + 1, len(sorted_nodes)):
                u, v = sorted_nodes[i], sorted_nodes[j]
                add_or_update_edge(
                    G,
                    u,
                    v,
                    new_edge_type="same_title",
                    weight=1.0,
                )

    # ---- adjacent_chunk edges ----
    for title, node_ids in title_to_nodes.items():
        sorted_nodes = sorted(
            node_ids,
            key=lambda nid: G.nodes[nid]["chunk_index"]
        )

        for i in range(len(sorted_nodes) - 1):
            u, v = sorted_nodes[i], sorted_nodes[i + 1]
            add_or_update_edge(
                G,
                u,
                v,
                new_edge_type="adjacent_chunk",
                weight=1.0,
            )

    # ---- semantic_knn edges ----
    semantic_edges = cosine_knn_edges(
        embeddings=embeddings,
        k=semantic_k,
        min_sim=semantic_min_sim,
    )

    for u, v, sim in semantic_edges:
        add_or_update_edge(
            G,
            u,
            v,
            new_edge_type="semantic_knn",
            weight=sim,
            semantic_weight=sim,
        )

    # ---- keyword_overlap edges ----
    keyword_sets = [get_keyword_set(chunk.page_content) for chunk in chunks]

    n = len(chunks)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = keyword_sets[i].intersection(keyword_sets[j])
            overlap_count = len(overlap)

            if overlap_count >= keyword_overlap_threshold:
                add_or_update_edge(
                    G,
                    i,
                    j,
                    new_edge_type="keyword_overlap",
                    weight=float(overlap_count),
                    keyword_overlap_count=overlap_count,
                )

    return G


def build_hybrid_graphs_for_all_examples(
    chunked_examples: List[Dict],
    semantic_k: int = 2,
    semantic_min_sim: float = 0.40,
    keyword_overlap_threshold: int = 3,
) -> List[Dict]:
    """
    Build and attach a hybrid graph to each example.
    """
    graph_examples = []

    for example in chunked_examples:
        G = build_hybrid_graph_for_example(
            example=example,
            semantic_k=semantic_k,
            semantic_min_sim=semantic_min_sim,
            keyword_overlap_threshold=keyword_overlap_threshold,
        )

        new_example = {
            **example,
            "graph": G,
        }
        graph_examples.append(new_example)

    return graph_examples


def graph_stats(G: nx.Graph) -> Dict:
    """
    Return useful graph statistics.
    """
    edge_type_counter = Counter()

    for _, _, data in G.edges(data=True):
        edge_types = data.get("edge_type", "")
        for et in edge_types.split("|"):
            if et:
                edge_type_counter[et] += 1

    supporting_nodes = sum(
        1 for _, data in G.nodes(data=True) if data.get("is_supporting", False)
    )

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "supporting_nodes": supporting_nodes,
        "edge_type_counts": dict(edge_type_counter),
    }


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

    for i, ex in enumerate(graph_examples[:3]):
        print("\n" + "=" * 70)
        print(f"Example {i+1}")
        print("Question ID:", ex["id"])
        print("Question:", ex["question"])
        print("Type:", ex["type"])
        print("Gold Titles:", ex["supporting_facts"]["title"])

        stats = graph_stats(ex["graph"])
        print("\nGraph Stats:")
        for k, v in stats.items():
            print(f"{k}: {v}")

        print("\nSample Nodes:")
        for nid, data in list(ex["graph"].nodes(data=True))[:3]:
            print(f"Node {nid}:")
            print("  title:", data["title"])
            print("  chunk_index:", data["chunk_index"])
            print("  is_supporting:", data["is_supporting"])

        print("\nSample Edges:")
        for u, v, data in list(ex["graph"].edges(data=True))[:5]:
            print(f"Edge {u} -- {v}: {data}")
