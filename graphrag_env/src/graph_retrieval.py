import numpy as np
from sentence_transformers import SentenceTransformer


def dense_seed_retrieval(
    example,
    model: SentenceTransformer,
    seed_k: int = 5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Dense retrieval over local chunks using precomputed chunk embeddings.

    Returns:
        {
            "seed_indices": List[int],
            "seed_scores": Dict[int, float],
            "query_embedding": np.ndarray
        }
    """
    question = example["question"]
    chunk_embeddings = example["context_chunk_embeddings"]
    chunks = example["context_chunks"]

    if len(chunks) == 0:
        return {
            "seed_indices": [],
            "seed_scores": {},
            "query_embedding": None,
        }

    query_text = query_prefix + question

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dense_scores = np.dot(chunk_embeddings, query_embedding)

    seed_k = min(seed_k, len(chunks))
    seed_indices = np.argsort(dense_scores)[::-1][:seed_k]

    return {
        "seed_indices": seed_indices.tolist(),
        "seed_scores": {int(i): float(dense_scores[i]) for i in seed_indices},
        "query_embedding": query_embedding,
    }


def graph_expand_candidates(
    example,
    seed_indices,
    neighbor_hops: int = 1,
):
    """
    Expand candidate nodes from dense seed nodes using graph neighbors.
    """
    G = example["graph"]

    candidates = set(seed_indices)
    frontier = set(seed_indices)

    for _ in range(neighbor_hops):
        new_frontier = set()

        for node in frontier:
            if node in G:
                new_frontier.update(G.neighbors(node))

        new_frontier = new_frontier - candidates
        candidates.update(new_frontier)
        frontier = new_frontier

    return sorted(candidates)


def graph_retrieve_simple(
    example,
    model: SentenceTransformer,
    seed_k: int = 5,
    final_k: int = 5,
    neighbor_hops: int = 1,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Simple graph retrieval:
    1. Dense top-k seed retrieval
    2. 1-hop or multi-hop graph expansion
    3. Re-rank expanded candidates using dense score only

    Important:
    Graph is used only for candidate expansion here.
    Dense similarity remains the final ranking signal.
    """
    chunks = example["context_chunks"]
    chunk_embeddings = example["context_chunk_embeddings"]

    if len(chunks) == 0:
        return {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "gold_titles": example["supporting_facts"]["title"],
            "retrieved_chunks": [],
            "retrieved_titles": [],
            "scores": [],
            "seed_indices": [],
            "candidate_indices": [],
        }

    seed_output = dense_seed_retrieval(
        example=example,
        model=model,
        seed_k=seed_k,
        query_prefix=query_prefix,
    )

    seed_indices = seed_output["seed_indices"]
    query_embedding = seed_output["query_embedding"]

    candidate_indices = graph_expand_candidates(
        example=example,
        seed_indices=seed_indices,
        neighbor_hops=neighbor_hops,
    )

    # Re-rank candidates using dense similarity only
    candidate_scores = {}
    for idx in candidate_indices:
        candidate_scores[idx] = float(np.dot(chunk_embeddings[idx], query_embedding))

    ranked_indices = sorted(
        candidate_indices,
        key=lambda idx: candidate_scores[idx],
        reverse=True,
    )[: min(final_k, len(candidate_indices))]

    retrieved_chunks = [chunks[i] for i in ranked_indices]
    retrieved_scores = [candidate_scores[i] for i in ranked_indices]

    # Deduplicate titles while preserving order
    retrieved_titles = []
    seen_titles = set()

    for chunk in retrieved_chunks:
        title = chunk.metadata["title"]
        if title not in seen_titles:
            retrieved_titles.append(title)
            seen_titles.add(title)

    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "type": example["type"],
        "gold_titles": example["supporting_facts"]["title"],
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "scores": retrieved_scores,
        "seed_indices": seed_indices,
        "candidate_indices": candidate_indices,
    }


def graph_retrieve_for_all_examples(
    graph_examples,
    model: SentenceTransformer,
    seed_k: int = 5,
    final_k: int = 5,
    neighbor_hops: int = 1,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Run simple graph retrieval for all examples.
    """
    results = []

    for example in graph_examples:
        result = graph_retrieve_simple(
            example=example,
            model=model,
            seed_k=seed_k,
            final_k=final_k,
            neighbor_hops=neighbor_hops,
            query_prefix=query_prefix,
        )
        results.append(result)

    return results