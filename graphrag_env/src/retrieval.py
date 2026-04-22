'''import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings import generate_chunk_embeddings


def retrieve_top_k(query, chunks, embeddings, model, k=5):
    """
    Retrieve top-k most similar chunks for a query using cosine similarity.

    Args:
        query: str
        chunks: List[Document]
        embeddings: np.ndarray
        model: SentenceTransformer
        k: int

    Returns:
        List[dict]
    """
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_vec
    top_k_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_k_idx:
        results.append(
            {
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "metadata": chunks[idx].metadata,
                "text": chunks[idx].page_content,
            }
        )

    return results


if __name__ == "__main__":
    chunks, embeddings, model = generate_chunk_embeddings(
        split="train",
        max_samples=10000,
        chunk_size=500,
        chunk_overlap=100,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    query = "Which magazine was started first Arthur's Magazine or First for Women?"
    results = retrieve_top_k(query, chunks, embeddings, model, k=5)

    for rank, item in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Score:", round(item["score"], 4))
        print("Metadata:", item["metadata"])
        print("Text:", item["text"][:400])'''

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .artifact_runtime import load_or_build_chunked_examples
except ImportError:
    from artifact_runtime import load_or_build_chunked_examples


def retrieve_top_k_chunks_for_example(
    example,
    model: SentenceTransformer,
    top_k: int = 5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Retrieve top-k chunks for one example using precomputed chunk embeddings.

    Args:
        example: dict with:
            - question
            - context_chunks
            - context_chunk_embeddings
            - supporting_facts
        model: SentenceTransformer
        top_k: int

    Returns:
        dict containing retrieval results
    """
    question = example["question"]
    chunks = example["context_chunks"]
    chunk_embeddings = example["context_chunk_embeddings"]

    if len(chunks) == 0:
        return {
            "id": example["id"],
            "question": question,
            "answer": example["answer"],
            "type": example["type"],
            "gold_titles": example["supporting_facts"]["title"],
            "retrieved_chunks": [],
            "retrieved_titles": [],
            "scores": [],
        }

    query_text = query_prefix + question

    query_embedding = model.encode(
        query_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # cosine similarity because both are normalized
    scores = np.dot(chunk_embeddings, query_embedding)

    top_k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = [chunks[i] for i in top_indices]
    retrieved_scores = [float(scores[i]) for i in top_indices]

    # deduplicate titles while preserving order
    retrieved_titles = []
    seen_titles = set()

    for chunk in retrieved_chunks:
        title = chunk.metadata["title"]
        if title not in seen_titles:
            retrieved_titles.append(title)
            seen_titles.add(title)

    return {
        "id": example["id"],
        "question": question,
        "answer": example["answer"],
        "type": example["type"],
        "gold_titles": example["supporting_facts"]["title"],
        "retrieved_chunks": retrieved_chunks,
        "retrieved_titles": retrieved_titles,
        "scores": retrieved_scores,
    }


def retrieve_for_all_examples(
    chunked_examples,
    model: SentenceTransformer,
    top_k: int = 5,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
):
    """
    Run retrieval for all question-wise examples using precomputed chunk embeddings.
    """
    all_results = []

    for example in chunked_examples:
        result = retrieve_top_k_chunks_for_example(
            example=example,
            model=model,
            top_k=top_k,
            query_prefix=query_prefix,
        )
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    chunked_examples, model = load_or_build_chunked_examples(
        split="train",
        max_samples=10000,
        chunk_size=300,
        chunk_overlap=50,
        min_text_length=20,
        model_name="BAAI/bge-base-en-v1.5",
        batch_size=64,
    )

    results = retrieve_for_all_examples(
        chunked_examples=chunked_examples,
        model=model,
        top_k=5,
    )

    for i, result in enumerate(results[:3]):
        print("\n" + "=" * 60)
        print(f"Example {i+1}")
        print("Question:", result["question"])
        print("Gold Titles:", result["gold_titles"])
        print("Retrieved Titles:", result["retrieved_titles"])

        print("\nTop Retrieved Chunks:")
        for rank, (chunk, score) in enumerate(
            zip(result["retrieved_chunks"], result["scores"]),
            start=1,
        ):
            print(f"\nRank {rank}")
            print("Title:", chunk.metadata["title"])
            print("Score:", round(score, 4))
            print("Is Supporting:", chunk.metadata.get("is_supporting", False))
            print("Text:", chunk.page_content[:250].replace("\n", " "))
