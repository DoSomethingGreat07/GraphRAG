'''import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import chunk_documents


def generate_chunk_embeddings(
    split="train",
    max_samples=1000,
    chunk_size=300,
    chunk_overlap=50,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Generate dense embeddings for chunked documents.

    Returns:
        chunks: List[Document]
        embeddings: np.ndarray of shape (num_chunks, embedding_dim)
        model: SentenceTransformer
    """
    chunks = chunk_documents(
        split=split,
        max_samples=max_samples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    model = SentenceTransformer(model_name)

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return chunks, embeddings, model


if __name__ == "__main__":
    chunks, embeddings, _ = generate_chunk_embeddings(
        split="train",
        max_samples=1000,
        chunk_size=300,
        chunk_overlap=50,
    )

    print("Total chunks:", len(chunks))
    print("Embedding matrix shape:", embeddings.shape)

    print("\nSample metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk text:")
    print(chunks[0].page_content[:300])

    print("\nFirst embedding vector shape:")
    print(embeddings[0].shape)

    print("\nEmbedding dtype:")
    print(embeddings.dtype)'''

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .chunking import chunk_hotpotqa_examples
    from .loading import load_hotpotqa_examples
except ImportError:
    from chunking import chunk_hotpotqa_examples
    from loading import load_hotpotqa_examples


def generate_chunk_embeddings(
    split="train",
    max_samples=10000,
    chunk_size=300,
    chunk_overlap=50,
    min_text_length=20,
    model_name="BAAI/bge-base-en-v1.5",
    batch_size=64,
):
    """
    Generate chunk embeddings once and store them inside each question example.

    Returns:
        chunked_examples: List[dict]
            Each example contains:
            - question
            - supporting_facts
            - context_chunks
            - context_chunk_embeddings  (np.ndarray: [num_chunks, dim])
        model: SentenceTransformer
    """
    examples = load_hotpotqa_examples(
        split=split,
        max_samples=max_samples,
    )

    chunked_examples = chunk_hotpotqa_examples(
        examples=examples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_text_length=min_text_length,
    )

    model = SentenceTransformer(model_name)

    for example in chunked_examples:
        chunks = example["context_chunks"]

        if not chunks:
            example["context_chunk_embeddings"] = np.empty((0, 0), dtype=np.float32)
            continue

        texts = [chunk.page_content for chunk in chunks]

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        example["context_chunk_embeddings"] = embeddings.astype(np.float32)

    return chunked_examples, model


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

    print("Total questions:", len(chunked_examples))

    sample = chunked_examples[0]
    print("\n===== SAMPLE QUESTION =====")
    print("Question ID:", sample["id"])
    print("Question:", sample["question"])
    print("Gold Titles:", sample["supporting_facts"]["title"])

    print("\nTotal chunks:", len(sample["context_chunks"]))
    print("Embedding matrix shape:", sample["context_chunk_embeddings"].shape)

    if sample["context_chunks"]:
        print("\nSample chunk metadata:")
        print(sample["context_chunks"][0].metadata)

        print("\nSample chunk text:")
        print(sample["context_chunks"][0].page_content[:300])

        print("\nFirst embedding vector shape:")
        print(sample["context_chunk_embeddings"][0].shape)

        print("\nEmbedding dtype:")
        print(sample["context_chunk_embeddings"].dtype)
