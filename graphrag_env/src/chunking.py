'''from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loading import load_hotpotqa_documents


def chunk_documents(
    split="train",
    max_samples=1000,
    chunk_size=300,
    chunk_overlap=50,
):
    docs = load_hotpotqa_documents(split=split, max_samples=max_samples)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    title_counts = defaultdict(int)

    for i, chunk in enumerate(chunks):
        title = chunk.metadata.get("title", "")
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_index_in_title"] = title_counts[title]
        title_counts[title] += 1

    return chunks


if __name__ == "__main__":
    chunks = chunk_documents()

    print("Total chunks:", len(chunks))
    print("\nSample chunk metadata:")
    print(chunks[0].metadata)

    print("\nSample chunk text:")
    print(chunks[0].page_content)'''

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from .loading import load_hotpotqa_examples
except ImportError:
    from loading import load_hotpotqa_examples


def build_text_splitter(
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter for HotpotQA documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_single_document(
    doc: Document,
    text_splitter: RecursiveCharacterTextSplitter,
    min_text_length: int = 20,
) -> List[Document]:
    """
    Chunk one document while preserving metadata.

    Strategy:
    - Skip extremely tiny/empty docs if desired
    - Split only within the current document
    - Preserve title/question metadata
    """
    text = doc.page_content.strip()

    if not text:
        return []

    # Optional: drop extremely tiny noise docs
    if len(text.split()) < min_text_length:
        return []

    split_docs = text_splitter.create_documents(
        texts=[text],
        metadatas=[doc.metadata],
    )

    chunked_docs = []
    for chunk_idx, chunk_doc in enumerate(split_docs):
        chunk_doc.metadata = {
            **chunk_doc.metadata,
            "chunk_index": chunk_idx,
            "chunk_id": (
                f"{doc.metadata['question_id']}::"
                f"{doc.metadata['title']}::"
                f"{chunk_idx}"
            ),
            "text_length_chars": len(chunk_doc.page_content),
            "text_length_words": len(chunk_doc.page_content.split()),
        }
        chunked_docs.append(chunk_doc)

    return chunked_docs


def chunk_hotpotqa_examples(
    examples: List[Dict[str, Any]],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    min_text_length: int = 20,
) -> List[Dict[str, Any]]:
    """
    Chunk each question's local context documents separately.

    Input:
        examples = [
            {
                "id": ...,
                "question": ...,
                "answer": ...,
                "type": ...,
                "level": ...,
                "supporting_facts": ...,
                "context_docs": [Document, Document, ...]
            },
            ...
        ]

    Output:
        Same question-level structure, plus:
            - context_chunks
    """
    text_splitter = build_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_examples = []

    for example in examples:
        context_chunks = []

        for doc in example["context_docs"]:
            doc_chunks = chunk_single_document(
                doc=doc,
                text_splitter=text_splitter,
                min_text_length=min_text_length,
            )
            context_chunks.extend(doc_chunks)

        chunked_example = {
            **example,
            "context_chunks": context_chunks,
        }
        chunked_examples.append(chunked_example)

    return chunked_examples


if __name__ == "__main__":
    examples = load_hotpotqa_examples(split="train", max_samples=10000)

    chunked_examples = chunk_hotpotqa_examples(
        examples,
        chunk_size=300,
        chunk_overlap=50,
        min_text_length=20,
    )

    print("Total questions:", len(chunked_examples))

    sample = chunked_examples[0]

    print("\n===== SAMPLE QUESTION =====")
    print("Question ID:", sample["id"])
    print("Question:", sample["question"])
    print("Type:", sample["type"])
    print("Gold Titles:", sample["supporting_facts"]["title"])

    print("\nOriginal local docs:", len(sample["context_docs"]))
    print("Total chunks:", len(sample["context_chunks"]))

    if sample["context_chunks"]:
        print("\nSample chunk metadata:")
        print(sample["context_chunks"][0].metadata)

        print("\nSample chunk text:")
        print(sample["context_chunks"][0].page_content[:500])

    print("\nChunks per title:")
    counts = {}
    for ch in sample["context_chunks"]:
        title = ch.metadata["title"]
        counts[title] = counts.get(title, 0) + 1

    for title, cnt in counts.items():
        print(f"{title}: {cnt}")


    
