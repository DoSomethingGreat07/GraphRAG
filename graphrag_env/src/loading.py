'''from datasets import load_dataset
from langchain_core.documents import Document


def load_hotpotqa_documents(split="train", max_samples=1000):
    """
    Load HotpotQA context paragraphs and deduplicate them by (title, text).

    Returns:
        List[Document]
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    documents = []
    seen = set()

    max_samples = min(max_samples, len(dataset))

    for item in dataset.select(range(max_samples)):
        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]

        for title, sentences in zip(titles, sentences_list):
            text = " ".join(sentences).strip()

            if not text:
                continue

            key = (title.strip(), text)
            if key in seen:
                continue
            seen.add(key)

            doc = Document(
                page_content=text,
                metadata={
                    "title": title,
                    "source": "HotpotQA",
                },
            )
            documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_hotpotqa_documents(split="train", max_samples=1000)

    print("Total unique documents loaded:", len(docs))
    print("\nSample metadata:")
    print(docs[0].metadata)

    print("\nSample text:")
    print(docs[0].page_content[:500])'''

from datasets import load_dataset
from langchain_core.documents import Document


def load_hotpotqa_examples(split="train", max_samples=10000):
    """
    Load HotpotQA distractor examples in a question-centric format.

    Each returned item contains:
    - question-level metadata
    - supporting facts
    - local context documents (usually 10 docs per question)

    Returns:
        List[dict]
    """
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    max_samples = min(max_samples, len(dataset))

    examples = []

    for item in dataset.select(range(max_samples)):
        question_id = item["id"]
        question = item["question"]
        answer = item["answer"]
        qtype = item["type"]
        level = item["level"]

        supporting_titles = list(dict.fromkeys(item["supporting_facts"]["title"]))
        supporting_sent_ids = item["supporting_facts"]["sent_id"]

        titles = item["context"]["title"]
        sentences_list = item["context"]["sentences"]

        context_docs = []

        for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
            text = " ".join(sentences).strip()

            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "question_id": question_id,
                    "question_type": qtype,
                    "level": level,
                    "title": title,
                    "context_index": idx,
                    "source": "HotpotQA",
                    "is_supporting": title in supporting_titles,
                },
            )
            context_docs.append(doc)

        example = {
            "id": question_id,
            "question": question,
            "answer": answer,
            "type": qtype,
            "level": level,
            "supporting_facts": {
                "title": supporting_titles,
                "sent_id": supporting_sent_ids,
            },
            "context_docs": context_docs,
        }

        examples.append(example)

    return examples


if __name__ == "__main__":
    examples = load_hotpotqa_examples(split="train", max_samples=10000)

    print("Total examples loaded:", len(examples))

    sample = examples[0]

    print("\n===== SAMPLE QUESTION =====")
    print("ID:", sample["id"])
    print("Question:", sample["question"])
    print("Answer:", sample["answer"])
    print("Type:", sample["type"])
    print("Level:", sample["level"])

    print("\nSupporting Titles:")
    print(sample["supporting_facts"]["title"])

    print("\nTotal Context Docs:")
    print(len(sample["context_docs"]))

    print("\nSample Context Doc Metadata:")
    print(sample["context_docs"][0].metadata)

    print("\nSample Context Doc Text:")
    print(sample["context_docs"][0].page_content[:500])