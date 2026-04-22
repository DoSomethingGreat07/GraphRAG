import os
import re
import json
import string
import collections
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

try:
    from .artifact_runtime import (
        load_artifact_bundle,
        load_gnn_from_checkpoint,
        load_or_build_graph_examples,
    )
    from .gnn_retrieval import gnn_retrieve_for_all_examples
    from .retrieval import retrieve_for_all_examples
    from .gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_all_examples
    from .pcst import pcst_retrieve_all
except ImportError:
    from artifact_runtime import (
        load_artifact_bundle,
        load_gnn_from_checkpoint,
        load_or_build_graph_examples,
    )
    from gnn_retrieval import gnn_retrieve_for_all_examples
    from retrieval import retrieve_for_all_examples
    from gnn_fusion_retreival import dense_gnn_fusion_retrieve_for_all_examples
    from pcst import pcst_retrieve_all


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Metrics
# =========================

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


# =========================
# Answer generation
# =========================

def build_context_from_chunks(retrieved_chunks, top_k: int = 5) -> str:
    context_parts = []

    for chunk in retrieved_chunks[:top_k]:
        title = chunk.metadata.get("title", "Unknown Title")
        text = chunk.page_content.strip()

        context_parts.append(f"Title: {title}\n{text}")

    return "\n\n".join(context_parts)


def generate_answer_openai(question: str, retrieved_chunks, top_k: int = 5) -> str:
    if client is None:
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=80,
    )

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            return "Insufficient evidence"
        return answer
    except Exception:
        return raw if raw else "Insufficient evidence"


def run_retrieval(
    retrieval_mode: str,
    chunked_examples,
    graph_examples,
    embed_model,
    gnn_model,
    device,
    top_k: int,
):
    if retrieval_mode == "dense":
        return retrieve_for_all_examples(
            chunked_examples=chunked_examples,
            model=embed_model,
            top_k=top_k,
        )

    if retrieval_mode == "gnn":
        return gnn_retrieve_for_all_examples(
            graph_examples=graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
        )

    if retrieval_mode == "fusion":
        return dense_gnn_fusion_retrieve_for_all_examples(
            graph_examples=graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            lambda_dense=0.5,
        )

    if retrieval_mode == "pcst":
        return pcst_retrieve_all(
            graph_examples=graph_examples,
            embed_model=embed_model,
            gnn_model=gnn_model,
            device=device,
            top_k=top_k,
            lambda_dense=0.5,
        )

    raise ValueError("Invalid RETRIEVAL_MODE. Use one of: dense, gnn, fusion, pcst")


# =========================
# Main
# =========================

if __name__ == "__main__":
    RETRIEVAL_MODE = "gnn"   # dense / gnn / fusion / pcst
    MAX_SAMPLES = 300
    TOP_K = 5
    SPLIT = "train"
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    output_file = f"llm_eval_results_{RETRIEVAL_MODE}.json"

    print("Loading data...")
    try:
        bundle = load_artifact_bundle(
            split=SPLIT,
            max_samples=MAX_SAMPLES,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunked_examples = bundle["chunked_examples"]
        graph_examples = bundle["graph_examples"]
        embed_model_name = bundle["manifest"]["model_name"]
        embed_model = SentenceTransformer(embed_model_name)
        print(f"Loaded artifacts from {bundle['paths']['artifacts_dir']}")
    except FileNotFoundError:
        graph_examples, embed_model = load_or_build_graph_examples(
            split=SPLIT,
            max_samples=MAX_SAMPLES,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_text_length=20,
            model_name="BAAI/bge-base-en-v1.5",
            batch_size=64,
            semantic_k=2,
            semantic_min_sim=0.40,
            keyword_overlap_threshold=3,
        )
        chunked_examples = [
            {
                key: value
                for key, value in example.items()
                if key != "graph"
            }
            for example in graph_examples
        ]

    gnn_model = None
    device = None

    if RETRIEVAL_MODE in ["gnn", "fusion", "pcst"]:
        gnn_model, device, checkpoint_path = load_gnn_from_checkpoint(
            graph_examples=graph_examples,
            embed_model=embed_model,
            split=SPLIT,
            max_samples=MAX_SAMPLES,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        print(f"Loaded GNN checkpoint from {checkpoint_path}")

    print("Running retrieval...")
    retrieval_results = run_retrieval(
        retrieval_mode=RETRIEVAL_MODE,
        chunked_examples=chunked_examples,
        graph_examples=graph_examples,
        embed_model=embed_model,
        gnn_model=gnn_model,
        device=device,
        top_k=TOP_K,
    )

    print("Running LLM evaluation...")

    example_map = {ex["id"]: ex for ex in chunked_examples}

    results_json = []

    total = 0
    em_total = 0.0
    f1_total = 0.0

    bridge_total = 0
    bridge_em_total = 0.0
    bridge_f1_total = 0.0

    comparison_total = 0
    comparison_em_total = 0.0
    comparison_f1_total = 0.0

    for result in retrieval_results:
        qid = result["id"]
        example = example_map[qid]

        question = example["question"]
        gold_answer = example["answer"]
        qtype = example["type"]
        retrieved_chunks = result.get("retrieved_chunks", [])

        prediction = generate_answer_openai(
            question=question,
            retrieved_chunks=retrieved_chunks,
            top_k=TOP_K,
        )

        em = exact_match(prediction, gold_answer)
        f1 = f1_score(prediction, gold_answer)

        total += 1
        em_total += em
        f1_total += f1

        if qtype == "bridge":
            bridge_total += 1
            bridge_em_total += em
            bridge_f1_total += f1
        elif qtype == "comparison":
            comparison_total += 1
            comparison_em_total += em
            comparison_f1_total += f1

        results_json.append(
            {
                "id": qid,
                "type": qtype,
                "question": question,
                "gold_answer": gold_answer,
                "prediction": prediction,
                "retrieved_titles": [
                    chunk.metadata["title"]
                    for chunk in retrieved_chunks[:TOP_K]
                ],
                "exact_match": em,
                "f1": f1,
            }
        )

        print(f"Processed {total}/{MAX_SAMPLES}")

    metrics = {
        "retrieval_mode": RETRIEVAL_MODE,
        "total_questions": total,
        "answer_em": em_total / total if total else 0.0,
        "answer_f1": f1_total / total if total else 0.0,
        "bridge_questions": bridge_total,
        "bridge_answer_em": bridge_em_total / bridge_total if bridge_total else 0.0,
        "bridge_answer_f1": bridge_f1_total / bridge_total if bridge_total else 0.0,
        "comparison_questions": comparison_total,
        "comparison_answer_em": comparison_em_total / comparison_total if comparison_total else 0.0,
        "comparison_answer_f1": comparison_f1_total / comparison_total if comparison_total else 0.0,
        "timestamp": str(datetime.now()),
    }

    final_output = {
        "metrics": metrics,
        "results": results_json,
    }

    with open(output_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print("\n===== LLM Evaluation Complete =====")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to: {output_file}")
