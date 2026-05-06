import os
import re
import json
import argparse
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
    from .pcst_dense_retrieval import pcst_dense_retrieve_all
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
    from pcst_dense_retrieval import pcst_dense_retrieve_all


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

RETRIEVAL_MODE_LABELS = {
    "dense": "FAISS-only retrieval",
    "pcst_dense": "FAISS + heuristic PCST",
    "gnn": "GNN retrieval",
    "fusion": "Dense retrieval + Query-Aware GraphSAGE",
    "pcst": "Dense retrieval + Query-Aware GraphSAGE + PCST (Main Method)",
}
PCST_LEARNED_SEED_K = 5
PCST_LEARNED_EXPANSION_FACTOR = 5
PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR = 3
PCST_LEARNED_BONUS = 0.08
PCST_LEARNED_PRESERVE_FUSION_TOP_K = 2
PCST_LEARNED_TITLE_DIVERSITY_BONUS = 0.03


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


def _clean_candidate_text(text: str) -> str:
    return " ".join((text or "").strip().split(" "))


def _extract_capitalized_candidates(text: str) -> list[str]:
    matches = re.findall(
        r"\b(?:[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)*)\b",
        text or "",
    )
    blocked = {
        "The",
        "A",
        "An",
        "He",
        "She",
        "They",
        "It",
        "This",
        "That",
        "These",
        "Those",
    }
    return [item for item in matches if item not in blocked and len(item.split()) <= 5]


def _select_best_candidate(question: str, candidates: list[str]) -> str:
    question_norm = normalize_answer(question)
    scored_candidates: collections.Counter[str] = collections.Counter()

    for candidate in candidates:
        cleaned = _clean_candidate_text(candidate)
        if not cleaned:
            continue
        candidate_norm = normalize_answer(cleaned)
        if not candidate_norm or candidate_norm in question_norm:
            continue
        scored_candidates[cleaned] += 1

    if not scored_candidates:
        return "Insufficient evidence"

    best_candidate, best_count = scored_candidates.most_common(1)[0]
    return best_candidate if best_count >= 2 else "Insufficient evidence"


def generate_retrieval_fallback_answer(question: str, retrieved_chunks, top_k: int = 5) -> str:
    selected_chunks = list(retrieved_chunks[:top_k]) if retrieved_chunks else []
    if not selected_chunks:
        return "Insufficient evidence"

    question_lower = (question or "").strip().lower()
    chunk_texts = [chunk.page_content.strip() for chunk in selected_chunks if chunk.page_content.strip()]
    titles = [chunk.metadata.get("title", "").strip() for chunk in selected_chunks if chunk.metadata.get("title")]
    combined_text = "\n".join(chunk_texts)

    if question_lower.startswith(("when ", "what year", "which year", "in what year")):
        date_match = re.search(
            r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
            combined_text,
        )
        if date_match:
            return date_match.group(0)
        year_match = re.search(r"\b(?:1[5-9]\d{2}|20\d{2})\b", combined_text)
        if year_match:
            return year_match.group(0)
        return "Insufficient evidence"

    if question_lower.startswith(("how many", "how much")):
        number_match = re.search(r"\b\d+(?:\.\d+)?\b", combined_text)
        return number_match.group(0) if number_match else "Insufficient evidence"

    if question_lower.startswith(("who ", "whom ", "whose ")):
        return _select_best_candidate(question, titles + _extract_capitalized_candidates(combined_text))

    if question_lower.startswith(("where ", "which city", "which country", "in which country")):
        location_matches = re.findall(
            r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b",
            combined_text,
        )
        return _select_best_candidate(question, location_matches + titles)

    if question_lower.startswith(("what album", "which album", "what film", "which film", "what song", "which song", "what book", "which book")):
        return _select_best_candidate(question, titles)

    generic_candidates = titles + _extract_capitalized_candidates(combined_text)
    return _select_best_candidate(question, generic_candidates)


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

def _truncate_chunk_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split()).strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated if truncated else text[:max_chars].strip()


def select_context_chunks(
    retrieved_chunks,
    top_k: int = 5,
    retrieval_mode: str | None = None,
):
    selected_chunks = []

    if retrieval_mode == "pcst":
        # For the main method, bias the context toward title diversity first.
        # This keeps the evidence chain broad instead of spending too much of
        # the prompt budget on multiple chunks from the same source title.
        seen_titles = set()
        remaining = []

        for chunk in retrieved_chunks[: max(top_k * 2, top_k)]:
            title = chunk.metadata.get("title", "Unknown Title")
            if title not in seen_titles and len(selected_chunks) < top_k:
                selected_chunks.append(chunk)
                seen_titles.add(title)
            else:
                remaining.append(chunk)

        for chunk in remaining:
            if len(selected_chunks) >= top_k:
                break
            selected_chunks.append(chunk)

        return selected_chunks[:top_k]

    return retrieved_chunks[:top_k]


def build_context_from_chunks(
    retrieved_chunks,
    top_k: int = 5,
    retrieval_mode: str | None = None,
) -> str:
    context_parts = []
    max_chars_per_chunk = 650 if retrieval_mode == "pcst" else 900

    for chunk in select_context_chunks(
        retrieved_chunks=retrieved_chunks,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    ):
        title = chunk.metadata.get("title", "Unknown Title")
        text = _truncate_chunk_text(chunk.page_content, max_chars=max_chars_per_chunk)

        context_parts.append(f"Title: {title}\n{text}")

    return "\n\n".join(context_parts)


def generate_answer_openai(
    question: str,
    retrieved_chunks,
    top_k: int = 5,
    retrieval_mode: str | None = None,
) -> str:
    if client is None:
        return "OPENAI_API_KEY not set"

    context = build_context_from_chunks(
        retrieved_chunks,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )

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
            seed_k=min(PCST_LEARNED_SEED_K, top_k),
            expansion_factor=PCST_LEARNED_EXPANSION_FACTOR,
            fusion_anchor_pool_factor=PCST_LEARNED_FUSION_ANCHOR_POOL_FACTOR,
            pcst_bonus=PCST_LEARNED_BONUS,
            preserve_fusion_top_k=min(PCST_LEARNED_PRESERVE_FUSION_TOP_K, top_k),
            title_diversity_bonus=PCST_LEARNED_TITLE_DIVERSITY_BONUS,
            lambda_dense=0.5,
        )

    if retrieval_mode == "pcst_dense":
        return pcst_dense_retrieve_all(
            graph_examples=graph_examples,
            embed_model=embed_model,
            top_k=top_k,
        )

    raise ValueError("Invalid RETRIEVAL_MODE. Use one of: dense, pcst_dense, gnn, fusion, pcst")


# =========================
# Main
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM answer evaluation for one of the five retrieval modes.")
    parser.add_argument(
        "--retrieval-mode",
        default="gnn",
        choices=["dense", "pcst_dense", "gnn", "fusion", "pcst"],
    )
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    RETRIEVAL_MODE = args.retrieval_mode
    RETRIEVAL_MODE_LABEL = RETRIEVAL_MODE_LABELS[RETRIEVAL_MODE]
    MAX_SAMPLES = args.max_samples
    TOP_K = args.top_k
    SPLIT = args.split
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    output_file = f"llm_eval_results_{RETRIEVAL_MODE}.json"

    print(f"Retrieval mode: {RETRIEVAL_MODE_LABEL} [{RETRIEVAL_MODE}]")

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
            retrieval_mode=RETRIEVAL_MODE,
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
        "retrieval_mode_label": RETRIEVAL_MODE_LABEL,
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
