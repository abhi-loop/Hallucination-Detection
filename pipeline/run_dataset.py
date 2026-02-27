"""
pipeline/run_dataset.py

Runs the EigenScore hallucination detection pipeline over TruthfulQA.
Saves results incrementally to data/results.csv.

Usage:
    python pipeline/run_dataset.py --limit 10
"""

import os
import csv
import argparse
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_loader import load_model
from models.generation import generate_k_answers
from models.hidden_extraction import extract_sentence_embedding
from metrics.eigenscore import compute_eigenscore
from data.loader import load_truthfulqa
from data.labeler import majority_label, label_response

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results.csv")
K = 5  # number of responses to generate per question


def already_processed(question: str) -> bool:
    """Check if this question already has a result in the CSV (for resuming)."""
    if not os.path.exists(RESULTS_PATH):
        return False
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["question"].strip() == question.strip():
                return True
    return False


def append_result(question: str, eigenscore: float, label: int):
    """Append a single result row to the CSV."""
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "eigenscore", "label"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"question": question, "eigenscore": eigenscore, "label": label})


def run(limit: int):
    print("Loading model...")
    tokenizer, model = load_model()

    print(f"Loading TruthfulQA dataset (limit={limit})...")
    records = load_truthfulqa(limit=limit)

    print(f"Processing {len(records)} questions...\n")

    for i, record in enumerate(records):
        question = record["question"]
        correct_answers = record["correct_answers"]

        if already_processed(question):
            print(f"[{i+1}/{len(records)}] Skipping (already processed): {question[:60]}...")
            continue

        print(f"[{i+1}/{len(records)}] {question[:80]}")

        # Generate K responses
        responses = generate_k_answers(model, tokenizer, question, k=K)

        # Extract embeddings and compute EigenScore
        embeddings = [extract_sentence_embedding(model, tokenizer, r) for r in responses]
        eigenscore = compute_eigenscore(embeddings)

        # Label via majority vote (pass question so prompt echo can be stripped)
        label = majority_label(responses, correct_answers, question=question)

        print(f"   Correct answers : {correct_answers}")
        for j, resp in enumerate(responses):
            per_label = label_response(resp, correct_answers, question=question)
            verdict_per = "Factual" if per_label == 0 else "Hallucination"
            print(f"   Response [{j+1}]: {resp.strip()[:120]}  →  {verdict_per}")

        verdict = "Factual" if label == 0 else "Hallucination"
        print(f"   EigenScore: {eigenscore:.4f} | Majority label: {label} ({verdict})")

        append_result(question, eigenscore, label)

    print(f"\nDone. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EigenScore pipeline over TruthfulQA")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to process")
    args = parser.parse_args()
    run(args.limit)
