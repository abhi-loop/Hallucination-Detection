import os
import csv
import torch

from models.llm_loader import load_model
from models.hidden_extraction import extract_sentence_embedding
from models.generation import generate_k_answers
from metrics.eigenscore import compute_eigenscore
from metrics.threshold import find_best_threshold


RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")

# Fallback placeholder data used only when results.csv does not exist yet.
_FALLBACK_SCORES = [-1.8, -1.5, -1.6, -1.7, -1.4, 2.0, 2.3, 2.8]
_FALLBACK_LABELS = [  0,    0,    0,    0,    0,   1,   1,   1  ]


def load_reference_scores():
    """Load EigenScores and labels from data/results.csv.
    Falls back to hardcoded placeholder data if file not found."""
    if not os.path.exists(RESULTS_PATH):
        print("[WARNING] data/results.csv not found. Using placeholder reference scores.")
        print("          Run: python pipeline/run_dataset.py --limit 50")
        return _FALLBACK_SCORES, _FALLBACK_LABELS

    scores, labels = [], []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["eigenscore"]))
            labels.append(int(row["label"]))

    if len(scores) < 2:
        print("[WARNING] results.csv has fewer than 2 rows. Using placeholder reference scores.")
        return _FALLBACK_SCORES, _FALLBACK_LABELS

    print(f"Loaded {len(scores)} reference scores from data/results.csv")
    return scores, labels


def main():
    print("Loading model...")
    tokenizer, model = load_model()

    question = "What is the capital of France? Answer in one word."

    # Generate K different responses to the question
    K = 5
    print(f"\nGenerating {K} different responses...")
    responses = generate_k_answers(model, tokenizer, question, k=K)

    print("Extracting embeddings from each response...")
    embeddings = []
    for i, response in enumerate(responses):
        print(f"  Response {i+1}: {response[:200]}...")
        embedding = extract_sentence_embedding(model, tokenizer, response)
        embeddings.append(embedding)

    print("\nComputing EigenScore...")
    score = compute_eigenscore(embeddings)
    print(f"EigenScore: {score:.4f}")

    # ── Threshold Calibration from real (or fallback) data ────────────────────
    print("\n" + "=" * 60)
    reference_scores, reference_labels = load_reference_scores()
    print("Finding best threshold from reference scores...")
    threshold, gmean = find_best_threshold(reference_scores, reference_labels)
    print(f"  Best Threshold : {threshold:.4f}")
    print(f"  G-Mean         : {gmean:.4f}")

    # Classify the live score
    prediction = 1 if score > threshold else 0
    verdict = "⚠ Likely Hallucination" if prediction == 1 else "✔ Likely Factual"
    print(f"\n  Live EigenScore : {score:.4f}")
    print(f"  Verdict         : {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
