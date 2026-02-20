import torch

from models.llm_loader import load_model
from models.hidden_extraction import extract_sentence_embedding
from models.generation import generate_k_answers
from metrics.eigenscore import compute_eigenscore
from metrics.threshold import find_best_threshold


# Pre-defined reference scores (EigenScore, label) used to calibrate threshold.
# label: 0 = factual/consistent, 1 = likely hallucination
# These represent typical score ranges observed at different temperatures.
REFERENCE_SCORES  = [-1.8, -1.5, -1.6, -1.7, -1.4, 2.0, 2.3, 2.8]
REFERENCE_LABELS  = [0,     0,     0,     0,     0,   1,   1,   1  ]


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

    # ── Threshold Integration ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Finding best threshold from reference scores...")
    threshold, gmean = find_best_threshold(REFERENCE_SCORES, REFERENCE_LABELS)
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
