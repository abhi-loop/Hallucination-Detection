"""
metrics/lexical_similarity.py

Pairwise Lexical Similarity across K responses using ROUGE-L F1.

Computes the mean ROUGE-L F1 score over all C(K, 2) unique response pairs.

Interpretation:
    • High → responses are lexically consistent with each other (model is certain)
    • Low  → responses are diverse / inconsistent (model is uncertain)

⚠️  For NQ-style short answers (often 1–4 tokens) this metric will tend toward
    either 0 (completely different words) or 1 (identical short answers).
    Treat as a consistency / diversity signal rather than an absolute quality score.
"""

from itertools import combinations
from rouge_score import rouge_scorer


def compute_lexical_similarity(responses: list[dict]) -> float:
    """
    Args:
        responses: list of K dicts, each with key "text" → str.

    Returns:
        Mean pairwise ROUGE-L F1 across all response pairs (float in [0, 1]).
        Returns 0.0 if fewer than 2 responses are available.
    """
    texts = [r.get("text", "") for r in responses]
    texts = [t.strip() for t in texts if t.strip()]

    if len(texts) < 2:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for t1, t2 in combinations(texts, 2):
        result = scorer.score(t1, t2)["rougeL"].fmeasure
        scores.append(result)

    return sum(scores) / len(scores)
