# STOPWORDS = {"what", "does", "this", "that", "with", "from", "have", "your", "they", "when",
#              "will", "been", "were", "there", "their", "about", "which", "would", "could"}

# def label_response(response: str, correct_answers: list[str], question: str = "") -> int:
#     """
#     Returns 0 (factual) if the response overlaps with any correct answer.
#     Three matching strategies (in order):
#       1. Correct answer is a substring of the response
#       2. Response is a substring of any correct answer
#       3. Keyword overlap: enough meaningful words from the correct answer appear in the response
#     """
#     # Strip the echoed prompt from the response
#     if question and response.lower().startswith(question.lower()):
#         response = response[len(question):].strip()

#     response_lower = response.lower().strip()
#     if not response_lower:
#         return 1

#     for answer in correct_answers:
#         answer_lower = answer.lower().strip()

#         # Strategy 1: exact substring match (either direction)
#         if answer_lower in response_lower or (len(response_lower) > 5 and response_lower in answer_lower):
#             return 0

#         # Strategy 3: keyword overlap
#         # Extract meaningful words (len > 3, not stopwords) from the correct answer
#         keywords = [w for w in answer_lower.split()
#                     if len(w) > 3 and w not in STOPWORDS]
#         if keywords:
#             matches = sum(1 for kw in keywords if kw in response_lower)
#             # If more than half the keywords appear in the response → factual
#             if matches / len(keywords) >= 0.5:
#                 return 0

#     return 1


# def majority_label(responses: list[str], correct_answers: list[str], question: str = "") -> int:
#     """
#     Labels each of the K responses and returns the majority vote.
#     Tie → treated as hallucination (conservative).
#     """
#     labels = [label_response(r, correct_answers, question=question) for r in responses]
#     hallucination_count = sum(labels)
#     factual_count = len(labels) - hallucination_count
#     return 0 if factual_count > hallucination_count else 1


import re
from rouge_score import rouge_scorer
import torch
from sentence_transformers import SentenceTransformer, util

# Load on CPU — labeling is not latency-critical; keeps GPU free for the LLM
_semantic_model = SentenceTransformer("nli-roberta-large", device="cpu")

# Paper thresholds
ROUGE_L_THRESHOLD = 0.5
SEMANTIC_SIM_THRESHOLD = 0.9


def _normalize(text: str) -> str:
    """
    Light normalization (paper does NOT over-normalize)
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def response_is_correct(response: str, correct_answers: list[str]) -> bool:
    """
    Check whether ONE response is factually correct
    using paper-defined correctness measures.
    """

    response = _normalize(response)
    if len(response) == 0:
        return False

    # -------- ROUGE-L --------
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for gt in correct_answers:
        gt = _normalize(gt)

        rouge_score = rouge.score(gt, response)["rougeL"].fmeasure
        if rouge_score >= ROUGE_L_THRESHOLD:
            return True

    # -------- Semantic similarity --------
    with torch.no_grad():
        resp_emb = _semantic_model.encode(response, convert_to_tensor=True)

        for gt in correct_answers:
            gt = _normalize(gt)
            gt_emb = _semantic_model.encode(gt, convert_to_tensor=True)

            sim = util.cos_sim(resp_emb, gt_emb).item()
            if sim >= SEMANTIC_SIM_THRESHOLD:
                return True

    return False


def label_question(responses: list[str], correct_answers: list[str]) -> int:
    """
    Question-level label.

    Returns:
        0 -> factual
        1 -> hallucination
    """

    for r in responses:
        if response_is_correct(r, correct_answers):
            return 0  # factual

    return 1  # hallucination