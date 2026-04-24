"""
metrics/avg_token_prob.py

Average Token Probability (approximation of MSP).

For each sampled token we have its log-probability under the model.
Converting back to probability and averaging gives a confidence estimate:

    avg_token_prob = (1/K) * sum_i [ (1/n_i) * sum_j exp(ℓ_{i,j}) ]

⚠️  This is NOT the classic Maximum Softmax Probability (MSP), which is
    defined as the maximum probability across the FULL vocabulary at each
    position.  True MSP would require storing the entire logit distribution
    (very expensive).  avg_token_prob is the correct name for what we compute:
    the mean probability the model assigned to its OWN generated tokens.

Interpretation:
    • Higher → model was confident in its sampled tokens (not necessarily correct)
    • Lower  → model was uncertain / less confident
"""

import math


def compute_avg_token_prob(responses: list[dict]) -> float:
    """
    Args:
        responses: list of K dicts, each with key "token_logprobs" → list[float].
                   Log-probs should be ≤ 0 (natural log).

    Returns:
        Mean average token probability across K responses (float in (0, 1]).
        Returns 0.0 if all responses have empty log-prob lists.
    """
    per_response_avg = []

    for resp in responses:
        logprobs = resp.get("token_logprobs", [])
        n = len(logprobs)
        if n == 0:
            continue

        avg_prob = sum(math.exp(lp) for lp in logprobs) / n
        per_response_avg.append(avg_prob)

    if not per_response_avg:
        return 0.0

    return sum(per_response_avg) / len(per_response_avg)
