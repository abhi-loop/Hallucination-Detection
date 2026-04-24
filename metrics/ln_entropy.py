"""
metrics/ln_entropy.py

Length-Normalized (Per-Token) Entropy.

For each of the K responses, compute the per-token entropy of the sampled
sequence, then average across all K responses.

Formula (per response i with n_i tokens and log-probs ℓ_{i,j}):
    H_i = -(1 / n_i) * sum_j( ℓ_{i,j} )
LN-Entropy = mean_i( H_i )

A higher value → the model was more uncertain when generating these responses
(each sampled token had a lower probability on average).
"""


def compute_ln_entropy(responses: list[dict]) -> float:
    """
    Args:
        responses: list of K dicts, each with key "token_logprobs" → list[float].
                   Log-probs should be ≤ 0 (natural log of a probability).

    Returns:
        Mean length-normalized entropy across K responses (float).
        Returns 0.0 if all responses have empty log-prob lists.
    """
    per_response_entropy = []

    for resp in responses:
        logprobs = resp.get("token_logprobs", [])
        n = len(logprobs)
        if n == 0:
            continue  # skip degenerate responses

        # H_i = -(1/n) * sum(log_probs)  — note logprobs are already ≤ 0
        H_i = -sum(logprobs) / n
        per_response_entropy.append(H_i)

    if not per_response_entropy:
        return 0.0

    return sum(per_response_entropy) / len(per_response_entropy)
