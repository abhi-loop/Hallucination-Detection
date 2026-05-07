"""
models/generation.py

Two generation modes:
  1. generate_k_answers_with_logprobs  — primary dataset builder
       Generates K sampled responses AND captures per-token log-probs.
       Used by pipeline/run_dataset.py.

  2. generate_k_answers                — legacy / EigenScore-only runs
  3. generate_canonical_answer         — deterministic; used by the API
"""

import torch
import torch.nn.functional as F

_EMPTY_FALLBACK = (
    "I'm sorry, I couldn't generate a response for that. "
    "Please try rephrasing your question."
)

# Default generation hyperparameters (stored in raw_data.jsonl)
DEFAULT_GEN_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.99,
    "top_k": 5,
    "max_new_tokens": 50,
}


def _wrap_prompt(prompt: str) -> str:
    """
    OPT is a raw completion model (not instruction-tuned).
    Framing the input as 'Q: ... \\nA:' gives the model a clear continuation
    target so it actually generates an answer instead of an empty/EOS output.
    """
    return f"Q: {prompt.strip()}\nA:"


# ---------------------------------------------------------------------------
# PRIMARY: generation with log-probs (for dataset building)
# ---------------------------------------------------------------------------

def generate_k_answers_with_logprobs(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
    gen_config: dict | None = None,
    clipper=None,
) -> tuple[list[dict], list, dict]:
    """
    Generate K diverse responses, capture per-token log-probabilities,
    and capture hidden-state embeddings from the generation pass itself.

    Args:
        model:      loaded LLM (HuggingFace CausalLM)
        tokenizer:  matching tokenizer
        prompt:     raw question string (will be wrapped internally)
        k:          number of responses to generate
        gen_config: dict of generation hyperparameters; defaults to DEFAULT_GEN_CONFIG
        clipper:    optional FeatureClipping instance applied to each embedding

    Returns:
        responses   — list of K dicts:
                        {
                          "text": str,                  # decoded response
                          "token_logprobs": [float, ...]  # per generated token
                        }
        embeddings  — list of K tensors of shape (d,), one per response;
                      extracted from the middle hidden layer during generation
                      (NOT re-encoded from decoded text)
        used_config — the generation config dict actually used (for JSONL storage)
    """
    if gen_config is None:
        gen_config = DEFAULT_GEN_CONFIG.copy()

    wrapped = _wrap_prompt(prompt)
    inputs = tokenizer(wrapped, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    responses = []
    embeddings = []

    for _ in range(k):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_config.get("max_new_tokens", 50),
                do_sample=True,
                temperature=gen_config.get("temperature", 0.5),
                top_p=gen_config.get("top_p", 0.99),
                top_k=gen_config.get("top_k", 5),
                output_scores=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # ── Decode text ──────────────────────────────────────────────────────
        generated_ids = out.sequences[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not text:
            text = _EMPTY_FALLBACK

        # ── Extract per-token log-probs ──────────────────────────────────────
        # out.scores is a tuple of length T (one logit tensor per generated step)
        # Each tensor has shape (batch=1, vocab_size)
        token_logprobs = []
        for step_idx, step_scores in enumerate(out.scores):
            if step_idx >= len(generated_ids):
                break
            token_id = generated_ids[step_idx].item()
            log_probs = F.log_softmax(step_scores[0], dim=-1)  # (vocab_size,)
            lp = log_probs[token_id].item()
            token_logprobs.append(lp)

        # ── Extract hidden-state embedding from generation ───────────────────
        # out.hidden_states: tuple of T steps, each step is a tuple of
        # (num_layers+1) tensors of shape (batch=1, seq_len, d).
        # We pick the middle layer and mean-pool across all generated token steps
        # to get a single (d,) sentence-level embedding.
        if out.hidden_states and len(out.hidden_states) > 0:
            num_layers = len(out.hidden_states[0])          # total layer count
            mid = num_layers // 2                            # middle layer index
            # Collect the middle-layer hidden state for each generated step
            # Each step's hidden state at the generated position: shape (1, 1, d)
            step_vecs = [
                step[mid][0, -1, :]                          # (d,)
                for step in out.hidden_states
                if step[mid].shape[1] > 0
            ]
            if step_vecs:
                # Last-token pooling: use only the final generated token's hidden state
                embedding = step_vecs[-1]                    # (d,)
            else:
                embedding = out.hidden_states[0][mid][0, -1, :]
        else:
            # Fallback: should not normally occur
            embedding = torch.zeros(model.config.hidden_size, device="cuda")

        if clipper is not None:
            embedding = clipper.clip(embedding)

        embeddings.append(embedding.detach())
        responses.append({"text": text, "token_logprobs": token_logprobs})

    return responses, embeddings, gen_config


# ---------------------------------------------------------------------------
# LEGACY: text-only generation (keep for backward compatibility / API)
# ---------------------------------------------------------------------------

def generate_k_answers(model, tokenizer, prompt, k=10):
    """
    Generate K diverse responses via temperature sampling.
    Used exclusively for EigenScore computation (legacy pipeline).
    """
    texts = []
    wrapped = _wrap_prompt(prompt)

    for _ in range(k):
        inputs = tokenizer(wrapped, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5,
                top_p=0.99,
                top_k=5
            )

        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
        texts.append(text if text else _EMPTY_FALLBACK)

    return texts


def generate_canonical_answer(model, tokenizer, prompt):
    """
    Generate a single deterministic (greedy) response for display in chat.
    do_sample=False picks the most likely token at each step.
    """
    wrapped = _wrap_prompt(prompt)
    inputs = tokenizer(wrapped, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
    return text if text else _EMPTY_FALLBACK
