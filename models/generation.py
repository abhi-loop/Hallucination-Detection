import torch

_EMPTY_FALLBACK = "I'm sorry, I couldn't generate a response for that. Please try rephrasing your question."

def _wrap_prompt(prompt: str) -> str:
    """
    OPT is a raw completion model (not instruction-tuned).
    Framing the input as 'Q: ... \\nA:' gives the model a clear continuation
    target so it actually generates an answer instead of an empty/EOS output.
    """
    return f"Q: {prompt.strip()}\nA:"


def generate_k_answers(model, tokenizer, prompt, k=10):
    """
    Generate K diverse responses via temperature sampling.
    Used exclusively for EigenScore computation.
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

