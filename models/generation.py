import torch


def generate_k_answers(model, tokenizer, prompt, k=10):
    """
    Generate K diverse responses via temperature sampling.
    Used exclusively for EigenScore computation.
    """
    texts = []

    for _ in range(k):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

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
        text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
        texts.append(text)

    return texts


def generate_canonical_answer(model, tokenizer, prompt):
    """
    Generate a single deterministic (greedy) response.
    Used as the displayed answer in the chat UI.
    do_sample=False ensures the most likely token is always picked.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
