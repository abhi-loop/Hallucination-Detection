import torch

def generate_k_answers(model, tokenizer, prompt, k=10):
    embeddings = []
    texts = []

    for _ in range(k):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        texts.append(text)

    return texts
