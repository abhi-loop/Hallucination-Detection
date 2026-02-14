import torch

def extract_sentence_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states
    middle_layer = hidden_states[len(hidden_states)//2]
    last_token_embedding = middle_layer[:, -1, :]

    return last_token_embedding.squeeze(0)
