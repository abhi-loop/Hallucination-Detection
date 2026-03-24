import torch

def extract_sentence_embedding(model, tokenizer, text, clipper=None):
    """
    Extract a single sentence-level embedding from the model's middle hidden layer.

    Args:
        model: the loaded LLM
        tokenizer: matching tokenizer
        text: the response text to embed
        clipper: optional FeatureClipping instance; if provided, FC is applied
                 to the raw embedding before returning (memory bank updated with
                 raw features, returned tensor is clipped).

    Returns:
        embedding tensor of shape (d,)
    """
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states = outputs.hidden_states
    middle_layer = hidden_states[len(hidden_states)//2]
    embedding = middle_layer[:, -1, :].squeeze(0)  # (d,)

    if clipper is not None:
        embedding = clipper.clip(embedding)

    return embedding
