import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="facebook/opt-6.7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return tokenizer, model
