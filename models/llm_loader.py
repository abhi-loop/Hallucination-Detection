import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name="facebook/opt-6.7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # New 4-bit config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16   # NEW API (instead of torch_dtype)
    )

    return tokenizer, model
