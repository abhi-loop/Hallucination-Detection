import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name="facebook/opt-6.7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 4-bit config with CPU offload enabled
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True   # allow overflow to CPU RAM
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        max_memory={0: "5GiB", "cpu": "20GiB"}  # 6GB GPU; nli-roberta is on CPU now
    )

    return tokenizer, model
