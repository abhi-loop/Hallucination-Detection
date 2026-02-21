import torch

def generate_k_answers(model, tokenizer, prompt, k=10):
    generations = []
    hidden_states_list = []

    for _ in range(k):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.5,     # PAPER VALUE
                top_k=5,             # PAPER VALUE
                top_p=0.99,          # PAPER VALUE
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        generations.append(text)
        hidden_states_list.append(output.hidden_states)

    return generations, hidden_states_list