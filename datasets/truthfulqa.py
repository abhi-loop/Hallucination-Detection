from datasets import load_dataset

def load_truthfulqa(split="validation"):
    dataset = load_dataset("truthful_qa", "generation", split=split)

    data = []
    for item in dataset:
        data.append({
            "question": item["question"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
        })
    return data