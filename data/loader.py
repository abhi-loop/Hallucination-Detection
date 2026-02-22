from datasets import load_dataset

def load_truthfulqa(split="validation", limit=None):
    """
    Load TruthfulQA from HuggingFace (generation config).
    Returns a list of dicts: {question, correct_answers, incorrect_answers}
    """
    dataset = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    for row in dataset:
        records.append({
            "question": row["question"],
            "correct_answers": row["correct_answers"],   # list of strings
            "incorrect_answers": row["incorrect_answers"] # list of strings
        })

    return records
