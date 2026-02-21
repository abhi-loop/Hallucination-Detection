from models.load_model import load_model
from models.generate_k_answers import generate_k_answers
from models.extract_sentence_embedding import extract_sentence_embedding

from datasets.truthfulqa import load_truthfulqa
from metrics.eigenscore import compute_eigenscore
from metrics.labels import compute_truthfulqa_label
from metrics.threshold import find_best_threshold

def main():
    tokenizer, model = load_model()
    dataset = load_truthfulqa(split="validation")

    scores = []
    labels = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        correct_answers = item["correct_answers"]

        generations, hidden_states_list = generate_k_answers(
            model, tokenizer, question, k=10
        )

        embeddings = [
            extract_sentence_embedding(hs)
            for hs in hidden_states_list
        ]

        eigenscore = compute_eigenscore(embeddings)
        label = compute_truthfulqa_label(generations[0], correct_answers)

        scores.append(eigenscore)
        labels.append(label)

        if idx % 10 == 0:
            print(f"[{idx}] EigenScore={eigenscore:.3f}, Label={label}")

    threshold, gmean, auroc = find_best_threshold(scores, labels)

    print("\n========== FINAL RESULTS ==========")
    print(f"AUROC      : {auroc:.4f}")
    print(f"Threshold  : {threshold:.4f}")
    print(f"G-Mean     : {gmean:.4f}")

if __name__ == "__main__":
    main()