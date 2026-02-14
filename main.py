import torch

from models.llm_loader import load_model
from models.hidden_extraction import extract_sentence_embedding
from metrics.eigenscore import compute_eigenscore


def main():
    print("Loading model...")
    tokenizer, model = load_model()

    question = "What is the capital of France?"

    print("Extracting embeddings...")

    embeddings = []

    # For now we simulate K=5 samples
    # Later you will generate different responses
    K = 5

    for i in range(K):
        embedding = extract_sentence_embedding(model, tokenizer, question)
        embeddings.append(embedding)

    print("Computing EigenScore...")
    score = compute_eigenscore(embeddings)

    print("\nFinal EigenScore:", score)


if __name__ == "__main__":
    main()
