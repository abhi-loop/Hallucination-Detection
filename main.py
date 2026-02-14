import torch

from models.llm_loader import load_model
from models.hidden_extraction import extract_sentence_embedding
from models.generation import generate_k_answers
from metrics.eigenscore import compute_eigenscore


def main():
    print("Loading model...")
    tokenizer, model = load_model()

    question = "What is the capital of France? Answer in one word."

    # Generate K different responses to the question
    K = 5
    print(f"Generating {K} different responses...")
    responses = generate_k_answers(model, tokenizer, question, k=K)

    print("Extracting embeddings from each response...")
    embeddings = []
    for i, response in enumerate(responses):
        print(f"  Response {i+1}: {response[:200]}...")  # Show first 80 chars
        embedding = extract_sentence_embedding(model, tokenizer, response)
        embeddings.append(embedding)

    print("Computing EigenScore...")
    score = compute_eigenscore(embeddings)

    print("\nFinal EigenScore:", score)


if __name__ == "__main__":
    main()
