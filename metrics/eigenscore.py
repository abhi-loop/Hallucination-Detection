import torch

def compute_eigenscore(embeddings, alpha=1e-3):
    # embeddings: list of tensors [4096]

    Z = torch.stack(embeddings).T  # shape (4096, K)

    Z_centered = Z - Z.mean(dim=1, keepdim=True)

    Sigma = Z_centered.T @ Z_centered

    K = Sigma.shape[0]
    Sigma += alpha * torch.eye(K).to(Sigma.device)

    eigenvalues = torch.linalg.eigvalsh(Sigma)

    score = torch.mean(torch.log(eigenvalues))

    return score.item()
