import torch

def compute_eigenscore(embeddings, alpha=1e-3, eps=1e-6):
    """
    embeddings: list of K tensors (d,)
    """
    Z = torch.stack(embeddings)              # (K, d)
    Z = Z - Z.mean(dim=0, keepdim=True)      # center
    Sigma = Z @ Z.T                          # (K, K)
    K = Sigma.shape[0]
    Sigma = Sigma + alpha * torch.eye(K, device=Z.device)

    eigenvalues = torch.linalg.eigvalsh(Sigma)
    eigenvalues = torch.clamp(eigenvalues, min=eps)

    return torch.mean(torch.log(eigenvalues)).item()