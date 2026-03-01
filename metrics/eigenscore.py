import torch

def compute_eigenscore(embeddings, alpha=1e-3, eps=1e-6):
    """
    embeddings: list of K tensors of shape (d,)
                OR tensor of shape (K, d)
    alpha: regularization strength (increased for stability)
    eps: minimum eigenvalue threshold to prevent log(0) or log(negative)
    """

    # Convert to tensor if list
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)

    # embeddings: (K, d)
    device = embeddings.device

    # Center across samples
    Z = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Compute covariance in K×K space
    Sigma = Z @ Z.T   # (K, K)

    K = Sigma.shape[0]

    # Regularization (same device!) - increased for stability
    Sigma = Sigma + alpha * torch.eye(K, device=device)

    # Eigenvalues (symmetric matrix → use eigvalsh)
    eigenvalues = torch.linalg.eigvalsh(Sigma)
    
    # Clamp eigenvalues to prevent log(0) or log(negative)
    eigenvalues_clamped = torch.clamp(eigenvalues, min=eps)

    # Log eigenvalues
    log_eigenvalues = torch.log(eigenvalues_clamped)
    
    score = torch.mean(log_eigenvalues)

    return score.item()
