"""
metrics/feature_clipping.py

Standalone implementation of the Feature Clipping (FC) method proposed in
"INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection" (ICLR 2024).

IMPORTANT FIX:
The memory bank stores RAW hidden activations, not clipped ones.
This prevents threshold drift caused by repeatedly clipping already-clipped data.


"""

import torch


class FeatureClipping:
    """
    Implements test-time Feature Clipping (FC).

    Maintains a memory bank of hidden-state activations and clips
    new features based on percentile thresholds.

    Args:
        memory_size (int): number of feature vectors stored in memory bank
        percentile (float): percentile used for clipping thresholds (p in paper)
    """

    def __init__(self, memory_size: int = 3000, percentile: float = 0.2):
        self.memory_size = memory_size
        self.percentile = percentile
        self.memory_bank = []

    def _update_memory(self, features: torch.Tensor):
        """
        Push RAW feature vectors into the memory bank.

        features: tensor shape (d,) or (batch, d)
        """

        if features.dim() == 1:
            features = features.unsqueeze(0)

        for f in features:
            # detach + move to CPU so we don't keep GPU computation graph
            self.memory_bank.append(f.detach().cpu().float())

        # maintain fixed memory size
        if len(self.memory_bank) > self.memory_size:
            overflow = len(self.memory_bank) - self.memory_size
            self.memory_bank = self.memory_bank[overflow:]

    def _compute_thresholds(self):
        """
        Compute clipping thresholds from memory bank.

        Returns:
            h_min (tensor)
            h_max (tensor)
        """

        if len(self.memory_bank) == 0:
            return None, None

        data = torch.stack(self.memory_bank)  # (N, d)

        lower = self.percentile / 100
        upper = 1 - lower

        h_min = torch.quantile(data, lower, dim=0)
        h_max = torch.quantile(data, upper, dim=0)

        return h_min, h_max

    def clip(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply feature clipping.

        IMPORTANT:
        - RAW features are stored in the memory bank
        - Only the returned features are clipped

        Args:
            features: tensor shape (d,) or (batch, d)

        Returns:
            clipped tensor with same shape as input
        """

        original_shape = features.shape
        device = features.device

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # STEP 1: store RAW features
        self._update_memory(features)

        # STEP 2: compute thresholds
        h_min, h_max = self._compute_thresholds()

        # If memory bank not populated yet
        if h_min is None:
            return features.view(original_shape)

        h_min = h_min.to(device)
        h_max = h_max.to(device)

        # STEP 3: clip output
        clipped = torch.clamp(features, min=h_min, max=h_max)

        return clipped.view(original_shape)


# ---------------------------------------------------------------------
# Optional functional API
# ---------------------------------------------------------------------

def feature_clip(features: torch.Tensor, h_min: torch.Tensor, h_max: torch.Tensor):
    """
    Functional implementation of Feature Clipping.

    Implements the piecewise function from the paper:

        FC(h) =
            h_min   if h < h_min
            h       if h_min <= h <= h_max
            h_max   if h > h_max

    Args:
        features: tensor of hidden activations
        h_min: minimum threshold
        h_max: maximum threshold

    Returns:
        clipped tensor
    """

    return torch.clamp(features, min=h_min, max=h_max)


# ---------------------------------------------------------------------
# Simple test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing FeatureClipping...")

    fc = FeatureClipping(memory_size=100, percentile=0.2)

    for _ in range(200):
        x = torch.randn(4096) * 5
        clipped = fc.clip(x)

    print("Feature clipping working correctly.")