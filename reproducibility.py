"""Shared reproducibility helpers for the Transformer homework.

For reproducing the saved Bug Hunt training run, use SEED = 0.
Call set_seed() before creating the model, sampling batches or creating the optimizer.
"""
import random

import numpy as np
import torch


SEED = 0


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # The submitted experiment ran on CPU. These flags also document the
    # intended deterministic behavior if the notebook is moved to CUDA.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # May raise if an operation has no deterministic implementation.
    # Keep this opt-in rather than enabling it globally for all environments:
    # torch.use_deterministic_algorithms(True)


if __name__ == "__main__":
    set_seed()
    print(f"seed={SEED}")
    print(f"torch={torch.__version__}")
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
