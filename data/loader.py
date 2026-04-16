"""
Causal sliding-window DataLoader for DINE.

DI(X→Y) = Σ_t I(X^t ; Y_t | Y^{t-1}), so the network at each step t needs:

    x_ctx  = X_{t-k+1}, …, X_t       shape (k,)   — causal X, *includes* current
    y_ctx  = Y_{t-k},   …, Y_{t-1}   shape (k,)   — past Y,  *excludes* current
    y_curr = Y_t                      shape ()      — target

Windows are formed for t ∈ [k, T), so the first k steps serve as burn-in.
Binary sequences → LongTensor (for embedding lookup).
Gaussian sequences → FloatTensor.

The DataLoader yields joint (x, y) windows.  The training loop forms the
marginal reference distribution by permuting x_ctx within each batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.generator import (
    BSCConfig,
    BinaryFeedbackConfig,
    GaussianAR1Config,
    generate_binary_feedback,
    generate_bsc,
    generate_gaussian_ar1,
)


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------

class CausalSample(NamedTuple):
    """One sliding-window sample returned by the Dataset and collated in batches.

    Fields
    ------
    x_ctx  : (context_len,)  causal X context, last element is X_t
    y_ctx  : (context_len,)  past Y context,   last element is Y_{t-1}
    y_curr : ()              current output Y_t (the DI target)
    """
    x_ctx:  Tensor
    y_ctx:  Tensor
    y_curr: Tensor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CausalWindowDataset(Dataset):
    """
    Wraps raw (X, Y) arrays into a CausalSample Dataset.

    Parameters
    ----------
    X, Y : ndarray, shape (n_trajectories, T)
        Raw sequences from any generator.  Must share the same shape.
    context_len : int
        Window length k.  Each window spans k steps.
    is_binary : bool
        True  → LongTensor  (alphabet indices, for embedding layers).
        False → FloatTensor (continuous values).

    Notes
    -----
    Valid time steps are t ∈ [k, T), giving (T − k) windows per trajectory.
    The window layout at time t:

        x_ctx  = X[traj, t-k+1 : t+1]   — length k, last entry is X[t]
        y_ctx  = Y[traj, t-k   : t  ]   — length k, last entry is Y[t-1]
        y_curr = Y[traj, t]

    PyTorch's default_collate preserves the NamedTuple type, so batches are
    CausalSample(x_ctx=(B,k), y_ctx=(B,k), y_curr=(B,)).
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        context_len: int,
        is_binary: bool,
    ) -> None:
        if X.shape != Y.shape:
            raise ValueError(f"X and Y shapes must match; got {X.shape} vs {Y.shape}")
        if X.ndim != 2:
            raise ValueError("X and Y must be 2-D arrays of shape (n_trajectories, T)")
        T = X.shape[1]
        if T <= context_len:
            raise ValueError(
                f"seq_len ({T}) must be strictly greater than context_len ({context_len})"
            )

        self.X = X
        self.Y = Y
        self.k = context_len
        self.is_binary = is_binary
        self._n_traj = X.shape[0]
        self._T = T
        self._per_traj = T - context_len   # valid windows per trajectory

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n_traj * self._per_traj

    def __getitem__(self, idx: int) -> CausalSample:
        traj = idx // self._per_traj
        t    = idx  % self._per_traj + self.k   # actual time index ∈ [k, T)

        x_raw  = self.X[traj, t - self.k + 1 : t + 1]   # (k,)
        y_raw  = self.Y[traj, t - self.k     : t    ]   # (k,)
        yc_raw = self.Y[traj, t]                          # scalar

        if self.is_binary:
            return CausalSample(
                x_ctx  = torch.as_tensor(x_raw,        dtype=torch.long),
                y_ctx  = torch.as_tensor(y_raw,        dtype=torch.long),
                y_curr = torch.tensor(int(yc_raw),     dtype=torch.long),
            )
        return CausalSample(
            x_ctx  = torch.as_tensor(x_raw.copy(),    dtype=torch.float32),
            y_ctx  = torch.as_tensor(y_raw.copy(),    dtype=torch.float32),
            y_curr = torch.tensor(float(yc_raw),      dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Loader config + factory
# ---------------------------------------------------------------------------

@dataclass
class LoaderConfig:
    context_len: int = 10    # k — history length fed to the network
    batch_size:  int = 256
    shuffle:     bool = True
    num_workers: int = 0
    drop_last:   bool = True   # keeps batch size uniform during training


def make_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    is_binary: bool,
    loader_cfg: LoaderConfig | None = None,
) -> DataLoader:
    """
    Wrap (X, Y) sequence arrays into a batched DataLoader of CausalSamples.

    Batch tensor shapes
    -------------------
    x_ctx  : (batch_size, context_len)   long or float32
    y_ctx  : (batch_size, context_len)   long or float32
    y_curr : (batch_size,)               long or float32

    The caller is responsible for constructing the marginal reference
    distribution (typically by shuffling x_ctx along dim=0 within the batch).
    """
    if loader_cfg is None:
        loader_cfg = LoaderConfig()
    dataset = CausalWindowDataset(
        X, Y,
        context_len=loader_cfg.context_len,
        is_binary=is_binary,
    )
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle,
        num_workers=loader_cfg.num_workers,
        drop_last=loader_cfg.drop_last,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Per-scenario convenience constructors
# ---------------------------------------------------------------------------

def bsc_loader(
    gen_cfg: BSCConfig,
    loader_cfg: LoaderConfig | None = None,
    rng: np.random.Generator | None = None,
) -> DataLoader:
    """Generate BSC data and return a causal-window DataLoader."""
    X, Y = generate_bsc(gen_cfg, rng)
    return make_dataloader(X, Y, is_binary=True, loader_cfg=loader_cfg)


def binary_feedback_loader(
    gen_cfg: BinaryFeedbackConfig,
    loader_cfg: LoaderConfig | None = None,
    rng: np.random.Generator | None = None,
) -> DataLoader:
    """Generate binary feedback channel data and return a causal-window DataLoader."""
    X, Y = generate_binary_feedback(gen_cfg, rng)
    return make_dataloader(X, Y, is_binary=True, loader_cfg=loader_cfg)


def gaussian_ar1_loader(
    gen_cfg: GaussianAR1Config,
    loader_cfg: LoaderConfig | None = None,
    rng: np.random.Generator | None = None,
) -> DataLoader:
    """Generate Gaussian AR(1) data and return a causal-window DataLoader."""
    X, Y = generate_gaussian_ar1(gen_cfg, rng)
    return make_dataloader(X, Y, is_binary=False, loader_cfg=loader_cfg)
