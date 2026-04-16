"""
Synthetic data generators for directed information (DI) estimation benchmarks.

Three canonical scenarios with known analytic DI ground truth:

  1. BSC(p)                  — memoryless, DI = 1 - H_b(p)
  2. Binary unifilar channel — input uses causal feedback, DI computed via
                               stationary Markov analysis; DI > marginal I(X;Y)
  3. Gaussian AR(1)          — continuous, analytic DI via Kalman steady-state
                               (spectral method)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _h2(p: float) -> float:
    """Binary entropy in bits. Safe at p=0 and p=1."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


# ---------------------------------------------------------------------------
# 1.  Binary Symmetric Channel (BSC)
# ---------------------------------------------------------------------------

@dataclass
class BSCConfig:
    crossover_prob: float = 0.1   # p in BSC(p)
    seq_len: int = 1_000
    n_samples: int = 1            # independent trajectories


def generate_bsc(
    cfg: BSCConfig,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary Symmetric Channel:

        X_t ~ Bern(0.5) i.i.d.
        Y_t = X_t XOR N_t,   N_t ~ Bern(p) i.i.d.

    Returns
    -------
    X : int8 array, shape (n_samples, seq_len)
    Y : int8 array, shape (n_samples, seq_len)

    Analytic DI (bits):  DI(X→Y) = 1 − H_b(p)
    """
    if rng is None:
        rng = np.random.default_rng()
    p = cfg.crossover_prob
    X = rng.integers(0, 2, size=(cfg.n_samples, cfg.seq_len), dtype=np.int8)
    N = (rng.random((cfg.n_samples, cfg.seq_len)) < p).astype(np.int8)
    Y = (X ^ N).astype(np.int8)
    return X, Y


def bsc_analytic_di(p: float) -> float:
    """DI = capacity = 1 − H_b(p)  (bits)."""
    return 1.0 - _h2(p)


# ---------------------------------------------------------------------------
# 2.  Binary unifilar channel with causal feedback
# ---------------------------------------------------------------------------

@dataclass
class BinaryFeedbackConfig:
    """
    Unifilar binary channel where the optimal input exploits causal feedback.

    Channel model
    -------------
    State       S_t = Y_{t-1}           (previous output)
    Transition  P(Y_t = 1 | X_t, S_t):

        chan[x, s] = P(Y_t = 1 | X_t = x, S_t = s)

        Default: asymmetric Z-type channel with memory
          chan[0,0]=0.05, chan[1,0]=0.90  (S=0: X=1 reliably gives Y=1)
          chan[0,1]=0.85, chan[1,1]=0.10  (S=1: X=0 reliably gives Y=1)

    Feedback input policy
    ---------------------
    P(X_t = 1 | S_t = 0) = alpha
    P(X_t = 1 | S_t = 1) = beta

    Default alpha=0.9, beta=0.1:
      When S=0 prefer X=1 (drives Y→1 with high prob).
      When S=1 prefer X=0 (drives Y→1 with high prob).
    This aggressive feedback policy creates a situation where
    the causal DI is strictly larger than the marginal I(X_t ; Y_t).
    """
    chan_00: float = 0.05   # P(Y=1 | X=0, S=0)
    chan_10: float = 0.90   # P(Y=1 | X=1, S=0)
    chan_01: float = 0.85   # P(Y=1 | X=0, S=1)
    chan_11: float = 0.10   # P(Y=1 | X=1, S=1)
    alpha: float = 0.90     # P(X=1 | S=0)  — feedback policy
    beta: float = 0.10      # P(X=1 | S=1)  — feedback policy
    seq_len: int = 1_000
    n_samples: int = 1
    y0: int = 0             # initial state S_1 = Y_0


def generate_binary_feedback(
    cfg: BinaryFeedbackConfig,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Binary unifilar channel with causal feedback.

    Returns
    -------
    X : int8 array, shape (n_samples, seq_len)
    Y : int8 array, shape (n_samples, seq_len)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, B = cfg.seq_len, cfg.n_samples

    # chan[x, s] = P(Y=1 | X=x, S=s)
    chan = np.array(
        [[cfg.chan_00, cfg.chan_01],
         [cfg.chan_10, cfg.chan_11]],
        dtype=np.float64,
    )
    # policy[s] = P(X=1 | S=s)
    policy = np.array([cfg.alpha, cfg.beta], dtype=np.float64)

    X_out = np.empty((B, T), dtype=np.int8)
    Y_out = np.empty((B, T), dtype=np.int8)
    S = np.full(B, cfg.y0, dtype=np.int8)

    u_x = rng.random((B, T))
    u_y = rng.random((B, T))

    for t in range(T):
        p_x1 = policy[S]                        # P(X=1 | S_t), shape (B,)
        X_t = (u_x[:, t] < p_x1).astype(np.int8)
        p_y1 = chan[X_t, S]                     # P(Y=1 | X_t, S_t), shape (B,)
        Y_t = (u_y[:, t] < p_y1).astype(np.int8)
        X_out[:, t] = X_t
        Y_out[:, t] = Y_t
        S = Y_t

    return X_out, Y_out


def _binary_feedback_stationary(cfg: BinaryFeedbackConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pi, p_y1_given_s) where pi[s] = P(S_t = s) in stationarity
    and p_y1_given_s[s] = P(Y_t = 1 | S_t = s).
    """
    chan = np.array(
        [[cfg.chan_00, cfg.chan_01],
         [cfg.chan_10, cfg.chan_11]],
    )
    policy = np.array([cfg.alpha, cfg.beta])

    # P(Y=1 | S=s) = (1 − policy[s])·chan[0,s] + policy[s]·chan[1,s]
    p_y1 = (1.0 - policy) * chan[0] + policy * chan[1]   # shape (2,)

    # Transition matrix on S: T[s, s'] = P(Y_t = s' | S_t = s)
    T_mat = np.column_stack([1.0 - p_y1, p_y1])          # shape (2, 2)

    # Stationary: π (T − I) = 0,  sum π = 1
    A = (T_mat - np.eye(2)).T
    A = np.vstack([A, np.ones(2)])
    b = np.array([0.0, 0.0, 1.0])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi, p_y1


def binary_feedback_analytic_di(cfg: BinaryFeedbackConfig) -> float:
    """
    Analytic directed-information rate (bits/step) for the unifilar channel.

        DI/step = I(X_t ; Y_t | S_t)
                = H(Y_t | S_t) − H(Y_t | X_t, S_t)

    Valid when the input policy is first-order Markov in S_t = Y_{t-1}.
    """
    chan = np.array(
        [[cfg.chan_00, cfg.chan_01],
         [cfg.chan_10, cfg.chan_11]],
    )
    policy = np.array([cfg.alpha, cfg.beta])
    pi, p_y1_given_s = _binary_feedback_stationary(cfg)

    # H(Y_t | S_t) = Σ_s π[s] · H_b(P(Y=1|S=s))
    h_y_s = float(sum(pi[s] * _h2(p_y1_given_s[s]) for s in range(2)))

    # H(Y_t | X_t, S_t) = Σ_{s,x} π[s]·P(X=x|S=s)·H_b(chan[x,s])
    h_y_xs = 0.0
    for s in range(2):
        for x in range(2):
            p_x = policy[s] if x == 1 else (1.0 - policy[s])
            h_y_xs += pi[s] * p_x * _h2(chan[x, s])

    return h_y_s - h_y_xs


def binary_feedback_analytic_marginal_mi(cfg: BinaryFeedbackConfig) -> float:
    """
    Marginal MI per step: I(X_t ; Y_t) — no causal conditioning.
    Compare with DI to observe the effect of the feedback structure.
    """
    chan = np.array(
        [[cfg.chan_00, cfg.chan_01],
         [cfg.chan_10, cfg.chan_11]],
    )
    policy = np.array([cfg.alpha, cfg.beta])
    pi, p_y1_given_s = _binary_feedback_stationary(cfg)

    p_x1 = float(pi @ policy)                      # marginal P(X=1)
    p_y1 = float(pi @ p_y1_given_s)                # marginal P(Y=1)

    # Joint P(X=x, Y=y) = Σ_s π[s]·P(X=x|s)·P(Y=y|X=x,s)
    pxy = np.zeros((2, 2))
    for s in range(2):
        for x in range(2):
            p_x = policy[s] if x == 1 else (1.0 - policy[s])
            for y in range(2):
                p_y = chan[x, s] if y == 1 else (1.0 - chan[x, s])
                pxy[x, y] += pi[s] * p_x * p_y

    px = np.array([1.0 - p_x1, p_x1])
    py = np.array([1.0 - p_y1, p_y1])
    mi = 0.0
    for x in range(2):
        for y in range(2):
            if pxy[x, y] > 1e-15:
                mi += pxy[x, y] * np.log2(pxy[x, y] / (px[x] * py[y]))
    return mi


# ---------------------------------------------------------------------------
# 3.  Gaussian AR(1) channel
# ---------------------------------------------------------------------------

@dataclass
class GaussianAR1Config:
    """
    Two modes, both with closed-form DI:

    'channel_ar':   AR(1) channel with i.i.d. input
        Y_t = phi·Y_{t-1} + X_t + sigma_n·eps_t,   X_t ~ N(0, sigma_x²) i.i.d.
        DI/step = ½ log₂(1 + sigma_x²/sigma_n²)          (AWGN rate)
        Channel memory cancels under causal conditioning on Y^{t-1}.

    'source_ar':    AR(1) source through memoryless AWGN channel
        X_t = rho·X_{t-1} + sqrt(1−rho²)·sigma_x·Z_t
        Y_t = X_t + sigma_n·eps_t
        DI/step = ½ log₂(1 + v/sigma_n²)
        where v is the steady-state Kalman prediction variance.
        DI depends on rho — illustrates spectral / Kalman analysis.
    """
    mode: str = "source_ar"      # "channel_ar" | "source_ar"
    phi: float = 0.8             # AR coeff for channel memory (channel_ar)
    rho: float = 0.9             # AR coeff for source memory (source_ar)
    sigma_x: float = 1.0        # std of X (innovations scaled so Var(X)=sigma_x²)
    sigma_n: float = 0.5        # std of channel noise
    seq_len: int = 1_000
    n_samples: int = 1


def generate_gaussian_ar1(
    cfg: GaussianAR1Config,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian AR(1) process in either mode.

    Returns
    -------
    X : float32 array, shape (n_samples, seq_len)
    Y : float32 array, shape (n_samples, seq_len)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, B = cfg.seq_len, cfg.n_samples
    eps = (rng.standard_normal((B, T)) * cfg.sigma_n).astype(np.float32)

    if cfg.mode == "channel_ar":
        X = (rng.standard_normal((B, T)) * cfg.sigma_x).astype(np.float32)
        Y = np.empty((B, T), dtype=np.float32)
        Y_prev = np.zeros(B, dtype=np.float32)
        for t in range(T):
            Y[:, t] = cfg.phi * Y_prev + X[:, t] + eps[:, t]
            Y_prev = Y[:, t]

    elif cfg.mode == "source_ar":
        innov_std = float(np.sqrt(max(1.0 - cfg.rho**2, 0.0))) * cfg.sigma_x
        innov = (rng.standard_normal((B, T)) * innov_std).astype(np.float32)
        X = np.empty((B, T), dtype=np.float32)
        X_prev = np.zeros(B, dtype=np.float32)
        for t in range(T):
            X[:, t] = cfg.rho * X_prev + innov[:, t]
            X_prev = X[:, t]
        Y = X + eps

    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}")

    return X, Y


def gaussian_ar1_analytic_di(cfg: GaussianAR1Config) -> float:
    """
    Analytic DI per time step (bits).

    channel_ar:
        Conditioning on Y^{t-1} removes phi·Y_{t-1}, leaving X_t + noise.
        DI/step = ½ log₂(1 + sigma_x²/sigma_n²)

    source_ar:
        Steady-state Kalman filter gives prediction variance v satisfying
            v² + v·(1−ρ²)·(σ_n²−σ_x²) − (1−ρ²)·σ_x²·σ_n² = 0
        DI/step = ½ log₂(1 + v/σ_n²)
    """
    if cfg.mode == "channel_ar":
        return 0.5 * np.log2(1.0 + cfg.sigma_x**2 / cfg.sigma_n**2)

    elif cfg.mode == "source_ar":
        r2 = cfg.rho**2
        sx2 = cfg.sigma_x**2
        sn2 = cfg.sigma_n**2
        # coefficients of: v² + b·v + c = 0
        b = (1.0 - r2) * (sn2 - sx2)
        c = -(1.0 - r2) * sx2 * sn2
        disc = b**2 - 4.0 * c          # always non-negative for valid params
        v = (-b + np.sqrt(disc)) / 2.0  # positive root (prediction variance)
        return 0.5 * np.log2(1.0 + v / sn2)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}")
