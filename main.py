"""
DINE / MINE benchmark on synthetic channels with known analytic DI/MI.

Usage examples
--------------
# DINE-MLP on Gaussian AR(1), default settings
python main.py

# DINE-LSTM on BSC
python main.py --scenario bsc --architecture lstm --crossover-prob 0.1

# MINE-MLP on Gaussian AR(1) channel_ar mode
python main.py --estimator mine --scenario gaussian_ar1 --mode channel_ar

# Full run with custom hyperparameters
python main.py --scenario binary_feedback --epochs 100 --lr 5e-4 --batch-size 512
"""

from __future__ import annotations

import argparse
import math
import os
import sys

# Make src/ and repo root importable regardless of working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import numpy as np
import torch
from torch.optim import Adam

from data.generator import (
    BSCConfig,
    BinaryFeedbackConfig,
    GaussianAR1Config,
    binary_feedback_analytic_di,
    binary_feedback_analytic_marginal_mi,
    bsc_analytic_di,
    gaussian_ar1_analytic_di,
    generate_binary_feedback,
    generate_bsc,
    generate_gaussian_ar1,
)
from data.loader import LoaderConfig, make_dataloader
from DINE.estimator import DINEEstimator, EstimatorConfig, MINEEstimator
from DINE.network import NetworkConfig, build_estimator_network

NATS_TO_BITS = 1.0 / math.log(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DINE or MINE on a synthetic channel and compare to the analytic ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scenario
    p.add_argument(
        "--scenario",
        choices=["bsc", "binary_feedback", "gaussian_ar1"],
        default="gaussian_ar1",
        help="Synthetic channel to benchmark.",
    )
    p.add_argument("--crossover-prob", type=float, default=0.1,
                   help="BSC crossover probability p.")
    p.add_argument("--mode", choices=["channel_ar", "source_ar"], default="source_ar",
                   help="Gaussian AR(1) mode.")
    p.add_argument("--phi",     type=float, default=0.8,  help="AR(1) channel memory coeff (channel_ar).")
    p.add_argument("--rho",     type=float, default=0.9,  help="AR(1) source memory coeff (source_ar).")
    p.add_argument("--sigma-x", type=float, default=1.0,  help="Signal std.")
    p.add_argument("--sigma-n", type=float, default=0.5,  help="Noise std.")

    # Estimator / architecture
    p.add_argument("--estimator",    choices=["dine", "mine"],      default="dine")
    p.add_argument("--architecture", choices=["mlp", "lstm"],       default="mlp")
    p.add_argument("--context-len",  type=int,          default=10, help="Sliding-window context length k.")
    p.add_argument("--hidden-dims",  type=int, nargs="+", default=[128, 128],
                   help="MLP hidden layer sizes.")
    p.add_argument("--hidden-dim",   type=int, default=128,  help="LSTM hidden size.")
    p.add_argument("--num-layers",   type=int, default=1,    help="LSTM depth.")
    p.add_argument("--dropout",      type=float, default=0.0)

    # Training
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--seq-len",    type=int,   default=5_000, help="Sequence length per trajectory.")
    p.add_argument("--n-samples",  type=int,   default=5,     help="Number of independent trajectories.")
    p.add_argument("--clip-grad",  type=float, default=5.0,   help="Gradient clip norm (0 to disable).")
    p.add_argument("--log-every",  type=int,   default=10,    help="Print every N epochs (0 to silence).")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int,   default=42)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _build_scenario(args: argparse.Namespace, rng: np.random.Generator):
    """Return (X, Y, is_binary, analytic_bits, extra_info_str, label)."""
    extra = ""
    if args.scenario == "bsc":
        cfg = BSCConfig(
            crossover_prob=args.crossover_prob,
            seq_len=args.seq_len,
            n_samples=args.n_samples,
        )
        X, Y = generate_bsc(cfg, rng)
        analytic = bsc_analytic_di(args.crossover_prob)
        label = f"BSC(p={args.crossover_prob})"
        return X, Y, True, analytic, extra, label

    if args.scenario == "binary_feedback":
        cfg = BinaryFeedbackConfig(seq_len=args.seq_len, n_samples=args.n_samples)
        X, Y = generate_binary_feedback(cfg, rng)
        analytic = binary_feedback_analytic_di(cfg)
        mi = binary_feedback_analytic_marginal_mi(cfg)
        extra = f"  Marginal MI (no feedback) : {mi:.4f} bits  (DI > MI ✓ shows feedback gains)"
        label = "BinaryFeedback"
        return X, Y, True, analytic, extra, label

    # gaussian_ar1
    cfg = GaussianAR1Config(
        mode=args.mode,
        phi=args.phi,
        rho=args.rho,
        sigma_x=args.sigma_x,
        sigma_n=args.sigma_n,
        seq_len=args.seq_len,
        n_samples=args.n_samples,
    )
    X, Y = generate_gaussian_ar1(cfg, rng)
    analytic = gaussian_ar1_analytic_di(cfg)
    label = f"GaussianAR1(mode={args.mode})"
    return X, Y, False, analytic, extra, label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*62}")
    print(f"  Scenario    : {args.scenario}")
    print(f"  Estimator   : {args.estimator.upper()}")
    print(f"  Architecture: {args.architecture.upper()}")
    print(f"  Device      : {args.device}")
    print(f"{'='*62}\n")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X, Y, is_binary, analytic_bits, extra_info, label = _build_scenario(args, rng)

    print(f"Scenario : {label}")
    print(f"  Analytic DI  : {analytic_bits:.4f} bits  ({analytic_bits / NATS_TO_BITS:.4f} nats)")
    if extra_info:
        print(f"  {extra_info}")
    print(f"  Data shape   : X={X.shape}, Y={Y.shape}\n")

    loader_cfg = LoaderConfig(
        context_len=args.context_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    loader = make_dataloader(X, Y, is_binary=is_binary, loader_cfg=loader_cfg)

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    # Map CLI estimator name to NetworkConfig estimator type
    estimator_key = "di" if args.estimator == "dine" else "mi"

    if args.estimator == "mine" and args.architecture == "lstm":
        print("Warning: LSTM is not supported for MINE — falling back to MLP.\n")
        args.architecture = "mlp"

    net_cfg = NetworkConfig(
        estimator=estimator_key,
        architecture=args.architecture,
        x_dim=1,
        y_dim=1,
        context_len=args.context_len,
        hidden_dims=tuple(args.hidden_dims),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    net = build_estimator_network(net_cfg)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Network : {net.__class__.__name__}  ({n_params:,} parameters)\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    clip = args.clip_grad if args.clip_grad > 0 else None
    est_cfg = EstimatorConfig(
        lr=args.lr,
        n_epochs=args.epochs,
        clip_grad_norm=clip,
        device=args.device,
        log_every=args.log_every,
    )
    opt = Adam(net.parameters(), lr=args.lr)

    if args.estimator == "dine":
        estimator = DINEEstimator(net, opt, est_cfg)
    else:
        estimator = MINEEstimator(net, opt, est_cfg)

    print(f"Training for {args.epochs} epochs …\n")
    estimator.fit(loader)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    neural_nats = estimator.estimate(loader)
    neural_bits = neural_nats * NATS_TO_BITS
    rel_err = abs(neural_bits - analytic_bits) / max(analytic_bits, 1e-12) * 100

    print(f"\n{'='*62}")
    print(f"  Analytic DI   : {analytic_bits:.4f} bits")
    print(f"  Neural est.   : {neural_bits:.4f} bits  ({neural_nats:.4f} nats)")
    print(f"  Absolute err  : {abs(neural_bits - analytic_bits):.4f} bits")
    print(f"  Relative err  : {rel_err:.1f}%")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
