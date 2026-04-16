"""
DINE / MINE training objectives and estimator wrappers.

Both use the Donsker–Varadhan (DV) lower bound on KL divergence:

    DI(X→Y)  ≥  E_p[T(x_ctx, y_ctx, y_curr)]  −  log E_q[e^T]
    MI(X;Y)  ≥  E_p[T(x, y)]                   −  log E_q[e^T]

where p is the joint distribution and q is the product-of-marginals.

Marginal construction
---------------------
DINE : permute x_ctx along dim=0 within the batch (per loader.py convention).
MINE : permute y along dim=0 within the batch.

Binary data
-----------
The loader returns LongTensor for binary sequences (intended for future
embedding layers).  Both estimators cast inputs to float32 before forwarding
through the network so that the MLP / LSTM backbones receive numeric tensors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from DINE.network import DirectedInformationNetwork, MutualInformationNetwork


# ---------------------------------------------------------------------------
# DV lower bound (shared by DINE and MINE)
# ---------------------------------------------------------------------------

def dv_lower_bound(t_joint: Tensor, t_marg: Tensor) -> Tensor:
    """Donsker–Varadhan lower bound:  E_p[T] − log E_q[exp(T)].

    Uses logsumexp for numerical stability:
        log E_q[exp(T)] = logsumexp(T) − log N

    Parameters
    ----------
    t_joint : (N,)  network scores for joint samples
    t_marg  : (M,)  network scores for marginal samples

    Returns
    -------
    Scalar tensor — the lower bound value (≥ 0 when X and Y are dependent).
    """
    log_mean_exp = t_marg.logsumexp(0) - math.log(t_marg.size(0))
    return t_joint.mean() - log_mean_exp


# ---------------------------------------------------------------------------
# Per-batch loss helpers
# ---------------------------------------------------------------------------

def dine_batch_loss(
    net: DirectedInformationNetwork,
    x_ctx: Tensor,
    y_ctx: Tensor,
    y_curr: Tensor,
) -> Tensor:
    """DINE loss for one batch (negated DV bound, for gradient descent).

    Marginal reference: x_ctx is permuted along the batch dimension.
    """
    x_ctx  = x_ctx.float()
    y_ctx  = y_ctx.float()
    y_curr = y_curr.float()
    perm = torch.randperm(x_ctx.size(0), device=x_ctx.device)
    t_joint = net(x_ctx,        y_ctx, y_curr).squeeze(-1)
    t_marg  = net(x_ctx[perm],  y_ctx, y_curr).squeeze(-1)
    return -dv_lower_bound(t_joint, t_marg)


def mine_batch_loss(
    net: MutualInformationNetwork,
    x: Tensor,
    y: Tensor,
) -> Tensor:
    """MINE loss for one batch (negated DV bound, for gradient descent).

    Marginal reference: y is permuted along the batch dimension.
    """
    x = x.float()
    y = y.float()
    perm = torch.randperm(y.size(0), device=y.device)
    t_joint = net(x, y).squeeze(-1)
    t_marg  = net(x, y[perm]).squeeze(-1)
    return -dv_lower_bound(t_joint, t_marg)


# ---------------------------------------------------------------------------
# Estimator config
# ---------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    lr: float = 1e-6
    n_epochs: int = 50
    clip_grad_norm: float | None = 5.0  # None to disable gradient clipping
    device: str = "cpu"
    log_every: int = 10  # print progress every N epochs; 0 to silence


# ---------------------------------------------------------------------------
# DINE estimator
# ---------------------------------------------------------------------------

class DINEEstimator:
    """Trains a DirectedInformationNetwork to estimate DI(X→Y).

    Parameters
    ----------
    network   : DirectedInformationNetwork
    optimizer : Optimizer  (caller constructs and owns)
    config    : EstimatorConfig
    """

    def __init__(
        self,
        network: DirectedInformationNetwork,
        optimizer: Optimizer,
        config: EstimatorConfig | None = None,
    ) -> None:
        self.net = network
        self.opt = optimizer
        self.cfg = config or EstimatorConfig()
        self.device = torch.device(self.cfg.device)
        self.net.to(self.device)
        self.train_losses: list[float] = []

    # ------------------------------------------------------------------

    def _to(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        return tuple(t.to(self.device) for t in tensors)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.net.train()
        total, n = 0.0, 0
        for batch in loader:
            x_ctx, y_ctx, y_curr = self._to(batch.x_ctx, batch.y_ctx, batch.y_curr)
            self.opt.zero_grad()
            loss = dine_batch_loss(self.net, x_ctx, y_ctx, y_curr)
            loss.backward()
            if self.cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_grad_norm)
            self.opt.step()
            total += loss.item()
            n += 1
        return total / n if n > 0 else float("nan")

    def fit(self, loader: DataLoader, n_epochs: int | None = None) -> list[float]:
        """Train for n_epochs (defaults to config.n_epochs).

        Returns
        -------
        Per-epoch mean training losses.
        """
        epochs = n_epochs if n_epochs is not None else self.cfg.n_epochs
        for epoch in range(1, epochs + 1):
            loss = self._train_epoch(loader)
            self.train_losses.append(loss)
            if self.cfg.log_every > 0 and epoch % self.cfg.log_every == 0:
                print(f"[DINE] epoch {epoch:>4d}/{epochs}  loss={loss:.4f}  DI≈{-loss:.4f}")
        return self.train_losses

    @torch.no_grad()
    def estimate(self, loader: DataLoader) -> float:
        """Evaluate DI(X→Y) over the full loader using the trained network.

        Returns
        -------
        DI estimate in nats (same units as network output).
        """
        self.net.eval()
        t_joints: list[Tensor] = []
        t_margs:  list[Tensor] = []
        for batch in loader:
            x_ctx, y_ctx, y_curr = self._to(batch.x_ctx, batch.y_ctx, batch.y_curr)
            x_ctx  = x_ctx.float()
            y_ctx  = y_ctx.float()
            y_curr = y_curr.float()
            perm = torch.randperm(x_ctx.size(0), device=self.device)
            t_joints.append(self.net(x_ctx,       y_ctx, y_curr).squeeze(-1))
            t_margs.append( self.net(x_ctx[perm], y_ctx, y_curr).squeeze(-1))
        return dv_lower_bound(torch.cat(t_joints), torch.cat(t_margs)).item()


# ---------------------------------------------------------------------------
# MINE estimator
# ---------------------------------------------------------------------------

class MINEEstimator:
    """Trains a MutualInformationNetwork to estimate MI(X;Y).

    The DataLoader may yield either:
    - ``(x, y)`` tensor tuples where x:(B, x_dim), y:(B, y_dim)
    - ``CausalSample`` batches, in which case x_ctx[:, -1] (current X) and
      y_curr are used as the i.i.d. pair fed to the network.

    Parameters
    ----------
    network   : MutualInformationNetwork
    optimizer : Optimizer
    config    : EstimatorConfig
    """

    def __init__(
        self,
        network: MutualInformationNetwork,
        optimizer: Optimizer,
        config: EstimatorConfig | None = None,
    ) -> None:
        self.net = network
        self.opt = optimizer
        self.cfg = config or EstimatorConfig()
        self.device = torch.device(self.cfg.device)
        self.net.to(self.device)
        self.train_losses: list[float] = []

    # ------------------------------------------------------------------

    def _unpack(self, batch) -> tuple[Tensor, Tensor]:
        """Return (x, y) float tensors from either (x, y) tuples or CausalSamples."""
        if hasattr(batch, "x_ctx"):
            x = batch.x_ctx[:, -1:].float().to(self.device)   # (B, 1)
            y = batch.y_curr.unsqueeze(-1).float().to(self.device)  # (B, 1)
        else:
            x, y = batch
            x = x.float().to(self.device)
            y = y.float().to(self.device)
        return x, y

    def _train_epoch(self, loader: DataLoader) -> float:
        self.net.train()
        total, n = 0.0, 0
        for batch in loader:
            x, y = self._unpack(batch)
            self.opt.zero_grad()
            loss = mine_batch_loss(self.net, x, y)
            loss.backward()
            if self.cfg.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.clip_grad_norm)
            self.opt.step()
            total += loss.item()
            n += 1
        return total / n if n > 0 else float("nan")

    def fit(self, loader: DataLoader, n_epochs: int | None = None) -> list[float]:
        """Train for n_epochs (defaults to config.n_epochs).

        Returns
        -------
        Per-epoch mean training losses.
        """
        epochs = n_epochs if n_epochs is not None else self.cfg.n_epochs
        for epoch in range(1, epochs + 1):
            loss = self._train_epoch(loader)
            self.train_losses.append(loss)
            if self.cfg.log_every > 0 and epoch % self.cfg.log_every == 0:
                print(f"[MINE] epoch {epoch:>4d}/{epochs}  loss={loss:.4f}  MI≈{-loss:.4f}")
        return self.train_losses

    @torch.no_grad()
    def estimate(self, loader: DataLoader) -> float:
        """Evaluate MI(X;Y) over the full loader using the trained network.

        Returns
        -------
        MI estimate in nats.
        """
        self.net.eval()
        t_joints: list[Tensor] = []
        t_margs:  list[Tensor] = []
        for batch in loader:
            x, y = self._unpack(batch)
            perm = torch.randperm(y.size(0), device=self.device)
            t_joints.append(self.net(x, y).squeeze(-1))
            t_margs.append( self.net(x, y[perm]).squeeze(-1))
        return dv_lower_bound(torch.cat(t_joints), torch.cat(t_margs)).item()
