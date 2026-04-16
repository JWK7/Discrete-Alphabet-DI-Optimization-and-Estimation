"""
Tests for data/generator.py and data/loader.py.

Run with:  pytest tests/test_data.py -v

Coverage
--------
Generator
  - Shape, dtype, and value constraints for all three channels
  - Empirical statistics match configured parameters (crossover rate, feedback
    policy weights, channel transition probabilities, AR autocorrelation)
  - Analytic DI formulas: boundary cases, monotonicity, and the key property
    DI > marginal I(X;Y) for the feedback channel

Loader
  - Dataset length formula: n_trajectories × (T − k)
  - Window alignment: x_ctx[-1] == X[t], y_ctx[-1] == Y[t-1], y_curr == Y[t]
  - Causal direction (no future Y leaks into y_ctx)
  - Tensor dtypes: LongTensor for binary, FloatTensor for Gaussian
  - Batch shapes from DataLoader
  - Multi-trajectory index mapping
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from data.generator import (
    BSCConfig,
    BinaryFeedbackConfig,
    GaussianAR1Config,
    _h2,
    binary_feedback_analytic_di,
    binary_feedback_analytic_marginal_mi,
    bsc_analytic_di,
    generate_binary_feedback,
    generate_bsc,
    generate_gaussian_ar1,
    gaussian_ar1_analytic_di,
)
from data.loader import (
    CausalSample,
    CausalWindowDataset,
    LoaderConfig,
    binary_feedback_loader,
    bsc_loader,
    gaussian_ar1_loader,
    make_dataloader,
)

# Shared RNG seed for reproducible statistical tests
_SEED = 42


# ==========================================================================
# Utility
# ==========================================================================

class TestH2:
    def test_zero(self):
        assert _h2(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_one(self):
        assert _h2(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_half(self):
        assert _h2(0.5) == pytest.approx(1.0, abs=1e-9)

    def test_tenth(self):
        expected = -0.1 * np.log2(0.1) - 0.9 * np.log2(0.9)
        assert _h2(0.1) == pytest.approx(expected, abs=1e-9)


# ==========================================================================
# 1. BSC generator
# ==========================================================================

class TestBSCGenerator:
    def _gen(self, p=0.1, T=1_000, B=1, seed=_SEED):
        cfg = BSCConfig(crossover_prob=p, seq_len=T, n_samples=B)
        return generate_bsc(cfg, np.random.default_rng(seed))

    # --- shape / dtype -------------------------------------------------------

    def test_shape(self):
        X, Y = self._gen(T=500, B=3)
        assert X.shape == (3, 500)
        assert Y.shape == (3, 500)

    def test_dtype_int8(self):
        X, Y = self._gen()
        assert X.dtype == np.int8
        assert Y.dtype == np.int8

    def test_values_binary(self):
        X, Y = self._gen()
        assert set(np.unique(X)).issubset({0, 1})
        assert set(np.unique(Y)).issubset({0, 1})

    # --- empirical statistics ------------------------------------------------

    def test_x_is_approximately_uniform(self):
        X, _ = self._gen(T=200_000)
        assert np.mean(X) == pytest.approx(0.5, abs=0.01)

    @pytest.mark.parametrize("p", [0.05, 0.1, 0.3, 0.5])
    def test_empirical_crossover_rate(self, p):
        X, Y = self._gen(p=p, T=200_000)
        empirical_p = np.mean(X != Y)
        assert empirical_p == pytest.approx(p, abs=0.01)

    def test_noise_independent_of_input(self):
        """X and the flip indicator N = X XOR Y should be independent."""
        X, Y = self._gen(p=0.2, T=100_000)
        N = X ^ Y                              # flip indicator
        # If independent, P(N=1 | X=0) ≈ P(N=1 | X=1) ≈ 0.2
        p_flip_given_x0 = np.mean(N[X == 0])
        p_flip_given_x1 = np.mean(N[X == 1])
        assert p_flip_given_x0 == pytest.approx(0.2, abs=0.02)
        assert p_flip_given_x1 == pytest.approx(0.2, abs=0.02)


class TestBSCAnalyticDI:
    def test_noiseless(self):
        assert bsc_analytic_di(0.0) == pytest.approx(1.0, abs=1e-6)

    def test_pure_noise(self):
        assert bsc_analytic_di(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        # DI(p) == DI(1-p)
        assert bsc_analytic_di(0.1) == pytest.approx(bsc_analytic_di(0.9), abs=1e-9)

    def test_monotone_in_noise(self):
        # More noise → less information
        assert bsc_analytic_di(0.05) > bsc_analytic_di(0.2) > bsc_analytic_di(0.4)

    def test_known_value(self):
        p = 0.1
        expected = 1.0 - _h2(p)
        assert bsc_analytic_di(p) == pytest.approx(expected, abs=1e-9)


# ==========================================================================
# 2. Binary feedback generator
# ==========================================================================

class TestBinaryFeedbackGenerator:
    def _gen(self, T=2_000, B=1, seed=_SEED, **kwargs):
        cfg = BinaryFeedbackConfig(seq_len=T, n_samples=B, **kwargs)
        return generate_binary_feedback(cfg, np.random.default_rng(seed)), cfg

    # --- shape / dtype -------------------------------------------------------

    def test_shape(self):
        (X, Y), _ = self._gen(T=300, B=4)
        assert X.shape == (4, 300)
        assert Y.shape == (4, 300)

    def test_dtype_int8(self):
        (X, Y), _ = self._gen()
        assert X.dtype == np.int8
        assert Y.dtype == np.int8

    def test_values_binary(self):
        (X, Y), _ = self._gen()
        assert set(np.unique(X)).issubset({0, 1})
        assert set(np.unique(Y)).issubset({0, 1})

    # --- empirical feedback policy -------------------------------------------

    def test_empirical_feedback_policy_alpha(self):
        """P(X_t = 1 | Y_{t-1} = 0) ≈ alpha."""
        (X, Y), cfg = self._gen(T=200_000)
        Y_prev = Y[0, :-1]
        X_curr = X[0, 1:]
        mask = Y_prev == 0
        empirical_alpha = np.mean(X_curr[mask])
        assert empirical_alpha == pytest.approx(cfg.alpha, abs=0.02)

    def test_empirical_feedback_policy_beta(self):
        """P(X_t = 1 | Y_{t-1} = 1) ≈ beta."""
        (X, Y), cfg = self._gen(T=200_000)
        Y_prev = Y[0, :-1]
        X_curr = X[0, 1:]
        mask = Y_prev == 1
        empirical_beta = np.mean(X_curr[mask])
        assert empirical_beta == pytest.approx(cfg.beta, abs=0.02)

    # --- empirical channel transitions ---------------------------------------

    def test_empirical_channel_p10(self):
        """P(Y_t = 1 | X_t = 1, Y_{t-1} = 0) ≈ chan_10."""
        (X, Y), cfg = self._gen(T=500_000)
        S = Y[0, :-1]           # Y_{t-1}
        Xt = X[0, 1:]
        Yt = Y[0, 1:]
        mask = (Xt == 1) & (S == 0)
        empirical = np.mean(Yt[mask])
        assert empirical == pytest.approx(cfg.chan_10, abs=0.02)

    def test_empirical_channel_p01(self):
        """P(Y_t = 1 | X_t = 0, Y_{t-1} = 1) ≈ chan_01."""
        (X, Y), cfg = self._gen(T=500_000)
        S = Y[0, :-1]
        Xt = X[0, 1:]
        Yt = Y[0, 1:]
        mask = (Xt == 0) & (S == 1)
        empirical = np.mean(Yt[mask])
        assert empirical == pytest.approx(cfg.chan_01, abs=0.02)


class TestBinaryFeedbackAnalyticDI:
    """Key property: causal DI strictly exceeds the marginal I(X_t ; Y_t)."""

    def test_di_positive(self):
        cfg = BinaryFeedbackConfig()
        assert binary_feedback_analytic_di(cfg) > 0.0

    def test_marginal_mi_positive(self):
        cfg = BinaryFeedbackConfig()
        assert binary_feedback_analytic_marginal_mi(cfg) > 0.0

    def test_di_greater_than_marginal_mi(self):
        """DI uses causal conditioning; marginal MI ignores the state."""
        cfg = BinaryFeedbackConfig()
        di  = binary_feedback_analytic_di(cfg)
        mi  = binary_feedback_analytic_marginal_mi(cfg)
        assert di > mi, (
            f"Expected DI ({di:.4f}) > marginal MI ({mi:.4f}) "
            "for the default asymmetric feedback policy."
        )

    def test_uniform_policy_reduces_to_channel_capacity(self):
        """With alpha=beta=0.5 the input ignores S; check DI is still valid."""
        cfg = BinaryFeedbackConfig(alpha=0.5, beta=0.5)
        di = binary_feedback_analytic_di(cfg)
        assert 0.0 < di <= 1.0

    def test_di_bounded(self):
        cfg = BinaryFeedbackConfig()
        di = binary_feedback_analytic_di(cfg)
        assert 0.0 < di <= 1.0


# ==========================================================================
# 3. Gaussian AR(1) generator
# ==========================================================================

class TestGaussianAR1Generator:
    def _gen(self, mode="channel_ar", T=5_000, B=1, seed=_SEED, **kwargs):
        cfg = GaussianAR1Config(mode=mode, seq_len=T, n_samples=B, **kwargs)
        return generate_gaussian_ar1(cfg, np.random.default_rng(seed)), cfg

    # --- shape / dtype -------------------------------------------------------

    def test_shape_channel_ar(self):
        (X, Y), _ = self._gen(T=400, B=2)
        assert X.shape == (2, 400)
        assert Y.shape == (2, 400)

    def test_dtype_float32(self):
        (X, Y), _ = self._gen()
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    # --- channel_ar: Y_t = phi*Y_{t-1} + X_t + noise ------------------------

    def test_channel_ar_x_is_iid(self):
        """X should be roughly uncorrelated across time."""
        (X, Y), _ = self._gen(mode="channel_ar", T=50_000)
        lag1_corr = np.corrcoef(X[0, :-1], X[0, 1:])[0, 1]
        assert abs(lag1_corr) < 0.05

    def test_channel_ar_innovations_are_white(self):
        """Residuals Y_t - phi*Y_{t-1} - X_t should be white noise."""
        phi = 0.8
        (X, Y), cfg = self._gen(mode="channel_ar", phi=phi, T=50_000)
        residuals = Y[0, 1:] - phi * Y[0, :-1] - X[0, 1:]
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        assert abs(lag1_corr) < 0.05

    def test_channel_ar_stationary_variance(self):
        """Var(Y) ≈ (sigma_x² + sigma_n²) / (1 - phi²) in stationarity."""
        phi, sx, sn = 0.6, 1.0, 0.5
        (X, Y), _ = self._gen(
            mode="channel_ar", phi=phi, sigma_x=sx, sigma_n=sn,
            T=200_000, B=1,
        )
        expected_var = (sx**2 + sn**2) / (1 - phi**2)
        # Use the tail to avoid the transient from Y_0 = 0
        empirical_var = np.var(Y[0, 1000:])
        assert empirical_var == pytest.approx(expected_var, rel=0.05)

    # --- source_ar: X_t = rho*X_{t-1} + ..., Y_t = X_t + noise -------------

    def test_source_ar_lag1_autocorrelation(self):
        """X lag-1 autocorrelation ≈ rho."""
        rho = 0.8
        (X, Y), _ = self._gen(mode="source_ar", rho=rho, T=100_000)
        lag1_corr = np.corrcoef(X[0, :-1], X[0, 1:])[0, 1]
        assert lag1_corr == pytest.approx(rho, abs=0.02)

    def test_source_ar_y_variance(self):
        """Var(Y) ≈ sigma_x² + sigma_n² (AWGN observation of AR source)."""
        sx, sn = 1.0, 0.5
        (X, Y), _ = self._gen(
            mode="source_ar", sigma_x=sx, sigma_n=sn, T=100_000,
        )
        expected_var = sx**2 + sn**2
        empirical_var = np.var(Y[0, 1000:])
        assert empirical_var == pytest.approx(expected_var, rel=0.05)

    def test_source_ar_observation_noise(self):
        """Y - X residuals should be white with variance sigma_n²."""
        sn = 0.5
        (X, Y), _ = self._gen(mode="source_ar", sigma_n=sn, T=100_000)
        noise = Y[0] - X[0]
        assert np.var(noise) == pytest.approx(sn**2, rel=0.05)
        lag1_corr = np.corrcoef(noise[:-1], noise[1:])[0, 1]
        assert abs(lag1_corr) < 0.05


class TestGaussianAR1AnalyticDI:
    def test_channel_ar_formula(self):
        cfg = GaussianAR1Config(mode="channel_ar", sigma_x=1.0, sigma_n=0.5)
        expected = 0.5 * np.log2(1.0 + 1.0 / 0.25)    # = 0.5 * log2(5) ≈ 1.161
        assert gaussian_ar1_analytic_di(cfg) == pytest.approx(expected, abs=1e-9)

    def test_channel_ar_phi_invariant(self):
        """DI for channel_ar does not depend on phi (AR memory cancels out)."""
        cfg_a = GaussianAR1Config(mode="channel_ar", phi=0.3)
        cfg_b = GaussianAR1Config(mode="channel_ar", phi=0.9)
        assert gaussian_ar1_analytic_di(cfg_a) == pytest.approx(
            gaussian_ar1_analytic_di(cfg_b), abs=1e-9
        )

    def test_source_ar_positive(self):
        cfg = GaussianAR1Config(mode="source_ar")
        assert gaussian_ar1_analytic_di(cfg) > 0.0

    def test_source_ar_iid_limit(self):
        """As rho → 0, source_ar DI → channel capacity (iid input)."""
        cfg_iid = GaussianAR1Config(mode="channel_ar", sigma_x=1.0, sigma_n=0.5)
        cfg_ar  = GaussianAR1Config(mode="source_ar",  rho=0.0, sigma_x=1.0, sigma_n=0.5)
        assert gaussian_ar1_analytic_di(cfg_ar) == pytest.approx(
            gaussian_ar1_analytic_di(cfg_iid), abs=1e-6
        )

    def test_source_ar_memory_reduces_di(self):
        """AR memory lets past Y predict X_t, reducing per-step DI."""
        cfg_iid = GaussianAR1Config(mode="source_ar", rho=0.0, sigma_x=1.0, sigma_n=0.5)
        cfg_ar  = GaussianAR1Config(mode="source_ar", rho=0.9, sigma_x=1.0, sigma_n=0.5)
        assert gaussian_ar1_analytic_di(cfg_ar) < gaussian_ar1_analytic_di(cfg_iid)

    def test_higher_snr_gives_higher_di(self):
        cfg_lo = GaussianAR1Config(mode="channel_ar", sigma_x=0.5, sigma_n=1.0)
        cfg_hi = GaussianAR1Config(mode="channel_ar", sigma_x=2.0, sigma_n=1.0)
        assert gaussian_ar1_analytic_di(cfg_hi) > gaussian_ar1_analytic_di(cfg_lo)

    def test_unknown_mode_raises(self):
        cfg = GaussianAR1Config(mode="bad_mode")
        with pytest.raises(ValueError, match="Unknown mode"):
            gaussian_ar1_analytic_di(cfg)


# ==========================================================================
# 4. CausalWindowDataset
# ==========================================================================

class TestCausalWindowDataset:
    """Use a deterministic float sequence so index math can be verified exactly."""

    def _make_ds(self, T=20, k=4, n_traj=1):
        # X[traj, t] = float(t),  Y[traj, t] = float(t + 100)
        X = np.tile(np.arange(T, dtype=np.float32), (n_traj, 1))
        Y = np.tile(np.arange(100, 100 + T, dtype=np.float32), (n_traj, 1))
        return CausalWindowDataset(X, Y, context_len=k, is_binary=False), X, Y

    # --- length --------------------------------------------------------------

    def test_length_single_traj(self):
        ds, _, _ = self._make_ds(T=20, k=4)
        assert len(ds) == 20 - 4   # 16 windows

    def test_length_multi_traj(self):
        ds, _, _ = self._make_ds(T=20, k=4, n_traj=3)
        assert len(ds) == 3 * (20 - 4)

    # --- window alignment ----------------------------------------------------

    def test_first_window(self):
        k = 4
        ds, X, Y = self._make_ds(T=20, k=k)
        # idx=0 → traj=0, t=4
        s = ds[0]
        t = k                          # first valid t
        assert s.x_ctx.tolist()  == X[0, t - k + 1 : t + 1].tolist()   # [1,2,3,4]
        assert s.y_ctx.tolist()  == Y[0, t - k     : t    ].tolist()   # [100,101,102,103]
        assert s.y_curr.item()   == Y[0, t]                            # 104.0

    def test_last_window(self):
        k, T = 4, 20
        ds, X, Y = self._make_ds(T=T, k=k)
        # idx = T-k-1 = 15 → traj=0, t=T-1=19
        s = ds[len(ds) - 1]
        t = T - 1
        assert s.x_ctx.tolist()  == X[0, t - k + 1 : t + 1].tolist()   # [16,17,18,19]
        assert s.y_ctx.tolist()  == Y[0, t - k     : t    ].tolist()   # [115,116,117,118]
        assert s.y_curr.item()   == Y[0, t]                            # 119.0

    def test_x_ctx_last_element_is_x_t(self):
        """x_ctx[-1] must be X[t] (current input, not future)."""
        k = 3
        ds, X, Y = self._make_ds(T=15, k=k)
        for idx in range(len(ds)):
            t = idx % (15 - k) + k
            assert ds[idx].x_ctx[-1].item() == X[0, t]

    def test_y_ctx_last_element_is_y_t_minus_1(self):
        """y_ctx[-1] must be Y[t-1], not Y[t] (causal — no future leak)."""
        k = 3
        ds, X, Y = self._make_ds(T=15, k=k)
        for idx in range(len(ds)):
            t = idx % (15 - k) + k
            assert ds[idx].y_ctx[-1].item()  == Y[0, t - 1]
            assert ds[idx].y_curr.item()     == Y[0, t]
            assert ds[idx].y_ctx[-1].item()  != ds[idx].y_curr.item()   # Y[t-1] ≠ Y[t] here

    def test_y_curr_not_in_y_ctx(self):
        """y_curr = Y[t] must differ from every element of y_ctx = Y[t-k..t-1]."""
        k = 4
        ds, X, Y = self._make_ds(T=20, k=k)
        for idx in range(len(ds)):
            s = ds[idx]
            assert s.y_curr.item() not in s.y_ctx.tolist()

    # --- multi-trajectory index mapping -------------------------------------

    def test_multi_traj_mapping(self):
        k, T = 3, 10
        n_traj = 2
        ds, X, Y = self._make_ds(T=T, k=k, n_traj=n_traj)
        per_traj = T - k    # 7

        # First index of the second trajectory
        idx = per_traj           # → traj=1, local_t=0, t=k=3
        s = ds[idx]
        assert s.x_ctx[-1].item() == X[1, k]
        assert s.y_curr.item()    == Y[1, k]

    # --- dtypes --------------------------------------------------------------

    def test_float_dtypes(self):
        ds, _, _ = self._make_ds()
        s = ds[0]
        assert s.x_ctx.dtype  == torch.float32
        assert s.y_ctx.dtype  == torch.float32
        assert s.y_curr.dtype == torch.float32

    def test_binary_dtypes(self):
        X = np.zeros((1, 20), dtype=np.int8)
        Y = np.zeros((1, 20), dtype=np.int8)
        ds = CausalWindowDataset(X, Y, context_len=4, is_binary=True)
        s = ds[0]
        assert s.x_ctx.dtype  == torch.long
        assert s.y_ctx.dtype  == torch.long
        assert s.y_curr.dtype == torch.long

    # --- validation ----------------------------------------------------------

    def test_seq_len_too_short_raises(self):
        X = np.zeros((1, 5), dtype=np.float32)
        Y = np.zeros((1, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="seq_len"):
            CausalWindowDataset(X, Y, context_len=5, is_binary=False)

    def test_shape_mismatch_raises(self):
        X = np.zeros((1, 20), dtype=np.float32)
        Y = np.zeros((1, 15), dtype=np.float32)
        with pytest.raises(ValueError, match="shapes must match"):
            CausalWindowDataset(X, Y, context_len=4, is_binary=False)


# ==========================================================================
# 5. DataLoader (batching)
# ==========================================================================

class TestDataLoader:
    def test_bsc_batch_shapes(self):
        cfg = BSCConfig(seq_len=500, n_samples=2)
        ldr_cfg = LoaderConfig(context_len=8, batch_size=32, shuffle=False)
        loader = bsc_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))
        batch = next(iter(loader))
        assert isinstance(batch, CausalSample)
        assert batch.x_ctx.shape  == (32, 8)
        assert batch.y_ctx.shape  == (32, 8)
        assert batch.y_curr.shape == (32,)

    def test_bsc_batch_dtypes(self):
        cfg = BSCConfig(seq_len=500, n_samples=1)
        ldr_cfg = LoaderConfig(context_len=5, batch_size=16, shuffle=False)
        batch = next(iter(bsc_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))))
        assert batch.x_ctx.dtype  == torch.long
        assert batch.y_ctx.dtype  == torch.long
        assert batch.y_curr.dtype == torch.long

    def test_gaussian_batch_shapes(self):
        cfg = GaussianAR1Config(mode="source_ar", seq_len=500, n_samples=2)
        ldr_cfg = LoaderConfig(context_len=10, batch_size=32, shuffle=False)
        loader = gaussian_ar1_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))
        batch = next(iter(loader))
        assert batch.x_ctx.shape  == (32, 10)
        assert batch.y_ctx.shape  == (32, 10)
        assert batch.y_curr.shape == (32,)

    def test_gaussian_batch_dtypes(self):
        cfg = GaussianAR1Config(mode="channel_ar", seq_len=500, n_samples=1)
        ldr_cfg = LoaderConfig(context_len=5, batch_size=16, shuffle=False)
        batch = next(iter(gaussian_ar1_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))))
        assert batch.x_ctx.dtype  == torch.float32
        assert batch.y_ctx.dtype  == torch.float32
        assert batch.y_curr.dtype == torch.float32

    def test_binary_feedback_batch_shapes(self):
        cfg = BinaryFeedbackConfig(seq_len=500, n_samples=1)
        ldr_cfg = LoaderConfig(context_len=6, batch_size=32, shuffle=False)
        batch = next(iter(binary_feedback_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))))
        assert batch.x_ctx.shape  == (32, 6)
        assert batch.y_ctx.shape  == (32, 6)
        assert batch.y_curr.shape == (32,)

    def test_total_windows_equals_dataset_len(self):
        """Iterating the full loader should yield exactly len(dataset) samples."""
        cfg = BSCConfig(seq_len=300, n_samples=3)
        ldr_cfg = LoaderConfig(context_len=5, batch_size=1, shuffle=False, drop_last=False)
        loader = bsc_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))
        n_windows = sum(1 for _ in loader)
        expected = 3 * (300 - 5)
        assert n_windows == expected

    def test_batch_values_are_binary_for_bsc(self):
        cfg = BSCConfig(seq_len=500, n_samples=1)
        ldr_cfg = LoaderConfig(context_len=5, batch_size=64, shuffle=False)
        batch = next(iter(bsc_loader(cfg, ldr_cfg, rng=np.random.default_rng(_SEED))))
        assert batch.x_ctx.max().item()  <= 1
        assert batch.x_ctx.min().item()  >= 0
        assert batch.y_curr.max().item() <= 1
        assert batch.y_curr.min().item() >= 0
