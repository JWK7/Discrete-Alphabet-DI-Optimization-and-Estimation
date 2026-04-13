"""JAX reimplementation of core DINE objectives for discrete alphabets."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def dv_lower_bound(t_joint: jnp.ndarray, t_reference: jnp.ndarray) -> jnp.ndarray:
    """Donsker-Varadhan lower bound used by DINE/MINE.

    Args:
        t_joint: Score samples from the joint distribution.
        t_reference: Score samples from the product/reference distribution.

    Returns:
        Scalar DV lower bound estimate.
    """
    return jnp.mean(t_joint) - jnp.log(jnp.mean(jnp.exp(t_reference)))


def estimate_di_from_scores(
    t_y: jnp.ndarray,
    t_y_ref: jnp.ndarray,
    t_xy: jnp.ndarray,
    t_xy_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Directed information estimate from two DV terms.

    Args:
        t_y: Score samples for Y history/current term.
        t_y_ref: Reference (contrastive) score samples for Y term.
        t_xy: Score samples for (X,Y) term.
        t_xy_ref: Reference (contrastive) score samples for (X,Y) term.

    Returns:
        Estimated directed information in nats.
    """
    return dv_lower_bound(t_xy, t_xy_ref) - dv_lower_bound(t_y, t_y_ref)


def dine_objective(
    t_y: jnp.ndarray,
    t_y_ref: jnp.ndarray,
    t_xy: jnp.ndarray,
    t_xy_ref: jnp.ndarray,
    subtract: bool = False,
) -> jnp.ndarray:
    """TensorFlow DINELoss equivalent in JAX.

    If subtract=False, returns stacked losses for each statistics network.
    If subtract=True, returns the optimization objective for the encoder/policy.
    """
    loss_y = -dv_lower_bound(t_y, t_y_ref)
    loss_xy = -dv_lower_bound(t_xy, t_xy_ref)
    if subtract:
        return loss_xy - loss_y
    return jnp.stack([loss_y, loss_xy])


def empirical_directed_information(
    x: jnp.ndarray,
    y: jnp.ndarray,
    alphabet_x: int,
    alphabet_y: int,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Empirical DI for one-step memoryless channels (equals empirical I(X;Y))."""
    x_oh = jax.nn.one_hot(x, alphabet_x)
    y_oh = jax.nn.one_hot(y, alphabet_y)

    joint = (x_oh.T @ y_oh) / x.shape[0]
    px = jnp.sum(joint, axis=1, keepdims=True)
    py = jnp.sum(joint, axis=0, keepdims=True)

    ratio = (joint + eps) / (px @ py + eps)
    return jnp.sum(joint * jnp.log(ratio))


def optimize_discrete_policy(
    channel: jnp.ndarray,
    num_steps: int = 300,
    learning_rate: float = 0.1,
) -> Dict[str, jnp.ndarray]:
    """Optimize input distribution for a discrete memoryless channel in JAX.

    Args:
        channel: Transition matrix P(y|x), shape [|X|, |Y|]. Rows sum to 1.
    """

    def mutual_information_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
        px = jax.nn.softmax(logits)
        py = px @ channel
        log_term = jnp.log(channel + 1e-12) - jnp.log(py + 1e-12)
        return jnp.sum(px[:, None] * channel * log_term)

    grad_fn = jax.grad(mutual_information_from_logits)
    logits = jnp.zeros(channel.shape[0])

    for _ in range(num_steps):
        logits = logits + learning_rate * grad_fn(logits)

    optimal_px = jax.nn.softmax(logits)
    estimated_di = mutual_information_from_logits(logits)

    return {
        "policy": optimal_px,
        "estimated_directed_information": estimated_di,
    }
