import importlib
import unittest


HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "jax is required for these tests")
class DineJaxTests(unittest.TestCase):
    def setUp(self):
        import jax.numpy as jnp

        self.jnp = jnp
        from dine_jax import (
            dine_objective,
            empirical_directed_information,
            estimate_di_from_scores,
            optimize_discrete_policy,
        )

        self.dine_objective = dine_objective
        self.empirical_directed_information = empirical_directed_information
        self.estimate_di_from_scores = estimate_di_from_scores
        self.optimize_discrete_policy = optimize_discrete_policy

    def test_estimate_di_from_scores_zero_when_scores_match(self):
        jnp = self.jnp
        t = jnp.array([0.1, -0.2, 0.3])
        di = self.estimate_di_from_scores(t, t, t, t)
        self.assertAlmostEqual(float(di), 0.0, places=6)

    def test_empirical_directed_information_independent_is_near_zero(self):
        jnp = self.jnp
        x = jnp.array([0, 0, 1, 1, 0, 1, 0, 1])
        y = jnp.array([0, 1, 0, 1, 1, 0, 0, 1])
        di = self.empirical_directed_information(x, y, alphabet_x=2, alphabet_y=2)
        self.assertLess(abs(float(di)), 0.25)

    def test_optimize_discrete_policy_returns_distribution(self):
        jnp = self.jnp
        channel = jnp.array(
            [
                [0.95, 0.05],
                [0.20, 0.80],
            ]
        )
        result = self.optimize_discrete_policy(channel, num_steps=200, learning_rate=0.2)
        policy = result["policy"]

        self.assertEqual(policy.shape, (2,))
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(policy.min()), 0.0)
        self.assertGreater(float(result["estimated_directed_information"]), 0.0)

    def test_dine_objective_shapes(self):
        jnp = self.jnp
        t = jnp.array([0.2, 0.4, 0.6])
        out = self.dine_objective(t, t, t, t, subtract=False)
        self.assertEqual(out.shape, (2,))


if __name__ == "__main__":
    unittest.main()
