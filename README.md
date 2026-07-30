# Discrete-Alphabet-DI-Optimization-and-Estimation

JAX reimplementation of core ideas from **"Data-Driven Optimization of Directed Information Over Discrete Alphabets"** and the reference implementation in `DorTsur/dine_ndt`.

## What's included

- `dine_jax.py`
  - Donsker-Varadhan (DV) lower bound
  - DINE-style directed information estimate from score samples
  - DINE objective (statistics-network mode and subtract/optimizer mode)
  - Empirical directed information estimator for memoryless discrete channels
  - JAX gradient-based optimization of a discrete input policy for a known channel matrix
- `tests/test_dine_jax.py`
  - Focused unit tests for the new implementation

## Run tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

> Note: JAX must be installed to execute the tests.
