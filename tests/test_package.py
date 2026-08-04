"""Task 1 acceptance: the package imports, and float64 is actually on.

This is deliberately the first test in the repo. Every later golden depends on
float64 being enabled by import alone, so if this breaks, nothing downstream
means anything.
"""

import jax.numpy as jnp
import pytest

import glomap_jax


def test_default_dtype_is_float64():
    assert jnp.zeros(1).dtype == jnp.float64


def test_explicit_float64_is_not_truncated():
    # Without jax_enable_x64 this silently downcasts to float32 with a warning.
    assert jnp.zeros(1, dtype=jnp.float64).dtype == jnp.float64


def test_thresholds_glomap_branches_on_are_representable():
    # ukca_solvecoagnucl_v branches on the discriminant against eps_d = 1e-40,
    # and mode num_eps values reach 1e-20. In float32 both underflow to zero,
    # which changes which closed-form branch is selected — not just precision.
    for threshold in (1.0e-20, 1.0e-40):
        assert jnp.asarray(threshold) > 0.0


def test_assert_x64_passes_when_enabled():
    glomap_jax._assert_x64()


def test_assert_x64_raises_when_disabled(monkeypatch):
    monkeypatch.setattr(jnp, "zeros", lambda *a, **k: jnp.array([0.0], dtype=jnp.float32))
    with pytest.raises(RuntimeError, match="requires float64"):
        glomap_jax._assert_x64()
