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


def test_float64_delivers_the_precision_the_tolerance_policy_gates_at():
    """The real reason for float64, encoded so it cannot rot.

    An earlier version of this test asserted that GLOMAP's 1e-20 / 1e-40
    thresholds "underflow in float32". That is false -- 1e-20 is a normal
    float32 number and 1e-40 is subnormal but non-zero -- and the assertion
    passed in float32, so it tested nothing.

    What is actually true: RTOL_STEP is 1e-11 and RTOL_ALGEBRAIC is 1e-13, and
    float32 carries ~7 significant digits, so a float32 run cannot even
    represent agreement at those tolerances.
    """
    one = jnp.asarray(1.0)
    # Smallest perturbation float64 can resolve on top of 1.0, vs float32's.
    assert float(one + 1e-13) != 1.0, "float64 must resolve RTOL_ALGEBRAIC"
    assert float(jnp.asarray(1.0, dtype=jnp.float32) + jnp.asarray(1e-13, dtype=jnp.float32)) == 1.0


def test_assert_x64_passes_when_enabled():
    glomap_jax._assert_x64()


def test_assert_x64_raises_when_disabled(monkeypatch):
    monkeypatch.setattr(jnp, "zeros", lambda *a, **k: jnp.array([0.0], dtype=jnp.float32))
    with pytest.raises(RuntimeError, match="requires float64"):
        glomap_jax._assert_x64()
