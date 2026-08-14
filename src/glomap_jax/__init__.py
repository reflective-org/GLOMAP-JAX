"""glomap_jax: GLOMAP-mode aerosol microphysics in Python/JAX.

A faithful port of the UKCA GLOMAP-mode box model — nucleation, condensation,
coagulation, ageing, mode merging and water uptake.

Importing this package enables 64-bit floating point in JAX.

The honest reason, stated carefully because an earlier version of this docstring
got it wrong. It is NOT that GLOMAP's thresholds underflow in float32: ``1e-20``
is a normal float32 number and ``eps_d = 1e-40`` is subnormal but still greater
than zero, and the reference Fortran runs in single precision and selects its
solver branches correctly. The reasons float64 is required are:

* **The reference is double precision.** Validation is against a
  ``-fdefault-real-8`` build; the measured single-vs-double spread on a 48-step
  run is 3.7e-4, four orders of magnitude above any tolerance worth gating on.
* **The tolerances are unreachable in float32.** ``RTOL_STEP = 1e-11`` and
  ``RTOL_ALGEBRAIC = 1e-13`` need more than float32's ~7 significant digits.
* **Cancellation.** ``ukca_binapara`` evaluates ``jveh = EXP(P)`` where ``P`` is
  a ~30-term polynomial whose O(1e3) terms cancel to O(10); float32 would lose
  most of the result to that cancellation.

Import ``glomap_jax`` before creating any JAX arrays, or set
``JAX_ENABLE_X64=1``.
"""

import jax

jax.config.update("jax_enable_x64", True)

__all__ = ["__version__"]

__version__ = "0.1.0"


def _assert_x64() -> None:
    """Fail fast if x64 got disabled after import (e.g. ``JAX_ENABLE_X64=0``).

    Called by the drivers before any numerics. Failing here with a clear message
    is much cheaper than discovering it as a mysterious golden-test mismatch, or
    worse, as a silently different solver branch.
    """
    import jax.numpy as jnp

    dtype = jnp.zeros(1).dtype
    if dtype != jnp.float64:
        raise RuntimeError(
            f"glomap_jax requires float64 but the JAX default dtype is {dtype}; "
            "do not disable jax_enable_x64 (check the JAX_ENABLE_X64 env var). "
            "The reference is a double-precision Fortran build and the tolerance "
            "policy gates at 1e-11 to 1e-13, which float32 cannot deliver."
        )
