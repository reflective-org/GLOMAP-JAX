"""glomap_jax: GLOMAP-mode aerosol microphysics in Python/JAX.

A faithful port of the UKCA GLOMAP-mode box model — nucleation, condensation,
coagulation, ageing, mode merging and water uptake.

Importing this package enables 64-bit floating point in JAX, and that is not
optional. The reference Fortran is validated in double precision, and several
GLOMAP thresholds sit below float32's range entirely: ``ukca_solvecoagnucl_v``
branches on a discriminant against ``eps_d = 1e-40`` and mode ``num_eps`` values
reach ``1e-20``. In float32 those underflow to zero, which does not merely lose
precision — it changes *which closed-form solution branch is selected*. Import
``glomap_jax`` before creating any JAX arrays, or set ``JAX_ENABLE_X64=1``.
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
            "GLOMAP branches on thresholds down to 1e-40, which float32 cannot "
            "represent."
        )
