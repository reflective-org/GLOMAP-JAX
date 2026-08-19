"""Transcendental and rounding primitives, written to match gfortran.

Not a convenience layer. Three of these have to be written a specific way, and
the obvious way is silently wrong — wrong in the sense of flipping a branch and
moving a trajectory by O(1), not of losing a digit.

Every claim below is measured, not assumed: ``validation/capture_leaf.py``
sweeps each primitive through the vendored Fortran over grids that land *on*
each hazard, and ``tests/test_numerics_reference.py`` asserts the results
against the committed golden. 15,382 points.

===============  ===================================  =========================
primitive        JAX vs gfortran                      verdict
===============  ===================================  =========================
``erf``          bit-identical, 4330/4330             use ``jax.scipy`` directly
``log``, ``1/x`` bit-identical                        use them directly
``x**(1/3)``     bit-identical, 1865/1865             **this** is ``cubrt_v``
``exp``          456/3199 differ, max 2.1e-16         1 ulp; inside tolerance
``jnp.cbrt``     1756/1865 differ, max 1.3e-14        **do not use** by default
``jnp.round``    64 of 129 ties differ                **do not use**
===============  ===================================  =========================

Two of those need saying out loud, because both look like improvements.

``jnp.cbrt`` is a better cube root than ``x ** (1.0/3.0)``. It is also not what
``ukca_um_legacy_mod``'s ``cubrt_v`` computes, and the difference reaches
1.3e-14 — a hundred times ``RTOL_ALGEBRAIC``. Its output is ``drydp``, which is
compared against ``dp_thresh1`` (merge or not) and ``ddplim0*0.1`` (rewrite
``md``/``mdt`` or not). Both are step changes, so a parcel within 1.3e-14 of
either threshold goes one way in the reference and the other in the port. They
also disagree about negatives: the power form is NaN, ``cbrt`` returns the real
root — which is *more* correct and is exactly why it cannot be the faithful
path. Available as ``FidelityConfig.cbrt_exact``.

``jnp.round`` rounds half to even; Fortran ``NINT`` rounds half away from zero.
Away from ties they agree exactly, so the shim below is narrow. It matters
because ``ukca_vapour.F90:226`` computes ``(NINT(wts/5))*5`` and uses the result
to **index a lookup table** — at ``wts`` in {42.5, 52.5, 62.5, 72.5, 82.5,
92.5} the naive version selects a different table entry, not a slightly
different number.

One hazard with no shim, because none is possible: XLA flushes the *result of
any arithmetic operation* to zero when it would be subnormal, and gfortran does
not. A subnormal constant survives conversion, so the value is representable
and just cannot be produced. Latent — ``num_eps`` bottoms out at 1e-20 and
``eps_d`` at 1e-40, both normal in float64 — but it would present as a gate-0
disagreement with no arithmetic explanation. Issue #15.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

# ukca_um_legacy_mod.F90:450 -- `y(i) = x(i) ** (1.0 / 3.0)`. Bound once so the
# literal is written down in exactly one place; under -fdefault-real-8 the
# Fortran constant is the float64 nearest a third, which is what Python gives.
_ONE_THIRD = 1.0 / 3.0

erf = jax.scipy.special.erf
"""Bit-identical to gfortran's ``ERF`` over the swept grid. No shim needed.

Reached through ``umErf`` in the Fortran (``ukca_remode.F90:245``), which is an
``ELEMENTAL`` wrapper around the intrinsic and is transparent.
"""


def cbrt(x: Array, *, exact: bool = False) -> Array:
    """Cube root, as ``cubrt_v`` computes it.

    ``exact=False`` (the default, and what ``FidelityConfig.cbrt_exact=False``
    selects) evaluates ``x ** (1.0/3.0)``, reproducing the Fortran bit for bit
    including ``NaN`` for negative ``x``.

    ``exact=True`` uses ``jnp.cbrt``: a genuinely better cube root, a real root
    for negatives, and a different answer on 94% of the swept grid. It is an
    order-2 option, not a fidelity setting — every order-1 golden is keyed to
    the power form.
    """
    if exact:
        return jnp.cbrt(x)
    return x**_ONE_THIRD


def nint(x: Array) -> Array:
    """Fortran ``NINT``: round half **away from zero**.

    ``jnp.round`` rounds half to even and disagrees on 64 of the 129 ties in the
    swept grid. The two coincide whenever rounding away from zero already lands
    on an even number, which is why only half the ties differ — and why reading
    the mismatch count as the tie count is an easy mistake to make.

    Note the formulation. The obvious ``sign(x) * floor(|x| + 0.5)`` is wrong at
    exactly two points in the swept grid: for ``x = ±0.49999999999999994``, the
    double immediately below a half, ``|x| + 0.5`` rounds *up* to exactly 1.0
    and the result is ±1 where Fortran gives 0. Comparing the fractional part
    against 0.5 avoids the intermediate rounding entirely — ``|x| - floor(|x|)``
    is exact.
    """
    magnitude = jnp.abs(x)
    whole = jnp.floor(magnitude)
    rounded = whole + jnp.where(magnitude - whole >= 0.5, 1.0, 0.0)
    return jnp.sign(x) * rounded


def vapour_round(x: Array) -> Array:
    """``ukca_vapour.F90:226`` exactly: ``(NINT(wts/5))*5``.

    Exposed as its own function rather than left to callers to compose, because
    the composition is what has to be right: the result indexes a lookup table,
    so a tie that rounds the other way selects a different entry.
    """
    return nint(x / 5.0) * 5.0


def safe_divide(numerator: Array, denominator: Array, where: Array) -> Array:
    """Divide under a mask without poisoning the gradient.

    The double-``where`` idiom. A single ``jnp.where(cond, a / b, 0.0)`` still
    evaluates ``a / b`` on the masked-out elements, and although the forward
    value is discarded the *cotangent* is not: reverse-mode differentiation
    propagates a ``NaN`` back through the branch that was not taken.

    Substituting a safe denominator first is the only formulation that gives
    both a correct value and a finite gradient, which order 2 needs.
    """
    safe = jnp.where(where, denominator, 1.0)
    return jnp.where(where, numerator / safe, 0.0)


def masked_sum(term: Array, where: Array, axis: int | None = None) -> Array:
    """Sum ``term`` over ``axis``, counting only ``where``.

    Written as ``jnp.where(mask, term, 0.0)`` before the reduction, never as
    ``mask * term``. Padding entries of the component axis are computed from
    ``ratio1 = mm / (avogadro * rhocomp)``, which is evaluated over the full
    extent, and ``0.0 * inf`` is ``NaN`` — so the multiplicative form
    contaminates the whole reduction.

    Matters at ``ukca_ageing.F90:308``, ``IF (SUM(totage) > 0.0)``: a whole-array
    reduction with no component mask, gating an entire transfer block.
    """
    return jnp.sum(jnp.where(where, term, 0.0), axis=axis)


__all__ = ["cbrt", "erf", "masked_sum", "nint", "safe_divide", "vapour_round"]
