"""`ukca_vapour` — H2SO4 weight percent and stratospheric solution density (task 38).

Two outputs, `wts` and `rhosol_strat`, from temperature, pressure and specific
humidity. Everything else the Fortran computes is dead; see below.

## What is actually live

`ukca_vapour.F90` declares `wts` and `rhosol_strat` `INTENT(OUT)` and nothing
else. After `wts_m40` is formed at `:190` the routine goes on to compute
`tmp4 = powr_v(wts_m40, 0.1)`, `muh2so4`, `tmp2 = 360/t`, `tmp2_out`,
`term1 = 2*surften*mmsul`, `kelvin` and `kelvin_out` — and **none of them reaches
an output**. `ph2so4` likewise. The chain is not ported, and
`tests/test_vapour.py` asserts the omission is safe by sweeping `rp`, the only
argument feeding it, and requiring both outputs to be byte-identical across the
sweep.

That matters for more than tidiness. The dead chain holds the only `EXP` and the
only fractional power in the routine, so dropping it takes the live path down to
`LOG` and `SQRT` — both bit-identical to gfortran on the capture platform. It is
why this task's acceptance is byte equality rather than the plan's
`RTOL_TRANSCENDENTAL`, which was four orders too loose for what remains.

## Faithful forms

* `b = ks3 + ks4*ust` is FMA-shaped, as are `a`, `c` and the `msb`/`ws`
  quotients. XLA contracts those under `jit` and gfortran does not, so this runs
  **eager** for byte equality (issue #23).
* `d = a*a - 4.0*b*c`, written in that order. `a*a`, not `a**2`.
* `1.0/t` for `ust`, matching `:141`, rather than folding it into the products
  that consume it.
* The `NINT` at `:226` goes through `numerics.vapour_round`, which rounds half
  **away from zero**. `jnp.round` rounds half to even and disagrees on seven of
  the ties in the swept range — and the result indexes a lookup table, so a tie
  going the wrong way selects a different density row rather than a slightly
  different number.

## The `wts` clamp is not [41, 99]

Only the `l_fix_neg_pvol_wat` arm has the 99 ceiling (`:184`). The default arm is
`MAX(41.0, ws*100)` with no ceiling (`:188`) and reaches 103.8 at
T = 303.65, bh2o = 2e-8. Both arms share the floor of 41.

The flag's numerical effect stops there. It does **not** reach `rhosol_strat`:
where the arms differ, `wts` is 99 in one and more in the other, and
`(NINT(wts/5))*5` sends both to 100 or above while `PERCENT` stops at 95 — so
neither matches and both fall through to the 1300.0 default. A both-settings
test on the density would be vacuous, which is why the one in `test_vapour.py`
is on `wts`.

`:182` reads `l_fix_neg_pvol_wat .OR. l_glomap_clim_radaer`, which has the shape
of a trap this repository has hit before — a second variable quietly making a
flag inert. It does not here: `l_glomap_clim_radaer` is a `PARAMETER = .FALSE.`
(`ukca_um_legacy_mod.F90:141`), so the flag is the sole control.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core import numerics
from ._vapour_literals import (
    BMAXATM,
    BMINATM,
    DATA253,
    K_DIFF,
    KS1,
    KS2,
    KS3,
    KS4,
    KS5,
    KS6,
    KS7,
    P0,
    PERCENT,
    XSB_EPS,
)

__all__ = ["vapour", "water_vapour_pressure", "weight_percent"]

_PERCENT = jnp.asarray(PERCENT, dtype=jnp.float64)
_DATA253 = jnp.asarray(DATA253, dtype=jnp.float64)
_K_DIFF = jnp.asarray(K_DIFF, dtype=jnp.float64)


def water_vapour_pressure(pmid: Array, s: Array) -> Array:
    """`bh2o`, in atmospheres, clamped to `[bminatm, bmaxatm]` (`:139-143`).

    The floor is what protects the `LOG` at `:149`: `s` can be zero or negative
    in a constructed fixture, and `1.609*s*patm` would then be non-positive.
    """
    # `:136` writes `patm = pmid(jl)/p0`, a division. numerics.true_divide,
    # because XLA turns `x / constant` into `x * (1/constant)` and 1/101325 is
    # inexact -- 28,534 of 200,000 sample values move. Inert today (`wts` comes
    # out bit-identical through the LOG and the quadratic) and one ulp from
    # live.
    patm = numerics.true_divide(pmid, P0)
    bh2o = 1.609 * s * patm
    bh2o = jnp.where(bh2o < BMINATM, BMINATM, bh2o)
    return jnp.where(bh2o > BMAXATM, BMAXATM, bh2o)


def weight_percent(t: Array, pmid: Array, s: Array, *, fix_neg_pvol_wat: bool) -> Array:
    """`wts`, the H2SO4 weight percent of the binary solution (`:150-190`).

    The Ayers fit, solved as a quadratic in the H2SO4 mole fraction. Three
    guards, each reproduced with its own comparison rather than folded together:
    the discriminant floor at `:170`, the mole-fraction floor at `:176`, and the
    weight-percent clamp at `:183-188`.
    """
    bh2o = water_vapour_pressure(pmid, s)
    ust = 1.0 / t
    tlog = jnp.log(t)
    pwlog = jnp.log(bh2o)

    a = KS1 + KS2 * ust
    b = KS3 + KS4 * ust
    c = KS5 + KS6 * ust + KS7 * tlog - pwlog

    d = a * a - 4.0 * b * c
    d = jnp.where(d < 0.0, 0.0, d)

    xsb = (-a - jnp.sqrt(d)) / (2.0 * b)
    xsb = jnp.where(xsb < XSB_EPS, XSB_EPS, xsb)

    msb = 55.51 * xsb / (1.0 - xsb)
    ws = msb * 0.098076 / (1.0 + msb * 0.098076)

    # numerics.fortran_max, not jnp.maximum: at the one temperature where the
    # Ayers denominator is exactly zero, `ws` is NaN and gfortran's MAX returns
    # 41.0 while jnp.maximum propagates. See the primitive's docstring.
    floored = numerics.fortran_max(41.0, ws * 100.0)
    return numerics.fortran_min(99.0, floored) if fix_neg_pvol_wat else floored


def solution_density(t: Array, wts: Array) -> Array:
    """`rhosol_strat` (`:220-237`).

    1300.0 unless `(NINT(wts/5))*5` hits one of the twelve tabulated weight
    percents, in which case the density at 253 K plus a linear temperature
    correction. The Fortran loops over the table and assigns on a match; since
    `PERCENT` has no duplicates, at most one row matches, so the loop is a
    `where` over a one-hot selection here.

    `t_diff = 253.0 - t` is written in that order, and the source comment notes
    that `t` is allowed to be negative.
    """
    rounded = numerics.vapour_round(wts)
    t_diff = 253.0 - t
    hit = rounded[..., None] == _PERCENT
    tabulated = _DATA253 + _K_DIFF * t_diff[..., None]
    return jnp.where(hit.any(axis=-1), jnp.sum(jnp.where(hit, tabulated, 0.0), axis=-1), 1300.0)


def vapour(t: Array, pmid: Array, s: Array, *, fix_neg_pvol_wat: bool) -> tuple[Array, Array]:
    """The routine's two outputs, in its argument order: `(wts, rhosol_strat)`."""
    wts = weight_percent(t, pmid, s, fix_neg_pvol_wat=fix_neg_pvol_wat)
    return wts, solution_density(t, wts)
