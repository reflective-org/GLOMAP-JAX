"""`ukca_cond_coff_v`: the condensation coefficient of one vapour on one mode.

Ported from `fortran/src/ukca/ukca_cond_coff_v.F90`, byte-equal to the compiled
routine on the capture platform over `tests/goldens/cond_coff.f64.leaf.npz`.

It returns two things per box: `cc`, the condensation coefficient in m^3 s-1,
which `ukca_conden.F90:336-338` multiplies by `nd` to get the loss rate of the
vapour; and `sinkarr`, which reaches `s_cond_s` and from there the boundary-layer
nucleation scheme. Both are zero wherever the mask is false, and the routine
sets them to zero before it does anything else (`:167-168`) -- so a masked box
is not merely unwritten, it is written zero, and a port that leaves the
unmasked value in place disagrees on every inactive mode.

Two switches, four closed forms, one of them ever run
----------------------------------------------------

`ifuchs` chooses Fuchs (1964) or Fuchs-Sutugin (1971); `idcmfp` chooses how the
diffusion coefficient and the vapour mean free path are formed. Every shipped
namelist runs `ifuchs = 1, idcmfp = 1`, so three of the four have no trajectory
reference at all and are validated only against the leaf sweep. `ModelConfig`
rejects anything outside `{1, 2}` for either, which is stricter than the
Fortran: nothing upstream validates them, and `idcmfp` out of range leaves
`dcoff_cp` never assigned while both Fuchs branches read it.

Because the switches are `ModelConfig` fields and not traced values, the
branches here are ordinary Python `if`s. That is not a violation of the no-
branching-on-traced-values rule -- these are static, and `jnp.where` over two
whole formulations would evaluate both.

`se` is inert at the value the model runs
-----------------------------------------

`ukca_conden.F90:235-237` sets `se_sol` and `se_ins` both to `1.0`, so
`1.0/se - 1.0` at `:210` is exactly `0.0`, `akn` is exactly `1.0`, and the
interfacial-transport correction the Fuchs-Sutugin branch exists to apply is
switched off by its own coefficient. The routine's own header (`:60`) documents
`se = 0.3`, which is not inert. The code is the specification; the sweep covers
both and `tests/test_cond_coff.py` pins the degeneracy.

Where the arithmetic had to be written a particular way
-------------------------------------------------------

* **The scalar terms are Python floats, not arrays.** `term1`-`term8` are
  scalars in the Fortran too, computed once per call from `mmcg`, `dmol` and
  `difvol`. Evaluating them in float64 Python performs the same IEEE
  operations, and sends `**` to the same libm gfortran uses; routing them
  through `jnp` instead would expose them to XLA's own lowering of `pow` for no
  gain.

* **`term8` squares by multiplication.** `(...)**2` in the Fortran is an
  integer literal exponent, which gfortran expands to a multiplication; `pow`
  and `x*x` are not required to agree. Phase D lost byte equality at exactly
  this shape, in the ZSR polynomial, on 35% and 55% of the live range.

* **`pmid/101325.0` goes through `numerics.true_divide`.** Dividing an array by
  a scalar constant is the site XLA rewrites into a multiply by the reciprocal.
  Seven of the sweep's eleven pressures see the difference.

* **Every division is `numerics.safe_divide` under the mask.** The masked rows
  of the fixture carry `inf`, `NaN` and zero on purpose, and a single `where`
  around a division returns the right value with a `NaN` cotangent.

* **`t ** 1.75` is left as a power.** It is the only non-integer exponent in
  the routine and `cubrt`'s rule does not apply: there is no alternative
  spelling that gfortran might have used.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from glomap_jax.core import numerics
from glomap_jax.core.constants import AVOGADRO, BOLTZMANN, PI, RGAS, RMOL

#: Diffusion volume of an air molecule (Fuller et al., Reid et al.), a local
#: assignment at `ukca_cond_coff_v.F90:166` rather than a module constant.
DAIR = 19.7

#: The standard atmosphere, inline at `:179` as the pressure the Fuller
#: diffusion coefficient is referenced to. Not `pref`, which
#: `ukca_config_constants_mod.F90:115` sets to 100000.0 for a different purpose.
P_STANDARD = 101325.0


class ScalarTerms(NamedTuple):
    """`term1`-`term8` and `mm_da`, in the source's own names.

    Exposed rather than inlined so a test can assert the degeneracy of `akn` at
    `se = 1.0`, and so `mm_da` can be checked against the derived value CLAUDE.md
    forbids caching in `core/constants`.
    """

    mm_da: float
    term1: float
    term2: float
    term5: float
    term6: float
    term7: float
    term8: float


def scalar_terms(mmcg: float, dmol: float, difvol: float) -> ScalarTerms:
    """`:149-167`, in order and with the Fortran's own association.

    `mm_da = avogadro*boltzmann/rgas` is computed here, at its point of use,
    and not stored in `core/constants`: a derived quantity in a constants table
    is a second source of truth. This is the site the constants module's
    docstring names.
    """
    term1 = math.sqrt(8.0 * RMOL / (PI * mmcg))

    mm_da = AVOGADRO * BOLTZMANN / RGAS
    zz = mmcg / mm_da
    term2 = 1.0 / (PI * math.sqrt(1.0 + zz) * dmol * dmol)

    term3 = 3.0 / (8.0 * AVOGADRO * dmol * dmol)
    term4 = math.sqrt((RGAS * mm_da * mm_da / (2.0 * PI)) * ((mmcg + mm_da) / mmcg))
    term5 = term3 * term4

    term6 = 4.0e6 * PI

    term7 = math.sqrt((1.0 / (mm_da * 1000.0)) + (1.0 / (mmcg * 1000.0)))
    # `(...)**2` with an integer literal exponent: gfortran multiplies, so so
    # does this. `base ** 2` would call `pow` and need not give the same double.
    base = DAIR ** (1.0 / 3.0) + difvol ** (1.0 / 3.0)
    term8 = base * base

    return ScalarTerms(
        mm_da=mm_da, term1=term1, term2=term2, term5=term5, term6=term6, term7=term7, term8=term8
    )


def cond_coff(
    mask: Array,
    rp: Array,
    tsqrt: Array,
    airdm3: Array,
    rhoa: Array,
    pmid: Array,
    t: Array,
    *,
    mmcg: float,
    se: float,
    dmol: float,
    difvol: float,
    ifuchs: int,
    idcmfp: int,
) -> tuple[Array, Array]:
    """Condensation coefficient and condensation sink, `(cc, sinkarr)`.

    `tsqrt` and `t` are separate arguments in the Fortran and are not required
    to agree; the caller passes `SQRT(t)` (`ukca_conden.F90:281`) but the callee
    never checks. `idcmfp = 1` reads only `tsqrt`, `idcmfp = 2` reads both, and
    the leaf fixture pins that by feeding them decoupled.
    """
    if ifuchs not in (1, 2):
        raise ValueError(f"ifuchs={ifuchs} must be 1 or 2")
    if idcmfp not in (1, 2):
        raise ValueError(f"idcmfp={idcmfp} must be 1 or 2")

    # Converted up front, not left to the first `jnp` call. With numpy inputs
    # the leading products would run in numpy and only the divisions in XLA,
    # so `t ** 1.75` -- the one non-integer power here -- would come from
    # numpy's libm on some paths and XLA's lowering on others. The two need not
    # agree, and which one a caller got would depend on the array type it
    # happened to pass.
    rp, tsqrt = jnp.asarray(rp), jnp.asarray(tsqrt)
    airdm3, rhoa = jnp.asarray(airdm3), jnp.asarray(rhoa)
    pmid, t = jnp.asarray(pmid), jnp.asarray(t)
    mask = jnp.asarray(mask)

    k = scalar_terms(mmcg, dmol, difvol)

    # `:171`. Thermal velocity of the condensable gas.
    vel_cp = k.term1 * tsqrt

    if idcmfp == 1:
        # `:174`, "as Gbin v1".
        dcoff_cp = numerics.safe_divide(k.term5 * tsqrt, rhoa, mask)
        # `:177`.
        mfp_cp = numerics.safe_divide(jnp.full_like(rhoa, k.term2), airdm3, mask)
    else:
        # `:178-180`, "as coded in GLOMAP-bin". `t` and `pmid` reach the
        # routine's outputs here and nowhere else.
        dcoff_cp = numerics.safe_divide(
            1.0e-7 * (t**1.75) * k.term7,
            numerics.true_divide(pmid, P_STANDARD) * k.term8,
            mask,
        )
        # `:181`, "as Gb v1_1".
        mfp_cp = numerics.safe_divide(3.0 * dcoff_cp, vel_cp, mask)

    if ifuchs == 1:
        # `:194-199`, basic Fuchs (1964).
        denom = numerics.safe_divide(4.0 * dcoff_cp, se * vel_cp * rp, mask) + numerics.safe_divide(
            rp, rp + mfp_cp, mask
        )
        cc = numerics.safe_divide(k.term6 * dcoff_cp * rp, denom, mask)
        sinkarr = numerics.safe_divide(rp * 1.0e6, denom, mask)
    else:
        # `:205-214`, Fuchs-Sutugin (1971).
        kn = numerics.safe_divide(mfp_cp, rp, mask)
        fkn = numerics.safe_divide(1.0 + kn, 1.0 + 1.71 * kn + 1.33 * kn * kn, mask)
        # At se = 1.0 the correction factor is exactly 1.0 and this whole term
        # is inert. See the module docstring; the model runs se = 1.0.
        akn = numerics.safe_divide(
            jnp.ones_like(kn), 1.0 + 1.33 * kn * fkn * (1.0 / se - 1.0), mask
        )
        cc = k.term6 * dcoff_cp * rp * fkn * akn
        sinkarr = rp * 1.0e6 * fkn * akn

    # `:167-168`: both outputs are zeroed before any branch runs, so a masked
    # box carries 0.0 and not a stale value. Selected, never multiplied --
    # the masked rows of the fixture hold inf and NaN, and `0.0 * inf` is NaN.
    zero = jnp.zeros_like(rp)
    return jnp.where(mask, cc, zero), jnp.where(mask, sinkarr, zero)
