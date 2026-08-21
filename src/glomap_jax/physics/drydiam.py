"""`ukca_calc_drydiam` — dry volume and geometric mean dry diameter (tasks 36, 37).

Runs five times per chemistry step at `nmts = 1` (`2 + 2*nmts`, plus the
driver's `update_size` call), which makes it the most frequently evaluated
routine in the model after the coefficient kernels.

## The mask is inverted from what its name suggests

`ukca_calc_drydiam.F90:206-212`:

    mask(i) = (nd(i,imode) > num_eps(imode))
    IF (mask(i)) THEN
      dvol(i,imode) = 0.0
    ELSE
      dvol(i,imode) = mmid(imode)*mmsul/(avogadro*rho_so4)
    END IF

So `mask` true means the mode **has** particles — and that is the branch that
sets `dvol` to *zero*, because the component masses are about to be accumulated
into it. The populated case starts empty; the empty case gets the `mmid`
fallback, a whole particle's worth of pure H2SO4. Reading `mask` as "use the
default here" inverts the routine, and both branches produce finite,
plausible-looking diameters.

The comparison is strict `>`, so `nd == num_eps` exactly takes the **`mmid`**
branch. That tie is not hypothetical: 625 of 3072 sampled `volume_mode` mask
points in the committed state goldens sit exactly on it.

## The undersize reset is modes 1-3, and only modes 1-3

`:245` reads `DO imode = mode_nuc_sol, mode_acc_sol`, i.e. 1 to 3 — the
soluble nucleation, Aitken and accumulation modes. Not all eight, and not
"the soluble ones" (mode 4, `cor_sol`, is soluble and excluded). The source
comment says "only do check for solvent modes nuc,ait,acc".

When `drydp < ddplim0*0.1` the routine rewrites `md`, `mdt`, `dvol` and
`drydp` from `mlo` — the mode's lower mass bound. Three details that a
paraphrase loses:

* `md` is rewritten **only for components the mode carries**. A nonzero mass
  parked in a non-member component survives the reset untouched.
* `drydp` is recomputed as `(sixovrpix*dvol)**(1.0/3.0)` written inline at
  `:257`, not by another `cubrt_v` call. Same value, and worth knowing when
  chasing a discrepancy.
* `dp_thresh1 = ddplim0(imode)*0.1` is a *product*, evaluated per mode. Not a
  literal, and not `ddplim0/10`.

**No shipped namelist ever reaches it** — 0 of 2160 records in the branch dump.
It needs constructed inputs, which is what task 35d's fixture is for.

## Faithful forms

* `sixovrpix = 6.0/(pi*x)`, exactly as `:231` writes it. The algebraically
  equal `1.0/(x*(pi/6))` differs by 2 ulp for two of the width parameters, and
  the cube root downstream turns that into a different double on 53-65% of
  points. Measured; see docs/porting-notes.md.
* `ratio1 = mm/(avogadro*rhocomp)` is formed once over components and then
  multiplied in, matching `:196`. Inlining it into the accumulation changes
  4408 of 12000 sampled masses.
* The `icp` accumulation is an ordered fold over components, not a sum: the
  Fortran adds in index order and a pairwise reduction associates differently.
* The mode-inactive branch writes `mmsul*mmid/(avogadro*rho_so4)` (`:225`)
  while the mask-false branch writes `mmid*mmsul/(avogadro*rho_so4)` (`:210`).
  Those are the same double — IEEE multiplication is commutative — but they are
  written as they appear rather than unified, because the next such pair might
  not be.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core import numerics
from ..core.constants import AVOGADRO, MMSUL, PI, RHO_SO4
from .modes import NMODES

__all__ = [
    "MODE_ACC_SOL",
    "MODE_NUC_SOL",
    "calc_drydiam",
    "component_ratio",
    "six_over_pi_x",
    "undersize_threshold",
]

# ukca_mode_setup.F90:86,88. The undersize reset spans these inclusive.
MODE_NUC_SOL = 1
MODE_ACC_SOL = 3


def component_ratio(mm: Array, rhocomp: Array) -> Array:
    """`ratio1 = mm/(avogadro*rhocomp)` (`:196`), formed once over components."""
    return mm / (AVOGADRO * rhocomp)


def undersize_threshold(ddplim0: Array) -> Array:
    """`dp_thresh1 = ddplim0(imode)*0.1` (`:251`).

    A product, and not `ddplim0/10.0`: those differ in the last ulp for one of
    setup 1's modes. Exposed as a function because the difference is a single
    ulp in a *threshold*, so no achievable input lands between the two spellings
    -- the only way to test it is to compare the expressions.
    """
    return ddplim0 * 0.1


def six_over_pi_x(x: Array) -> Array:
    """`6.0/(pi*x)` (`:231`). Not `1.0/(x*(pi/6))`; see the module docstring."""
    return 6.0 / (PI * x)


def calc_drydiam(tables, nd: Array, md: Array, mdt: Array) -> tuple[Array, Array, Array, Array]:
    """Return `(drydp, dvol, md, mdt)`.

    `md` and `mdt` are `INTENT(IN OUT)` in the Fortran and are rewritten by the
    undersize reset. They are returned rather than mutated, so a caller can see
    whether the reset fired by comparing what it passed against what came back.

    `tables` is a `ModeTables` from `physics.modes`.
    """
    nbox = nd.shape[0]
    ncp = tables.ncp
    # `.at[].set()` needs jax arrays; a caller passing numpy is normal.
    nd, md, mdt = jnp.asarray(nd), jnp.asarray(md), jnp.asarray(mdt)
    ratio1 = component_ratio(tables.mm[:ncp], tables.rhocomp[:ncp])
    sixovrpix = six_over_pi_x(tables.x)

    dvol_cols = []
    for imode in range(NMODES):
        if not bool(tables.mode[imode]):
            # Inactive mode: one value everywhere, no mask and no accumulation.
            dvol_cols.append(jnp.full((nbox,), MMSUL * tables.mmid[imode] / (AVOGADRO * RHO_SO4)))
            continue

        # Strict `>`: nd exactly equal to num_eps takes the mmid branch.
        mask = nd[:, imode] > tables.num_eps[imode]
        empty = tables.mmid[imode] * MMSUL / (AVOGADRO * RHO_SO4)
        column = jnp.where(mask, 0.0, empty)

        # Ordered fold over components, in index order, skipping the ones this
        # mode does not carry -- exactly the Fortran's two nested guards.
        for icp in range(ncp):
            if not bool(tables.component[imode, icp]):
                continue
            column = column + jnp.where(mask, ratio1[icp] * md[:, imode, icp], 0.0)
        dvol_cols.append(column)

    dvol = jnp.stack(dvol_cols, axis=1)
    ddpcub = sixovrpix[None, :] * dvol
    drydp = numerics.cbrt(ddpcub)

    return _undersize_reset(tables, sixovrpix, drydp, dvol, md, mdt)


def _undersize_reset(
    tables, sixovrpix: Array, drydp: Array, dvol: Array, md: Array, mdt: Array
) -> tuple[Array, Array, Array, Array]:
    """`:245-262`. Modes 1-3 only, and unreachable from any shipped namelist."""
    for imode in range(MODE_NUC_SOL - 1, MODE_ACC_SOL):
        if not bool(tables.mode[imode]):
            continue
        # A product per mode, not a literal and not a division by ten.
        dp_thresh1 = undersize_threshold(tables.ddplim0[imode])
        fired = drydp[:, imode] < dp_thresh1

        mlo = tables.mlo[imode]
        reset_dvol = mlo * MMSUL / (AVOGADRO * RHO_SO4)
        # `:257` spells the cube root inline rather than calling cubrt_v again.
        reset_drydp = (sixovrpix[imode] * reset_dvol) ** (1.0 / 3.0)

        for icp in range(tables.ncp):
            if not bool(tables.component[imode, icp]):
                # Not a member: a mass parked here survives the reset.
                continue
            md = md.at[:, imode, icp].set(
                jnp.where(fired, mlo * tables.mfrac_0[imode, icp], md[:, imode, icp])
            )
        mdt = mdt.at[:, imode].set(jnp.where(fired, mlo, mdt[:, imode]))
        dvol = dvol.at[:, imode].set(jnp.where(fired, reset_dvol, dvol[:, imode]))
        drydp = drydp.at[:, imode].set(jnp.where(fired, reset_drydp, drydp[:, imode]))

    return drydp, dvol, md, mdt
