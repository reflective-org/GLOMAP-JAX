"""`ukca_remode`: merging the top tail of a mode into the next one up.

Ported from `fortran/src/ukca/ukca_remode.F90`. When a mode's dry diameter has
grown past a threshold, the fraction of its number and mass above that
threshold moves into the mode above.

**Issue #12: no shipped namelist ever merges a mode**, so this routine has zero
trajectory coverage and its fixture is entirely constructed. `drydp` is handed
in rather than computed -- it is `calc_drydiam`'s output, already byte-equal --
which is what makes constructing the merge condition legitimate.

The mode loop is loop-carried, and reachably so
-----------------------------------------------

`:206` runs `DO imode=1,nmodemax_merge`, and each pass writes `nd(jl,imode+1)`
and `md(jl,imode+1,:)`. The next pass reads both: `dp_ip1` at `:213` and the
receiving mode's state at `:373`. So a box whose nucleation *and* Aitken modes
have both grown past threshold merges twice, and the second merge sees the
first's result.

This is the second of CLAUDE.md's five loops shown to be live -- and unlike
`ukca_coagwithnucl`'s, it needs nothing exotic to reach: the fixture has 210
boxes that merge twice. `broadcast=True` runs every mode against the entry
state and `tests/test_remode.py` requires it to differ.

`imerge` picks the threshold, and one setting always merges
-----------------------------------------------------------

1 is the next mode's mid-point, 2 its lower bin edge, 3 the geometric mean of
the two diameters. `:234` is `(dp > dp_thresh1) .OR. (imerge == 3)`, so
`imerge = 3` merges **unconditionally** -- it is the only setting that does
anything on a distribution no namelist would merge. Outside `{1,2,3}` both
thresholds are read never having been assigned (`:215-231` assigns them only
inside the three `IF`s), so this raises.

Two silent clamps, surfaced
---------------------------

`frac_n < 0.5` and `frac_m < 0.001` are floored at `:248` and `:259` with no
diagnostic -- the first caps the transferred number at half the mode, the second
the transferred mass at 99.9%. CLAUDE.md requires caps to be reported, so both
come back as masks.

`umErf` is the only primitive the merge fraction depends on, and the numerics
sweep found it bit-identical between gfortran and JAX on the capture platform.
`EXP` appears once, at `:252`, so issue #28 has exactly one place to bite.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.physics._remode_literals import MERGE_BUDGET_SITES
from glomap_jax.physics.budget_indices import NOT_CARRIED

__all__ = ["FRAC_M_FLOOR", "FRAC_N_FLOOR", "P_STRAT", "RemodeResult", "remode"]

#: `:204`. Below this only the nucleation and Aitken modes may merge.
P_STRAT = 1.0e4

#: `:248` and `:259`, both silent upstream.
FRAC_N_FLOOR = 0.5
FRAC_M_FLOOR = 0.001

#: sqrt(2) as a Python float, so the scalar divisions stay concrete under jit
#: (a jnp constant becomes a tracer inside jax.jit; float() of it then fails).
_SQRT2 = float(np.sqrt(np.float64(2.0)))


class RemodeResult(NamedTuple):
    """`nd`, `md`, `mdt`, `bud_aer_mas`, the merge count, and the two clamps."""

    nd: Array
    md: Array
    mdt: Array
    bud_aer_mas: Array
    n_merge: Array
    frac_n_clamped: Array
    frac_m_clamped: Array


def remode(
    tables,
    budget,
    nd: Array,
    md: Array,
    mdt: Array,
    drydp: Array,
    pmid: Array,
    bud_aer_mas: Array,
    *,
    imerge: int,
    broadcast: bool = False,
    fidelity: FidelityConfig = FidelityConfig(),
) -> RemodeResult:
    """One mode-merging pass.

    `broadcast=True` evaluates every mode against the state at entry instead of
    sequentially. It exists for the test that shows the sequential form is
    required, and must never be used for a trajectory.
    """
    if imerge not in (1, 2, 3):
        raise ValueError(
            f"imerge={imerge} must be 1, 2 or 3: ukca_remode.F90:215-231 assigns "
            "dp_thresh1 and dp_thresh2 only inside those three branches, so any "
            "other value reads them never having been assigned"
        )
    del fidelity  # no upstream quirk here needs a choice

    nd, md, mdt = jnp.asarray(nd), jnp.asarray(md), jnp.asarray(mdt)
    drydp, pmid = jnp.asarray(drydp), jnp.asarray(pmid)
    bud_aer_mas = jnp.asarray(bud_aer_mas)
    nbox, nmodes = nd.shape
    ncp = md.shape[2]
    mode = [bool(m) for m in tables.mode]
    component = [[bool(tables.component[i][c]) for c in range(ncp)] for i in range(nmodes)]

    entry_nd, entry_md = nd, md
    n_merge = jnp.zeros((nbox, nmodes), dtype=jnp.int32)
    frac_n_clamped = jnp.zeros((nbox, nmodes), dtype=bool)
    frac_m_clamped = jnp.zeros((nbox, nmodes), dtype=bool)

    # `:203-204`. Three modes in the troposphere, two in the stratosphere.
    in_range = [pmid >= P_STRAT if i == 2 else jnp.ones(nbox, dtype=bool) for i in range(3)]

    for imode in range(3):
        if not mode[imode]:
            continue
        src_nd = entry_nd if broadcast else nd
        src_md = entry_md if broadcast else md

        dp = drydp[:, imode]
        dp_ip1 = drydp[:, imode + 1]
        if imerge == 1:
            thresh = jnp.full(nbox, float(tables.ddpmid[imode + 1]))
        elif imerge == 2:
            thresh = jnp.full(nbox, float(tables.ddplim0[imode + 1]))
        else:
            thresh = jnp.sqrt(dp * dp_ip1)

        criterion = in_range[imode] & ((dp > thresh) | (imerge == 3))
        enough = criterion & (src_nd[:, imode] > float(tables.num_eps[imode]))
        n_merge = n_merge.at[:, imode].add(jnp.where(enough, 1, 0))

        log_sigma = float(np.log(np.float64(tables.sigmag[imode])))
        # `lnratn/SQRT(2.0)/LOG(sigmag)`: two divisions by scalar constants, so
        # two `true_divide`s.
        lnratn = jnp.log(numerics.safe_divide(thresh, dp, enough))
        erfnum = numerics.true_divide(
            numerics.true_divide(lnratn, _SQRT2), log_sigma
        )
        frac_n = 0.5 * (1.0 + jax_erf(erfnum))
        clamp_n = enough & (frac_n < FRAC_N_FLOOR)
        frac_n = jnp.where(frac_n < FRAC_N_FLOOR, FRAC_N_FLOOR, frac_n)
        frac_n_clamped = frac_n_clamped.at[:, imode].set(frac_n_clamped[:, imode] | clamp_n)

        deln = src_nd[:, imode] * (1.0 - frac_n)
        log2sg = log_sigma * log_sigma
        dp2 = jnp.exp(jnp.log(dp) + 3.0 * log2sg)
        lnratm = jnp.log(numerics.safe_divide(thresh, dp2, enough))
        erfmas = numerics.true_divide(
            numerics.true_divide(lnratm, _SQRT2), log_sigma
        )
        frac_m = 0.5 * (1.0 + jax_erf(erfmas))
        clamp_m = enough & (frac_m < FRAC_M_FLOOR)
        frac_m = jnp.where(frac_m < FRAC_M_FLOOR, FRAC_M_FLOOR, frac_m)
        frac_m_clamped = frac_m_clamped.at[:, imode].set(frac_m_clamped[:, imode] | clamp_m)

        newn = src_nd[:, imode] - deln
        newnp1 = src_nd[:, imode + 1] + deln
        transfers = enough & (newn > float(tables.num_eps[imode]))
        # `:389-396`: the criterion fired but there was nothing to merge, so
        # the composition goes back to the mode's initial fractions.
        resets = criterion & ~(src_nd[:, imode] > float(tables.num_eps[imode]))

        dm = []
        for icp in range(ncp):
            if component[imode][icp]:
                dm.append(src_md[:, imode, icp] * src_nd[:, imode] * (1.0 - frac_m))
            else:
                dm.append(jnp.zeros(nbox, dtype=nd.dtype))

        for site_i, _site_j, site_cp, name in MERGE_BUDGET_SITES:
            if site_i - 1 != imode:
                continue
            icp = site_cp - 1
            if icp >= ncp or not component[imode][icp]:
                continue
            slot = budget.slot(name)
            if slot == NOT_CARRIED:
                continue
            bud_aer_mas = bud_aer_mas.at[:, slot].add(jnp.where(transfers, dm[icp], 0.0))

        # `:350-361`: the donor mode loses the transferred mass, then its number.
        donor_total = jnp.zeros(nbox, dtype=nd.dtype)
        for icp in range(ncp):
            if component[imode][icp]:
                grown = numerics.safe_divide(
                    src_nd[:, imode] * src_md[:, imode, icp] - dm[icp], newn, transfers
                )
                md = md.at[:, imode, icp].set(jnp.where(transfers, grown, md[:, imode, icp]))
                donor_total = donor_total + jnp.where(transfers, md[:, imode, icp], 0.0)
            else:
                md = md.at[:, imode, icp].set(jnp.where(transfers, 0.0, md[:, imode, icp]))
        mdt = mdt.at[:, imode].set(jnp.where(transfers, donor_total, mdt[:, imode]))
        nd = nd.at[:, imode].set(jnp.where(transfers, newn, nd[:, imode]))

        # `:369-380`: the receiving mode gains it.
        recv_total = jnp.zeros(nbox, dtype=nd.dtype)
        for icp in range(ncp):
            if component[imode + 1][icp]:
                grown = numerics.safe_divide(
                    src_nd[:, imode + 1] * src_md[:, imode + 1, icp] + dm[icp],
                    newnp1,
                    transfers,
                )
                md = md.at[:, imode + 1, icp].set(
                    jnp.where(transfers, grown, md[:, imode + 1, icp])
                )
                recv_total = recv_total + jnp.where(transfers, md[:, imode + 1, icp], 0.0)
            else:
                md = md.at[:, imode + 1, icp].set(jnp.where(transfers, 0.0, md[:, imode + 1, icp]))
        mdt = mdt.at[:, imode + 1].set(jnp.where(transfers, recv_total, mdt[:, imode + 1]))
        nd = nd.at[:, imode + 1].set(jnp.where(transfers, newnp1, nd[:, imode + 1]))

        # `:389-396`, the other arm.
        for icp in range(ncp):
            if component[imode][icp]:
                md = md.at[:, imode, icp].set(
                    jnp.where(
                        resets,
                        float(tables.mmid[imode]) * float(tables.mfrac_0[imode][icp]),
                        md[:, imode, icp],
                    )
                )
        mdt = mdt.at[:, imode].set(jnp.where(resets, float(tables.mmid[imode]), mdt[:, imode]))

    return RemodeResult(nd, md, mdt, bud_aer_mas, n_merge, frac_n_clamped, frac_m_clamped)


def jax_erf(x: Array) -> Array:
    """`umErf`. Bit-identical to gfortran's on the capture platform -- the phase
    B numerics sweep measured that over 4,330 points, which is why the merge
    fraction can be gated exactly rather than to a tolerance."""
    from jax.scipy.special import erf

    return erf(x)
