"""`ukca_ageing`: insoluble particles becoming soluble.

Ported from `fortran/src/ukca/ukca_ageing.F90`. Consumes `ageterm1` from
`conden` (condensation onto insoluble modes) and `ageterm2` from `coagwithnucl`
(soluble modes coagulating into them), works out how many particles have
accumulated enough soluble material, and moves them -- with their mass -- into
the corresponding soluble mode.

Two of CLAUDE.md's claims about this routine, corrected
------------------------------------------------------

CLAUDE.md lists `ukca_ageing` twice among five loops needing sequential
treatment: "over modes (7->4 and 8->4 collide) and over `jv`".

**The mode collision is structurally real and unreachable.** `:219-223` sets
`tmode = imode - 3` below `mode_sup_insol` and `imode - 4` at it, so modes 7 and
8 both target mode 4. But `mode_sup_insol` needs setup 12 or 13, which the box
model does not implement. Measured across all seven supported setups at both
`l_dust_mp_ageing` settings, the modes entering the loop are a subset of
`{5, 6, 7}` and their targets are always distinct. So the mode loop is
parallelisable in every configuration this project can validate, and no
"broadcast differs" test for it can exist. `validation/capture_ageing_leaf.py`
asserts that, and raises if a future setup ever activates mode 8 -- which is
the point at which the port's mode loop must become sequential.

**The `jv` collision is real but static.** `cp_coag_added(icp)` at `:265` stops
the coagulation term being counted twice when two condensable gases share a
component, which `msec_org` and `msec_orgi` both do (`cp_oc`). It depends only
on `condensable_choice`, a compile-time table, so the gas that adds the term is
just the lowest `jv` with that component -- resolved here at trace time.
`broadcast=True` drops the guard, and `tests/test_ageing.py` requires that to
change the answer.

UP-3: the rescale is a no-op
---------------------------

`:302-306`::

    IF (naged > nd(jl,imode)) THEN
      naged=nd(jl,imode)
      totage(:)=totage(:)*nd(jl,imode)/naged
    END IF

`naged` is overwritten *before* it is used as the divisor, so `nd/naged` is
exactly `1.0` and the rescale does nothing. The comment claims it "reduces
ageing if limited by insoluble particles". `ageing_totage_rescale_noop`
defaults to reproducing that; flipping it uses the pre-clamp `naged` as the
divisor, which is what the comment describes -- and CLAUDE.md records that the
naive fix loses mass, which the flag's test measures rather than repeats.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.physics._ageing_literals import AGEING_BUDGET_SITES
from glomap_jax.physics.budget_indices import NOT_CARRIED

__all__ = ["ageing", "target_mode"]

#: `:212`. Divide by ten so a particle ages after ten monolayers, not one.
MONOLAYERS = 10.0


def target_mode(imode: int) -> int:
    """`:219-223`, 0-based in and out. Modes 7 and 8 (1-based) both give 4."""
    one_based = imode + 1
    return (one_based - 3 if one_based < 8 else one_based - 4) - 1


def ageing(
    tables,
    gas,
    budget,
    nd: Array,
    md: Array,
    mdt: Array,
    ageterm1: Array,
    ageterm2: Array,
    wetdp: Array,
    bud_aer_mas: Array,
    *,
    broadcast: bool = False,
    fidelity: FidelityConfig = FidelityConfig(),
) -> tuple[Array, Array, Array, Array]:
    """`(nd, md, mdt, bud_aer_mas)`.

    `broadcast=True` drops the `cp_coag_added` guard, counting each component's
    coagulation term once per condensable gas that maps to it. It exists for
    the test that shows the guard is load-bearing.
    """
    nd, md, mdt = jnp.asarray(nd), jnp.asarray(md), jnp.asarray(mdt)
    ageterm1, ageterm2 = jnp.asarray(ageterm1), jnp.asarray(ageterm2)
    wetdp, bud_aer_mas = jnp.asarray(wetdp), jnp.asarray(bud_aer_mas)
    nbox, nmodes = nd.shape
    ncp = md.shape[2]
    topmode = int(tables.topmode)
    mode = [bool(m) for m in tables.mode]
    component = [[bool(tables.component[i][c]) for c in range(ncp)] for i in range(nmodes)]

    condensable = [j for j in range(ageterm1.shape[2]) if bool(gas.condensable[j])]
    # `:265`, resolved statically: the lowest condensable `jv` with a given
    # component is the one that adds the coagulation term.
    first_for_cp: dict[int, int] = {}
    for jv in condensable:
        icp = int(gas.condensable_choice[jv])
        first_for_cp.setdefault(icp, jv)

    for imode in range(4, topmode):
        if not mode[imode]:
            continue
        tmode = target_mode(imode)

        # --- accumulate over gases, `:230-283` ---
        totage = [jnp.zeros(nbox, dtype=nd.dtype) for _ in range(ncp)]
        totage1 = [jnp.zeros(nbox, dtype=nd.dtype) for _ in range(ncp)]
        f_mm = [0.0] * ncp
        naged = jnp.zeros(nbox, dtype=nd.dtype)

        for jv in condensable:
            icp = int(gas.condensable_choice[jv])
            f_mm[icp] = float(gas.mm_gas[jv]) / float(tables.mm[icp])
            # Every division by a scalar constant goes through `true_divide`:
            # `f_mm`, `dimen` and the monolayer count are all Python floats, and
            # XLA rewrites `x / c` into `x * (1/c)` for any of them. The port's
            # first draft used plain `/` for `f_mm` and `dimen` and disagreed
            # with the reference on 12 elements -- 1 to 4 ulp, in the only
            # configuration that runs more than one insoluble mode.
            totage_jv = numerics.true_divide(ageterm1[:, imode - 4, jv], f_mm[icp])
            totage[icp] = totage[icp] + totage_jv
            totage1[icp] = totage1[icp] + totage_jv

            adds_coag = broadcast or first_for_cp[icp] == jv
            if adds_coag:
                # `:270-280` accumulates term by term into BOTH `totage_jv` and
                # `totage(icp)`, so the association is
                # `(((t0 + a1) + a2) + a3) + a4` and not `t0 + (a1+a2+a3+a4)`.
                # Summing the four first and adding once is the tidier form and
                # differs by an ulp -- 12 elements of setup 8's dust-ageing
                # arm, which is the only configuration running more than one
                # insoluble mode.
                for jmode in range(4):
                    term = numerics.true_divide(ageterm2[:, jmode, imode - 4, icp], f_mm[icp])
                    totage_jv = totage_jv + term
                    totage[icp] = totage[icp] + term

            # `:290-296`. `dimen` is the condensable's molecular diameter.
            dimen = float(gas.dimen[jv])
            age1ptcl = numerics.true_divide(
                numerics.true_divide(wetdp[:, imode] * wetdp[:, imode], dimen), dimen
            )
            naged = naged + numerics.true_divide(
                numerics.safe_divide(totage_jv, age1ptcl, jnp.ones(nbox, dtype=bool)),
                MONOLAYERS,
            )

        outer = nd[:, imode] > float(tables.num_eps[imode])
        significant = outer & (naged > float(tables.num_eps[imode]))

        # `:302-306`, UP-3. The clamp is real; the rescale is not.
        over = significant & (naged > nd[:, imode])
        naged_pre = naged
        naged = jnp.where(over, nd[:, imode], naged)
        # `totage(:)*nd(jl,imode)/naged` associates as `(totage*nd)/naged`, and
        # that is NOT `totage * (nd/naged)`: with `naged` already set to `nd`
        # the second form is exactly `totage * 1.0` and the first is
        # `(totage*nd)/nd`, which differs by up to 4 ulp. The port's first
        # draft factored the ratio out and disagreed with the reference on 12
        # elements -- the same reassociation phase C recorded for the mode
        # masses. So the divisor is applied to the product, as written.
        divisor = naged if fidelity.ageing_totage_rescale_noop else naged_pre
        totage = [
            jnp.where(over, numerics.safe_divide(t * nd[:, imode], divisor, over), t)
            for t in totage
        ]

        ndinsnew = nd[:, imode] - naged
        ndsolnew = nd[:, tmode] + naged
        any_material = significant & (sum(totage) > 0.0)

        # --- budgets and the soluble mode's composition, `:311-455` ---
        for site_i, site_t, site_cp, name, kind in AGEING_BUDGET_SITES:
            if site_i - 1 != imode:
                continue
            icp = site_cp - 1
            if icp >= ncp or not component[tmode][icp]:
                continue
            slot = budget.slot(name)
            if slot == NOT_CARRIED:
                continue
            value = totage[icp] * f_mm[icp] if kind == "totage" else naged * md[:, imode, icp]
            bud_aer_mas = bud_aer_mas.at[:, slot].add(jnp.where(any_material, value, 0.0))

        for icp in range(ncp):
            if not component[tmode][icp]:
                continue
            carried = naged * md[:, imode, icp] if component[imode][icp] else 0.0
            grown = numerics.safe_divide(
                nd[:, tmode] * md[:, tmode, icp] + carried + totage1[icp] * f_mm[icp],
                ndsolnew,
                any_material,
            )
            md = md.at[:, tmode, icp].set(jnp.where(any_material, grown, md[:, tmode, icp]))

        # --- `:459-476`: numbers and totals, insoluble then soluble ---
        nd = nd.at[:, imode].set(jnp.where(significant, ndinsnew, nd[:, imode]))
        ins_total = jnp.zeros(nbox, dtype=nd.dtype)
        for icp in range(ncp):
            if component[imode][icp]:
                ins_total = ins_total + md[:, imode, icp]
        mdt = mdt.at[:, imode].set(jnp.where(significant, ins_total, mdt[:, imode]))

        nd = nd.at[:, tmode].set(jnp.where(significant, ndsolnew, nd[:, tmode]))
        sol_total = jnp.zeros(nbox, dtype=nd.dtype)
        for icp in range(ncp):
            if component[tmode][icp]:
                sol_total = sol_total + md[:, tmode, icp]
        mdt = mdt.at[:, tmode].set(jnp.where(significant, sol_total, mdt[:, tmode]))

    return nd, md, mdt, bud_aer_mas
