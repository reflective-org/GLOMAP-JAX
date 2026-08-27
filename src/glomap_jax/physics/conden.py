"""`ukca_conden`: condensation of vapour onto the pre-existing aerosol.

Ported from `fortran/src/ukca/ukca_conden.F90`. The widest routine in the port:
four `INTENT(IN OUT)` arrays, three `INTENT(OUT)`, a loop over gases containing
two loops over modes, and thirty budget write sites.

Those thirty sites are **not transcribed**. `_conden_literals.py` is generated
by `validation/extract_conden_literals.py`, which parses them out of the
Fortran and asserts what it found: thirty sites, twenty-nine distinct
`(name, kind)` pairs, every `*ins` name taking `deltami` and every soluble name
`deltams`, every site inside an `imode ==` block, and every `ageterm1` ordinal
equal to `nc_mode - 4`. Four hundred lines of near-identical `WHERE` blocks is
exactly where a hand port loses one, or attaches it to the wrong mode.

A budget index gates the physics
--------------------------------

`deltams` and `deltami` are zeroed at `:363-364` and assigned **only inside**
`IF (nmascond... > 0)`, and `md`/`mdt` are then updated from `deltams`. So an
uncarried budget slot does not merely lose a diagnostic -- it loses the mass.
Reproduced exactly, because the port would be wrong not to; measured inert
across all seven setups (54 of 54 pairs carried) by `tests/test_conden.py`,
which re-derives the alignment on every run. Issue #30.

`ageterm1`'s second index is a soluble mode name used as an ordinal
-------------------------------------------------------------------

`ageterm1` is `(nbox, nmodes_ins, nchemg)` -- its second index runs over the
four *insoluble* modes. The Fortran writes `ageterm1(:,mode_nuc_sol,jv)` for
the Aitken-insoluble transfer and `ageterm1(:,mode_ait_sol,jv)` for the
accumulation-insoluble one: soluble-mode PARAMETERs standing in for the
ordinals 1 and 2. Correct, and it reads as a bug. The generated table stores
the ordinal and the extractor asserts the relation, so the misleading name
cannot be copied across.

Two upstream quirks, both flagged
---------------------------------

* `conden_insol_num_eps_by_sol_mode` (UP-10): the insoluble masks are gated by
  `num_eps` indexed by the enclosing **soluble** mode.
* `conden_ocaccins_double_count` (issue #29): `:576-602` is the same block
  twice, so `nmascondocaccins` accumulates twice. Diagnostic-only -- `ageterm1`
  is an assignment and `md`/`mdt` take `deltams`.

Byte equality is not attainable: `EXP` appears in `y2`, in `delgc_cond` and
inside `cond_coff`'s Cunningham correction (issue #28).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.core.constants import CONC_EPS
from glomap_jax.physics._conden_literals import DUPLICATED, WRITE_SITES
from glomap_jax.physics.budget_indices import NOT_CARRIED
from glomap_jax.physics.cond_coff import cond_coff

__all__ = ["AA_MODES", "DIFVOL_H2SO4", "DIFVOL_SEC_ORG", "SE_INS", "SE_SOL", "conden"]

#: `:235-237`. Both `1.0`, whatever the headers say -- see UP-9 and issue #27,
#: where the same stale `0.3` makes the Fuchs-Sutugin correction inert.
SE_SOL = 1.0
SE_INS = 1.0

#: `:325-327`, hard-coded per condensable rather than tabulated.
DIFVOL_H2SO4 = 51.96
DIFVOL_SEC_ORG = 204.14

#: `:268-278`. `icondiam = 1` takes the geometric-mean number radius;
#: `icondiam = 2` the per-mode moment of Lehtinen et al (2003), continuum at
#: the coarse end and molecular at the fine end.
AA_MODES = {
    1: (0.0,) * 8,
    2: (2.0, 1.9, 1.5, 1.1, 1.9, 1.5, 1.1, 1.1),
}

#: `(soluble mode, insoluble partner)` 0-based, and whether the partner is
#: behind the `topmode > mode_ait_insol` guard. From `:369-387`; the pairs are
#: the same ones `_conden_literals` records, kept here because the *masks* are
#: set before any write site is consulted.
INSOLUBLE_PARTNER = {
    1: (4, False),  # ait_sol  -> ait_insol, ungated
    2: (5, True),  # acc_sol  -> acc_insol
    3: (6, True),  # cor_sol  -> cor_insol
}
#: `:384-387`, the `mask4i` pair. Needs `mode_sup_insol`, which no supported
#: setup activates.
SUPER_COARSE_PARTNER = (3, 7)


def _difvol(jv: int, gas) -> float:
    """`:324-330`. Anything else reaches `ereport('DIFVOL remains undefined')`."""
    if jv == gas.mh2so4:
        return DIFVOL_H2SO4
    if jv in (gas.msec_org, gas.msec_orgi) and jv >= 0:
        return DIFVOL_SEC_ORG
    raise ValueError(
        f"difvol is undefined for gas index {jv}: ukca_conden.F90:329 ereports "
        "'DIFVOL remains undefined' for any condensable that is neither H2SO4 "
        "nor a secondary organic"
    )


def conden(
    tables,
    gas,
    budget,
    nd: Array,
    tsqrt: Array,
    rhoa: Array,
    airdm3: Array,
    wetdp: Array,
    pmid: Array,
    t: Array,
    md: Array,
    mdt: Array,
    gc: Array,
    bud_aer_mas: Array,
    *,
    dtz: float,
    ifuchs: int,
    idcmfp: int,
    icondiam: int,
    fidelity: FidelityConfig = FidelityConfig(),
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """`(md, mdt, gc, bud_aer_mas, delgc_cond, ageterm1, s_cond_s)`.

    `bud_aer_mas` is `(nbox, nbudaer + 1)` with column 0 the hole every
    uncarried index points at -- the Fortran declares it `(nbox, 0:nbudaer)`.
    """
    if icondiam not in AA_MODES:
        raise ValueError(
            f"icondiam={icondiam} must be 1 or 2: ukca_conden.F90:279-283 ereports "
            "'Unexpected ICONDIAM value' for anything else"
        )

    nd, wetdp = jnp.asarray(nd), jnp.asarray(wetdp)
    md, mdt, gc = jnp.asarray(md), jnp.asarray(mdt), jnp.asarray(gc)
    bud_aer_mas = jnp.asarray(bud_aer_mas)
    tsqrt, rhoa, airdm3 = jnp.asarray(tsqrt), jnp.asarray(rhoa), jnp.asarray(airdm3)
    pmid, t = jnp.asarray(pmid), jnp.asarray(t)

    nbox = nd.shape[0]
    nchemg = gc.shape[1]
    aa_modes = AA_MODES[icondiam]
    topmode = int(tables.topmode)

    delgc_cond = jnp.zeros((nbox, nchemg), dtype=gc.dtype)
    ageterm1 = jnp.zeros((nbox, 4, nchemg), dtype=gc.dtype)
    s_cond_s = jnp.zeros(nbox, dtype=gc.dtype)

    for jv in range(nchemg):
        if not bool(gas.condensable[jv]):
            continue
        icp = int(gas.condensable_choice[jv])
        dmol = float(gas.dimen[jv])
        mmcg = float(gas.mm_gas[jv])
        difvol = _difvol(jv, gas)
        is_sec_orgi = gas.msec_orgi >= 0 and jv == gas.msec_orgi

        mask1 = gc[:, jv] > CONC_EPS
        nc = jnp.zeros((nbox, 8), dtype=gc.dtype)
        sumnc = jnp.zeros(nbox, dtype=gc.dtype)

        # `:299` -- to `topmode`, not to `nmodes`. With `l_dust_mp_ageing` off
        # that is `mode_ait_insol`, so `nc` is never assigned for modes 6-8 and
        # the second loop's blocks that would read them carry the same guard.
        for imode in range(topmode):
            if not bool(tables.mode[imode]):
                continue
            aa = aa_modes[imode]
            log_sigma = jnp.log(jnp.asarray(float(tables.sigmag[imode])))
            y2 = jnp.exp(0.5 * aa * aa * log_sigma * log_sigma)
            mask2 = mask1 & (nd[:, imode] > float(tables.num_eps[imode]))
            rp = wetdp[:, imode] * 0.5 * y2
            se = SE_SOL if int(tables.modesol[imode]) == 1 else SE_INS
            cc, sinkarr = cond_coff(
                mask2,
                rp,
                tsqrt,
                airdm3,
                rhoa,
                pmid,
                t,
                mmcg=mmcg,
                se=se,
                dmol=dmol,
                difvol=difvol,
                ifuchs=ifuchs,
                idcmfp=idcmfp,
            )
            nc = nc.at[:, imode].set(jnp.where(mask2, nd[:, imode] * cc, 0.0))
            sumnc = jnp.where(mask2, sumnc + nc[:, imode], sumnc)
            if jv == gas.mh2so4:
                # `:341-343`, UNMASKED: `sinkarr` is zero outside mask2, so the
                # contribution is zero there and the sum is the same. Written
                # as the Fortran writes it.
                s_cond_s = s_cond_s + nd[:, imode] * sinkarr

        # `:348-349`.
        delgc = jnp.where(mask1, gc[:, jv] * (1.0 - jnp.exp(-sumnc * dtz)), 0.0)
        mask2 = mask1 & (delgc > CONC_EPS)
        # `:353-355`, UP-4: `delgc_cond/gc` where `= gc` was intended. Cannot
        # fire -- `delgc = gc*(1 - EXP(-sumnc*dtz))` and the exponential is
        # non-negative -- and reproduced anyway, because "cannot fire" is an
        # argument and the invariant test is what makes it a measurement.
        up4 = mask2 & (delgc > gc[:, jv])
        delgc = jnp.where(up4, numerics.safe_divide(delgc, gc[:, jv], up4), delgc)
        delgc_cond = delgc_cond.at[:, jv].set(delgc)
        gc = gc.at[:, jv].set(jnp.where(mask2, gc[:, jv] - delgc, gc[:, jv]))

        # `:359` -- the soluble modes only.
        for imode in range(4):
            if not bool(tables.mode[imode]):
                continue
            mask3 = mask2 & (nd[:, imode] > float(tables.num_eps[imode]))
            masks = {"soluble": mask3}

            # `:369-387`. The insoluble threshold is `num_eps` indexed by the
            # enclosing SOLUBLE mode -- UP-10.
            def insoluble_mask(partner: int, gated: bool, imode: int = imode) -> Array:
                if gated and not topmode > 5:
                    return jnp.zeros(nbox, dtype=bool)
                if not bool(tables.mode[partner]):
                    return jnp.zeros(nbox, dtype=bool)
                eps_mode = imode if fidelity.conden_insol_num_eps_by_sol_mode else partner
                return mask2 & (nd[:, partner] > float(tables.num_eps[eps_mode]))

            if imode in INSOLUBLE_PARTNER:
                partner, gated = INSOLUBLE_PARTNER[imode]
                masks["insoluble"] = insoluble_mask(partner, gated)
            else:
                masks["insoluble"] = jnp.zeros(nbox, dtype=bool)
            if imode == SUPER_COARSE_PARTNER[0]:
                masks["super_coarse_insoluble"] = insoluble_mask(SUPER_COARSE_PARTNER[1], True)
            else:
                masks["super_coarse_insoluble"] = jnp.zeros(nbox, dtype=bool)

            deltams = jnp.zeros(nbox, dtype=gc.dtype)
            seen: set[str] = set()
            for site_imode, site_icp, sec_orgi, kind, nc_mode, name, ageterm in WRITE_SITES:
                if site_imode - 1 != imode or site_icp - 1 != icp:
                    continue
                if sec_orgi != is_sec_orgi:
                    continue
                slot = budget.slot(name)
                if slot == NOT_CARRIED:
                    # Issue #30: the mass goes with the diagnostic.
                    continue
                if name in DUPLICATED and name in seen:
                    if not fidelity.conden_ocaccins_double_count:
                        continue
                seen.add(name)
                mask = masks[kind]
                delta = jnp.where(
                    mask,
                    numerics.safe_divide(delgc_cond[:, jv] * nc[:, nc_mode - 1], sumnc, mask),
                    0.0,
                )
                bud_aer_mas = bud_aer_mas.at[:, slot].add(delta)
                if kind == "soluble":
                    deltams = jnp.where(mask, delta, deltams)
                elif ageterm:
                    ageterm1 = ageterm1.at[:, ageterm - 1, jv].set(
                        jnp.where(mask, delta, ageterm1[:, ageterm - 1, jv])
                    )

            # `:762-767`. Only `deltams` -- the insoluble gain is deliberately
            # left to `ukca_ageing`; see the comment block at `:769-778`.
            grown = numerics.safe_divide(
                md[:, imode, icp] * nd[:, imode] + deltams, nd[:, imode], mask3
            )
            md = md.at[:, imode, icp].set(jnp.where(mask3, grown, md[:, imode, icp]))
            grown_t = numerics.safe_divide(
                mdt[:, imode] * nd[:, imode] + deltams, nd[:, imode], mask3
            )
            mdt = mdt.at[:, imode].set(jnp.where(mask3, grown_t, mdt[:, imode]))

    return md, mdt, gc, bud_aer_mas, delgc_cond, ageterm1, s_cond_s
