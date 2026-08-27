"""`ukca_aero_step`: the microphysics sequence, in the box model's configuration.

Ported from `fortran/src/ukca/ukca_aero_step.F90`. Every routine below it has
been checked against the compiled reference **on its own**; none had been
checked against the others. This is the module that makes that claim testable:
an initial re-mode, then `nmts` outer substeps each containing the coagulation
kernel and `nzts` competition substeps of gas production, condensation,
nucleation, coagulation and ageing, then re-mode and two size recalculations.

Eager, and permanently so
-------------------------

`drivers/__init__.py` says the eager driver stays because it is the debugger --
it can raise a real exception where a scan can only return a flag. This is that
driver. `nmts` and `nzts` are Python loops over static counts, so the whole
sequence unrolls at trace time; the `scan` variant is order 2's problem.

What is fixed, and where that comes from
----------------------------------------

The Fortran routine takes 96 arguments and the box model varies about 40 of
them. Everything else is fixed here exactly as `glomap_box.F90:140-163` passes
it: scavenging, deposition, cloud processing, wet oxidation and nitrate all
off, `dryox_in_aer = 1` so the gas-production term inside the competition loop
is live. That is the configuration `docs/unsupported.md` records and the only
one with a reference.

The units conversion is the part that is easy to get wrong
-----------------------------------------------------------

`s0g` is a volume mixing ratio of the *gas*; `gc` is a molecular concentration
expressed in units of the *aerosol component's* molar mass. `:846` converts one
to the other with `s0g_to_gc = mm_gas(jv)/mm(icp)`, and `:1192` converts back.
`ukca_calcnucrate` then needs the gas again, so `:1015` divides `gc` back by the
same ratio for H2SO4 and `:1027` by a *different* one for secondary organic --
`mm_gas(msec_org)/mm(cp_sec_org)`, which the source comment at `:1017-1023`
spells out because 150/16.8 is not 1.

Three of the routines below are gated
-------------------------------------

`cond_on`, `nucl_on` and `coag_on` each switch a call off. With `cond_on = 0`
and `nucl_on = 1` the Fortran reads `s_cond_s` never having been assigned --
UP-6, `fidelity.s_cond_s_zero_when_cond_off`, whose default reproduces the
zero gfortran's `.bss` happens to give.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.physics.ageing import ageing
from glomap_jax.physics.coag_kernel import calc_coag_kernel
from glomap_jax.physics.coagwithnucl import coagwithnucl
from glomap_jax.physics.conden import conden
from glomap_jax.physics.drydiam import calc_drydiam
from glomap_jax.physics.nucleation import calcnucrate
from glomap_jax.physics.remode import remode
from glomap_jax.physics.volume_mode import volume_mode

__all__ = ["AeroState", "AeroStepResult", "aero_step"]


class AeroState(NamedTuple):
    """The prognostic state `ukca_aero_step` advances."""

    nd: Array
    md: Array
    mdt: Array
    s0g: Array


class AeroStepResult(NamedTuple):
    """State plus every diagnostic field the routine hands back."""

    nd: Array
    md: Array
    mdt: Array
    s0g: Array
    drydp: Array
    dvol: Array
    wetdp: Array
    wvol: Array
    mdwat: Array
    rhopar: Array
    pvol: Array
    pvol_wat: Array
    bud_aer_mas: Array
    n_merge: Array


def aero_step(
    tables,
    gas,
    budget,
    coag_mode_table: Array,
    state: AeroState,
    env: dict,
    bud_aer_mas: Array,
    *,
    dtc: float,
    dtz: float,
    nmts: int,
    nzts: int,
    cond_on: int = 1,
    nucl_on: int = 1,
    coag_on: int = 1,
    bln_on: int = 0,
    icoag: int = 1,
    imerge: int = 1,
    ifuchs: int = 1,
    idcmfp: int = 1,
    icondiam: int = 1,
    ibln: int = 1,
    i_nuc_method: int = 2,
    ichem: int = 1,
    intraoff: int = 0,
    interoff: int = 0,
    fidelity: FidelityConfig = FidelityConfig(),
) -> AeroStepResult:
    """One chemistry timestep of microphysics."""
    del dtc  # reaches only the processes this configuration switches off
    nd, md, mdt, s0g = (jnp.asarray(x) for x in state)
    bud_aer_mas = jnp.asarray(bud_aer_mas)
    nbox = nd.shape[0]
    ncp = md.shape[2]
    nchemg = env["s0g_dot"].shape[1]
    n_merge = jnp.zeros(nd.shape, dtype=jnp.int32)

    vm_kw = {
        "fix_water_content": fidelity.l_fix_ukca_water_content,
        "fix_neg_pvol_wat": fidelity.l_fix_neg_pvol_wat,
    }

    def sizes(nd, md, mdt):
        """`calc_drydiam` then `volume_mode`, the pair the routine calls four
        times. `calc_drydiam` may rewrite `md`/`mdt` -- the undersize reset --
        so its outputs are threaded through rather than discarded."""
        drydp, dvol, md_r, mdt_r = calc_drydiam(tables, nd, md, mdt)
        mdwat, wvol, wetdp, rhopar, pvol, pvol_wat = volume_mode(
            tables,
            nd,
            md_r,
            mdt_r,
            env["rh_clr"],
            dvol,
            drydp,
            env["t"],
            env["pmid"],
            env["s"],
            **vm_kw,
        )
        return drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat, md_r, mdt_r

    # `:541-545`: dry size, then the initial re-mode that catches advection
    # having taken drydp out of bounds.
    drydp, dvol, md, mdt = calc_drydiam(tables, nd, md, mdt)
    r = remode(tables, budget, nd, md, mdt, drydp, env["pmid"], bud_aer_mas, imerge=imerge)
    nd, md, mdt, bud_aer_mas = r.nd, r.md, r.mdt, r.bud_aer_mas
    n_merge = n_merge + r.n_merge
    drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat, md, mdt = sizes(nd, md, mdt)

    # Which gases are condensable, and the molar-mass ratio each carries.
    condensable = [j for j in range(nchemg) if bool(gas.condensable[j])]
    ratio = {
        jv: float(gas.mm_gas[jv]) / float(tables.mm[int(gas.condensable_choice[jv])])
        for jv in condensable
    }

    for _imts in range(nmts):
        gc = jnp.zeros((nbox, nchemg), dtype=nd.dtype)
        if ichem == 1:
            # `:843-856`. Only where s0g is positive; elsewhere gc is zero.
            for jv in condensable:
                positive = s0g[:, jv] > 0.0
                gc = gc.at[:, jv].set(
                    jnp.where(
                        positive,
                        numerics.safe_divide(
                            ratio[jv] * s0g[:, jv] * env["aird"], env["sm"], positive
                        ),
                        0.0,
                    )
                )
        gcold = gc

        kii_arr, kij_arr = calc_coag_kernel(
            tables.mode,
            tables.modesol,
            drydp,
            dvol,
            wetdp,
            wvol,
            rhopar,
            env["mfpa"],
            env["dvisc"],
            env["t"],
            coag_on=coag_on,
            icoag=icoag,
        )

        for _izts in range(nzts):
            if ichem == 1:
                # `:901-902`, dryox_in_aer = 1.
                for jv in condensable:
                    gc = gc.at[:, jv].add(dtz * env["s0g_dot"][:, jv] * env["aird"] * ratio[jv])

            s_cond_s = jnp.zeros(nbox, dtype=nd.dtype)
            ageterm1 = jnp.zeros((nbox, 4, nchemg), dtype=nd.dtype)
            if cond_on == 1:
                md, mdt, gc, bud_aer_mas, _delgc, ageterm1, s_cond_s = conden(
                    tables,
                    gas,
                    budget,
                    nd,
                    env["tsqrt"],
                    env["rhoa"],
                    env["airdm3"],
                    wetdp,
                    env["pmid"],
                    env["t"],
                    md,
                    mdt,
                    gc,
                    bud_aer_mas,
                    dtz=dtz,
                    ifuchs=ifuchs,
                    idcmfp=idcmfp,
                    icondiam=icondiam,
                    fidelity=fidelity,
                )
            elif not fidelity.s_cond_s_zero_when_cond_off:
                raise ValueError(
                    "cond_on = 0 with s_cond_s_zero_when_cond_off disabled: "
                    "ukca_aero_step reads s_cond_s never having been assigned (UP-6)"
                )

            delgc_nucl = jnp.zeros((nbox, nchemg), dtype=nd.dtype)
            if nucl_on == 1:
                if gas.mh2so4 < 0:
                    raise ValueError("nucl_on = 1 needs mh2so4; :1069 ereports otherwise")
                r_h2so4 = ratio[gas.mh2so4]
                positive = gc[:, gas.mh2so4] > 0.0
                h2so4 = jnp.where(
                    positive,
                    numerics.true_divide(gc[:, gas.mh2so4], r_h2so4),
                    0.0,
                )
                # `:1017-1029`: a DIFFERENT ratio, and the source comment says
                # why -- calcnucrate wants gas-phase Sec_Org at 150 g/mol, not
                # the organic-carbon component at 16.8.
                if gas.msec_org >= 0:
                    r_org = float(gas.mm_gas[gas.msec_org]) / float(
                        tables.mm[int(gas.condensable_choice[gas.msec_org])]
                    )
                    sec_org = numerics.true_divide(gc[:, gas.msec_org], r_org)
                else:
                    sec_org = jnp.zeros(nbox, dtype=nd.dtype)

                h2so4, delh2so4 = calcnucrate(
                    env["t"],
                    env["s"],
                    env["rh"],
                    env["aird"],
                    h2so4,
                    sec_org,
                    env["height"],
                    env["htpblg"],
                    s_cond_s,
                    dtz=dtz,
                    bln_on=bln_on,
                    ibln=ibln,
                    i_nuc_method=i_nuc_method,
                )
                gc = gc.at[:, gas.mh2so4].set(jnp.where(h2so4 > 0.0, h2so4 * r_h2so4, 0.0))
                delgc_nucl = delgc_nucl.at[:, gas.mh2so4].set(delh2so4 * r_h2so4)
                slot = budget.slot("nmasnuclsunucsol")
                if slot != 0:
                    bud_aer_mas = bud_aer_mas.at[:, slot].add(delgc_nucl[:, gas.mh2so4])

            if coag_on == 1 or nucl_on == 1:
                cr = coagwithnucl(
                    tables,
                    coag_mode_table,
                    budget,
                    gas,
                    nd,
                    md,
                    mdt,
                    delgc_nucl,
                    kii_arr,
                    kij_arr,
                    bud_aer_mas,
                    dtz=dtz,
                    intraoff=intraoff,
                    interoff=interoff,
                    fidelity=fidelity,
                )
                nd, md, mdt, bud_aer_mas, ageterm2 = (
                    cr.nd,
                    cr.md,
                    cr.mdt,
                    cr.bud_aer_mas,
                    cr.ageterm2,
                )
            else:
                ageterm2 = jnp.zeros((nbox, 4, 4, ncp), dtype=nd.dtype)

            nd, md, mdt, bud_aer_mas = ageing(
                tables,
                gas,
                budget,
                nd,
                md,
                mdt,
                ageterm1,
                ageterm2,
                wetdp,
                bud_aer_mas,
                fidelity=fidelity,
            )

        # `:1146-1152`, `:1156`, `:1171-1175`: size, merge, size again.
        drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat, md, mdt = sizes(nd, md, mdt)
        r = remode(tables, budget, nd, md, mdt, drydp, env["pmid"], bud_aer_mas, imerge=imerge)
        nd, md, mdt, bud_aer_mas = r.nd, r.md, r.mdt, r.bud_aer_mas
        n_merge = n_merge + r.n_merge
        drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat, md, mdt = sizes(nd, md, mdt)

        if ichem == 1:
            # `:1185-1197`. The gas goes back to a mixing ratio, and the change
            # is clamped so it cannot take s0g negative.
            for jv in condensable:
                delta = numerics.true_divide(
                    (gc[:, jv] - gcold[:, jv])
                    * numerics.safe_divide(env["sm"], env["aird"], jnp.ones(nbox, dtype=bool)),
                    ratio[jv],
                )
                delta = jnp.where(delta < -s0g[:, jv], -s0g[:, jv], delta)
                s0g = s0g.at[:, jv].add(delta)

    return AeroStepResult(
        nd=nd,
        md=md,
        mdt=mdt,
        s0g=s0g,
        drydp=drydp,
        dvol=dvol,
        wetdp=wetdp,
        wvol=wvol,
        mdwat=mdwat,
        rhopar=rhopar,
        pvol=pvol,
        pvol_wat=pvol_wat,
        bud_aer_mas=bud_aer_mas,
        n_merge=n_merge,
    )
