"""The box model: environment, initial state, and the chemistry-step loop.

Ported from `fortran/src/box/`, which is new BSD-3 code this repository owns
rather than vendored UKCA -- so unlike everything else in the port, the source
here may be extended. It is ported anyway, and faithfully, because it is the
only thing that turns `aero_step` into a runnable model and therefore the only
route to **gate C**: the committed trajectory goldens, which are the one gate
that can say the *model* agrees rather than its parts.

Three pieces:

* `box_env` -- `set_box_env`. Derives every environmental field `aero_step`
  needs from seven namelist scalars. `spec_humid_from_rh` fills `s` when the
  namelist gives `spec_humid < 0`, which `boundary_layer.nml` does.
* `init_state` -- seeds every active mode at its mid-point mass so
  `calc_drydiam` has a positive `dvol` to work with, then imposes the
  requested number and size on the populated ones.
* `run` -- zero the budgets, `aero_step`, recompute size, record.

`update_size` passes `rh`, not `rh_clr`
---------------------------------------

`glomap_box_state_mod`'s `update_size` gives `volume_mode` `env%rh` while
`aero_step` gives it `RH_clr`. They are the same array in this model
(`set_box_env` sets `rh_clr = rh`), so nothing turns on it -- but the two call
sites disagree in the source and a port that quietly unified them would be
relying on a coincidence.

The seeding is the inverse of `calc_drydiam`
--------------------------------------------

`init_state` builds `md` from a requested `dp_init` by inverting
`dvol = (pi/6)*dp^3*x`, and the source comment says so. That is why a literal
zero-filled state does not work: `ukca_calc_drydiam` aborts on `dvol <= 0`, so
every active mode is seeded at `mmid*mfrac_0` with `nd = num_eps` first,
whether the namelist populates it or not.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core.constants import AVOGADRO, BOLTZMANN, PI, REPSILON, RGAS
from glomap_jax.drivers.aero_step import AeroState, aero_step
from glomap_jax.physics.drydiam import calc_drydiam
from glomap_jax.physics.volume_mode import volume_mode

__all__ = ["BoxState", "box_env", "diagnostics", "init_state", "run", "spec_humid_from_rh"]

#: `glomap_box_env_mod`'s PARAMETERs. `ma` is the mass of an air molecule;
#: the rest are Sutherland's law for the dynamic viscosity of air.
MA = 4.78e-26
DVISC_REF = 1.83e-5
T_SUTH = 120.0
T_SUTH_NUM = 416.16
T_REF_VIS = 296.16


class BoxState(NamedTuple):
    """Everything the model carries between chemistry steps."""

    nd: Array
    md: Array
    mdt: Array
    s0g: Array
    s0g_dot: Array
    drydp: Array
    dvol: Array
    wetdp: Array
    wvol: Array
    mdwat: Array
    rhopar: Array
    pvol: Array
    pvol_wat: Array


def spec_humid_from_rh(t: float, p: float, rh: float) -> float:
    """`glomap_box_env_mod`'s Magnus formula, used when `spec_humid < 0`."""
    esat = 610.94 * np.exp(17.625 * (t - 273.15) / (t - 273.15 + 243.04))
    e = rh * esat
    return REPSILON * e / (p - (1.0 - REPSILON) * e)


def box_env(
    nbox: int,
    *,
    t: float,
    pmid: float,
    rh: float,
    spec_humid: float,
    height: float,
    pbl_height: float,
    box_volume: float,
) -> dict:
    """`set_box_env`. Every field is uniform across boxes."""
    rh_clamped = min(max(rh, 0.0), 1.0)
    s = spec_humid if spec_humid >= 0.0 else spec_humid_from_rh(t, pmid, rh_clamped)
    aird = pmid / (BOLTZMANN * 1.0e6 * t)
    rhoa = pmid / (RGAS * t)
    dvisc = DVISC_REF * (T_SUTH_NUM / (t + T_SUTH)) * (np.sqrt(t / T_REF_VIS) ** 3)
    vba = np.sqrt(8.0 * BOLTZMANN * t / (PI * MA))
    ones = jnp.ones(nbox)
    return {
        "t": ones * t,
        "tsqrt": ones * float(np.sqrt(t)),
        "pmid": ones * pmid,
        "pupper": ones * (0.95 * pmid),
        "plower": ones * (1.05 * pmid),
        "rh": ones * rh_clamped,
        "rh_clr": ones * rh_clamped,
        "s": ones * float(s),
        "aird": ones * aird,
        "airdm3": ones * (aird * 1.0e6),
        "rhoa": ones * rhoa,
        "dvisc": ones * float(dvisc),
        "mfpa": ones * float(2.0 * dvisc / (rhoa * vba)),
        "sm": ones * (rhoa * box_volume),
        "height": ones * height,
        "htpblg": ones * pbl_height,
    }


def _update_size(tables, nd, md, mdt, env, fidelity, *, dry: bool = False):
    """`update_size`. Note `env["rh"]`, not `rh_clr` -- see the module docstring."""
    drydp, dvol, md, mdt = calc_drydiam(tables, nd, md, mdt)
    mdwat, wvol, wetdp, rhopar, pvol, pvol_wat = volume_mode(
        tables,
        nd,
        md,
        mdt,
        env["rh"],
        dvol,
        drydp,
        env["t"],
        env["pmid"],
        env["s"],
        dry=dry,
        fix_water_content=fidelity.l_fix_ukca_water_content,
        fix_neg_pvol_wat=fidelity.l_fix_neg_pvol_wat,
    )
    return md, mdt, drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat


def init_state(
    tables,
    gas,
    env: dict,
    *,
    nbox: int,
    nadvg: int,
    nchemg: int,
    nd_init,
    dp_init,
    mfrac_init=None,
    h2so4_init: float = 0.0,
    sec_org_init: float = 0.0,
    h2so4_prod: float = 0.0,
    sec_org_prod: float = 0.0,
    fidelity: FidelityConfig = FidelityConfig(),
    dry: bool = False,
) -> BoxState:
    """`init_state`, including its closing `update_size`."""
    ncp = int(tables.ncp)
    nmodes = len(tables.mode)
    nd = np.zeros((nbox, nmodes))
    md = np.zeros((nbox, nmodes, ncp))

    for imode in range(nmodes):
        if not bool(tables.mode[imode]):
            continue
        # Seed at mid-point mass so dvol > 0 even for an empty mode.
        for icp in range(ncp):
            if bool(tables.component[imode][icp]):
                md[:, imode, icp] = float(tables.mmid[imode]) * float(tables.mfrac_0[imode][icp])
        nd[:, imode] = float(tables.num_eps[imode])

        if nd_init[imode] > float(tables.num_eps[imode]):
            nd[:, imode] = nd_init[imode]
            if dp_init[imode] > 0.0:
                frac = np.array(
                    mfrac_init[imode][:ncp] if mfrac_init is not None else [0.0] * ncp,
                    dtype=np.float64,
                )
                if frac.sum() <= 0.0:
                    frac = np.array([float(tables.mfrac_0[imode][c]) for c in range(ncp)])
                for icp in range(ncp):
                    if not bool(tables.component[imode][icp]):
                        frac[icp] = 0.0
                if frac.sum() > 0.0:
                    frac = frac / frac.sum()
                    dvol_target = (PI / 6.0) * (dp_init[imode] ** 3) * float(tables.x[imode])
                    vol_per_kg = sum(
                        frac[c] / float(tables.rhocomp[c]) for c in range(ncp) if frac[c] > 0.0
                    )
                    mass_tot = dvol_target / vol_per_kg
                    for icp in range(ncp):
                        md[:, imode, icp] = frac[icp] * mass_tot * AVOGADRO / float(tables.mm[icp])

    mdt = md.sum(axis=2)
    s0g = np.zeros((nbox, nadvg))
    s0g_dot = np.zeros((nbox, nchemg))
    aird = np.asarray(env["aird"])
    sm = np.asarray(env["sm"])
    if gas.mh2so4 >= 0:
        s0g[:, gas.mh2so4] = (h2so4_init / aird) * sm
        s0g_dot[:, gas.mh2so4] = h2so4_prod / aird
    if gas.msec_org >= 0:
        s0g[:, gas.msec_org] = (sec_org_init / aird) * sm
        s0g_dot[:, gas.msec_org] = sec_org_prod / aird

    nd_j, md_j, mdt_j = jnp.asarray(nd), jnp.asarray(md), jnp.asarray(mdt)
    md_j, mdt_j, drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat = _update_size(
        tables, nd_j, md_j, mdt_j, env, fidelity, dry=dry
    )
    return BoxState(
        nd=nd_j,
        md=md_j,
        mdt=mdt_j,
        s0g=jnp.asarray(s0g),
        s0g_dot=jnp.asarray(s0g_dot),
        drydp=drydp,
        dvol=dvol,
        wetdp=wetdp,
        wvol=wvol,
        mdwat=mdwat,
        rhopar=rhopar,
        pvol=pvol,
        pvol_wat=pvol_wat,
    )


def diagnostics(tables, gas, state: BoxState, env: dict, time_s: float) -> list[float]:
    """One output row, in `write_output`'s column order.

    Number in cm-3, diameters in nm, density in kg m-3, component mass in
    ug m-3, and the two gases back as concentrations. The mass conversion --
    `md * nd * (mm/avogadro) * 1e6 * 1e9` -- is per-component and only for
    components the mode carries, which is what makes the column count
    setup-dependent.
    """
    row = [time_s, time_s / 3600.0]
    nd = np.asarray(state.nd)
    for imode in range(len(tables.mode)):
        if not bool(tables.mode[imode]):
            continue
        row += [
            float(nd[0, imode]),
            float(np.asarray(state.drydp)[0, imode] * 1.0e9),
            float(np.asarray(state.wetdp)[0, imode] * 1.0e9),
            float(np.asarray(state.rhopar)[0, imode]),
        ]
    md = np.asarray(state.md)
    for imode in range(len(tables.mode)):
        if not bool(tables.mode[imode]):
            continue
        for icp in range(int(tables.ncp)):
            if not bool(tables.component[imode][icp]):
                continue
            row.append(
                float(
                    md[0, imode, icp]
                    * nd[0, imode]
                    * (float(tables.mm[icp]) / AVOGADRO)
                    * 1.0e6
                    * 1.0e9
                )
            )
    s0g = np.asarray(state.s0g)
    sm = float(np.asarray(env["sm"])[0])
    aird = float(np.asarray(env["aird"])[0])
    row.append(float(s0g[0, gas.mh2so4] / sm * aird) if gas.mh2so4 >= 0 else 0.0)
    row.append(float(s0g[0, gas.msec_org] / sm * aird) if gas.msec_org >= 0 else 0.0)
    return row


def run(
    tables,
    gas,
    budget,
    coag_mode_table: Array,
    state: BoxState,
    env: dict,
    *,
    nsteps: int,
    dt_chem: float,
    nmts: int,
    nzts: int,
    output_every: int = 1,
    nbudaer: int,
    fidelity: FidelityConfig = FidelityConfig(),
    **switches,
) -> tuple[BoxState, np.ndarray]:
    """The chemistry-step loop, returning the final state and the output rows.

    `bud_aer_mas` and `n_merge` are zeroed at the top of every step, as
    `glomap_box.F90` does -- they are per-step diagnostics, not accumulators.
    """
    nbox = state.nd.shape[0]
    dtz = dt_chem / (nmts * nzts)
    rows = [diagnostics(tables, gas, state, env, 0.0)]

    for istep in range(1, nsteps + 1):
        result = aero_step(
            tables,
            gas,
            budget,
            coag_mode_table,
            AeroState(state.nd, state.md, state.mdt, state.s0g),
            {**env, "s0g_dot": state.s0g_dot},
            jnp.zeros((nbox, nbudaer + 1)),
            dtc=dt_chem,
            dtz=dtz,
            nmts=nmts,
            nzts=nzts,
            fidelity=fidelity,
            **switches,
        )
        md, mdt, drydp, dvol, wetdp, wvol, mdwat, rhopar, pvol, pvol_wat = _update_size(
            tables, result.nd, result.md, result.mdt, env, fidelity
        )
        state = BoxState(
            nd=result.nd,
            md=md,
            mdt=mdt,
            s0g=result.s0g,
            s0g_dot=state.s0g_dot,
            drydp=drydp,
            dvol=dvol,
            wetdp=wetdp,
            wvol=wvol,
            mdwat=mdwat,
            rhopar=rhopar,
            pvol=pvol,
            pvol_wat=pvol_wat,
        )
        if istep % max(output_every, 1) == 0 or istep == nsteps:
            rows.append(diagnostics(tables, gas, state, env, istep * dt_chem))

    return state, np.array(rows, dtype=np.float64)
