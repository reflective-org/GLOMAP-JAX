"""`ukca_volume_mode` — aerosol water, wet volume and wet diameter (tasks 41-45).

The widest leaf in phase D: it consumes `ukca_vapour` and `ukca_water_content_v`,
turns `(nd, md, rh, t, pmid, s)` into the wet particle, and its `wetdp`/`wvol`/
`rhopar` feed `ukca_conden`, `ukca_ageing` and the coagulation kernel.

## Task 41 — the soluble branch's water content

`mdwat` is the first output the routine produces and the only one that needs the
whole ZSR chain. What is here so far is the path to it: the ion assembly at
`:350-423`, the charge balance, the `ukca_water_content_v` call at `:429`, and
the `WHERE (mask)` / `ELSE WHERE` pair that zeroes it outside the mask.

**The stratospheric override at `:434-438` is deliberately not here** — it is
task 43. Until then `mdwat` is byte-equal only for `pmid >= putls`, which is
every row any shipped namelist has ever run.

## The three SO4 increments are applied in source order, and it is not `icp` order

`cl(-2)` is built by three statements in the order `cp_su` (`:368`, an
assignment), `cp_so` (`:372`), `cp_oc` (`:381`) — component indices **1, 6, 3**.
An ascending `icp` loop would apply `cp_oc` before `cp_so` and produce a
different double wherever a mode carries both, which is setups 4 and 5.

Each increment is `(fhyg_aom/avogadro)*(md/f_ao)`: two quotients formed first,
then multiplied. `fhyg_aom*md/(avogadro*f_ao)` is the same number in exact
arithmetic and a different one in float64.

`f_ao = mm_age_org/mm_pom` is recomputed at `:332` *inside* the mode loop. It is
computed here per mode too, and it is not cached in `core/constants.py`: a
derived quantity in a constants table is a second source of truth (CLAUDE.md).

## The charge balance keeps all six terms

`:422`:

    cl(1) = MAX(2.0*cl(-2) + cl(-1) + cl(-3) + cl(-4) - cl(2) - cl(3), 0.0)

Three of those six are identically zero in every supported setup — `cl(-1)` is
never written at all, and `cl(2)`/`cl(-3)` sit behind the nitrate block at
`:402`, whose guard `UBOUND(component,DIM=2) >= cp_no3` is false because
`ncp = 6 < cp_no3 = 7`. Dropping them *and reassociating* the survivors is a
different fold. They are written out.

`MAX` goes through `numerics.fortran_max` in the Fortran's argument order, so a
`NaN` sum comes back `NaN` exactly as gfortran's `MAX(NaN, 0.0)` does here.

## `ions` is built from the original `cl`, not from what ZSR consumes

`:425-427` snapshots `cl > 0.0` *before* `ukca_water_content_v` runs, and that
routine's pair loop then draws both ion pools down as it goes. A pair whose
cation was exhausted by an earlier pair still passes the presence mask and
contributes `clp = 0.0`. Rebuilding the mask from the depleted `cli` would skip
it, which is a different model rather than a faster one.

## H+ can be absent

A sulfate-free sea-salt mode gives `cl(3) = cl(-4)` bit-for-bit (both are
`md(cp_cl)/avogadro`, `:396-398`), so the charge balance is exactly `0.0 - 0.0`
and `ions(1)` is FALSE. Every H+ pair then contributes nothing. Assuming H+ is
present whenever any anion is inverts that.

## `mdcopy` and setup 11

`:295` copies `md` into `mdcopy`, and `:356-364` overwrites three of its
components for `i_mode_setup == 11` — all three right-hand sides reading
`cp_su`. That setup is **not constructible here**: `glomap_box_config_mod`'s
`init_indices` has no `CASE` for it and ereports instead, and `modes.build`
refuses it. The port raises rather than pretending, and `test_volume_mode.py`
asserts the unreachability from both ends. Everywhere else `mdcopy` is `md`,
which is why the three `mdcopy` reads below are spelled as `md`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core import numerics
from ..core.constants import AVOGADRO
from . import water_tables as wt
from .modes import NMODES, ModeTables
from .water_content import water_content

__all__ = [
    "CP_CL",
    "CP_NH4",
    "CP_NN",
    "CP_NO3",
    "CP_OC",
    "CP_SO",
    "CP_SU",
    "FHYG_AOM",
    "MM_AGE_ORG",
    "MM_POM",
    "SETUP_SOLINSOL",
    "aged_organic_moles",
    "charge_balance",
    "corrected_humidity",
    "ion_concentrations",
    "mdwat",
]

# ukca_mode_setup.F90:75-83, one-based as the Fortran writes them. Only these
# four are read by ukca_volume_mode; the other five component slots reach it
# only through `component`.
CP_SU = 1
CP_OC = 3
CP_CL = 4
CP_SO = 6
CP_NO3 = 7
CP_NN = 8
CP_NH4 = 9

# ukca_volume_mode.F90:254-256, PARAMETERs local to the routine. Not in
# core/constants.py for the same reason EPS_AB nearly was not: they are locals,
# not module constants, so they cannot be extracted by name. They are checked
# against the source text instead -- see
# test_volume_mode.py::test_the_local_parameters_still_read_as_the_source_writes_them.
FHYG_AOM = 0.65
MM_AGE_ORG = 0.150
MM_POM = 0.0168

# ukca_config_specification_mod's i_solinsol_6mode. See the module docstring.
SETUP_SOLINSOL = 11


def corrected_humidity(rh: Array) -> Array:
    """`corrh`, clamped to [0.1, 0.9] (`:305-307`).

    Two sequential `WHERE`s in that order, high then low, not a `clip`. The
    order is inert here because the bounds do not cross, but `jnp.clip`
    propagates `NaN` differently from a pair of comparisons and this is the only
    place in the routine where `rh` is touched at all -- `rh` itself goes
    nowhere else, so a clamp that failed to fire would be invisible in `rh`.

    **The clamps have never fired in a validated run.** The highest `rel_humid`
    in any shipped namelist is exactly 0.90 and the test at `:306` is a strict
    `>`, so even that row passes through. Reaching them needs a constructed
    fixture, which is what `test_volume_mode.py`'s `rh` axis is for.
    """
    corrh = jnp.asarray(rh, dtype=jnp.float64)
    corrh = jnp.where(corrh > 0.9, 0.9, corrh)
    return jnp.where(corrh < 0.1, 0.1, corrh)


def aged_organic_moles() -> float:
    """`f_ao = mm_age_org/mm_pom` (`:332`).

    A function, not a module constant, because `:332` sits *inside* the mode
    loop and CLAUDE.md forbids caching a derived quantity in a constants table.
    """
    return MM_AGE_ORG / MM_POM


def _hygroscopic_increment(md_cp: Array, f_ao: float) -> Array:
    """`(fhyg_aom/avogadro)*(md/f_ao)` (`:372`, `:381`).

    Two quotients, then one product. Written out because
    `fhyg_aom*md/(avogadro*f_ao)` is a different double.
    """
    return (FHYG_AOM / AVOGADRO) * (md_cp / f_ao)


def charge_balance(cl: Array) -> Array:
    """`cl(1)`, the H+ concentration that closes the charge balance (`:422-423`).

    Six terms, left to right, inside `MAX(..., 0.0)`. See the module docstring
    for why the three provably-zero ones are kept.
    """
    total = 2.0 * cl[:, wt.ion_slot(-2)]
    total = total + cl[:, wt.ion_slot(-1)]
    total = total + cl[:, wt.ion_slot(-3)]
    total = total + cl[:, wt.ion_slot(-4)]
    total = total - cl[:, wt.ion_slot(2)]
    total = total - cl[:, wt.ion_slot(3)]
    return numerics.fortran_max(total, 0.0)


def ion_concentrations(tables: ModeTables, md: Array, imode: int) -> Array:
    """`cl(:, -nanion:ncation)` for one mode (`:350-423`), shape `(nbox, 8)`.

    Slot order is `water_tables.ion_slot`'s: species -4 .. +3 at columns 0 .. 7.
    Every entry starts at 0.0 (`:351-353`) and only the components this mode
    carries write to it.
    """
    if tables.setup == SETUP_SOLINSOL:
        # `:356-364` rewrites mdcopy(cp_su), mdcopy(cp_cl) and mdcopy(cp_oc)
        # from solinsol_hygro_ratio * md(cp_su). Unreachable: init_indices has
        # no CASE for setup 11 and modes.build refuses it, so the ratio table is
        # not ported and there is nothing to be faithful to. Raise rather than
        # silently take the setup-1 path -- CLAUDE.md, "failing loudly".
        raise NotImplementedError(
            "i_mode_setup = 11 rewrites mdcopy at ukca_volume_mode.F90:356-364 "
            "from glomap_config%solinsol_hygro_ratio, which is not ported "
            "because glomap_box_config_mod's init_indices has no CASE for it"
        )

    md = jnp.asarray(md, dtype=jnp.float64)
    nbox = md.shape[0]
    slots = [jnp.zeros((nbox,), dtype=jnp.float64) for _ in range(wt.NANION + wt.NCATION + 1)]
    component = tables.component
    f_ao = aged_organic_moles()

    # Source order, not icp order: cp_su assigns, then cp_so and cp_oc
    # increment. Indices 1, 6, 3.
    if component[imode, CP_SU - 1]:
        slots[wt.ion_slot(-2)] = md[:, imode, CP_SU - 1] / AVOGADRO
    if component[imode, CP_SO - 1]:
        slots[wt.ion_slot(-2)] = slots[wt.ion_slot(-2)] + _hygroscopic_increment(
            md[:, imode, CP_SO - 1], f_ao
        )
    if component[imode, CP_OC - 1]:
        slots[wt.ion_slot(-2)] = slots[wt.ion_slot(-2)] + _hygroscopic_increment(
            md[:, imode, CP_OC - 1], f_ao
        )
    if component[imode, CP_CL - 1]:
        # Complete dissociation: Na+ and Cl- are the same double, which is what
        # makes a sulfate-free sea-salt mode cancel exactly in charge_balance.
        slots[wt.ion_slot(3)] = md[:, imode, CP_CL - 1] / AVOGADRO
        slots[wt.ion_slot(-4)] = md[:, imode, CP_CL - 1] / AVOGADRO

    if tables.ncp >= CP_NO3:
        # `:402`, dead in every supported setup (ncp = 6, cp_no3 = 7). Kept as
        # a guard rather than deleted, and asserted dead in the tests.
        raise NotImplementedError(
            "the nitrate block at ukca_volume_mode.F90:402-419 needs cp_no3, "
            f"cp_nn and cp_nh4, which no supported setup has (ncp = {tables.ncp})"
        )

    cl = jnp.stack(slots, axis=1)
    return cl.at[:, wt.ion_slot(1)].set(charge_balance(cl))


def _soluble_mdwat(
    tables: ModeTables,
    md: Array,
    imode: int,
    mask: Array,
    corrh: Array,
    *,
    fix_water_content: bool,
) -> Array:
    """`mdwat(:,imode)` for a soluble mode, tropospheric only (`:350-431`, `:448`).

    `wc*avogadro` under `mask`, 0.0 outside -- the `ELSE WHERE` at `:447-452`
    is what supplies the zero, several statements after the `WHERE` at `:431`
    supplies the value.
    """
    cl = ion_concentrations(tables, md, imode)
    ions = cl > 0.0
    wc = water_content(cl, corrh, ions, mask, fix_water_content=fix_water_content)
    return jnp.where(mask, wc * AVOGADRO, 0.0)


def mdwat(
    tables: ModeTables,
    nd: Array,
    md: Array,
    rh: Array,
    *,
    fix_water_content: bool,
) -> Array:
    """`mdwat`, the water content in molecules per particle, shape `(nbox, nmodes)`.

    Task 41: the soluble branch below `putls` only. An insoluble mode (`:641`)
    and an absent mode (`:679`) both give exactly 0.0, which is already faithful;
    the stratospheric override at `:434-438` is task 43 and is **not** applied,
    so a `pmid < putls` row is not covered by this function.
    """
    nd = jnp.asarray(nd, dtype=jnp.float64)
    md = jnp.asarray(md, dtype=jnp.float64)
    corrh = corrected_humidity(rh)
    nbox = nd.shape[0]

    columns = []
    for imode in range(NMODES):
        if not bool(tables.mode[imode]) or tables.modesol[imode] != 1:
            columns.append(jnp.zeros((nbox,), dtype=jnp.float64))
            continue
        # Strict `>`; the box seeds nd exactly at num_eps, so the tie is live.
        mask = nd[:, imode] > tables.num_eps[imode]
        columns.append(
            _soluble_mdwat(tables, md, imode, mask, corrh, fix_water_content=fix_water_content)
        )
    return jnp.stack(columns, axis=1)
