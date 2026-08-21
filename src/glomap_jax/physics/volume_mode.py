"""`ukca_volume_mode` — aerosol water, wet volume and wet diameter (tasks 41-45).

The widest leaf in phase D: it consumes `ukca_vapour` and `ukca_water_content_v`,
turns `(nd, md, rh, t, pmid, s)` into the wet particle, and its `wetdp`/`wvol`/
`rhopar` feed `ukca_conden`, `ukca_ageing` and the coagulation kernel.

## Every array-by-constant division goes through `numerics.true_divide`

`ukca_volume_mode` divides an array by a scalar constant at five places --
`:368`, `:372`/`:381`, `:396-398`, `:435`, `:437` -- plus `:294`'s `mm_ovravc`,
which is seven call sites here because `:396-398` is two divisions.
XLA rewrites `divide(x, broadcast(c))` into `multiply(x, broadcast(1/c))` for
**any** scalar constant, and `1/c` is inexact for every constant here, so the
plain spelling is a different double from the division gfortran performs.

Whether the rewrite happens eagerly depends on the JAX version: jax 0.9.2 emits
a true divide and **jax 0.11.0 does not**. That is not a hypothetical -- this
module was first validated byte-equal on 0.9.2 and produced 73 failures at 1-2
ulp on 0.11.0, which is the repository's canonical interpreter
(`Makefile:18`, `.venv`). See `core.numerics.true_divide` for the counts and for
why `jnp.asarray(c)` and `lax.optimization_barrier` do not help.

## Task 41 — the soluble branch's water content

`mdwat` is the first output the routine produces and the only one that needs the
whole ZSR chain. What is here so far is the path to it: the ion assembly at
`:350-423`, the charge balance, the `ukca_water_content_v` call at `:429`, and
the `WHERE (mask)` / `ELSE WHERE` pair that zeroes it outside the mask.

## Task 43 — the stratospheric branch, which has never executed

`putls = 1.5e4` (`:258`) and the four shipped namelists run `pressure` in
{1e5, 2e4, 1e5, 1e5}, so **neither override has run in any validated
trajectory**. Two of them, under two different masks:

* `:434-438`, under **`mask`**: throw the ZSR water content away and rebuild
  `mdwat` from `wts`, the H2SO4 weight percent `ukca_vapour` returned.
* `:584-586`, under **`mask_sol`**: replace the solution density with
  `rhosol_strat`.

Unifying the two masks changes `mdwat` at a `mask_nosol` point, where `mask` is
true and `mask_sol` is not.

The test is a strict `<`, so `pmid == 1.5e4` exactly is **tropospheric**. And
the override is per point, not per call: the fixture uses `nbox > 1` with a
mixed `pmid` column, which the box model cannot produce because `pmid` there is
a run-level scalar (`glomap_box_env_mod.F90:75`, `nbox = 1`).

`:435` writes `md*mm(cp_su)/avogadro` although `mm_ovravc(cp_su)` is in scope
and is exactly that quotient. They are different doubles and the inline form is
what is reproduced. `md(:,imode,cp_su)` is read **unconditionally** -- there is
no `component` guard on it -- so a mass parked in a non-member `cp_su` slot
reaches `mdwat` here.

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

## Task 44 — the insoluble and inactive branches

The insoluble branch (`:638-673`) is not the soluble one with the water taken
out. Three differences a symmetric rewrite would erase:

* **`pvol` at `:647` is unmasked and has no default.** `:597` and `:613` both
  carry an `ELSE WHERE pvol = dvol*mfrac_0`, and `:686` writes the same default
  for an absent mode. `:647` has neither: it assigns `md*mm_ovravcrhocp` on
  every point, including where `nd <= num_eps`. Adding the symmetric default
  changes `pvol` for an empty insoluble mode.
* **`wvol` at `:642` is exactly `dvol`**, set before the `pvol` loop and never
  accumulated into. So `wetdp` on an insoluble mode is a pure function of
  `dvol`; the partial volumes just computed do not enter it.
* **The guarded division at `:665` is guarded by `mask`, not `mask_sol`.**
  Reusing `:627`'s guard changes `rhopar` at every masked point of an insoluble
  mode, where `mask_sol` is not computed at all.

The branch also has **no `denom2 <= 0` diagnostic**: the 100-line `ereport`
block at `:476-575` lives only inside the soluble arm, so `denom2 == 0` at
`:665` simply divides. The gap is real and unreachable, and provably so -- see
`test_the_insoluble_branch_has_no_denom2_diagnostic`.

`rhopar` on this branch cannot be shown to be a mass-weighted **mean** by any
supported setup: no insoluble mode carries two different densities. Mode 5 is
bc + oc at 1500 each, and modes 6 and 7 are dust alone at 2650. What byte
equality still tests there is the fold order, since the quotient of two
rounded sums is not exactly the common density.

The inactive branch's `pvol` default at `:686` is **dead code**.
`ukca_mode_setup.F90:693-704` sets `mode = (mode_choice > 0)` and makes
`component(imode,icp)` true only when `mode_choice(imode) == 1`, so `component`
implies `mode`, while `:686` sits inside `ELSE` on `mode(imode)`. Reproduced and
asserted dead rather than dropped or "fixed".

## Task 45 — `wetdp`, and the three diagnostic blocks

`:292-293` writes

    piovrsix     = pi/6.0
    sixovrpix(:) = 1.0/(x(:)*piovrsix)

while `ukca_calc_drydiam.F90:230` writes `6.0/(pi*x(imode))` from the **same**
`x` array. The two are algebraically identical and 2 ulp apart for sigma_g = 2.0
and 1.8, and the cube root downstream turns that into a different double on
53.2% (mode 4), 53.3% (mode 7) and 64.5% (mode 8) of sampled volumes. So the two
spellings are kept in the two modules and must never be factored into a shared
helper.

`cubrt_v` at `:696` is called on `nmodes*nbox` elements, i.e. the whole array by
sequence association, and `:698` then *discards* the result for inactive modes
and copies `drydp` in. So `wetdp` for an absent mode is `drydp` bit for bit, not
a cube root of anything.

**The three `ereport` blocks are not ported**, and the reasons differ:

* `:476-575`, `denom <= 0` or `denom2 <= 0` under `mask_sol`. A diagnostic:
  it prints and aborts, and computes nothing that reaches an output.
* `:703-880`, the five-way `MINVAL(wetdp/drydp/wvol/dvol/rhopar) <= 0` guard.
  **Unreachable**, not merely unreached: the `WRITE` at `:856-876` formats
  `'(5(A,E15.6,A,I0),A)'` into a `CHARACTER(LEN=errormessagelength)` buffer,
  overruns it, and gfortran raises "Fortran runtime error: End of record" and
  kills the process at `:876` -- before `ereport` at `:877` ever runs. So there
  is no reference behaviour to reproduce; there is only a crash. Recorded, not
  built a fixture for.
* `:882-898`, `MINVAL(pvol_wat) < 0` or `MINVAL(mdwat) < 0`. This one is
  **always active in the box**: `l_glomap_clim_radaer` is a
  `LOGICAL, PARAMETER, .FALSE.` (`ukca_um_legacy_mod.F90:141`), so the
  disjunction at `:882` reduces to `l_fix_neg_pvol_wat`, and
  `glomap_box_config_mod.F90:323` pins that `.TRUE.`. Omitting it means the
  port silently accepts a negative `mdwat` where the reference dies. The port
  reproduces the *computation* and not the abort; the omission is asserted in
  `test_the_diagnostic_ereport_blocks_are_omitted` rather than left implicit.
  Note that if it were implemented, gfortran's `MINVAL` **skips** `NaN` while
  `jnp.min` propagates, so it would need `nanmin` semantics.

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
from ..core.constants import AVOGADRO, MMW, PI, RHO_SO4, RHO_WATER
from . import water_tables as wt
from .modes import NMODES, ModeTables
from .vapour import vapour
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
    "PUTLS",
    "SETUP_SOLINSOL",
    "aged_organic_moles",
    "charge_balance",
    "corrected_humidity",
    "ion_concentrations",
    "mdwat",
    "partial_volumes",
    "six_over_pi_x",
    "solubility_masks",
    "soluble_mass",
    "stratospheric",
    "volume_mode",
    "wet_diameter",
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
PUTLS = 1.5e4

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
    return (FHYG_AOM / AVOGADRO) * numerics.true_divide(md_cp, f_ao)


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
        slots[wt.ion_slot(-2)] = numerics.true_divide(md[:, imode, CP_SU - 1], AVOGADRO)
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
        slots[wt.ion_slot(3)] = numerics.true_divide(md[:, imode, CP_CL - 1], AVOGADRO)
        slots[wt.ion_slot(-4)] = numerics.true_divide(md[:, imode, CP_CL - 1], AVOGADRO)

    if tables.ncp >= CP_NO3:
        # `:402`, dead in every supported setup (ncp = 6, cp_no3 = 7). Kept as
        # a guard rather than deleted, and asserted dead in the tests.
        raise NotImplementedError(
            "the nitrate block at ukca_volume_mode.F90:402-419 needs cp_no3, "
            f"cp_nn and cp_nh4, which no supported setup has (ncp = {tables.ncp})"
        )

    cl = jnp.stack(slots, axis=1)
    return cl.at[:, wt.ion_slot(1)].set(charge_balance(cl))


def stratospheric(pmid: Array) -> Array:
    """`pmid < putls` (`:434`, `:584`). Strict, so `putls` itself is FALSE."""
    return jnp.asarray(pmid, dtype=jnp.float64) < PUTLS


def _strat_mdwat(
    tables: ModeTables,
    md: Array,
    imode: int,
    mdwat_col: Array,
    where: Array,
    wts: Array,
    scales,
) -> Array:
    """`:434-438`. `mdwat` rebuilt from the H2SO4 weight percent.

    Three statements, each kept as written:

        massh2so4kg = md(:,imode,cp_su)*mm(cp_su)/avogadro
        masswaterkg = (100.0/wts - 1.0)*massh2so4kg
        mdwat       = masswaterkg/mmwovravc

    `mm(cp_su)/avogadro` is *not* replaced by the in-scope `mm_ovravc(cp_su)`,
    and the last line divides by a precomputed quotient rather than multiplying
    by `avogadro/mmw`.

    With `l_fix_neg_pvol_wat` off, `wts` has no 99% ceiling and reaches 103.8,
    so `100.0/wts - 1.0` goes **negative** and `mdwat` with it. The same flag
    also disables the `:882-898` abort that would catch that, so flipping it
    changes the failure mode rather than a number. This port reproduces the
    silence: it returns the negative water content, and the omitted guard is
    recorded in `test_volume_mode.py` rather than raised here.
    """
    massh2so4kg = numerics.true_divide(md[:, imode, CP_SU - 1] * tables.mm[CP_SU - 1], AVOGADRO)
    masswaterkg = (100.0 / wts - 1.0) * massh2so4kg
    return jnp.where(where, numerics.true_divide(masswaterkg, scales.mmwovravc), mdwat_col)


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
    t: Array,
    pmid: Array,
    s: Array,
    *,
    fix_water_content: bool,
    fix_neg_pvol_wat: bool,
) -> Array:
    """`mdwat`, the water content in molecules per particle, shape `(nbox, nmodes)`.

    The soluble branch's water content including the `:434-438` override. An
    insoluble mode (`:641`) and an absent mode (`:679`) both give exactly 0.0,
    which is already faithful, so this column is complete for every mode.
    """
    nd = jnp.asarray(nd, dtype=jnp.float64)
    md = jnp.asarray(md, dtype=jnp.float64)
    corrh = corrected_humidity(rh)
    scales = _Scales(tables)
    wts, _ = vapour(t, pmid, s, fix_neg_pvol_wat=fix_neg_pvol_wat)
    strat = stratospheric(pmid)
    nbox = nd.shape[0]

    columns = []
    for imode in range(NMODES):
        if not bool(tables.mode[imode]) or tables.modesol[imode] != 1:
            columns.append(jnp.zeros((nbox,), dtype=jnp.float64))
            continue
        # Strict `>`; the box seeds nd exactly at num_eps, so the tie is live.
        mask = nd[:, imode] > tables.num_eps[imode]
        column = _soluble_mdwat(tables, md, imode, mask, corrh, fix_water_content=fix_water_content)
        columns.append(_strat_mdwat(tables, md, imode, column, mask & strat, wts, scales))
    return jnp.stack(columns, axis=1)


# ---------------------------------------------------------------------------
# Task 42 -- density, rhopar, pvol, pvol_wat
# ---------------------------------------------------------------------------


def soluble_mass(tables: ModeTables, md: Array, imode: int, mask: Array) -> Array:
    """`mdsol`, the total soluble mass per particle (`:317-326`).

    An ordered fold over components under `mask`, seeded at 0.0.
    """
    md = jnp.asarray(md, dtype=jnp.float64)
    total = jnp.zeros(jnp.shape(mask), dtype=jnp.float64)
    for icp in range(tables.ncp):
        if tables.component[imode, icp] and tables.soluble[icp]:
            total = jnp.where(mask, total + md[:, imode, icp], total)
    return total


def solubility_masks(mdsol: Array, mask: Array) -> tuple[Array, Array]:
    """`(mask_sol, mask_nosol)` (`:329-330`).

    `mask_nosol` is `mdsol == 0.0` **exactly**, not `~mask_sol`, and the two do
    not partition `mask`: `mdsol < 0` is a third state with no name in the
    source. A row in it takes the `ELSE WHERE` at `:597` -- so `pvol` keeps the
    `dvol*mfrac_0` default rather than being zeroed by `:601` -- and the
    `ELSE WHERE` at `:631`, giving `pvol_wat = 0`, `wvol = dvol` and
    `rhopar = rho_so4`. Writing `mask_nosol = mask & ~mask_sol` changes `pvol`
    there and nothing else, which is exactly the kind of difference a tolerance
    would hide.
    """
    return mask & (mdsol > 0.0), mask & (mdsol == 0.0)


class _Scales:
    """The per-component quotients formed once at `:294-299`.

    Two of them are the kind of expression a reader "simplifies":

    * `mm_ovravcrhocp = (mm/avogadro)/rhocomp` -- **two divisions**, not
      `mm/(avogadro*rhocomp)`.
    * `mm_rhocp = mm*rhocomp`, a product, and the only place `rhocomp`
      multiplies rather than divides.
    """

    __slots__ = ("mm_ovravc", "mm_ovravcrhocp", "mm_rhocp", "mmwovravc", "mmwrhow")

    def __init__(self, tables: ModeTables):
        mm = jnp.asarray(tables.mm[: tables.ncp], dtype=jnp.float64)
        rhocomp = jnp.asarray(tables.rhocomp[: tables.ncp], dtype=jnp.float64)
        self.mm_ovravc = numerics.true_divide(mm, AVOGADRO)
        self.mmwovravc = MMW / AVOGADRO
        self.mm_ovravcrhocp = self.mm_ovravc / rhocomp
        self.mm_rhocp = mm * rhocomp
        self.mmwrhow = MMW * RHO_WATER


def _density_accumulators(
    tables: ModeTables,
    md: Array,
    imode: int,
    mask: Array,
    mdwat_col: Array,
    scales: _Scales,
) -> tuple[Array, Array, Array, Array]:
    """`(rhotmp, denom, rhotmp2, denom2)` (`:439-475`).

    Four ordered folds seeded from the **water** term and then run over `icp`
    ascending. `rhotmp2` and `denom2` are seeded by assignment *from* `rhotmp`
    and `denom` (`:443`, `:445`), inside a `WHERE` construct whose statements
    execute in order -- so they read the value written one line earlier, not the
    value from before the construct.

    The unprimed pair sums soluble components only; the primed pair sums all of
    them. Both live under the same `IF (component)` guard and the soluble
    `WHERE` comes first, which is why the two share a single loop here.

    Not `jnp.sum` over the component axis, and not "water last": both associate
    differently, and the outputs are `rhosol` (a divisor of every soluble
    `pvol`) and `rhopar`.
    """
    md = jnp.asarray(md, dtype=jnp.float64)
    rhotmp = jnp.where(mask, mdwat_col * scales.mmwrhow, 0.0)
    rhotmp2 = rhotmp
    denom = jnp.where(mask, mdwat_col * MMW, 0.0)
    denom2 = denom

    for icp in range(tables.ncp):
        if not tables.component[imode, icp]:
            continue
        term_rho = md[:, imode, icp] * scales.mm_rhocp[icp]
        term_den = md[:, imode, icp] * tables.mm[icp]
        if tables.soluble[icp]:
            rhotmp = rhotmp + jnp.where(mask, term_rho, 0.0)
            denom = denom + jnp.where(mask, term_den, 0.0)
        rhotmp2 = rhotmp2 + jnp.where(mask, term_rho, 0.0)
        denom2 = denom2 + jnp.where(mask, term_den, 0.0)
    return rhotmp, denom, rhotmp2, denom2


def _soluble_mode(
    tables: ModeTables,
    imode: int,
    nd: Array,
    md: Array,
    dvol: Array,
    corrh: Array,
    scales: _Scales,
    strat: Array,
    wts: Array,
    rhosol_strat: Array,
    *,
    fix_water_content: bool,
) -> dict:
    """One soluble mode (`:314-636`)."""
    mask = nd[:, imode] > tables.num_eps[imode]
    mdsol = soluble_mass(tables, md, imode, mask)
    mask_sol, mask_nosol = solubility_masks(mdsol, mask)

    mdwat_col = _soluble_mdwat(tables, md, imode, mask, corrh, fix_water_content=fix_water_content)
    # `:434`, under `mask` -- and before the density accumulators, which read
    # the overridden value.
    mdwat_col = _strat_mdwat(tables, md, imode, mdwat_col, mask & strat, wts, scales)
    rhotmp, denom, rhotmp2, denom2 = _density_accumulators(
        tables, md, imode, mask, mdwat_col, scales
    )

    # `:579`, double-where on mask_sol. `rhosol` is never initialised in the
    # Fortran and carries the previous imode's value outside mask_sol -- it is
    # assigned only at :579/:585 and read only at :593/:623, both under
    # WHERE(mask_sol) -- so the 0.0 this produces is a value nothing reads.
    # `denom` is exactly 0.0 on the :447 ELSE WHERE, which is what makes the
    # guard load-bearing rather than defensive.
    rhosol = numerics.safe_divide(rhotmp, denom, mask_sol)
    # `:584`, under `mask_sol` -- a DIFFERENT mask from `:434`'s. At a
    # mask_nosol point the water is overridden and the density is not.
    rhosol = jnp.where(mask_sol & strat, rhosol_strat, rhosol)

    dvol_col = dvol[:, imode]
    wvol = jnp.zeros(jnp.shape(mask), dtype=jnp.float64)
    pvol: dict[int, Array] = {}
    for icp in range(tables.ncp):
        if not tables.component[imode, icp]:
            # `pvol` is the only conditionally written output. A non-member
            # component is never assigned by the Fortran and keeps whatever the
            # caller had; both callers pre-zero it
            # (glomap_box_state_mod.F90:86, and leaf_volume_mode), so this port
            # leaves the zero it started with. Recorded, not inferred.
            continue
        default = dvol_col * tables.mfrac_0[imode, icp]
        if tables.soluble[icp]:
            # `:593`: (md*mm_ovravc)/rhosol, left to right.
            value = numerics.safe_divide(
                md[:, imode, icp] * scales.mm_ovravc[icp], rhosol, mask_sol
            )
            column = jnp.where(mask_sol, value, default)
            wvol = wvol + jnp.where(mask_sol, column, 0.0)
            # `:601`, AFTER the accumulation, and on mask_nosol only: an
            # mdsol < 0 row keeps the default the ELSE WHERE just wrote.
            column = jnp.where(mask_nosol, 0.0, column)
        else:
            column = jnp.where(mask, md[:, imode, icp] * scales.mm_ovravcrhocp[icp], default)
            wvol = wvol + jnp.where(mask, column, 0.0)
        pvol[icp] = column

    # `:623`: (mdwat*mmwovravc)/rhosol. Water is added to wvol LAST.
    pvol_wat = numerics.safe_divide(mdwat_col * scales.mmwovravc, rhosol, mask_sol)
    wvol = jnp.where(mask_sol, wvol + pvol_wat, dvol_col)
    pvol_wat = jnp.where(mask_sol, pvol_wat, 0.0)
    rhopar = jnp.where(mask_sol, numerics.safe_divide(rhotmp2, denom2, mask_sol), RHO_SO4)

    return {
        "mdwat": mdwat_col,
        "wvol": wvol,
        "rhopar": rhopar,
        "pvol": pvol,
        "pvol_wat": pvol_wat,
        "mask": mask,
        "mask_sol": mask_sol,
        "mask_nosol": mask_nosol,
    }


def _insoluble_mode(
    tables: ModeTables,
    imode: int,
    nd: Array,
    md: Array,
    dvol: Array,
    scales: _Scales,
) -> dict:
    """One insoluble mode (`:638-673`).

    See the module docstring for the three ways this is not the soluble branch
    with the water removed.
    """
    mask = nd[:, imode] > tables.num_eps[imode]
    dvol_col = dvol[:, imode]

    pvol: dict[int, Array] = {}
    for icp in range(tables.ncp):
        if tables.component[imode, icp]:
            # `:647`: no WHERE and no dvol*mfrac_0 default. Assigned on every
            # point, including where nd <= num_eps.
            pvol[icp] = md[:, imode, icp] * scales.mm_ovravcrhocp[icp]

    # `:656-671`. Seeded at 0.0 -- no water term here -- and folded in icp order
    # under `mask`.
    rhotmp2 = jnp.zeros(jnp.shape(mask), dtype=jnp.float64)
    denom2 = jnp.zeros(jnp.shape(mask), dtype=jnp.float64)
    for icp in range(tables.ncp):
        if not tables.component[imode, icp]:
            continue
        rhotmp2 = rhotmp2 + jnp.where(mask, md[:, imode, icp] * scales.mm_rhocp[icp], 0.0)
        denom2 = denom2 + jnp.where(mask, md[:, imode, icp] * tables.mm[icp], 0.0)

    # `:665`. The guard is `mask`, a different one from `:579` and `:627`, and
    # it guards on the MASK and not on `denom2 != 0`: a masked point with
    # denom2 exactly zero divides here, in the port as in the reference, because
    # this branch has no diagnostic to stop it.
    rhopar = jnp.where(mask, numerics.safe_divide(rhotmp2, denom2, mask), RHO_SO4)

    zero = jnp.zeros(jnp.shape(mask), dtype=jnp.float64)
    return {
        "mdwat": zero,
        "wvol": dvol_col,
        "rhopar": rhopar,
        "pvol": pvol,
        "pvol_wat": zero,
        "mask": mask,
    }


def _inactive_mode(tables: ModeTables, imode: int, dvol: Array) -> dict:
    """One absent mode (`:675-690`).

    `pvol_wat = 0`, `wvol = dvol`, `mdwat = 0`, `rhopar = rho_so4`, all
    unconditional -- no `mask` is even computed here. The `pvol` default at
    `:686` is dead code; it is written anyway, and the tests assert it can never
    execute.
    """
    dvol_col = dvol[:, imode]
    zero = jnp.zeros_like(dvol_col)
    pvol = {
        icp: dvol_col * tables.mfrac_0[imode, icp]
        for icp in range(tables.ncp)
        if tables.component[imode, icp]
    }
    return {
        "mdwat": zero,
        "wvol": dvol_col,
        "rhopar": jnp.full_like(dvol_col, RHO_SO4),
        "pvol": pvol,
        "pvol_wat": zero,
    }


def partial_volumes(
    tables: ModeTables,
    nd: Array,
    md: Array,
    rh: Array,
    dvol: Array,
    t: Array,
    pmid: Array,
    s: Array,
    *,
    fix_water_content: bool,
    fix_neg_pvol_wat: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    """`(mdwat, wvol, rhopar, pvol, pvol_wat)` for every mode.

    The mode loop of `:310-692`, all three branches. `wetdp` is not here: it is
    computed after the loop closes, over the whole array at once, and belongs to
    task 45.

    `t`, `pmid` and `s` enter for one purpose: the `ukca_vapour` call at `:287`,
    whose two results are read only inside the two `pmid < putls` blocks. The
    call is made once, outside the mode loop, exactly as `:288` does -- `wts`
    and `rhosol_strat` are independent of particle size and composition.
    """
    nd = jnp.asarray(nd, dtype=jnp.float64)
    md = jnp.asarray(md, dtype=jnp.float64)
    dvol = jnp.asarray(dvol, dtype=jnp.float64)
    corrh = corrected_humidity(rh)
    scales = _Scales(tables)
    # `:286`: rp is a dummy 100.0e-9 and reaches no output of ukca_vapour, so
    # the port does not pass it -- see physics/vapour.py.
    wts, rhosol_strat = vapour(t, pmid, s, fix_neg_pvol_wat=fix_neg_pvol_wat)
    strat = stratospheric(pmid)
    nbox = nd.shape[0]

    mdwat_cols, wvol_cols, rhopar_cols, pvol_wat_cols = [], [], [], []
    pvol = jnp.zeros((nbox, NMODES, tables.ncp), dtype=jnp.float64)
    for imode in range(NMODES):
        if not bool(tables.mode[imode]):
            out = _inactive_mode(tables, imode, dvol)
        elif tables.modesol[imode] != 1:
            out = _insoluble_mode(tables, imode, nd, md, dvol, scales)
        else:
            out = _soluble_mode(
                tables,
                imode,
                nd,
                md,
                dvol,
                corrh,
                scales,
                strat,
                wts,
                rhosol_strat,
                fix_water_content=fix_water_content,
            )
        mdwat_cols.append(out["mdwat"])
        wvol_cols.append(out["wvol"])
        rhopar_cols.append(out["rhopar"])
        pvol_wat_cols.append(out["pvol_wat"])
        for icp, column in out["pvol"].items():
            pvol = pvol.at[:, imode, icp].set(column)

    return (
        jnp.stack(mdwat_cols, axis=1),
        jnp.stack(wvol_cols, axis=1),
        jnp.stack(rhopar_cols, axis=1),
        pvol,
        jnp.stack(pvol_wat_cols, axis=1),
    )


# ---------------------------------------------------------------------------
# Task 45 -- wetdp, and the whole routine
# ---------------------------------------------------------------------------


def six_over_pi_x(x: Array) -> Array:
    """`sixovrpix = 1.0/(x*(pi/6.0))` (`:292-293`).

    **Not** `drydiam.six_over_pi_x`, which is `6.0/(pi*x)` from `:230` of the
    other routine. Same `x`, 2 ulp apart for two of the width parameters, and
    the cube root turns that into a different `wetdp` on more than half the
    domain. Never factor the two into one helper.
    """
    piovrsix = PI / 6.0
    return 1.0 / (jnp.asarray(x, dtype=jnp.float64) * piovrsix)


def wet_diameter(tables: ModeTables, wvol: Array, drydp: Array) -> Array:
    """`wetdp` (`:694-700`).

    `cubrt_v` runs over the whole `(nbox, nmodes)` array -- `:696` passes
    `nmodes*nbox` and relies on sequence association -- and `:698` then
    overwrites the inactive columns with `drydp`. Reproduced in that order: the
    cube root is taken everywhere and thrown away where the mode is absent.
    """
    sixovrpix = six_over_pi_x(tables.x)
    tmp1 = sixovrpix[None, :] * jnp.asarray(wvol, dtype=jnp.float64)
    wetdp = numerics.cbrt(tmp1)
    drydp = jnp.asarray(drydp, dtype=jnp.float64)
    for imode in range(NMODES):
        if not bool(tables.mode[imode]):
            wetdp = wetdp.at[:, imode].set(drydp[:, imode])
    return wetdp


def volume_mode(
    tables: ModeTables,
    nd: Array,
    md: Array,
    mdt: Array,
    rh: Array,
    dvol: Array,
    drydp: Array,
    t: Array,
    pmid: Array,
    s: Array,
    *,
    fix_water_content: bool,
    fix_neg_pvol_wat: bool,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """`(mdwat, wvol, wetdp, rhopar, pvol, pvol_wat)`, in the Fortran's order.

    `mdt` is accepted and not used, exactly as the reference uses it: its only
    reader in this routine is the diagnostic `WRITE` at `:842`, inside the
    unreachable five-way block. It is in the signature so a caller cannot pass
    the arguments in the wrong order without noticing, and so the omission is
    visible here rather than inferred.

    None of the three `ereport` blocks is ported; see the module docstring for
    which, and why each.
    """
    del mdt  # `:842` only, inside the block that cannot run. See the docstring.
    mdwat_, wvol_, rhopar_, pvol_, pvol_wat_ = partial_volumes(
        tables,
        nd,
        md,
        rh,
        dvol,
        t,
        pmid,
        s,
        fix_water_content=fix_water_content,
        fix_neg_pvol_wat=fix_neg_pvol_wat,
    )
    wetdp_ = wet_diameter(tables, wvol_, drydp)
    return mdwat_, wvol_, wetdp_, rhopar_, pvol_, pvol_wat_
