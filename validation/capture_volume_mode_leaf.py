#!/usr/bin/env python3
"""Sweep `ukca_volume_mode` over the inputs the physics can reach (task 35e).

    python validation/capture_volume_mode_leaf.py --dry-run
    python validation/capture_volume_mode_leaf.py    # writes tests/goldens/

`ukca_volume_mode` turns (nd, md, rh, t, pmid, s) into the wet particle -- water
content, wet volume, wet diameter, particle density and the per-component
partial volumes. It is the widest leaf in phase D and the only one that is both
setup-dependent *and* environment-dependent, so it is the only place several
whole branches of the routine can be reached at all.

Why this grid, branch by branch
-------------------------------

**The stratospheric override has never executed.** `putls = 1.5e4` (`:258`),
and the four shipped namelists run `pressure` in {1e5, 2e4, 1e5, 1e5} -- so
`WHERE (mask .AND. pmid < putls)` at `:434-438`, which throws away the ZSR
water content and rebuilds `mdwat` from `wts`, and `WHERE (mask_sol .AND.
pmid < putls)` at `:584-586`, which replaces the solution density with
`rhosol_strat`, have run zero times in any validated trajectory. The `pmid`
axis therefore straddles `putls` and includes `1.5e4` itself with both
neighbouring doubles, because the predicate is a strict `<` and `1.5e4` must
come out FALSE.

That axis is swept **inside one call with a mixed column**, not one call per
pressure. In the box model `pmid` is a per-run scalar, so a per-run sweep
cannot distinguish "the override is applied at the points where it should be"
from "the override is applied to the whole call". A mixed column can, and
`check_axes` refuses a grid whose main block does not contain rows below, at,
and above the threshold.

**`mask_nosol` has never executed either.** Reconstructing `mdsol` from `nd`,
`md` and the setup's `component`/`soluble` tables at every `volume_mode`
snapshot in the four committed state goldens -- 768 snapshots, 2447 points where
`nd > num_eps` -- gives 0 `mdsol == 0` and 0 `mdsol < 0`. Reaching it needs
`md(:,m,cp_su)` and `md(:,m,cp_cl)` both exactly 0.0 with `nd > num_eps` -- the
`nosol` composition variant -- and it is only *safe* to reach in a mode that
still carries an insoluble component, because a mode whose mass is entirely
soluble would have `dvol = 0` and trip `ukca_calc_drydiam`'s own guard before
`volume_mode` is ever called. Setups 1
and 6 therefore cannot reach it at all and are recorded as expected-zero: every
active soluble mode of setup 1 carries only `su` and `cl`, and setup 6 has no
active soluble mode.

**The relative-humidity clamps at `:306-307` have never fired.** The highest
`rel_humid` in any namelist is exactly 0.90 and the test is strict `>`. The
`rh` axis is dense on [0.1, 0.9], carries 0.1 and 0.9 exactly with both
neighbouring doubles, and steps outside to 0.05 and 0.95.

**`mdsol < 0` is a third state with no name in the source.** `mask_sol` and
`mask_nosol` are `mdsol > 0` and `mdsol == 0`; a negative soluble mass is
neither, and the row then takes the `ELSE WHERE` arms at `:597` and `:631`
without ever being `mask_nosol`. The `negsu` variant reaches it.

**`t` and `s` reach the outputs only through the stratosphere.** `t`, `pmid`
and `s` enter `ukca_volume_mode` for exactly one purpose -- the `ukca_vapour`
call at `:287` -- and its two results, `wts` and `rhosol_strat`, are read only
inside the two `pmid < putls` blocks. So the `t` and `s` axes are swept at
stratospheric pressure, where they matter, and the `t` axis is chosen to walk
`(NINT(wts/5))*5` across every one of `ukca_vapour`'s twelve `percent` entries
and past the top of the table into the `rhosol_strat = 1300.0` fall-through.

Measured, not assumed
---------------------

Three things this capture asserts about the Fortran, each of which was measured
here and each of which would be a finding if it changed:

* `l_fix_ukca_water_content` moves `mdwat` in **six of the seven** setups, and
  not for the reason the coefficient patch suggests. Its `y(1,-3,6)` fix at
  `:235` really is unreachable here -- `cl(:,-3)` is assigned only at
  `:409`/`:415` inside `IF (UBOUND(component,DIM=2) >= cp_no3)` and
  `ncp = 6 < cp_no3 = 7` in every supported setup -- but the flag has a second
  arm at `:271-322` that is not about nitrate at all. In the unfixed arm `aw` is
  clamped to `rh_min(ic,ia)` *cumulatively* across the ion-pair loop, so by the
  time Na+/Cl- (`rh_min = 47`) is reached `aw` has already been ratcheted to
  `0.62` by NH4+/NO3-, whose ions are not even present. Every mode carrying
  `cp_cl` therefore gets a different water content at `rh < 0.62`. The one
  setup where the two settings must agree is 6, which has no active soluble
  mode and so never calls `ukca_water_content_v` -- named in
  FIX_WATER_IDENTICAL and *required* by `check_varied` rather than assumed away.
  Both settings get their own subprocess regardless: the flag is a one-way latch
  on a SAVEd table (#22).
* `l_fix_neg_pvol_wat` reaches an output **only** through the stratospheric
  branch. It changes `wts` (`ukca_vapour.F90:184` vs `:188`), and `wts` is read
  only at `:436`. Its own guard at `:882-898` is a check, not a computation.
  Hence the dedicated strat block at both settings.
* `denom <= 0` with `mask_sol` is unreachable in the **troposphere** and only
  reachable in the **stratosphere**, which is the opposite of the obvious guess.
  Below `putls` the ZSR water is `0.0886*md_su` for H2SO4 and `0.04314*md_cl`
  for NaCl (measured here, at `rh <= 0.3` where the NaCl pair sits on its
  `molal_max`), and `mm_cl + 0.04314 = 0.10158 > mm_su = 0.098` closes both
  mixed-sign arrangements by 3.6%. Above it, `:436` replaces the water with
  `(100/wts - 1)*md_su*mm_su/mmw`, so `denom = md_su*mm_su*(100/wts) +
  md_cl*mm_cl` and a negative `md_su` opens the window. The full derivation is
  above `_ABORT_BODY`.

The grids are exactly reproducible
----------------------------------

No abscissa comes from a libm call. Every grid value is either a decimal string
literal (`float("1.4999e4")` -- IEEE 754 requires decimal-to-binary conversion
to be correctly rounded, so every machine parses it identically), integer
arithmetic, or `np.nextafter`, which is exact. `np.logspace` is `10.0 **
linspace(...)`, i.e. a libm `pow` per point, and is not used: four abscissae of
the numerics sweep were themselves platform-dependent before that rule existed.
The composition grids are built from `mmid` and `mfrac_0`, which come out of the
Fortran, so they move only if the mode tables move -- which `modes.f64.tables`
already gates.

What is NOT in this grid, and why
---------------------------------

* An exact `NINT(wts/5)` tie. `wts` is a transcendental function of `t` and `s`
  (`LOG`, `SQRT`, `**`), so a `t` that lands `wts/5` exactly on a half-integer
  on this host would not do so on another, and the tie would be a platform
  accident rather than a grid point. The tie behaviour is pinned where it is
  reproducible: `capture_leaf.py`'s `vapour_round` grid.
* `i_mode_setup == 11` (`:357`, `:366`, `:380`, `:395`). `glomap_box_config_mod`
  has no CASE for setup 11, so the box model cannot construct it; recorded as
  expected-zero.
* The nitrate block at `:402-419`. `ncp = 6` in every supported setup and
  `cp_no3 = 7`, so `UBOUND(component,DIM=2) >= cp_no3` is false everywhere;
  recorded as expected-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_modes import render_namelist
from leaf_common import NAMELISTS, check_varied, run_child

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "volume_mode.f64.leaf.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)
FIX_WATER = (0, 1)
FIX_NEG_PVOL = (0, 1)

# The two switch combinations worth a volume_mode record. Both move a density
# this routine divides by, and both are invisible in some setups: `nacl_off`
# changes rhocomp(cp_cl), which setup 6 never touches, and `bc_mg_mix` changes
# rhocomp(cp_bc), which no active mode of setup 1 carries. Setup 2 carries both
# cl (modes 3-4) and bc (modes 2-5), so it is the one setup where each change
# is decidable from rhopar and pvol alone. Note the default rhocomp already
# makes bc and oc indistinguishable at 1500.0 each, which is why the record
# that moves bc is worth having at all.
COMBOS = ("nacl_off", "bc_mg_mix")
COMBO_SETUP = 2
COMBO_FIX_WATER = 1

# Component slots, 0-based (Fortran cp_su=1 .. cp_so=6). ncp is 6 in every
# supported setup, which is why cp_no3=7, cp_nn=8 and cp_nh4=9 have no slot.
# Which of them are soluble is read out of the Fortran, never assumed here.
CP_SU, CP_BC, CP_OC, CP_CL, CP_DU, CP_SO = range(6)

# Relative mass weights for the `typical` composition, as decimal literals.
# Normalised in the child over whichever components the setup actually gives
# the mode, so one recipe produces a sensible particle for all seven setups.
# `mfrac_0` is NOT used for this: in every supported setup it is [1,0,0,0,0,0]
# for modes 1-3 and [0,0,0,1,0,0] for mode 4, i.e. zero for bc, oc and so in
# every SOLUBLE mode, so a mfrac_0-weighted particle would carry no insoluble
# mass where it matters and could never reach mask_nosol.
TYPICAL_WEIGHTS = ("0.50", "0.10", "0.20", "0.30", "0.40", "0.15")

PUTLS = 1.5e4  # ukca_volume_mode.F90:258
MMW = 0.0180154  # ukca_constants.F90:60, the molar mass of water

# The pmid axis. 1.5e4 is putls itself and must come out FALSE (`<`, not `<=`);
# its two neighbouring doubles bracket it.
PMID_AXIS = (
    1.0e4,
    1.4999e4,
    np.nextafter(PUTLS, 0.0),
    PUTLS,
    np.nextafter(PUTLS, np.inf),
    2.0e4,
    1.0e5,
)

# The rh axis. 0.1 and 0.9 are the clamp boundaries at :306-307 and the tests
# are strict, so both must appear exactly and with both neighbours; 0.05 and
# 0.95 are outside. validate_config admits [0, 1] (glomap_box_config_mod:287),
# so 0.05 and 0.95 are configurations the box model would accept -- the
# specification's claim that it admits "up to 0.95" understates the range.
RH_AXIS = (
    0.05,
    np.nextafter(0.1, 0.0),
    0.1,
    np.nextafter(0.1, 1.0),
    0.15,
    0.2,
    0.3,
    0.4,
    0.47,
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    np.nextafter(0.9, 0.0),
    0.9,
    np.nextafter(0.9, 1.0),
    0.95,
)

# The t axis, at stratospheric pressure where t reaches an output. Each value
# was measured to put (NINT(wts/5))*5 in a different bin: 180.0 sits on the
# wts = 41.0 floor (round 40), 204.0 .. 288.5 walk percent(1..12), and 306.0,
# 320.0, 326.5 fall off the top of the table so rhosol_strat keeps 1300.0.
T_AXIS = (
    180.0,
    204.0,
    206.5,
    209.5,
    213.0,
    218.0,
    224.5,
    232.5,
    242.5,
    255.5,
    271.0,
    288.5,
    306.0,
    320.0,
    326.5,
)

# The s axis. bh2o = 1.609*s*pmid/p0 is clamped to [2e-8, 2e-6]
# (ukca_vapour.F90:141-142); 1.0e-8 is below the floor at every pressure here
# and 1.0e-2 above the ceiling, so both clamps fire.
S_AXIS = (1.0e-8, 2.0e-7, 2.0e-6, 1.0e-5, 4.0e-5, 1.0e-4, 3.0e-3, 1.0e-2)

# Reference points for the one-factor-at-a-time rows.
T_REF, S_REF, RH_REF = 213.0, 1.0e-2, 0.6
P_TROP, P_STRAT = 1.0e5, 1.0e4

# Composition variants. Applied per mode, and skipped for a mode where they
# would be unreachable or would void the call -- the child records which.
VARIANTS = ("typical", "su_only", "cl_rich", "insol_rich", "nosol", "negsu", "tiny")

# nd variants. num_eps is the mask threshold at :312 and the test is strict, so
# num_eps itself must come out FALSE and only the double above it TRUE.
ND_KINDS = ("bulk", "eps_below", "eps_exact", "eps_above", "zero")

# Predicates counted in every block, in archive order. `hits` is committed
# inside the archive and every entry is asserted non-zero unless it is named in
# EXPECTED_ZERO with a reason.
PREDICATES = (
    "corrh_clamp_high",  # :306  rh > 0.9
    "corrh_clamp_low",  # :307  rh < 0.1
    "corrh_unclamped",
    "mask_true",  # :312  nd > num_eps
    "mask_false",
    "mode_soluble",  # :314  modesol == 1
    "mode_insoluble",  # :638
    "mode_absent",  # :675
    "mask_sol",  # :329  mdsol > 0
    "mask_nosol",  # :330  mdsol == 0
    "mdsol_negative",  # the unnamed third state
    "strat_mdwat_override",  # :434  mask .AND. pmid < putls
    "strat_rhosol_override",  # :584  mask_sol .AND. pmid < putls
    "trop_no_override",  # :434 false
    "at_putls_not_overridden",  # pmid == putls exactly, must be FALSE
    "cp_su_present",  # :366
    "cp_cl_present",  # :395
    "cp_oc_present",  # :380
    "cp_so_present",  # :371
    "pvol_soluble_default",  # :597  .NOT. mask_sol on a soluble cpt
    "pvol_insoluble_default",  # :613
    "undersize_reset",  # ukca_calc_drydiam:250
    "wts_floor_41",  # ukca_vapour:184/:188  MAX(41.0, ...)
    "wts_at_99_ceiling",  # ukca_vapour:184  the l_fix_neg_pvol_wat clamp, active
    "wts_above_99",  # ukca_vapour:188  what that clamp would have caught
    "bh2o_clamp_low",  # ukca_vapour:141
    "bh2o_clamp_high",  # ukca_vapour:142
    "rhosol_strat_lut_hit",  # ukca_vapour:233  round == percent(k)
    "rhosol_strat_fallthrough",  # :223  no match, stays 1300.0
    "setup_11_branch",  # :357 / :366 / :380 / :395
    "nitrate_branch",  # :402
    "denom_le_zero",  # :476/:481
    "denom2_le_zero",  # :533
    "minval_guard",  # :704-708
    "neg_pvol_wat_guard",  # :884
)

# Predicates that must be zero, each with the reason it cannot be reached.
# A non-zero count here is as much a finding as a zero count elsewhere.
EXPECTED_ZERO = {
    "setup_11_branch": (
        "glomap_box_config_mod's init_indices has no CASE for i_mode_setup 11, "
        "so the box model cannot construct it"
    ),
    "nitrate_branch": (
        "ncp = 6 in every supported setup and cp_no3 = 7, so "
        "UBOUND(component,DIM=2) >= cp_no3 is false everywhere"
    ),
    "minval_guard": "no swept row drives wetdp/drydp/wvol/dvol/rhopar to <= 0",
    "neg_pvol_wat_guard": (
        "no swept row has both l_fix_neg_pvol_wat on and a negative md(cp_su) "
        "below putls; the `negsu` variant is tropospheric for exactly that "
        "reason, since :436 gives a negative mdwat for any wts <= 100 and the "
        "99% clamp does not help"
    ),
    "denom_le_zero": "reached only by the deliberate-abort block, counted separately",
    "denom2_le_zero": "reached only by the deliberate-abort block, counted separately",
}

# Predicates required to be non-zero for EVERY setup, not merely in the total.
# A total-only check cannot notice a grid that lost the exact-putls point from
# the main block while the strat block still carried one -- measured, by
# removing it and watching the capture stay green.
PER_SETUP_REQUIRED = (
    "corrh_clamp_high",
    "corrh_clamp_low",
    "mask_true",
    "mask_false",
    "mode_absent",
)
# The same, but only meaningful where the setup has an active soluble mode.
PER_SOLUBLE_SETUP_REQUIRED = (
    "mask_sol",
    "strat_mdwat_override",
    "strat_rhosol_override",
    "trop_no_override",
    "at_putls_not_overridden",
)
NO_SOLUBLE_MODE = {6: "mode = F,F,F,F,F,T,T,F against modesol = 1,1,1,1,0,0,0,0"}

# Setups that cannot reach mask_nosol however the composition is chosen, with
# the reason. Recorded rather than omitted: if a future setup table changes,
# the count moves off zero and check_hits refuses to write.
NOSOL_UNREACHABLE = {
    1: (
        "every active soluble mode of setup 1 carries only cp_su and cp_cl, both "
        "soluble, so zeroing them leaves dvol = 0 and ukca_calc_drydiam aborts first"
    ),
    6: "setup 6 has no active soluble mode (mode = F,F,F,F,F,T,T,F)",
}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def check_axes() -> None:
    """The grid points whose *identity* is the point, not just their presence.

    `putls` itself must be on the pmid axis, because the predicate is a strict
    `<` and the only way to show it is strict is to sit on it; the same for the
    rh clamps, whose tests are strict `>` and `<`. A grid that lost one of these
    would still reach the branch from a neighbouring point and every hit count
    would stay non-zero, so the identity is asserted directly.
    """
    problems = []
    if PUTLS not in PMID_AXIS:
        problems.append(f"pmid axis does not contain putls = {PUTLS!r} exactly")
    for side in (0.0, np.inf):
        nb = np.nextafter(PUTLS, side)
        if nb not in PMID_AXIS:
            problems.append(f"pmid axis is missing putls' neighbour {nb!r}")
    if not any(p < PUTLS for p in PMID_AXIS) or not any(p > PUTLS for p in PMID_AXIS):
        problems.append("pmid axis does not straddle putls")
    for bound in (0.1, 0.9):
        if bound not in RH_AXIS:
            problems.append(f"rh axis does not contain the clamp bound {bound!r} exactly")
        for side in (0.0, 1.0):
            nb = np.nextafter(bound, side)
            if nb not in RH_AXIS:
                problems.append(f"rh axis is missing {bound!r}'s neighbour {nb!r}")
    if not any(r > 0.9 for r in RH_AXIS) or not any(r < 0.1 for r in RH_AXIS):
        problems.append("rh axis never leaves [0.1, 0.9], so neither clamp fires")
    rows = row_grid()
    pmids = [r["pmid"] for r in rows]
    if PUTLS not in pmids:
        problems.append("no main row sits exactly on putls")
    if not (any(p < PUTLS for p in pmids) and any(p > PUTLS for p in pmids)):
        problems.append(
            "the main block is one call, and it must contain rows on both sides "
            "of putls -- a single-sided column cannot show the override is "
            "applied per point rather than per call"
        )
    if problems:
        raise SystemExit("grid axes:\n  " + "\n  ".join(problems))


def _row(t, pmid, s, rh, variant="typical", nd_kind="bulk"):
    return {"t": t, "pmid": pmid, "s": s, "rh": rh, "variant": variant, "nd": nd_kind}


def row_grid() -> list[dict]:
    """The main block: a small factorial core plus one factor at a time.

    A full cross product is not affordable and is not the right shape either --
    T(15) x p(7) x RH(18) x s(8) x variants(6) x nd(5) x 7 setups is 4.8e7 rows.
    What the routine actually needs is every branch reached, every axis walked,
    and both sides of `putls` present in the same column so the override can be
    shown to be per point.
    """
    rows: list[dict] = []

    # (a) the pmid axis at two humidities, so every block straddles putls.
    for pmid in PMID_AXIS:
        for rh in (0.3, 0.9):
            rows.append(_row(T_REF, pmid, S_REF, rh))

    # (b) the pmid axis again in the wts > 99 corner, where l_fix_neg_pvol_wat
    #     is the difference between a positive and a negative strat mdwat.
    for pmid in PMID_AXIS:
        rows.append(_row(288.0, pmid, 1.0e-8, RH_REF))

    # (c) the rh axis. Tropospheric: in the stratosphere mdwat is overwritten
    #     at :434 before rh can reach an output, which the two strat rows below
    #     are here to demonstrate.
    for rh in RH_AXIS:
        rows.append(_row(T_REF, P_TROP, S_REF, rh))
    for rh in (0.2, 0.8):
        rows.append(_row(T_REF, P_STRAT, S_REF, rh))

    # (d) the t axis, at stratospheric pressure where t reaches an output.
    for t in T_AXIS:
        rows.append(_row(t, P_STRAT, S_REF, RH_REF))

    # (e) the s axis, likewise, at a temperature high enough that the bh2o
    #     ceiling and floor give different wts.
    for s in S_AXIS:
        rows.append(_row(303.65, P_STRAT, s, RH_REF))

    # (f) the nd axis on both sides of putls -- num_eps exactly must be FALSE.
    for kind in ND_KINDS:
        for pmid in (P_TROP, P_STRAT):
            rows.append(_row(T_REF, pmid, S_REF, RH_REF, nd_kind=kind))

    # (g) composition variants on both sides of putls and both sides of the
    #     ZSR-relevant humidity range.
    #
    #     `negsu` is tropospheric only, and that is a measurement rather than a
    #     preference. A negative md(cp_su) in the stratosphere gives
    #     mdwat = (100/wts - 1)*md_su*mm_su/mmw < 0 at :436 for ANY wts <= 100,
    #     so the l_fix_neg_pvol_wat clamp does not save it and the guard at :884
    #     fires -- correctly, since a negative water mass is exactly what that
    #     guard exists to catch. One such row voids the whole call, so the
    #     mdsol < 0 state is reached where it can be recorded.
    for variant in VARIANTS:
        if variant == "typical":
            continue
        pressures = (P_TROP,) if variant == "negsu" else (P_TROP, P_STRAT)
        for pmid in pressures:
            for rh in (0.3, 0.9):
                rows.append(_row(T_REF, pmid, S_REF, rh, variant=variant))

    return rows


def strat_grid() -> list[dict]:
    """The l_fix_neg_pvol_wat block: the only rows where that flag moves a number.

    `wts` is the flag's whole numerical effect and `wts` is read only at `:436`,
    inside `pmid < putls`. Every row here is therefore stratospheric except the
    control rows at 1.0e5, which must come out identical at both settings.
    """
    rows: list[dict] = []
    hot = (288.0, 298.0, 303.65, 310.0, 320.0)  # wts > 99 at s = 1e-8
    for t in hot:
        for pmid in (P_STRAT, PUTLS, np.nextafter(PUTLS, np.inf), P_TROP):
            rows.append(_row(t, pmid, 1.0e-8, RH_REF))
    for t in (213.0, 242.5):  # wts well below 99: the flag must NOT move these
        for pmid in (P_STRAT, P_TROP):
            rows.append(_row(t, pmid, S_REF, RH_REF))
    return rows


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------

# Everything below runs in a subprocess, one per configuration: ukca_mode_setup
# never deallocates and l_fix_ukca_water_content patches a SAVEd table in place
# and never restores it (#22), so a switch swept in one process compares a
# configuration against itself and passes.
_BODY = """
sys.path.insert(0, {validation!r})
from leaf_common import bind_call
call = bind_call(g)

ROWS = json.loads({rows!r})
PUTLS = {putls!r}
MMW = {mmw!r}
CP_SU, CP_BC, CP_OC, CP_CL, CP_DU, CP_SO = range(6)
SOLUBLE_CP = (CP_SU, CP_CL)
WEIGHTS = np.array([float(w) for w in {weights!r}])
PREDICATES = {predicates!r}

# The Fortran's own opinion of which setup it is running, read a second way:
# CHILD_PREAMBLE already checked wrap_get_config_flags, this is wrap_sizes.
_sizes = g.wrap_sizes()
assert int(_sizes[-1]) == 0, ("wrap_sizes", _sizes[-1])
nbox_cfg, nmodes, ncp = int(_sizes[0]), int(_sizes[1]), int(_sizes[2])
assert int(_sizes[7]) == _setup, ("wrap_sizes setup", int(_sizes[7]), _setup)

def _mode_real(f):
    v, e = g.wrap_mode_real(f, nmodes); assert int(e) == 0, (f, e); return v
def _mode_int(f):
    v, e = g.wrap_mode_int(f, nmodes); assert int(e) == 0, (f, e); return v
def _cp_real(f):
    v, e = g.wrap_cp_real(f, ncp); assert int(e) == 0, (f, e); return v
def _cp_int(f):
    v, e = g.wrap_cp_int(f, ncp); assert int(e) == 0, (f, e); return v

num_eps = _mode_real("num_eps")
mmid    = _mode_real("mmid")
mode    = _mode_int("mode").astype(bool)
modesol = _mode_int("modesol")
mm      = _cp_real("mm")
soluble = _cp_int("soluble").astype(bool)
comp, e = g.wrap_mode_cp_int("component", nmodes, ncp); assert int(e) == 0
comp = comp.astype(bool)
mfrac0, e = g.wrap_mode_cp_real("mfrac_0", nmodes, ncp); assert int(e) == 0

nrows = len(ROWS)
t    = np.array([r["t"] for r in ROWS])
pmid = np.array([r["pmid"] for r in ROWS])
s    = np.array([r["s"] for r in ROWS])
rh   = np.array([r["rh"] for r in ROWS])

# ---- nd -------------------------------------------------------------------
nd = np.zeros((nrows, nmodes))
for i, r in enumerate(ROWS):
    kind = r["nd"]
    if kind == "bulk":
        nd[i, :] = 1000.0
    elif kind == "zero":
        nd[i, :] = 0.0
    elif kind == "eps_exact":
        nd[i, :] = num_eps
    elif kind == "eps_below":
        nd[i, :] = np.nextafter(num_eps, 0.0)
    elif kind == "eps_above":
        nd[i, :] = np.nextafter(num_eps, np.inf)
    else:
        raise SystemExit("unknown nd kind " + kind)

# ---- md -------------------------------------------------------------------
# One recipe per mode, normalised over the components the setup gives it. A
# variant that would leave the mode with no dry volume at all, or with no
# insoluble mass to carry a nosol/negsu row, falls back to `typical` and is
# recorded in variant_applied so the hit counts stay honest.
variant_applied = [[""] * nmodes for _ in range(nrows)]
md = np.zeros((nrows, nmodes, ncp))
for m in range(nmodes):
    present = np.where(comp[m])[0]
    if len(present) == 0:
        continue
    insol = [c for c in present if not soluble[c]]
    sol   = [c for c in present if soluble[c]]
    w = np.zeros(ncp); w[present] = WEIGHTS[present]
    base = mmid[m] * w / w.sum()
    for i, r in enumerate(ROWS):
        want = r["variant"]
        got = want
        if want == "typical":
            md[i, m, :] = base
        elif want == "su_only":
            if CP_SU in present:
                md[i, m, CP_SU] = mmid[m]
            else:
                md[i, m, :] = base; got = "typical"
        elif want == "cl_rich":
            if CP_CL in present and CP_SU in present:
                md[i, m, CP_CL] = mmid[m] * 0.9
                md[i, m, CP_SU] = mmid[m] * 0.1
            else:
                md[i, m, :] = base; got = "typical"
        elif want == "insol_rich":
            if insol:
                md[i, m, :] = base
                for c in sol:
                    md[i, m, c] = base[c] * 0.01
            else:
                md[i, m, :] = base; got = "typical"
        elif want == "nosol":
            # mask_nosol needs md(su) and md(cl) EXACTLY 0.0 with dvol > 0,
            # which only a mode carrying an insoluble component can supply.
            if insol and modesol[m] == 1:
                for c in insol:
                    md[i, m, c] = mmid[m] / len(insol)
                for c in sol:
                    md[i, m, c] = 0.0
            else:
                md[i, m, :] = base; got = "typical"
        elif want == "tiny":
            # Under ddplim0*0.1, so ukca_calc_drydiam's undersize reset at :250
            # rewrites md and mdt and volume_mode is driven by the rewrite.
            md[i, m, :] = base * 1.0e-9
        elif want == "negsu":
            # mdsol < 0: the state that is neither mask_sol nor mask_nosol.
            if insol and CP_SU in present:
                for c in insol:
                    md[i, m, c] = mmid[m] / len(insol)
                for c in sol:
                    md[i, m, c] = 0.0
                md[i, m, CP_SU] = -mmid[m] * 0.01
            else:
                md[i, m, :] = base; got = "typical"
        else:
            raise SystemExit("unknown variant " + want)
        variant_applied[i][m] = got
mdt = md.sum(axis=2)

# ---- the two driver calls -------------------------------------------------
drydp, dvol, md_out, mdt_out, ierr = call(
    "leaf_drydiam", g.leaf_drydiam, nd, md, mdt)
mdwat, wvol, wetdp, rhopar, pvol, pvol_wat, ierr = call(
    "leaf_volume_mode", g.leaf_volume_mode,
    nd, md_out, mdt_out, rh, dvol, drydp, t, pmid, s)

# `wts` and `rhosol_strat` as ukca_volume_mode itself computes them: the same
# routine, on the same (t, pmid, s), with the same dummy rp(:) = 100.0e-9
# (:286). Used to CHECK the strat override against the Fortran's own mdwat,
# not to replace it.
rp = np.full(nrows, 100.0e-9)
wts, rhosol_strat, ierr = call("leaf_vapour", g.leaf_vapour, t, pmid, s, rp)

# ---- nbox = 1 replication -------------------------------------------------
# Byte equality, row by row. ukca_volume_mode has no cross-row coupling that
# can reach an output -- SUM(ierr) and the MINVAL guards gate ereports only --
# so a difference here would be a finding about the routine, not about the grid.
nbox1_bad = []
nbox1 = {{"mdwat": [], "wvol": [], "wetdp": [], "rhopar": [], "pvol": [],
          "pvol_wat": [], "drydp": [], "dvol": []}}
for i in range(nrows):
    sl = slice(i, i + 1)
    d1, v1, mo1, mt1, e1 = call(
        "leaf_drydiam(nbox=1)", g.leaf_drydiam, nd[sl], md[sl], mdt[sl])
    a1, b1, c1, r1, p1, w1, e1 = call(
        "leaf_volume_mode(nbox=1)", g.leaf_volume_mode,
        nd[sl], mo1, mt1, rh[sl], v1, d1, t[sl], pmid[sl], s[sl])
    for _k, _v in (("mdwat", a1), ("wvol", b1), ("wetdp", c1), ("rhopar", r1),
                   ("pvol", p1), ("pvol_wat", w1), ("drydp", d1), ("dvol", v1)):
        nbox1[_k].append(_v[0].tolist())
    same = (
        np.array_equal(d1, drydp[sl]) and np.array_equal(v1, dvol[sl])
        and np.array_equal(mo1, md_out[sl]) and np.array_equal(mt1, mdt_out[sl])
        and np.array_equal(a1, mdwat[sl]) and np.array_equal(b1, wvol[sl])
        and np.array_equal(c1, wetdp[sl]) and np.array_equal(r1, rhopar[sl])
        and np.array_equal(p1, pvol[sl]) and np.array_equal(w1, pvol_wat[sl])
    )
    if not same:
        nbox1_bad.append(i)

# ---- branch reconstruction ------------------------------------------------
# Every predicate is reconstructed from the inputs and then TIED to an output,
# so a reconstruction that drifted from the Fortran shows up as a failed
# cross-check rather than as a plausible hit count.
hits = dict.fromkeys(PREDICATES, 0)
strat = pmid < PUTLS

hits["corrh_clamp_high"] = int((rh > 0.9).sum())
hits["corrh_clamp_low"] = int((rh < 0.1).sum())
hits["corrh_unclamped"] = int(((rh >= 0.1) & (rh <= 0.9)).sum())
hits["setup_11_branch"] = int(nrows * nmodes) if _setup == 11 else 0
hits["nitrate_branch"] = int(nrows * nmodes) if ncp >= 7 else 0
hits["wts_floor_41"] = int((wts == 41.0).sum())
hits["wts_at_99_ceiling"] = int((wts == 99.0).sum())
hits["wts_above_99"] = int((wts > 99.0).sum())
bh2o = 1.609 * s * pmid / 101325.0
hits["bh2o_clamp_low"] = int((bh2o < 2.0e-8).sum())
hits["bh2o_clamp_high"] = int((bh2o > 2.0e-6).sum())
hits["rhosol_strat_lut_hit"] = int((rhosol_strat != 1300.0).sum())
hits["rhosol_strat_fallthrough"] = int((rhosol_strat == 1300.0).sum())
hits["undersize_reset"] = int((md_out != md).any(axis=2).sum())
hits["minval_guard"] = 0
hits["neg_pvol_wat_guard"] = 0
for m in range(nmodes):
    lo = min(np.min(wetdp[:, m]), np.min(drydp[:, m]), np.min(wvol[:, m]),
             np.min(dvol[:, m]), np.min(rhopar[:, m]))
    hits["minval_guard"] += int(lo <= 0.0)
    if _fn == 1:
        hits["neg_pvol_wat_guard"] += int(
            min(np.min(pvol_wat[:, m]), np.min(mdwat[:, m])) < 0.0)

checks = []
for m in range(nmodes):
    if not mode[m]:
        hits["mode_absent"] += nrows
        continue
    mask = nd[:, m] > num_eps[m]
    hits["mask_true"] += int(mask.sum())
    hits["mask_false"] += int((~mask).sum())
    for c in range(ncp):
        if comp[m, c]:
            if c == CP_SU: hits["cp_su_present"] += nrows
            if c == CP_CL: hits["cp_cl_present"] += nrows
            if c == CP_OC: hits["cp_oc_present"] += nrows
            if c == CP_SO: hits["cp_so_present"] += nrows
    if modesol[m] != 1:
        hits["mode_insoluble"] += nrows
        continue
    hits["mode_soluble"] += nrows
    mdsol = np.zeros(nrows)
    for c in range(ncp):
        if comp[m, c] and soluble[c]:
            mdsol = np.where(mask, mdsol + md_out[:, m, c], mdsol)
    m_sol = mask & (mdsol > 0.0)
    m_nosol = mask & (mdsol == 0.0)
    m_neg = mask & (mdsol < 0.0)
    hits["mask_sol"] += int(m_sol.sum())
    hits["mask_nosol"] += int(m_nosol.sum())
    hits["mdsol_negative"] += int(m_neg.sum())
    hits["strat_mdwat_override"] += int((mask & strat).sum())
    hits["strat_rhosol_override"] += int((m_sol & strat).sum())
    hits["trop_no_override"] += int((mask & ~strat).sum())
    hits["at_putls_not_overridden"] += int((mask & (pmid == PUTLS)).sum())
    for c in range(ncp):
        if not comp[m, c]:
            continue
        if soluble[c]:
            hits["pvol_soluble_default"] += int((~m_sol).sum())
        else:
            hits["pvol_insoluble_default"] += int((~mask).sum())

    # Cross-checks that tie the reconstruction to the Fortran's own output.
    if m_nosol.any():
        for c in range(ncp):
            if comp[m, c] and soluble[c]:
                checks.append(["nosol pvol", m, c,
                               bool((pvol[m_nosol, m, c] == 0.0).all())])
        checks.append(["nosol wvol", m, -1,
                       bool(np.array_equal(wvol[m_nosol, m], dvol[m_nosol, m]))])
    sm = mask & strat
    if sm.any() and comp[m, CP_SU]:
        # :435-437, with the avogadro of massh2so4kg and the one hidden in
        # mmwovravc cancelled: mdwat = (100/wts - 1)*md_su*mm_su/mmw.
        h2so4 = md_out[sm, m, CP_SU] * mm[CP_SU]
        want = (100.0 / wts[sm] - 1.0) * h2so4 / MMW
        checks.append(["strat mdwat", m, -1,
                       bool(np.allclose(mdwat[sm, m], want, rtol=1e-12, atol=0.0))])
    if (~mask).any():
        checks.append(["nomask mdwat", m, -1,
                       bool((mdwat[~mask, m] == 0.0).all())])

out = {{
    "setup": int(_setup), "fix_water": int(_fw), "fix_neg_pvol": int(_fn),
    "nmodes": nmodes, "ncp": ncp, "nrows": nrows,
    "nd": nd.tolist(), "md_in": md.tolist(), "mdt_in": mdt.tolist(),
    "drydp": drydp.tolist(), "dvol": dvol.tolist(),
    "md_out": md_out.tolist(), "mdt_out": mdt_out.tolist(),
    "mdwat": mdwat.tolist(), "wvol": wvol.tolist(), "wetdp": wetdp.tolist(),
    "rhopar": rhopar.tolist(), "pvol": pvol.tolist(),
    "pvol_wat": pvol_wat.tolist(),
    "wts": wts.tolist(), "rhosol_strat": rhosol_strat.tolist(),
    "hits": [int(hits[p]) for p in PREDICATES],
    "checks": checks, "nbox1_bad": nbox1_bad, "nbox1": nbox1,
    "num_eps": num_eps.tolist(), "mmid": mmid.tolist(), "mm": mm.tolist(),
    "rhocomp": _cp_real("rhocomp").tolist(),
    "mfrac_0": [list(r) for r in mfrac0],
    "mode": [int(v) for v in mode], "modesol": [int(v) for v in modesol],
    "soluble": [int(v) for v in soluble],
    "component": [[int(v) for v in r] for r in comp],
    "variant_applied": variant_applied,
}}
print("@@RESULT@@" + json.dumps(out))
"""


# The deliberate-abort block, and why it is where it is.
#
# `denom` at :444 is `mdwat*mmw + SUM_soluble md*mm`, and `mask_sol` is
# `SUM_soluble md > 0`. In the troposphere those two cannot disagree in sign,
# whatever md is: only cp_su and cp_cl are soluble in any supported setup, the
# ZSR water content is linear in the ion concentrations, and its measured
# coefficients here are `mdwat*mmw = 0.0886*md_su` for H2SO4 and
# `0.04314*md_cl` for NaCl at rh <= 0.3 (the pair is at `molal_max` below its
# `rh_min` of 47, so the NaCl figure is a floor). Both mixed-sign arrangements
# then fail:
#
#   md_su < 0 < md_cl : denom <= 0 wants md_cl*(mm_cl + 0.04314) <= |md_su|*mm_su,
#                       i.e. md_cl <= 0.9648*|md_su|, while mdsol > 0 wants
#                       md_cl > |md_su|.
#   md_cl < 0 < md_su : denom <= 0 wants |md_cl|*mm_cl >= md_su*(mm_su + 0.0886),
#                       i.e. |md_cl| >= 3.19*md_su, while mdsol > 0 wants
#                       |md_cl| < md_su.
#
# The margin in the first is 3.6% and it is the wrong way round --
# `mm_cl + kappa_cl = 0.10158 > mm_su = 0.098` -- so `denom <= 0` with
# `mask_sol` is UNREACHABLE below `putls`. This is a property of the water
# coefficients, not of the grid.
#
# In the stratosphere it opens up, because :434-438 throws the ZSR water away
# and rebuilds mdwat from md_su alone. Substituting :436 into :444,
#
#     denom = md_su*mm_su*(100/wts) + md_cl*mm_cl
#
# and with md_su < 0 the first term is scaled by 100/wts > 1 while mdsol is
# not, so `md_su < 0 < md_cl < 1.677*(100/wts)*|md_su|` satisfies both. That
# also forces `l_fix_neg_pvol_wat = 0` for this block: md_su < 0 makes the
# strat mdwat negative for any wts, and the guard at :884 would fire first.
#
# `denom2 <= 0` is separable from `denom <= 0` because setup 8 is the only
# supported setup whose soluble modes carry dust, and rhocomp(cp_du) = 2650
# against rhocomp(cp_bc) = 1500 makes `SUM md*mm` and `SUM md*mm/rhocomp`
# point in different directions: negative dust drives denom2 below zero while
# positive bc keeps dvol above it, so `ukca_calc_drydiam` still accepts the row.
#
# Two rows, one call, nbox = 2, both in mode 3 of setup 8:
#   row 1  denom > 0, denom2 <= 0 -- reported ONLY because row 2 opened the
#                                    gate: :481 is `SUM(ierr) > 0` and ierr is
#                                    built from denom alone (:476), so on its
#                                    own this row is silent
#   row 2  denom <= 0 with mask_sol -- the gate itself
# so exactly two fatal ereports, and the last message is row 2's.
#
# What this block deliberately does NOT reach is the five-way guard at
# :704-708. Its `CALL ereport` at :877 is preceded by a WRITE at :856-876 whose
# format is `'(5(A,E15.6,A,I0),A)'` into a `CHARACTER(LEN=errormessagelength)`;
# it overruns, and gfortran raises "Fortran runtime error: End of record" and
# kills the process before ereport is ever called. Measured, by tripping it.
# The path is unusable, which is why `minval_guard` is expected-zero everywhere
# and not given a block of its own. Issue #16's "dies on 19 of 20 error paths".
_ABORT_BODY = """
sys.path.insert(0, {validation!r})
from leaf_common import bind_call
call = bind_call(g)

CP_SU, CP_BC, CP_OC, CP_CL, CP_DU, CP_SO = range(6)
_sizes = g.wrap_sizes()
assert int(_sizes[-1]) == 0
nmodes, ncp = int(_sizes[1]), int(_sizes[2])
assert int(_sizes[7]) == _setup
mmid, e = g.wrap_mode_real("mmid", nmodes); assert int(e) == 0
comp, e = g.wrap_mode_cp_int("component", nmodes, ncp); assert int(e) == 0
comp = comp.astype(bool)
mm, e = g.wrap_cp_real("mm", ncp); assert int(e) == 0
soluble, e = g.wrap_cp_int("soluble", ncp); assert int(e) == 0
soluble = soluble.astype(bool)
WEIGHTS = np.array([float(w) for w in {weights!r}])
MMW = {mmw!r}

MODE = {mode0!r}          # 0-based index of the mode carrying both defects
for _c in (CP_SU, CP_BC, CP_OC, CP_CL, CP_DU):
    assert comp[MODE, _c], ("the abort block needs su, bc, oc, cl and du", MODE, _c)

nbox = 2
nd = np.full((nbox, nmodes), 1000.0)
md = np.zeros((nbox, nmodes, ncp))
for m in range(nmodes):
    present = np.where(comp[m])[0]
    if len(present) == 0:
        continue
    w = np.zeros(ncp); w[present] = WEIGHTS[present]
    md[:, m, :] = mmid[m] * w / w.sum()

base = mmid[MODE]
# Row 1: denom2 <= 0 < denom, and dvol > 0 so ukca_calc_drydiam still accepts it.
md[0, MODE, :] = 0.0
md[0, MODE, CP_SU] = base * {a_su!r}
md[0, MODE, CP_BC] = base * {a_bc!r}
md[0, MODE, CP_DU] = base * {a_du!r}
# Row 2: denom <= 0 with mdsol > 0, and enough insoluble mass that dvol, wvol
# and rhopar all stay positive so :704 -- which cannot survive its own WRITE --
# does not also fire.
md[1, MODE, :] = 0.0
md[1, MODE, CP_SU] = base * {b_su!r}
md[1, MODE, CP_CL] = base * {b_cl!r}
md[1, MODE, CP_BC] = base * {b_bc!r}
md[1, MODE, CP_OC] = base * {b_oc!r}
mdt = md.sum(axis=2)

t    = np.array([{t!r}, {t!r}])
pmid = np.array([{pmid!r}, {pmid!r}])
s    = np.array([{s!r}, {s!r}])
rh   = np.array([{rh!r}, {rh!r}])
assert (pmid < {putls!r}).all(), "the abort block only works below putls"

# drydiam must be clean: if it aborts, dvol and drydp are the outputs of a call
# that took an error path and the whole block is void.
drydp, dvol, md_out, mdt_out, ierr = call("leaf_drydiam", g.leaf_drydiam, nd, md, mdt)

before = tuple(int(v) for v in g.wrap_ereport_count())
mdwat, wvol, wetdp, rhopar, pvol, pvol_wat, ierr = g.leaf_volume_mode(
    nd, md_out, mdt_out, rh, dvol, drydp, t, pmid, s)
after = tuple(int(v) for v in g.wrap_ereport_count())
status, routine, message = g.wrap_ereport_last()
routine = routine.decode() if isinstance(routine, bytes) else str(routine)
message = message.decode() if isinstance(message, bytes) else str(message)

# denom and denom2 as :444/:445 and :461-467 build them, for the record.
denom = mdwat[:, MODE] * MMW
denom2 = denom.copy()
for c in range(ncp):
    if comp[MODE, c]:
        if soluble[c]:
            denom = denom + md_out[:, MODE, c] * mm[c]
        denom2 = denom2 + md_out[:, MODE, c] * mm[c]

print("@@RESULT@@" + json.dumps({{
    "setup": int(_setup), "mode0": MODE, "ierr": int(ierr),
    "fatal_delta": after[0] - before[0],
    "warning_delta": after[1] - before[1],
    "info_delta": after[2] - before[2],
    "status": int(status), "routine": routine.strip(), "message": message.strip(),
    "nd": nd.tolist(), "md_in": md.tolist(), "mdt_in": mdt.tolist(),
    "md_out": md_out.tolist(), "mdt_out": mdt_out.tolist(),
    "dvol": dvol.tolist(), "drydp": drydp.tolist(),
    "denom": denom.tolist(), "denom2": denom2.tolist(),
    "t": t.tolist(), "pmid": pmid.tolist(), "s": s.tolist(), "rh": rh.tolist(),
}}))
"""

ABORT_SETUP = 8
ABORT_MODE0 = 2  # Fortran imode 3: su, bc, oc, cl and du all present in setup 8
ABORT_FIX_NEG_PVOL = 0
# Coefficients on mmid(imode). Every one of them is forced by the inequalities
# in the comment above, evaluated at wts(213.0 K, 1.0e4 Pa, 1.0e-2) = 57.57, so
# 100/wts = 1.7369:
#   row 1  denom2 <= 0   needs  0.1*|a_du| >= 0.098*1.7369*a_su + 0.012*a_bc
#          dvol   >  0   needs  a_su*0.098/1769 + a_bc*0.012/1500 > |a_du|*0.1/2650
#   row 2  mdsol  >  0   needs  b_cl > |b_su|
#          denom  <= 0   needs  b_cl*0.05844 <= |b_su|*0.098*1.7369
#          wvol   >  0   needs  the bc and oc volumes to beat denom/rhosol_strat
ABORT_COEFFS = {
    "a_su": 0.5,
    "a_bc": 6.0,
    "a_du": -1.6,
    "b_su": -1.0,
    "b_cl": 1.3,
    "b_bc": 6.0,
    "b_oc": 6.0,
}
ABORT_ENV = {"t": 213.0, "pmid": 1.0e4, "s": 1.0e-2, "rh": 0.3}
ABORT_MESSAGE = "Demoninator <= 0 for i=2"


# ---------------------------------------------------------------------------
# Driving
# ---------------------------------------------------------------------------


def _body(rows: list[dict]) -> str:
    return _BODY.format(
        validation=str(REPO / "validation"),
        rows=json.dumps(rows),
        putls=PUTLS,
        mmw=MMW,
        weights=TYPICAL_WEIGHTS,
        predicates=PREDICATES,
    )


def _abort_body() -> str:
    return _ABORT_BODY.format(
        validation=str(REPO / "validation"),
        mmw=MMW,
        weights=TYPICAL_WEIGHTS,
        mode0=ABORT_MODE0,
        putls=PUTLS,
        **ABORT_COEFFS,
        **ABORT_ENV,
    )


def _run(rows, setup, combo, fix_water, fix_neg_pvol, label):
    source = (NAMELISTS / "boundary_layer.nml").read_text()
    rec = run_child(
        _body(rows),
        namelist_text=render_namelist(source, setup, combo),
        setup=setup,
        fix_water=fix_water,
        fix_neg_pvol=fix_neg_pvol,
        label=label,
    )
    if rec["nbox1_bad"]:
        raise SystemExit(
            f"{label}: rows {rec['nbox1_bad']} differ between nbox={rec['nrows']} "
            "and nbox=1 -- ukca_volume_mode has cross-row coupling that reaches an "
            "output, which would invalidate every vectorised comparison downstream"
        )
    bad = [c for c in rec["checks"] if not c[-1]]
    if bad:
        raise SystemExit(
            f"{label}: branch reconstruction disagrees with the Fortran output at "
            f"{bad} -- the hit counts below it cannot be trusted"
        )
    if int(rec["setup"]) != setup or int(rec["fix_water"]) != fix_water:
        raise SystemExit(f"{label}: child reported {rec['setup']}/{rec['fix_water']}")
    return rec


# Fields stacked into the archive, and the axes they are stacked along.
PER_SETUP = ("nd", "md_in", "mdt_in", "drydp", "dvol", "md_out", "mdt_out")
# The mode and component tables, carried so the fixture tests can decide which
# (row, mode) is mask_nosol or mask_sol without loading a second archive.
TABLES_REAL = ("num_eps", "mmid", "mm", "rhocomp", "mfrac_0")
TABLES_INT = ("mode", "modesol", "soluble", "component")
PER_RECORD = (
    "mdwat",
    "wvol",
    "wetdp",
    "rhopar",
    "pvol",
    "pvol_wat",
    "wts",
    "rhosol_strat",
)


def check_hits(total: dict[str, int], per_setup: dict[int, dict[str, int]]) -> None:
    """Refuse to write an archive in which a predicate never ran.

    Criterion 3 of task 35: one integer per predicate, every entry asserted
    non-zero or named in EXPECTED_ZERO with a reason. This runs before
    `savez_compressed`, so a grid that stopped reaching a branch cannot be
    committed and then discovered later.
    """
    problems = []
    for name in PREDICATES:
        n = total[name]
        if name in EXPECTED_ZERO:
            if n != 0:
                problems.append(f"{name} = {n}, expected 0 ({EXPECTED_ZERO[name]})")
        elif n == 0:
            problems.append(f"{name} = 0 -- no row in the grid reaches it")
    for setup, hits in per_setup.items():
        for name in PER_SETUP_REQUIRED:
            if hits[name] == 0:
                problems.append(f"setup {setup}: {name} = 0")
        for name in PER_SOLUBLE_SETUP_REQUIRED:
            if setup in NO_SOLUBLE_MODE:
                if hits[name] != 0:
                    problems.append(
                        f"setup {setup}: {name} = {hits[name]}, expected 0 "
                        f"({NO_SOLUBLE_MODE[setup]})"
                    )
            elif hits[name] == 0:
                problems.append(f"setup {setup}: {name} = 0")
        want_zero = setup in NOSOL_UNREACHABLE
        got = hits["mask_nosol"]
        if want_zero and got != 0:
            problems.append(
                f"setup {setup}: mask_nosol = {got}, expected 0 ({NOSOL_UNREACHABLE[setup]})"
            )
        if not want_zero and got == 0:
            problems.append(f"setup {setup}: mask_nosol = 0, and it should be reachable")
    if problems:
        raise SystemExit(
            "branch coverage regressed:\n  " + "\n  ".join(problems) + "\n"
            "A fixture whose grid stops reaching a branch is a coverage loss that "
            "no byte-equality test can notice, so it is refused here rather than "
            "written and discovered later."
        )


# The one collision that is a finding about the Fortran rather than a capture
# bug. l_fix_ukca_water_content reaches an output only through
# ukca_water_content_v, and setup 6 never calls it: mode = F,F,F,F,F,T,T,F and
# modesol = 1,1,1,1,0,0,0,0, so no active mode takes the `modesol(imode) == 1`
# branch that contains the call. Required rather than ignored -- if it stops
# colliding, either the mode table or the call site has changed.
FIX_WATER_IDENTICAL = {
    6: (
        "setup 6 has no active soluble mode, so the modesol(imode) == 1 branch "
        "that calls ukca_water_content_v never runs"
    )
}


def identical_pairs() -> list[tuple[str, str]]:
    return [(f"s{s}_fw0", f"s{s}_fw1") for s in sorted(FIX_WATER_IDENTICAL)]


def capture(out_dir: Path, quiet: bool = False) -> Path:
    check_axes()
    rows = row_grid()
    strat_rows = strat_grid()
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}

    total = dict.fromkeys(PREDICATES, 0)
    per_setup: dict[int, dict[str, int]] = {}
    per_setup_hits = []

    stacks: dict[str, list] = {f: [] for f in PER_SETUP}
    rec_stacks: dict[str, list] = {f: [] for f in PER_RECORD}
    tables: dict[str, list] = {f: [] for f in TABLES_REAL + TABLES_INT}

    for setup in SETUPS:
        for fw in FIX_WATER:
            label = f"main setup {setup} fix_water={fw}"
            rec = _run(rows, setup, "default", fw, 1, label)
            key = f"s{setup}_fw{fw}"
            records[key] = {f: rec[f] for f in PER_RECORD + PER_SETUP}
            for f in PER_RECORD:
                rec_stacks[f].append(rec[f])
            if fw == FIX_WATER[0]:
                for f in PER_SETUP:
                    stacks[f].append(rec[f])
                for f in TABLES_REAL + TABLES_INT:
                    tables[f].append(rec[f])
                hits = dict(zip(PREDICATES, rec["hits"]))
                per_setup[setup] = hits
                per_setup_hits.append(rec["hits"])
                for name in PREDICATES:
                    total[name] += hits[name]
                if not quiet:
                    print(
                        f"  setup {setup}: mask_sol={hits['mask_sol']:>5} "
                        f"nosol={hits['mask_nosol']:>4} mdsol<0={hits['mdsol_negative']:>4} "
                        f"strat={hits['strat_mdwat_override']:>5}"
                    )

    for f in PER_SETUP:
        arrays[f"main_{f}"] = np.array(stacks[f], dtype=np.float64)
    for f in TABLES_REAL:
        arrays[f"tab_{f}"] = np.array(tables[f], dtype=np.float64)
    for f in TABLES_INT:
        arrays[f"tab_{f}"] = np.array(tables[f], dtype=np.int32)
    for f in PER_RECORD:
        arrays[f"main_{f}"] = np.array(rec_stacks[f], dtype=np.float64).reshape(
            len(SETUPS), len(FIX_WATER), *np.shape(rec_stacks[f][0])
        )
    arrays["main_t"] = np.array([r["t"] for r in rows], dtype=np.float64)
    arrays["main_pmid"] = np.array([r["pmid"] for r in rows], dtype=np.float64)
    arrays["main_s"] = np.array([r["s"] for r in rows], dtype=np.float64)
    arrays["main_rh"] = np.array([r["rh"] for r in rows], dtype=np.float64)
    arrays["main_variant"] = np.array([r["variant"] for r in rows], dtype=np.str_)
    arrays["main_nd_kind"] = np.array([r["nd"] for r in rows], dtype=np.str_)

    # The two switch combinations, on setup 2 only: the archive records what a
    # density change does to rhopar and pvol, which is the only thing about the
    # mode tables that volume_mode can see.
    for combo in COMBOS:
        label = f"combo {combo} setup {COMBO_SETUP}"
        rec = _run(rows, COMBO_SETUP, combo, COMBO_FIX_WATER, 1, label)
        records[f"combo_{combo}"] = {f: rec[f] for f in PER_RECORD + PER_SETUP}
        for f in PER_RECORD + PER_SETUP:
            arrays[f"combo_{combo}_{f}"] = np.array(rec[f], dtype=np.float64)
        arrays[f"combo_{combo}_rhocomp"] = np.array(rec["rhocomp"], dtype=np.float64)
        if not quiet:
            print(f"  {combo:<12} captured on setup {COMBO_SETUP}")

    # l_fix_neg_pvol_wat, on the only rows where it reaches an output.
    for fn in FIX_NEG_PVOL:
        label = f"strat setup {COMBO_SETUP} fix_neg_pvol={fn}"
        rec = _run(strat_rows, COMBO_SETUP, "default", COMBO_FIX_WATER, fn, label)
        records[f"strat_fn{fn}"] = {f: rec[f] for f in PER_RECORD}
        for f in PER_RECORD + PER_SETUP:
            arrays[f"strat_fn{fn}_{f}"] = np.array(rec[f], dtype=np.float64)
        # The nbox = 1 replication, committed so the archive can re-prove the
        # equality the capture asserted rather than only recording that it did.
        for f, v in rec["nbox1"].items():
            arrays[f"strat_fn{fn}_nbox1_{f}"] = np.array(v, dtype=np.float64)
        # The strat block is the only place wts is left unclamped, so it is the
        # only place `wts_above_99` can be counted. Its hits join the total.
        for name, n in zip(PREDICATES, rec["hits"]):
            total[name] += n
    arrays["strat_t"] = np.array([r["t"] for r in strat_rows], dtype=np.float64)
    arrays["strat_pmid"] = np.array([r["pmid"] for r in strat_rows], dtype=np.float64)
    arrays["strat_s"] = np.array([r["s"] for r in strat_rows], dtype=np.float64)
    arrays["strat_rh"] = np.array([r["rh"] for r in strat_rows], dtype=np.float64)

    # The deliberate abort. Not run through _run: this one is EXPECTED to reach
    # ereport, and the expectation is on the exact count and the message.
    source = (NAMELISTS / "boundary_layer.nml").read_text()
    ab = run_child(
        _abort_body(),
        namelist_text=render_namelist(source, ABORT_SETUP, "default"),
        setup=ABORT_SETUP,
        fix_water=COMBO_FIX_WATER,
        fix_neg_pvol=ABORT_FIX_NEG_PVOL,
        label="deliberate abort",
    )
    if (ab["fatal_delta"], ab["warning_delta"], ab["info_delta"]) != (2, 0, 0):
        raise SystemExit(
            "the deliberate-abort block moved the ereport counters by "
            f"({ab['fatal_delta']}, {ab['warning_delta']}, {ab['info_delta']}), "
            "wanted exactly (2, 0, 0): one report for the denom2 row, which is "
            "only visible because the denom row opened the :481 gate, and one for "
            "the denom row itself"
        )
    if ABORT_MESSAGE not in ab["message"]:
        raise SystemExit(
            f"the last ereport said {ab['message']!r}, wanted a message containing "
            f"{ABORT_MESSAGE!r}"
        )
    if not (ab["denom2"][0] <= 0.0 < ab["denom"][0]):
        raise SystemExit(f"abort row 1 wanted denom > 0 >= denom2, got {ab['denom']}")
    if not (ab["denom"][1] <= 0.0 < ab["denom2"][1]):
        raise SystemExit(f"abort row 2 wanted denom <= 0 < denom2, got {ab['denom2']}")
    for f in ("nd", "md_in", "mdt_in", "md_out", "mdt_out", "dvol", "drydp",
              "denom", "denom2", "t", "pmid", "s", "rh"):  # fmt: skip
        arrays[f"abort_{f}"] = np.array(ab[f], dtype=np.float64)
    arrays["abort_counts"] = np.array(
        [ab["fatal_delta"], ab["warning_delta"], ab["info_delta"]], dtype=np.int64
    )
    arrays["abort_message"] = np.array(ab["message"], dtype=np.str_)
    arrays["abort_routine"] = np.array(ab["routine"], dtype=np.str_)
    arrays["abort_setup"] = np.array(ABORT_SETUP, dtype=np.int32)
    arrays["abort_mode0"] = np.array(ABORT_MODE0, dtype=np.int32)
    arrays["abort_fix_neg_pvol"] = np.array(ABORT_FIX_NEG_PVOL, dtype=np.int32)
    if not quiet:
        print(f"  abort       {ab['fatal_delta']} fatal, {ab['message'][:48]!r}")

    # Everything that can refuse a bad archive runs BEFORE savez_compressed.
    check_hits(total, per_setup)
    check_varied(
        records,
        expected_identical=identical_pairs(),
        what="volume_mode configurations",
    )
    if not quiet:
        print(
            f"  witness : {len(records)} records pairwise distinct except the "
            f"{len(FIX_WATER_IDENTICAL)} fix_water pair(s) required to collide"
        )

    arrays["hits"] = np.array([total[p] for p in PREDICATES], dtype=np.int64)
    arrays["hits_by_setup"] = np.array(per_setup_hits, dtype=np.int64)
    arrays["hits_abort"] = np.array(
        [sum(1 for v in ab["denom"] if v <= 0.0), sum(1 for v in ab["denom2"] if v <= 0.0)],
        dtype=np.int64,
    )
    if list(arrays["hits_abort"]) != [1, 1]:
        raise SystemExit(
            f"the abort block wanted one denom <= 0 row and one denom2 <= 0 row, "
            f"got {list(arrays['hits_abort'])}"
        )
    arrays["_predicates"] = np.array(PREDICATES, dtype=np.str_)
    arrays["_expected_zero"] = np.array(sorted(EXPECTED_ZERO), dtype=np.str_)
    arrays["_nosol_unreachable"] = np.array(sorted(NOSOL_UNREACHABLE), dtype=np.int32)
    arrays["_fix_water_identical"] = np.array(sorted(FIX_WATER_IDENTICAL), dtype=np.int32)
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_fix_water"] = np.array(FIX_WATER, dtype=np.int32)
    arrays["_fix_neg_pvol"] = np.array(FIX_NEG_PVOL, dtype=np.int32)
    arrays["_combos"] = np.array(COMBOS, dtype=np.str_)
    arrays["_putls"] = np.array(PUTLS, dtype=np.float64)
    arrays["_case"] = np.array("volume_mode")
    arrays["_mode"] = np.array("leaf")
    arrays["_variant"] = np.array("f64")
    arrays["_rows"] = np.array(len(rows), dtype=np.int64)
    arrays["_strat_rows"] = np.array(len(strat_rows), dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grid and stop")
    args = parser.parse_args(argv)

    check_axes()
    rows = row_grid()
    strat_rows = strat_grid()
    children = len(SETUPS) * len(FIX_WATER) + len(COMBOS) + len(FIX_NEG_PVOL) + 1
    if args.dry_run:
        print(f"volume_mode leaf sweep -> {args.out / ARCHIVE}")
        print(f"  {len(rows)} main rows x {len(SETUPS)} setups x {len(FIX_WATER)} flags")
        print(f"  {len(strat_rows)} strat rows x {len(FIX_NEG_PVOL)} l_fix_neg_pvol_wat")
        print(f"  {len(COMBOS)} switch combinations on setup {COMBO_SETUP}")
        print(f"  1 deliberate-abort block on setup {ABORT_SETUP}")
        print(f"  {children} subprocesses, one per configuration")
        for name, axis in (
            ("pmid", PMID_AXIS),
            ("rh", RH_AXIS),
            ("t", T_AXIS),
            ("s", S_AXIS),
        ):
            print(f"  {name:<6} {len(axis):>3} points  [{min(axis):.6g}, {max(axis):.6g}]")
        print(f"  variants {', '.join(VARIANTS)}")
        print(f"  nd       {', '.join(ND_KINDS)}")
        print(f"  {len(PREDICATES)} predicates counted, {len(EXPECTED_ZERO)} expected zero")
        return 0

    print(f"sweeping ukca_volume_mode -> {args.out}  ({children} subprocesses)")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
