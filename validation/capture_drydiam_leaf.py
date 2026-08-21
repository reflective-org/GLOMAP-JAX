#!/usr/bin/env python3
"""Sweep `ukca_calc_drydiam` over a grid chosen to reach its branches (task 35d).

    python validation/capture_drydiam_leaf.py --dry-run
    python validation/capture_drydiam_leaf.py   # writes tests/goldens/drydiam.f64.leaf.npz

Why a leaf fixture at all
-------------------------
The undersize reset at `ukca_calc_drydiam.F90:245-262` is unreachable from any
shipped namelist, and that is measured rather than assumed. The four committed
`*.f64.branches.npz` archives carry the `drydiam`/`undersize` predicate 3456
times between them -- 1296 for `bl_nmts3` and 720 each for the other three --
and the recorded value is **0 in every one**. (The plan said 2160; the count is
3456. The zero is what matters and it holds.) The neighbouring
`drydiam`/`nd_gt_eps` predicate is well covered by contrast: 3934 true and 914
false of 4848.

The reset is the only thing in the routine that writes `md` and `mdt`, so the
whole `INTENT(IN OUT)` half of the interface -- the half task 37 has to port --
has no reference data from any trajectory. This script makes some.

The grid is built backwards from the predicates rather than forwards from
anything physical. There are seven of them:

* `mode(imode)` (`:204`) and its `ELSE` (`:224`) -- inactive slots take a wholly
  different formula, `ratio2 = mmsul*mmid/(avogadro*rho_so4)`;
* `mask(i) = nd > num_eps` (`:206`), a STRICT `>`, so `nd` exactly equal to
  `num_eps` takes the `mmid` arm. Both representable neighbours of `num_eps`
  are in the grid, as is `num_eps` itself;
* `component(imode,icp)` (`:215`) -- mass is added only for members, so a
  non-member's `md` has to be ignored;
* `mode(imode)` again in the reset loop (`:246`), which runs only over
  `mode_nuc_sol..mode_acc_sol`, i.e. modes 1 to 3;
* `dp < ddplim0(imode)*0.1` (`:250`) -- the reset;
* `component(imode,icp)` again at `:252`, which decides which components the
  reset overwrites -- and, by omission, which it must leave alone;
* `MINVAL(dvol) <= 0 .OR. MINVAL(drydp) <= 0` (`:268`), the deliberate abort.

Each gets a hit count computed at capture time and stored as `*_hits`; the
capture refuses to write if one is zero without a recorded reason.
`tests/test_drydiam_fixtures.py` re-asserts them.

Reaching the reset
------------------
The reset needs an active mode in 1..3 with `nd > num_eps` and a dry diameter
below `0.1*ddplim0`. Mask-false rows can never reach it: they get
`drydp` = 3.163e-9 / 3.163e-8 / 2.237e-7 for modes 1/2/3, which is 31.6x, 31.6x
and 22.4x their thresholds. So the rows that reach it are mask-true with a very
small `md`.

The threshold is **derived, not transcribed**. `_reset_md_threshold` inverts
the routine's own chain -- `ratio1 = mm/(avogadro*rhocomp)` (`:195`),
`sixovrpix = 6/(pi*x)` (`:230`), `dp = (sixovrpix*dvol)**(1/3)` (`:237`) --
against `dp_thresh1 = ddplim0*0.1` (`:249`), using the mode tables from
`glomap_jax.physics.modes.build`. Two reasons it is derived rather than typed
in:

* the plan quoted setup-1 thresholds of 0.014980188168490213 /
  14.98018816849021 / 9473.211907304352. Only the middle one survives: written
  the way the Fortran writes a cube, `d*d*d` rather than numpy's `pow`, they
  come out 2 ulp, 0 ulp and 1 ulp higher (0.014980188168490217 /
  14.98018816849021 / 9473.211907304354). A transcribed constant carries the
  wrong spelling's rounding into the golden and nothing notices;
* a derived threshold tracks the table. `nacl_off` and `bc_mg_mix` move
  `rhocomp`, and a frozen literal would quietly stop straddling.

These abscissae are the one deliberate exception to the "decimal literals only"
rule of `capture_leaf.py`. The quantity being straddled is a function of the
mode table, so the sample points have to be as well -- a frozen literal would
be reproducible and wrong. What keeps that honest is that the child asserts,
byte for byte, that the tables it was handed are the tables the Fortran holds
(`_check_tables` in the child body), so if `x` or `mmid` ever moved the grid
would move with the threshold rather than away from it. Everything else --
`nd`, the `mdt` garbage, the non-member pollution -- is decimal literals and
`nextafter`.

One measurement that changes the grid. The closed form is not where the
predicate flips: `(k*md)**(1/3)` is not exactly invertible, and the flip sits
**15 to 35 ulp below** `md*` across all seven setups, all five combinations and
every member component -- so the closed form itself never fires. A `+-1 ulp`
probe would therefore have landed entirely on the non-firing side and recorded
nothing. The ladder runs to `+-256 ulp` for that reason, and
`check_straddle` refuses to write an archive in which any member's ladder
stopped bracketing its flip.

`mdt` is written and never read
-------------------------------
`mdt` is `INTENT(IN OUT)` and appears at exactly four lines: the argument list
(`:40`), the declaration (`:135`), the OpenMP `SHARED` clause (`:243`) and the
single assignment inside the reset (`:256`). Nothing reads it.

That is measured rather than asserted from the grep: **the whole grid is run
twice**, with two disjoint `mdt` arrays (NaN, negatives, 1e300, signed zero,
subnormal-ish decimals), and `drydp`, `dvol` and `md_out` must come back bit
identical. The same pair of runs is how the reset is *detected* without
instrumenting the Fortran -- `mdt_out` is passed straight through unless the
reset fired, so `mdt_out_a == mdt_out_b` at a slot exactly when both were
overwritten with `mlo(imode)`.

The abort block is captured separately, and has to be
-----------------------------------------------------
`:265-267` reduces `MINVAL` over the **whole** `nbox` extent, so a single row
with `dvol <= 0` voids the entire call under the ereport check. The
deliberate-abort rows are therefore not in the main grid: they are separate
`nbox = 1` calls, each asserted to move the fatal counter by exactly one and to
leave `' dvol or drydp <= 0'` in `wrap_ereport_last`.

Note which modes can abort. An active mode 1..3 with every member `md = 0` gets
`dvol = 0` and `drydp = 0`, but the reset then fires and restores both -- so
modes 1..3 *escape*, and that escape is a non-aborting row in the main grid.
Modes 4 and up have no such rescue. Setups 1, 3 and 5 have no active mode above
slot 4, but `mode_cor_sol` (slot 4) is itself outside the reset loop, so every
supported setup has an aborting mode.

Setup 6 is the control
----------------------
`mode_choice = [0,0,0,0,0,1,1,0]`, so modes 1 to 3 are inactive and `:245-262`
is dead. It is dead because of `mode()` alone: `ddplim0`, `num_eps` and
`mfrac_0` are byte-identical to every other setup's, which
`tests/test_drydiam_fixtures.py` checks -- a port that keys the deadness off
any of those three would pass the control for the wrong reason.

Configurations
--------------
Seven setups x five switch combinations, one subprocess each (35). No
environment axis: the argument list is `(nbox, glomap_variables_local, nd, md,
mdt, drydp, dvol)` and the `USE` block reaches no environment module.

`nacl_off` and `bc_mg_mix` both move `rhocomp`, hence `ratio1`, hence
`mmid`/`mlo`. `hygro_off` moves only `no_ions` and `dust_ageing` only
`topmode`, neither of which appears anywhere in `ukca_calc_drydiam.F90`, so
both are captured as *expected-identical* controls rather than dropped: a
missing collision would mean the routine had grown a dependency and an
unexpected one would mean a switch never reached the Fortran. Both raise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
sys.path.insert(0, str(REPO / "src"))

import capture_modes as cm  # noqa: E402
from leaf_common import NAMELISTS, check_varied, run_child  # noqa: E402

from glomap_jax.core.constants import AVOGADRO, MMSUL, PI, RHO_SO4  # noqa: E402
from glomap_jax.physics import modes  # noqa: E402

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "drydiam.f64.leaf.npz"

SETUPS = cm.SETUPS
NMODES = modes.NMODES

# The reset loop runs mode_nuc_sol..mode_acc_sol -- ukca_mode_setup.F90:86-88,
# i.e. slots 1, 2 and 3 one-based. Zero-based here.
RESET_MODES = (0, 1, 2)

# Five of `capture_modes.COMBOS`, reused rather than redefined so there is one
# asserted namelist rewrite in this repository rather than two. The first three
# are expected to move the answer; the last two are the controls.
COMBOS = ("default", "nacl_off", "bc_mg_mix", "hygro_off", "dust_ageing")
EXPECTED_IDENTICAL = ("default", "hygro_off", "dust_ageing")

# How far either side of the derived threshold the fine straddle reaches, in
# representable steps. Measured: the flip is 15-35 ulp below the closed form
# everywhere, so +-1 ulp alone would sit entirely on the non-firing side. 256
# brackets the worst case four times over, and the 24/32/40 rungs narrow the
# bracket to <= 8 ulp of md, which is ~3 ulp of `drydp`. `check_straddle` is
# what notices if the offset ever grows past the widest rung.
#
# What this does NOT sample is the row where `drydp` lands exactly on
# `dp_thresh1`, which is the one input that separates `<` from `<=`. Finding it
# needs a libm `pow` in the abscissa, which would make the sample point itself
# platform-dependent, so it is left out on purpose and recorded here as a gap.
ULP_STEPS = (1, 4, 16, 24, 32, 40, 64, 256)

# `mdt` garbage, two disjoint sets. Every position differs between them, which
# is what makes "mdt_out_a == mdt_out_b" mean "the reset overwrote both". NaN
# is in both at different positions, deliberately: a slot where A is NaN and B
# is not must still compare unequal.
GARBAGE_A = (0.0, float("nan"), -1.0, 1e300, 3.5, -0.0, 1e-300)
GARBAGE_B = (7.0, 1e300, float("nan"), -2.5, -1e-300, 1234.5, 0.5)

# The predicates counted for every (setup, combo), in source order.
PREDICATES = (
    "mode_true",
    "mode_false",
    "mask_true",
    "mask_false",
    "component_true",
    "component_false",
    "reset_mode_true",
    "reset_mode_false",
    "reset_fired",
    "reset_not_fired",
    "reset_component_true",
    "reset_component_false",
    "abort_true",
    "abort_false",
)

# Arrays the child returns for every configuration.
OUTPUTS = ("drydp", "dvol", "md_out", "mdt_out_a", "mdt_out_b")


def tables(setup: int, combo: str) -> modes.ModeTables:
    """The mode tables for one configuration.

    `capture_modes.COMBOS` holds the switch overrides and `modes.build` takes
    them by keyword, so the namelist the child is handed and the tables the
    grid is derived from cannot drift apart.
    """
    return modes.build(setup, **cm.COMBOS[combo])


def _ulp(value: float, steps: int) -> float:
    """`value` moved `steps` representable positions (signed)."""
    towards = np.inf if steps > 0 else -np.inf
    out = np.float64(value)
    for _ in range(abs(steps)):
        out = np.nextafter(out, towards)
    return float(out)


def _reset_md_threshold(t: modes.ModeTables, imode: int, icp: int) -> float:
    """The `md(:,imode,icp)` at which a pure-`icp` particle meets `dp_thresh1`.

    Inverted from the routine's own chain, in the routine's own spelling:

        ratio1(icp)      = mm(icp)/(avogadro*rhocomp(icp))      :195
        dvol             = ratio1(icp)*md                       :218
        sixovrpix(imode) = 6.0/(pi*x(imode))                    :230
        ddpcub           = sixovrpix*dvol                       :232
        dp               = ddpcub**(1.0/3.0)                    :237 (cubrt_v)
        dp_thresh1       = ddplim0(imode)*0.1                   :249

    so `md* = dp_thresh1**3 / (sixovrpix*ratio1)`. The cube is written `d*d*d`
    and not `d**3`: gfortran expands an integer literal exponent into repeated
    multiplication while numpy's `**` calls `pow()`, and on setup 1 mode 3 the
    two answers differ by one ulp. Nothing here reaches libm.
    """
    thresh = 0.1 * float(t.ddplim0[imode])
    sixovrpix = 6.0 / (PI * float(t.x[imode]))
    ratio1 = float(t.mm[icp]) / (AVOGADRO * float(t.rhocomp[icp]))
    return (thresh * thresh * thresh) / (sixovrpix * ratio1)


def _nd_points(num_eps: float) -> list[float]:
    """The six `nd` values swept per mode.

    `:206` is a strict `>`, so `num_eps` itself must take the mask-FALSE arm and
    only the next representable value above it takes mask-TRUE. Both are here,
    along with the one below, because a port written with `>=` passes a grid
    that merely brackets the threshold loosely.
    """
    return [
        0.0,
        float(np.nextafter(num_eps, 0.0)),
        float(num_eps),
        float(np.nextafter(num_eps, np.inf)),
        1e-6,
        1e3,
    ]


def mode_cases(t: modes.ModeTables, imode: int) -> list[tuple[float, list[float], str]]:
    """`(nd, md, tag)` for one mode column, before rows are assembled.

    Rows are built by cycling each mode's list independently (see `build_grid`),
    so within one mode the batch mixes mask-true with mask-false and reset with
    no-reset -- which is what catches a port that decides a whole column at
    once. The lists have different lengths on purpose; a common length would
    lock the modes into step.
    """
    ncp = t.ncp
    scale = float(t.mmid[imode])
    zeros = [0.0] * ncp
    nds = _nd_points(float(t.num_eps[imode]))

    def pure(icp: int, value: float) -> list[float]:
        md = [0.0] * ncp
        md[icp] = value
        return md

    if not bool(t.mode[imode]):
        # Inactive: `:224-228` ignores nd and md entirely and writes ratio2.
        # Swept anyway, with every component polluted, so the golden shows the
        # column is constant rather than merely plausible.
        polluted = [scale * (k + 1) for k in range(ncp)]
        cases = [(nd, polluted, f"inactive/nd{j}") for j, nd in enumerate(nds)]
        cases.append((1e3, zeros, "inactive/md0"))
        return cases

    members = [c for c in range(ncp) if bool(t.component[imode, c])]
    others = [c for c in range(ncp) if not bool(t.component[imode, c])]
    if not members:
        raise SystemExit(
            f"setup {t.setup} mode {imode + 1} is active with no component -- every "
            "mask-true row would give dvol = 0 and void the whole call at :268"
        )

    equal = [scale / len(members) if c in members else 0.0 for c in range(ncp)]
    cases = [(nd, equal, f"nd{j}/equal_mass") for j, nd in enumerate(nds)]
    cases += [(1e-6, pure(c, scale), f"pure/cp{c}") for c in members]

    if others:
        polluted = list(equal)
        polluted[others[0]] = scale
        cases.append((1e3, polluted, f"nonmember/cp{others[0]}/mask_true"))
        cases.append((0.0, polluted, f"nonmember/cp{others[0]}/mask_false"))

    if imode in RESET_MODES:
        # dvol = 0 with the mask on: drydp = 0, the reset fires and puts dvol
        # back above zero -- which is exactly why modes 1..3 never reach the
        # abort at :268 the way mode 4 does.
        cases.append((1e-6, zeros, "reset/md0"))
        for rank, icp in enumerate(members):
            m0 = _reset_md_threshold(t, imode, icp)
            points = [(0.9 * m0, "0.9x"), (1.1 * m0, "1.1x")]
            if rank == 0:
                points.append((m0, "1.0x"))
                for step in ULP_STEPS:
                    points.append((_ulp(m0, -step), f"-{step}ulp"))
                    points.append((_ulp(m0, step), f"+{step}ulp"))
            cases += [(1e-6, pure(icp, v), f"reset/cp{icp}/{lab}") for v, lab in points]
        if others:
            # A reset row that also carries non-member mass. Without this the
            # ":253 leaves non-members alone" check is vacuous: every other
            # reset row has zeros in the non-member slots, so "unchanged" and
            # "overwritten with mlo*mfrac_0 = 0" are the same number.
            polluted = pure(members[0], 0.9 * _reset_md_threshold(t, imode, members[0]))
            polluted[others[0]] = scale
            cases.append((1e-6, polluted, f"reset/nonmember/cp{others[0]}"))
    else:
        # Slot 4 and up have no reset to rescue them, so a zero-md mask-true row
        # here would abort. Kept mask-false; the aborting version has its own
        # block.
        cases.append((0.0, zeros, "md0/mask_false"))

    return cases


def build_grid(t: modes.ModeTables) -> dict[str, np.ndarray]:
    """The main grid for one setup: `nd`, `md`, and the two `mdt` garbage sets.

    Identical for all five combinations of a setup -- `rhocomp(cp_su)` and `x`
    are what set the reset thresholds and no switch touches either -- so the
    inputs are stored once per setup and only the outputs are per combination.
    `capture` asserts that rather than assuming it.
    """
    cases = [mode_cases(t, imode) for imode in range(NMODES)]
    rows = max(len(c) for c in cases)

    nd = np.zeros((rows, NMODES), dtype=np.float64)
    md = np.zeros((rows, NMODES, t.ncp), dtype=np.float64)
    tags: list[list[str]] = []
    for r in range(rows):
        tags.append([])
        for imode in range(NMODES):
            case_nd, case_md, tag = cases[imode][r % len(cases[imode])]
            nd[r, imode] = case_nd
            md[r, imode, :] = case_md
            tags[-1].append(tag)

    index = (np.arange(rows)[:, None] + np.arange(NMODES)[None, :]) % len(GARBAGE_A)
    return {
        "nd": nd,
        "md_in": md,
        "mdt_in_a": np.asarray(GARBAGE_A, dtype=np.float64)[index],
        "mdt_in_b": np.asarray(GARBAGE_B, dtype=np.float64)[index],
        "tags": np.array(tags, dtype=np.str_),
    }


def abort_mode(t: modes.ModeTables) -> int:
    """The lowest active mode outside the reset loop, i.e. the one that aborts.

    Modes 1..3 cannot: `dvol = 0` there triggers the reset, which restores it.
    Slot 4 (`mode_cor_sol`) is already outside the loop, so every supported
    setup has one -- setup 6, whose first five slots are off, uses slot 6.
    """
    for imode in range(max(RESET_MODES) + 1, NMODES):
        if bool(t.mode[imode]):
            return imode
    raise SystemExit(f"setup {t.setup} has no active mode above slot {max(RESET_MODES) + 1}")


def abort_rows(t: modes.ModeTables) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """One-row calls that must each move the fatal counter by exactly one.

    Every other mode is left mask-false with zero `md`, which gives
    `dvol = mmid*mmsul/(avogadro*rho_so4) > 0`, so precisely one mode offends
    and "exactly one fatal" is a real expectation rather than "at least one".

    Two rows, because `:268` is a disjunction of two comparisons and `dvol = 0`
    only reaches the first. A negative `md` drives `dvol` below zero, and with
    it `ddpcub`, so `cubrt_v` returns NaN and `min_drydp <= 0` is FALSE -- the
    abort still fires, through the other arm.
    """
    imode = abort_mode(t)
    icp = next(c for c in range(t.ncp) if bool(t.component[imode, c]))
    scale = float(t.mmid[imode])
    tags, nd, md = [], [], []
    for tag, value in (("dvol_zero", 0.0), ("dvol_negative", -scale)):
        row_nd = np.zeros((1, NMODES), dtype=np.float64)
        row_md = np.zeros((1, NMODES, t.ncp), dtype=np.float64)
        row_nd[0, imode] = 1e-6
        row_md[0, imode, icp] = value
        tags.append(f"{tag}/mode{imode + 1}/cp{icp}")
        nd.append(row_nd)
        md.append(row_md)
    if not tags:
        raise SystemExit(
            f"setup {t.setup}: the abort block is empty, so `:268` is never taken and "
            "the only deliberately-fatal path in the routine has no reference"
        )
    return (
        tags,
        np.concatenate(nd),
        np.concatenate(md),
        np.zeros((len(tags), NMODES), dtype=np.float64),
    )


def branch_counts(t: modes.ModeTables, grid: dict, reset: np.ndarray, aborts: int) -> dict:
    """Hits per predicate for one (setup, combo), over the batched main call.

    Everything except `reset_fired` is a function of the tables and the grid, so
    it is arithmetic and not observation; `reset_fired` comes from the
    `mdt_out_a == mdt_out_b` detector, which is the only part the Fortran gets a
    vote on. `abort_*` come from the separate one-row block, where every call
    evaluates `:268` once per mode.
    """
    rows = grid["nd"].shape[0]
    active = [i for i in range(NMODES) if bool(t.mode[i])]
    mask = np.zeros((rows, NMODES), dtype=bool)
    for i in active:
        mask[:, i] = grid["nd"][:, i] > t.num_eps[i]

    members = {i: int(t.component[i].sum()) for i in range(NMODES)}
    reset_active = [i for i in RESET_MODES if bool(t.mode[i])]
    fired_members = sum(int(reset[:, i].sum()) * members[i] for i in reset_active)
    fired_others = sum(int(reset[:, i].sum()) * (t.ncp - members[i]) for i in reset_active)

    return {
        "mode_true": rows * len(active),
        "mode_false": rows * (NMODES - len(active)),
        "mask_true": int(mask.sum()),
        "mask_false": rows * len(active) - int(mask.sum()),
        "component_true": rows * sum(members[i] for i in active),
        "component_false": rows * sum(t.ncp - members[i] for i in active),
        "reset_mode_true": rows * len(reset_active),
        "reset_mode_false": rows * (len(RESET_MODES) - len(reset_active)),
        "reset_fired": int(reset.sum()),
        "reset_not_fired": rows * len(reset_active) - int(reset.sum()),
        "reset_component_true": fired_members,
        "reset_component_false": fired_others,
        "abort_true": aborts,
        "abort_false": aborts * (NMODES - 1),
    }


def expected_zero(t: modes.ModeTables) -> dict[str, str]:
    """Predicates that must be zero for this setup, and why.

    Both arms of `:246` are per-setup facts about `mode_choice` and nothing
    else. Six setups carry all three of modes 1-3, so its FALSE arm is
    unreachable there and setup 6 is the only place it is exercised; setup 6
    carries none of them, so its TRUE arm and everything downstream is dead
    there. Between them the two arms are covered, and neither is silently
    dropped.

    Setup 6's deadness is keyed off `mode_choice` alone, deliberately:
    `ddplim0`, `num_eps` and `mfrac_0` are byte-identical to every other
    setup's, so a port that decided the reset was dead from one of those would
    pass this control for the wrong reason.
    """
    zeros: dict[str, str] = {}
    active = [i for i in RESET_MODES if bool(t.mode[i])]
    if len(active) == len(RESET_MODES):
        zeros["reset_mode_false"] = (
            f"setup {t.setup}: mode_choice has all of modes 1-3 on, so :246 never "
            "takes its false arm -- setup 6 is where that arm is covered"
        )
    if not active:
        reason = f"setup {t.setup}: mode_choice leaves modes 1-3 off, so :245-262 is dead"
        zeros.update(
            {
                "reset_mode_true": reason,
                "reset_fired": reason,
                "reset_not_fired": reason,
                "reset_component_true": reason,
                "reset_component_false": reason,
            }
        )
    return zeros


def check_hits(label: str, counts: dict, zeros: dict[str, str]) -> None:
    """Refuse to write an archive with an unexplained dead predicate.

    The point of a hit count is that removing coverage takes it to zero and
    stops the capture. Deleting the reset straddle from `mode_cases` takes
    `reset_fired` to 0 for the six setups with active modes 1-3, and this is
    what turns that into a refusal rather than a quietly smaller golden.
    """
    missing = sorted(set(PREDICATES) - set(counts))
    if missing:
        raise SystemExit(f"{label}: no hit count for {missing}")
    dead = sorted(p for p, n in counts.items() if n == 0 and p not in zeros)
    if dead:
        raise SystemExit(
            f"{label}: predicate(s) {dead} were never reached and are not recorded as "
            "expected-zero -- the grid no longer covers the routine, so the golden "
            "would pin a subset of it and every test against it would pass"
        )
    wrong = {p: counts[p] for p in zeros if counts.get(p, 0) != 0}
    if wrong:
        raise SystemExit(
            f"{label}: predicate(s) {sorted(wrong)} are recorded as expected-zero but "
            f"fired {wrong} -- the reason recorded for them is no longer true"
        )


def check_straddle(label: str, t: modes.ModeTables, grid: dict, reset: np.ndarray) -> None:
    """Every reset ladder has to bracket its flip, or the archive is one-sided.

    The ladder is placed from a closed form and the flip is 15-35 ulp away from
    it, so "the straddle straddles" is a measurement and not a construction. If
    the offset ever grows past `max(ULP_STEPS)` the ladder silently lands
    entirely on the non-firing side, `reset_fired` stays positive because of the
    `0.9x` point, and the fine structure the golden exists for is gone. This is
    what notices.
    """
    tags = grid["tags"]
    for imode in RESET_MODES:
        if not bool(t.mode[imode]):
            continue
        for icp in range(t.ncp):
            if not bool(t.component[imode, icp]):
                continue
            rows = np.char.startswith(tags[:, imode], f"reset/cp{icp}/")
            if not rows.any():
                raise SystemExit(f"{label}: mode {imode + 1} cp{icp} has no straddle row")
            fired = reset[rows, imode]
            if not fired.any() or fired.all():
                where = "all" if fired.all() else "none"
                raise SystemExit(
                    f"{label}: the reset straddle for mode {imode + 1} cp{icp} fired on "
                    f"{where} of its {int(rows.sum())} rows -- the ladder no longer "
                    f"brackets the flip, so +-{max(ULP_STEPS)} ulp is not wide enough"
                )


def check_closed_forms(label: str, t: modes.ModeTables, grid: dict, data) -> None:
    """The two arms that are pure arithmetic, checked before the archive lands.

    `:210` and `:225` are multiplications and one division -- no libm -- so
    exact equality is the right comparison and holds on any IEEE 754 host.
    Checking them here means a `dvol` column that is quietly the wrong table
    entry never reaches the disk. The `drydp` that follows goes through
    `cubrt_v` and is therefore platform-dependent; it is left to the test,
    which knows whether it is on the capture platform.
    """
    ratio2 = MMSUL * t.mmid / (AVOGADRO * RHO_SO4)
    dvol = np.asarray(data["dvol"], dtype=np.float64)
    for imode in range(NMODES):
        if bool(t.mode[imode]):
            rows = grid["nd"][:, imode] <= t.num_eps[imode]
        else:
            rows = np.ones(dvol.shape[0], dtype=bool)
        if not rows.any():
            raise SystemExit(f"{label}: mode {imode + 1} has no row on the mmid arm")
        if not np.array_equal(dvol[rows, imode], np.full(int(rows.sum()), ratio2[imode])):
            raise SystemExit(
                f"{label}: mode {imode + 1} dvol on the mmid arm is not "
                f"mmsul*mmid/(avogadro*rho_so4) = {ratio2[imode]!r}; got "
                f"{sorted(set(dvol[rows, imode].tolist()))}"
            )


def capture(out_dir: Path, quiet: bool = False) -> Path:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, str]] = {}
    total_rows = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        source = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

        for setup in SETUPS:
            default_grid = build_grid(tables(setup, "default"))
            total_rows += int(default_grid["nd"].shape[0])

            for combo in COMBOS:
                t = tables(setup, combo)
                label = f"setup {setup} ({combo})"

                # One grid per combination, not one per setup. `mmid` sets the
                # mass scale and moves with `l_fix_nacl_density`, and the reset
                # threshold for a black-carbon member moves with `i_tune_bc`, so
                # a shared grid would place the straddle by the wrong table for
                # two of the five combinations and stop bracketing the flip.
                grid = build_grid(t)
                rows = int(grid["nd"].shape[0])
                if grid["nd"].shape != default_grid["nd"].shape:
                    raise SystemExit(
                        f"{label}: {grid['nd'].shape[0]} rows against the default "
                        f"combination's {default_grid['nd'].shape[0]} -- the switch "
                        "moved the shape of the grid, not just its values"
                    )
                # The controls move no table this routine reads, so their grids
                # must be byte-identical to the default's. If one is not, the
                # expected collision below would fail for the wrong reason.
                if combo in EXPECTED_IDENTICAL:
                    for key in ("nd", "md_in", "mdt_in_a", "mdt_in_b"):
                        if not np.array_equal(grid[key], default_grid[key], equal_nan=True):
                            raise SystemExit(
                                f"{label}: {key} differs from the default combination's "
                                "grid, but this combination is recorded as moving "
                                "nothing the routine reads"
                            )

                grid_path = tmpdir / f"grid_s{setup}_{combo}.npz"
                np.savez(grid_path, **{k: grid[k] for k in ("nd", "md_in", "mdt_in_a", "mdt_in_b")})

                tags, abort_nd, abort_md, abort_mdt = abort_rows(t)
                abort_path = tmpdir / f"abort_s{setup}_{combo}.npz"
                np.savez(
                    abort_path,
                    tags=np.array(tags, dtype=np.str_),
                    nd=abort_nd,
                    md=abort_md,
                    mdt=abort_mdt,
                )

                tables_path = tmpdir / f"tables_s{setup}_{combo}.npz"
                np.savez(
                    tables_path,
                    mode=np.asarray(t.mode, dtype=np.int32),
                    component=np.asarray(t.component, dtype=np.int32),
                    num_eps=t.num_eps,
                    ddplim0=t.ddplim0,
                    mmid=t.mmid,
                    mlo=t.mlo,
                    x=t.x,
                    mm=t.mm,
                    rhocomp=t.rhocomp,
                    mfrac_0=t.mfrac_0,
                )

                out_path = tmpdir / f"out_s{setup}_{combo}.npz"
                meta = run_child(
                    _body(grid_path, abort_path, tables_path, out_path),
                    namelist_text=cm.render_namelist(source, setup, combo),
                    setup=setup,
                    label=label,
                )
                data = np.load(out_path, allow_pickle=False)

                reset = np.asarray(data["reset"], dtype=bool)
                _check_reset_is_where_it_can_be(label, t, reset)
                check_closed_forms(label, t, grid, data)
                if any(bool(t.mode[i]) for i in RESET_MODES):
                    check_straddle(label, t, grid, reset)

                counts = branch_counts(t, grid, reset, len(tags))
                check_hits(label, counts, expected_zero(t))

                if int(meta["nbox1_mismatch"]):
                    raise SystemExit(
                        f"{label}: the nbox={rows} call disagrees with {rows} nbox=1 "
                        f"calls in {meta['nbox1_mismatch']} elements -- the routine is "
                        "not row-independent, so a batched golden is not the reference "
                        "a per-row port should be compared against"
                    )
                if int(meta["mdt_invariance_mismatch"]):
                    raise SystemExit(
                        f"{label}: changing mdt changed {meta['mdt_invariance_mismatch']} "
                        "output elements -- mdt IS read somewhere, which contradicts "
                        ":40/:135/:243/:256 and invalidates the reset detector"
                    )

                prefix = f"s{setup}_{combo}_"
                for key in ("nd", "md_in", "mdt_in_a", "mdt_in_b", "tags"):
                    arrays[prefix + key] = grid[key]
                for key in OUTPUTS:
                    arrays[prefix + key] = data[key]
                arrays[prefix + "reset"] = reset.astype(np.int8)
                arrays[prefix + "hits"] = np.array([counts[p] for p in PREDICATES], np.int64)
                arrays[prefix + "nbox1_mismatch"] = np.array(meta["nbox1_mismatch"], np.int64)
                arrays[prefix + "mdt_mismatch"] = np.array(
                    meta["mdt_invariance_mismatch"], np.int64
                )
                arrays[prefix + "abort_tags"] = np.array(tags, dtype=np.str_)
                arrays[prefix + "abort_fatal"] = np.array(meta["abort_fatal"], np.int64)
                arrays[prefix + "abort_message"] = np.array(meta["abort_message"], np.str_)

                records[f"s{setup}.{combo}"] = _digest(data, reset)
                data.close()
                if not quiet:
                    print(
                        f"  setup {setup} {combo:<12} rows={rows:<3} "
                        f"reset={counts['reset_fired']:<4} "
                        f"mask_true={counts['mask_true']:<4} "
                        f"aborts={counts['abort_true']}"
                    )

    # Before anything is written. A collapsed archive that reaches the disk is
    # one every byte-equality test then agrees with.
    check_varied(
        records,
        expected_identical=[
            (f"s{s}.{a}", f"s{s}.{b}")
            for s in SETUPS
            for i, a in enumerate(EXPECTED_IDENTICAL)
            for b in EXPECTED_IDENTICAL[i + 1 :]
        ],
        what="drydiam (setup, combination) records",
    )
    print(
        f"  witness : {len(SETUPS)} setups x {len(COMBOS)} combinations all distinct "
        f"except the {len(EXPECTED_IDENTICAL)} recorded controls per setup"
    )

    arrays["_case"] = np.array("drydiam")
    arrays["_mode"] = np.array("leaf")
    arrays["_variant"] = np.array("f64")
    arrays["_rows"] = np.array(total_rows * len(COMBOS), dtype=np.int64)
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_combos"] = np.array(COMBOS, dtype=np.str_)
    arrays["_identical_combos"] = np.array(EXPECTED_IDENTICAL, dtype=np.str_)
    arrays["_predicates"] = np.array(PREDICATES, dtype=np.str_)
    arrays["_expected_zero"] = np.array(
        json.dumps({f"s{s}": expected_zero(tables(s, "default")) for s in SETUPS}, sort_keys=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"wrote {path.name}  {path.stat().st_size / 1e3:.0f} kB")
    return path


def _check_reset_is_where_it_can_be(label: str, t: modes.ModeTables, reset: np.ndarray) -> None:
    """`:245-246` cannot rewrite anything outside an active mode 1..3.

    The detector is `mdt_out_a == mdt_out_b`, which is a statement about two
    runs and not about the loop bounds. If it ever reported a hit in slot 4 or
    in an inactive slot, the detector -- not the Fortran -- would be what had
    gone wrong, and every reset row in the archive would be suspect.
    """
    allowed = np.zeros(NMODES, dtype=bool)
    for imode in RESET_MODES:
        allowed[imode] = bool(t.mode[imode])
    stray = np.argwhere(reset & ~allowed[None, :])
    if stray.size:
        raise SystemExit(
            f"{label}: the reset detector fired at (row, mode) "
            f"{[(int(r), int(m) + 1) for r, m in stray[:8]]}, outside the active part of "
            "mode_nuc_sol..mode_acc_sol -- mdt is being read, or the two garbage sets "
            "are not disjoint"
        )


def _digest(data, reset: np.ndarray) -> dict[str, str]:
    """A hashable summary of one record, over raw bytes rather than values.

    Bytes, because `mdt_out` legitimately carries NaN and `NaN != NaN` would
    make every record differ from itself. `check_varied` would then never report
    a collision, and the guard would be one that cannot fail.
    """
    out = {
        key: hashlib.sha256(np.ascontiguousarray(data[key], dtype=np.float64).tobytes()).hexdigest()
        for key in OUTPUTS
    }
    out["reset"] = hashlib.sha256(reset.astype(np.int8).tobytes()).hexdigest()
    return out


def _body(grid_path: Path, abort_path: Path, tables_path: Path, out_path: Path) -> str:
    """The child: the tables check, the grid twice, then the abort rows singly.

    Source text rather than a module because it runs in a subprocess with only
    the extension importable. `tests/test_capture_scripts.py` compiles it and
    checks the setup readback it inherits from `leaf_common.CHILD_PREAMBLE`.
    """
    return f'''
_grid = np.load({str(grid_path)!r}, allow_pickle=False)
_want = np.load({str(tables_path)!r}, allow_pickle=False)
_abort = np.load({str(abort_path)!r}, allow_pickle=False)

_nmodes = int(_want["mode"].shape[0])
_ncp = int(_want["mm"].shape[0])


def _fail(payload):
    print("@@FAIL@@" + json.dumps(payload))
    raise SystemExit(0)


def _bits(a):
    """Bit patterns, so -0.0 and a passed-through NaN compare as themselves."""
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def _differs(a, b):
    return int(np.count_nonzero(_bits(a) != _bits(b)))


# The grid's abscissae are derived from these tables, so if the Fortran holds
# different ones the sample points no longer straddle anything. Byte equality,
# not a tolerance: one ulp on `x` moves the reset threshold by ~3 ulp.
def _check_tables():
    for _name, _pair in (
        ("mode", g.wrap_mode_int("mode", _nmodes)),
        ("component", g.wrap_mode_cp_int("component", _nmodes, _ncp)),
        ("num_eps", g.wrap_mode_real("num_eps", _nmodes)),
        ("ddplim0", g.wrap_mode_real("ddplim0", _nmodes)),
        ("mmid", g.wrap_mode_real("mmid", _nmodes)),
        ("mlo", g.wrap_mode_real("mlo", _nmodes)),
        ("x", g.wrap_mode_real("x", _nmodes)),
        ("mm", g.wrap_cp_real("mm", _ncp)),
        ("rhocomp", g.wrap_cp_real("rhocomp", _ncp)),
        ("mfrac_0", g.wrap_mode_cp_real("mfrac_0", _nmodes, _ncp)),
    ):
        _got, _e = _pair
        if int(_e) != 0:
            _fail({{"stage": "wrap table", "field": _name, "ierr": int(_e)}})
        _got = np.asarray(_got)
        if not np.array_equal(_got, _want[_name].astype(_got.dtype)):
            _fail({{"stage": "table mismatch", "field": _name,
                    "fortran": _got.ravel().tolist(),
                    "python": _want[_name].ravel().tolist()}})


_check_tables()


def _counts():
    return tuple(int(v) for v in g.wrap_ereport_count())


def _last_message():
    _s, _r, _m = g.wrap_ereport_last()
    return (_m.decode() if isinstance(_m, bytes) else _m).strip()


def call(what, fn, *args):
    """One driver call that must not reach the shim."""
    _before = _counts()
    _res = fn(*args)
    _after = _counts()
    if _after != _before:
        _fail({{"stage": "ereport", "what": what, "before": _before,
                "after": _after, "message": _last_message()}})
    if int(_res[-1]) != 0:
        _fail({{"stage": "ierr", "what": what, "ierr": int(_res[-1])}})
    return _res


def abort_call(what, fn, *args):
    """One driver call that MUST reach the shim, exactly once, fatally."""
    _before = _counts()
    fn(*args)
    _after = _counts()
    _msg = _last_message()
    _moved = tuple(a - b for a, b in zip(_after, _before))
    if _moved != (1, 0, 0) or "dvol or drydp <= 0" not in _msg:
        _fail({{"stage": "expected abort", "what": what,
                "moved": _moved, "message": _msg}})
    return _moved[0], _msg


_nd, _md = _grid["nd"], _grid["md_in"]
_mdt_a, _mdt_b = _grid["mdt_in_a"], _grid["mdt_in_b"]
_rows = int(_nd.shape[0])

_dp_a, _dv_a, _md_a, _mdt_out_a, _ = call("main/mdt_a", g.leaf_drydiam, _nd, _md, _mdt_a)
_dp_b, _dv_b, _md_b, _mdt_out_b, _ = call("main/mdt_b", g.leaf_drydiam, _nd, _md, _mdt_b)

# mdt is written and never read (:40, :135, :243, :256). Two disjoint garbage
# arrays must leave every other output bit identical.
_mdt_mismatch = (_differs(_dp_a, _dp_b) + _differs(_dv_a, _dv_b) + _differs(_md_a, _md_b))

# mdt_out is passed straight through unless the reset overwrote it with
# mlo(imode), so the two runs agree at a slot exactly where the reset fired.
_reset = _mdt_out_a == _mdt_out_b

# :265-267 reduces over the whole nbox extent, so a batched call and a stack of
# one-row calls are not obviously the same computation. Compared bit-wise.
_mismatch = 0
for _r in range(_rows):
    _one = call("row%d" % _r, g.leaf_drydiam, _nd[_r:_r + 1], _md[_r:_r + 1], _mdt_a[_r:_r + 1])
    for _got, _ref in zip(_one[:4], (_dp_a, _dv_a, _md_a, _mdt_out_a)):
        _mismatch += _differs(_got, _ref[_r:_r + 1])

_fatal, _messages = [], []
for _i, _tag in enumerate(_abort["tags"]):
    _n, _msg = abort_call("abort/%s" % _tag, g.leaf_drydiam, _abort["nd"][_i:_i + 1],
                          _abort["md"][_i:_i + 1], _abort["mdt"][_i:_i + 1])
    _fatal.append(_n)
    _messages.append(_msg)

np.savez({str(out_path)!r}, drydp=_dp_a, dvol=_dv_a, md_out=_md_a,
         mdt_out_a=_mdt_out_a, mdt_out_b=_mdt_out_b, reset=_reset)

print("@@RESULT@@" + json.dumps({{
    "rows": _rows,
    "nbox1_mismatch": _mismatch,
    "mdt_invariance_mismatch": _mdt_mismatch,
    "abort_fatal": _fatal,
    "abort_message": _messages,
}}))
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grid and stop")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(
            f"{len(SETUPS)} setups x {len(COMBOS)} combinations = "
            f"{len(SETUPS) * len(COMBOS)} subprocesses -> {args.out / ARCHIVE}"
        )
        for setup in SETUPS:
            t = tables(setup, "default")
            grid = build_grid(t)
            reset_modes = [i + 1 for i in RESET_MODES if bool(t.mode[i])]
            print(
                f"  setup {setup}: {grid['nd'].shape[0]:>2} rows, "
                f"active {[i + 1 for i in range(NMODES) if bool(t.mode[i])]}, "
                f"reset modes {reset_modes or 'none (control)'}, "
                f"abort mode {abort_mode(t) + 1}"
            )
            for imode in reset_modes:
                icp = next(c for c in range(t.ncp) if bool(t.component[imode - 1, c]))
                print(
                    f"      mode {imode} md threshold (cp{icp}) = "
                    f"{_reset_md_threshold(t, imode - 1, icp)!r}"
                )
        return 0

    print(f"sweeping leaf_drydiam -> {args.out / ARCHIVE}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
