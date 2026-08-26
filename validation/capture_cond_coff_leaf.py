#!/usr/bin/env python3
"""Sweep `ukca_cond_coff_v` over the inputs the physics can reach (task 46).

    python validation/capture_cond_coff_leaf.py --dry-run
    python validation/capture_cond_coff_leaf.py       # writes tests/goldens/

`ukca_cond_coff_v` turns a particle radius and the local air state into the
condensation coefficient `cc` (m^3 s-1) and the condensation sink `sinkarr`, for
one condensable vapour. It is the first routine of phase E and the narrowest:
two outputs, no loop-carried state, no per-setup table, and no fidelity flag.

What makes it worth a leaf sweep anyway is that **three of its four closed
forms have never executed**. Every shipped namelist and every scenario in
`inputs/` sets `ifuchs = 1` and `idcmfp = 1`; the branch dump therefore covers
one corner of a 2x2. `ifuchs = 2` (Fuchs-Sutugin 1971) and `idcmfp = 2` (the
GLOMAP-bin diffusion coefficient) are reachable configurations that no
trajectory golden constrains at all.

Which argument reaches which output
-----------------------------------

The routine's two switches do not merely select a formula, they select which
*arguments* are read. That is the structure this capture is built to pin,
because it is what the port has to get right and what a trajectory fixture --
running one corner -- cannot check:

| argument | idcmfp = 1 | idcmfp = 2 |
|---|---|---|
| `tsqrt`  | read (`vel_cp`, `dcoff_cp`) | read (`vel_cp`, then `mfp_cp`) |
| `rhoa`   | read (`:174`)               | **not read** |
| `airdm3` | read (`:177`)               | **not read** |
| `t`      | **not read**                | read (`:179`) |
| `pmid`   | **not read**                | read (`:179`) |
| `dmol`   | read (`term2`, `term3`)     | **not read** |
| `difvol` | **not read**                | read (`term8`) |

Six of those seven cells are asserted here as byte equalities between calls or
between rows -- an inert argument must move nothing at all, and a live one must
move something. `se` is the seventh and is handled separately below.

`tsqrt` and `t` are separate arguments and are **not** required to agree. The
caller passes `SQRT(t)` (`ukca_conden.F90:281`), but the callee never checks,
so a `tsqrt_decoupled` block feeds `tsqrt = SQRT(t_b)` with `t = 999.0`. At
`idcmfp = 1` those rows must be byte-equal to the consistent `t = t_b` rows,
which proves `t` is unread rather than merely un-influential on this grid; at
`idcmfp = 2` they must differ.

`se` is inert at the value the model runs
-----------------------------------------

`se` enters `ifuchs = 2` only through

    akn = 1.0/(1.0 + 1.33*kn*fkn*(1.0/se - 1.0))            (`:210`)

and `ukca_conden.F90:235-237` sets **both** `se_sol` and `se_ins` to `1.0`, so
`1.0/se - 1.0` is exactly `0.0` and `akn` is exactly `1.0` on every row of
every validated run. The interfacial-transport correction the Fuchs-Sutugin
branch exists to apply is switched off by its own coefficient.

That is not what the routine's own header says. `ukca_cond_coff_v.F90:60`
documents `SE : Sticking efficiency [taken as 0.3 from Raes et al 1992]`, and
0.3 gives `1.0/se - 1.0 = 2.333...`, which is not inert. It is the same class
of defect as `ukca_conden`'s own header, which CLAUDE.md already records as
claiming `se_ins = 0.3` while the live value is `1.0`: the comment describes a
configuration the code does not run. Both `1.0` (live) and `0.3` (documented)
are swept, and the capture asserts they differ -- so the claim that the header
is wrong is re-derivable from the committed golden rather than from this
paragraph.

Setup independence, measured
----------------------------

`ukca_cond_coff_v` takes no `glomap_variables` argument and reads no per-setup
table, which is an argument about the signature. The measurement is that the
whole grid is run twice, under `i_mode_setup` 1 and 4, and the two results are
required to be byte-equal -- the treatment `coag_mode` got in phase C, and for
the same reason: a table that is *believed* to be setup-independent and a table
that is *shown* to be are different states of knowledge.

Setup 4 rather than 2: it is the first setup whose gas table carries
`msec_org`, so it is the one where the second live scalar set (`mmcg = 0.15`,
`difvol = 204.14`) is a configuration the model can actually reach.

The pressure axis carries the divide-by-a-constant hazard
---------------------------------------------------------

`:179` divides by a scalar literal, `pmid(:)/101325.0`. That is exactly the
site XLA rewrites into `multiply(pmid, 1/101325.0)` -- eagerly, and only on
some jax versions -- which cost phase D 73 tests and produced
`numerics.true_divide`. The rewrite is not detectable on most inputs: of twelve
plausible pressures only two, `9.0e4` and `1.05e5`, give a different double
under the reciprocal form. So the `pmid` axis is chosen to include seven values
that discriminate, alongside `101325.0` itself, where the quotient is exactly
`1.0` and the two forms agree. A grid that sampled round pressures would pass
against a port that had the rewrite in it, which is precisely how the phase-D
failures happened.

What is refused rather than captured
------------------------------------

`ifuchs` and `idcmfp` outside `{1, 2}` are refused by the leaf driver and are
not in this golden. Neither switch is validated anywhere upstream --
`glomap_box_config_mod.F90:155` reads both from the namelist and
`validate_config` does not constrain them -- and out of range `idcmfp` leaves
`dcoff_cp` never assigned while both Fuchs branches read it. Capturing that
would commit uninitialised memory as a reference, stable enough within a
process to pass every byte-equality test written against it; see issue #19 and
`validation/f2py/glomap_leaf_mod.F90`'s `leaf_cond_coff`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_modes import render_namelist
from leaf_common import CHILD_PREAMBLE, F2PY_DIR, NAMELISTS, REPO, check_varied, run_child

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "cond_coff.f64.leaf.npz"

SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_cond_coff_v.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

# One process each. Nothing in this routine latches, but `ukca_mode_setup`
# never deallocates, so a second `wrap_init` in the same process would return
# the first setup's tables and the two arms would compare setup 1 against
# itself -- byte-equal, green, and meaningless.
SETUPS: tuple[int, ...] = (1, 4)

# --- the row grid -----------------------------------------------------------
#
# Decimal literals throughout, never `np.logspace`. `logspace` is
# `10.0 ** linspace(...)`, which puts the *sample points* through libm and made
# four of the phase-C numerics abscissae platform-dependent before anyone
# noticed the results were being compared at grids that did not match.

BASE_RP = float("5.0e-8")
BASE_T = float("283.0")
BASE_AIRDM3 = float("2.5e25")
BASE_RHOA = float("1.2")
BASE_PMID = float("1.0e5")

# `rp` is `wetdp*0.5*y2` (`ukca_conden.F90:311`), so it is a wet radius scaled
# by a mode-width factor: nucleation-mode wet radii reach below 1 nm and the
# coarse insoluble mode above 1 um.
RP_AXIS = tuple(
    float(s)
    for s in (
        "1.0e-10",
        "3.0e-10",
        "5.0e-10",
        "1.0e-9",
        "3.0e-9",
        "5.0e-9",
        "1.0e-8",
        "2.0e-8",
        "5.0e-8",
        "1.0e-7",
        "2.0e-7",
        "5.0e-7",
        "1.0e-6",
        "2.0e-6",
        "5.0e-6",
        "1.0e-5",
        "2.0e-5",
    )
)

# Temperature spans the four namelists (273.15-292.0) and then the range a
# column would see, because `t**1.75` at `:179` is the only place an argument
# reaches an output through a non-integer power and it is worth sweeping wide.
T_AXIS = tuple(
    float(s)
    for s in (
        "180.0",
        "200.0",
        "220.0",
        "240.0",
        "260.0",
        "273.15",
        "283.0",
        "292.0",
        "303.0",
        "313.0",
        "320.0",
    )
)

AIRDM3_AXIS = tuple(
    float(s) for s in ("1.0e23", "5.0e23", "1.0e24", "5.0e24", "1.0e25", "2.5e25", "3.0e25")
)

RHOA_AXIS = tuple(float(s) for s in ("0.05", "0.1", "0.3", "0.5", "0.8", "1.0", "1.2", "1.4"))

# Seven of these differ between `p/101325.0` and `p*(1/101325.0)`; the
# remainder do not. `verify_pmid_axis_discriminates` re-derives the split
# rather than trusting this comment, and requires at least five.
PMID_AXIS = tuple(
    float(s)
    for s in (
        "1.2e4",
        "1.5e4",
        "2.0e4",
        "2.4e4",
        "4.5e4",
        "6.5e4",
        "8.4e4",
        "9.0e4",
        "9.6e4",
        "1.0e5",
        "101325.0",
    )
)

# The `t` a decoupled row carries. Absurd on purpose: if `idcmfp = 1` ever
# starts reading `t`, a plausible temperature would move the answer a little
# and 999.0 moves it a lot.
DECOUPLED_T = float("999.0")

BLOCKS = ("main", "rp", "t", "airdm3", "rhoa", "pmid", "tsqrt_decoupled", "masked_off")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def build_grid() -> dict[str, np.ndarray]:
    """Every row the sweep runs, with the block each belongs to.

    One flat set of rows, run whole by every call, rather than one grid per
    configuration. That is what lets a row in the `t` block and its partner in
    `tsqrt_decoupled` be compared *within* one call, with every other argument
    and every scalar term identical by construction.
    """
    rp: list[float] = []
    t: list[float] = []
    tsqrt: list[float] = []
    airdm3: list[float] = []
    rhoa: list[float] = []
    pmid: list[float] = []
    mask: list[int] = []
    block: list[int] = []

    def add(*, r, temp, ts, ad, rh, pm, m, blk):
        rp.append(r)
        t.append(temp)
        tsqrt.append(ts)
        airdm3.append(ad)
        rhoa.append(rh)
        pmid.append(pm)
        mask.append(m)
        block.append(_block_code(blk))

    # `np.sqrt` is IEEE-correctly-rounded and so is gfortran's SQRT, so the
    # abscissae themselves are exact on both sides -- unlike a power.
    def sq(x: float) -> float:
        return float(np.sqrt(np.float64(x)))

    for r in RP_AXIS:
        for temp in T_AXIS:
            add(
                r=r,
                temp=temp,
                ts=sq(temp),
                ad=BASE_AIRDM3,
                rh=BASE_RHOA,
                pm=BASE_PMID,
                m=1,
                blk="main",
            )

    for r in RP_AXIS:
        add(
            r=r,
            temp=BASE_T,
            ts=sq(BASE_T),
            ad=BASE_AIRDM3,
            rh=BASE_RHOA,
            pm=BASE_PMID,
            m=1,
            blk="rp",
        )

    for temp in T_AXIS:
        add(
            r=BASE_RP,
            temp=temp,
            ts=sq(temp),
            ad=BASE_AIRDM3,
            rh=BASE_RHOA,
            pm=BASE_PMID,
            m=1,
            blk="t",
        )

    for ad in AIRDM3_AXIS:
        add(
            r=BASE_RP,
            temp=BASE_T,
            ts=sq(BASE_T),
            ad=ad,
            rh=BASE_RHOA,
            pm=BASE_PMID,
            m=1,
            blk="airdm3",
        )

    for rh in RHOA_AXIS:
        add(
            r=BASE_RP,
            temp=BASE_T,
            ts=sq(BASE_T),
            ad=BASE_AIRDM3,
            rh=rh,
            pm=BASE_PMID,
            m=1,
            blk="rhoa",
        )

    for pm in PMID_AXIS:
        add(
            r=BASE_RP,
            temp=BASE_T,
            ts=sq(BASE_T),
            ad=BASE_AIRDM3,
            rh=BASE_RHOA,
            pm=pm,
            m=1,
            blk="pmid",
        )

    # Row i here pairs with row i of the `t` block: same tsqrt, absurd t.
    for temp in T_AXIS:
        add(
            r=BASE_RP,
            temp=DECOUPLED_T,
            ts=sq(temp),
            ad=BASE_AIRDM3,
            rh=BASE_RHOA,
            pm=BASE_PMID,
            m=1,
            blk="tsqrt_decoupled",
        )

    # Masked-off rows carry values that would produce inf or NaN if the mask
    # leaked. `WHERE` never evaluates them; a port that multiplies by the mask
    # instead of selecting on it gets `0.0 * inf = NaN` and fails here.
    poison = (
        0.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        float("1.0e308"),
        float("5.0e-324"),
    )
    for p in poison:
        add(r=p, temp=p, ts=p, ad=p, rh=p, pm=p, m=0, blk="masked_off")
    # ... and one row that is ordinary in every way except that it is masked
    # out, so a port that ignores the mask entirely also fails rather than
    # merely a port that multiplies by it.
    add(
        r=BASE_RP,
        temp=BASE_T,
        ts=float(np.sqrt(BASE_T)),
        ad=BASE_AIRDM3,
        rh=BASE_RHOA,
        pm=BASE_PMID,
        m=0,
        blk="masked_off",
    )

    return {
        "rp": np.array(rp, dtype=np.float64),
        "t": np.array(t, dtype=np.float64),
        "tsqrt": np.array(tsqrt, dtype=np.float64),
        "airdm3": np.array(airdm3, dtype=np.float64),
        "rhoa": np.array(rhoa, dtype=np.float64),
        "pmid": np.array(pmid, dtype=np.float64),
        "mask": np.array(mask, dtype=np.int32),
        "block": np.array(block, dtype=np.int32),
    }


# --- the call grid ----------------------------------------------------------
#
# `mmcg`, `dmol` and `difvol` are per-condensable scalars, not free parameters.
# `ukca_conden.F90:292-293` reads `dmol = dimen(jv)` and `mmcg = mm_gas(jv)`
# from the gas table, and `:324-328` hard-codes `difvol` to 51.96 for H2SO4 and
# 204.14 for the two secondary-organic slots. `mm_gas` gives 0.098 and 0.15,
# and `dimen` gives 4.5e-10 for BOTH -- so `dmol` has exactly one live value
# and the second one below exists only to show the argument is read at all.

SCALARS: tuple[dict, ...] = (
    dict(name="h2so4", mmcg=0.098, se=1.0, dmol=4.5e-10, difvol=51.96, live=1),
    dict(name="sec_org", mmcg=0.15, se=1.0, dmol=4.5e-10, difvol=204.14, live=1),
    # Differs from `h2so4` in `difvol` ALONE: inert at idcmfp=1, live at 2.
    dict(name="h2so4_difvol_org", mmcg=0.098, se=1.0, dmol=4.5e-10, difvol=204.14, live=0),
    # Differs from `h2so4` in `dmol` ALONE: live at idcmfp=1, inert at 2.
    dict(name="h2so4_dmol_small", mmcg=0.098, se=1.0, dmol=3.0e-10, difvol=51.96, live=0),
    # The routine header's `se`, which the model does not run.
    dict(name="h2so4_se_header", mmcg=0.098, se=0.3, dmol=4.5e-10, difvol=51.96, live=0),
    dict(name="h2so4_se_half", mmcg=0.098, se=0.5, dmol=4.5e-10, difvol=51.96, live=0),
)

SWITCHES: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2))


def build_calls() -> list[dict]:
    calls = []
    for sc in SCALARS:
        for ifuchs, idcmfp in SWITCHES:
            calls.append({**sc, "ifuchs": ifuchs, "idcmfp": idcmfp})
    return calls


def call_label(c: dict) -> str:
    return f"{c['name']}|ifuchs={c['ifuchs']}|idcmfp={c['idcmfp']}"


# --- the child --------------------------------------------------------------

CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_cond_coff_leaf import build_grid, build_calls, call_label\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "calls = build_calls()\n"
    "cc_rows, sink_rows = [], []\n"
    "for _c in calls:\n"
    "    _cc, _sink, _ierr = call(call_label(_c), g.leaf_cond_coff,\n"
    "                             grid['mask'], grid['rp'], grid['tsqrt'],\n"
    "                             grid['airdm3'], grid['rhoa'], grid['pmid'],\n"
    "                             grid['t'], _c['mmcg'], _c['se'], _c['dmol'],\n"
    "                             _c['difvol'], _c['ifuchs'], _c['idcmfp'])\n"
    "    cc_rows.append(np.asarray(_cc))\n"
    "    sink_rows.append(np.asarray(_sink))\n"
    "\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, cc=np.stack(cc_rows), sinkarr=np.stack(sink_rows))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

# Built at import so `tests/test_capture_scripts.py` can compile it without a
# built extension: a syntax error here would otherwise surface only during a
# capture, two subprocesses deep.
_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification, all of it before np.savez_compressed ---------------------


def verify_pmid_axis_discriminates(minimum: int = 5) -> int:
    """The `pmid` axis must be able to see the divide-by-a-constant rewrite.

    `:179` computes `pmid(:)/101325.0`. XLA rewrites that into a multiply by
    the reciprocal, which is a different double on *some* inputs and not on
    most. Counting how many of this axis's points discriminate is the only way
    to know the grid can fail against a port that has the rewrite in it -- and
    a count, not a boolean, because "at least one" is how phase D's six-value
    humidity grid passed while missing 481 of 1301 points.
    """
    c = 101325.0
    inv = 1.0 / c
    n = sum(1 for p in PMID_AXIS if (p / c) != (p * inv))
    if n < minimum:
        raise SystemExit(
            f"only {n} of {len(PMID_AXIS)} pmid values distinguish p/101325.0 from "
            f"p*(1/101325.0); at least {minimum} are needed for this axis to be "
            "able to catch the XLA rewrite (issue #23, numerics.true_divide)"
        )
    return n


def verify_source_literals() -> dict[str, int]:
    """Re-read the Fortran for every literal this capture's reasoning rests on.

    Each of these appears in the module docstring as a fact. If the vendored
    tree is ever updated and one of them moves, the docstring becomes a wrong
    claim about committed data, which is the failure mode CLAUDE.md records as
    recurring in every review so far. So they are parsed, not remembered.
    """
    text = SOURCE.read_text(encoding="utf-8")
    conden = (REPO / "fortran" / "src" / "ukca" / "ukca_conden.F90").read_text(encoding="utf-8")
    required = {
        "dair=19.7": "dair=19.7" in text.replace(" ", ""),
        "101325.0": "101325.0" in text,
        "t**1.75": "t(:)**1.75" in text.replace(" ", ""),
        "fkn 1.71/1.33": "1.71" in text and "1.33" in text,
        "term6 4.0e6*pi": "term6=4.0e6*pi" in text.replace(" ", ""),
        "header se 0.3": "taken as 0.3 from Raes" in text,
        "se_sol=1.0": "se_sol=1.0" in conden.replace(" ", ""),
        "se_ins=1.0": "se_ins=1.0" in conden.replace(" ", ""),
        "difvol 51.96": "difvol=51.96" in conden.replace(" ", ""),
        "difvol 204.14": "difvol=204.14" in conden.replace(" ", ""),
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit(
            "the vendored source no longer contains: "
            + ", ".join(missing)
            + " -- this capture's docstring describes a routine that has changed"
        )
    return {"checked": len(required)}


def _idx(grid: dict[str, np.ndarray], block: str) -> np.ndarray:
    return np.flatnonzero(grid["block"] == _block_code(block))


def _pair(calls: list[dict], name_a: str, name_b: str, ifuchs: int, idcmfp: int) -> tuple[int, int]:
    def find(name):
        for i, c in enumerate(calls):
            if c["name"] == name and c["ifuchs"] == ifuchs and c["idcmfp"] == idcmfp:
                return i
        raise SystemExit(f"no call {name} at ifuchs={ifuchs}, idcmfp={idcmfp}")

    return find(name_a), find(name_b)


# (argument, the call that differs in it alone, the idcmfp at which it is INERT)
INERT_AT = (
    ("difvol", "h2so4_difvol_org", 1),
    ("dmol", "h2so4_dmol_small", 2),
)

# (block, the idcmfp at which every row of that block must give the same answer)
CONSTANT_BLOCK_AT = (
    ("airdm3", 2),
    ("rhoa", 2),
    ("pmid", 1),
)


def verify_outputs(cc: np.ndarray, sinkarr: np.ndarray, grid: dict, calls: list[dict]) -> dict:
    """Everything the golden claims, checked against the golden itself.

    Runs before the archive is written, never as a test afterwards. A guard
    that lives in the test suite catches a collapsed capture exactly one run too
    late -- phase C wrote one, committed it, and passed every byte-equality test
    against it.
    """
    nsetup, ncall, nrow = cc.shape
    if (nsetup, ncall, nrow) != (len(SETUPS), len(calls), grid["rp"].size):
        want = (len(SETUPS), len(calls), grid["rp"].size)
        raise SystemExit(f"cc has shape {cc.shape}, expected {want}")

    report: dict[str, int] = {}

    # 1. The mask is a select, not a multiply.
    off = _idx(grid, "masked_off")
    for arr, what in ((cc, "cc"), (sinkarr, "sinkarr")):
        bad = np.flatnonzero(arr[:, :, off] != 0.0)
        if bad.size:
            raise SystemExit(f"{what} is non-zero on {bad.size} masked-off entries")
    report["masked_rows"] = int(off.size)

    # 2. Every live row produced a positive, finite coefficient.
    on = np.flatnonzero(grid["mask"] == 1)
    for arr, what in ((cc, "cc"), (sinkarr, "sinkarr")):
        live = arr[:, :, on]
        if not np.all(np.isfinite(live)):
            n_bad = int((~np.isfinite(live)).sum())
            raise SystemExit(f"{what} is not finite on {n_bad} live entries")
        if not np.all(live > 0.0):
            raise SystemExit(f"{what} is not positive on {int((live <= 0.0).sum())} live entries")
    report["live_rows"] = int(on.size)

    # 3. Setup independence, measured rather than argued.
    if not (np.array_equal(cc[0], cc[1]) and np.array_equal(sinkarr[0], sinkarr[1])):
        raise SystemExit(
            f"i_mode_setup {SETUPS[0]} and {SETUPS[1]} disagree -- ukca_cond_coff_v "
            "reads a per-setup table after all, and every claim in this file's "
            "docstring about setup independence is wrong"
        )
    report["setups_compared"] = len(SETUPS)

    # 4/5. An inert argument moves nothing; the same argument, at the other
    #      switch setting, must move something. Both halves, because only the
    #      second one can fail if the call never reached the Fortran at all.
    checked = 0
    for arg, other, inert_idcmfp in INERT_AT:
        for ifuchs in (1, 2):
            for idcmfp in (1, 2):
                a, b = _pair(calls, "h2so4", other, ifuchs, idcmfp)
                same = np.array_equal(cc[0, a], cc[0, b]) and np.array_equal(
                    sinkarr[0, a], sinkarr[0, b]
                )
                want_same = idcmfp == inert_idcmfp
                if same != want_same:
                    raise SystemExit(
                        f"{arg} at ifuchs={ifuchs}, idcmfp={idcmfp}: outputs "
                        f"{'agree' if same else 'differ'}, expected to "
                        f"{'agree' if want_same else 'differ'}"
                    )
                checked += 1
    report["inert_argument_pairs"] = checked

    # 6. `se` is read on both branches, so the header's 0.3 and the live 1.0
    #    must differ everywhere. This is what makes "the header is wrong"
    #    re-derivable from the committed golden.
    for ifuchs in (1, 2):
        for idcmfp in (1, 2):
            a, b = _pair(calls, "h2so4", "h2so4_se_header", ifuchs, idcmfp)
            if np.array_equal(cc[0, a], cc[0, b]):
                raise SystemExit(
                    f"se=1.0 and se=0.3 give identical cc at ifuchs={ifuchs}, "
                    f"idcmfp={idcmfp} -- se is not being read"
                )

    # 7. A block whose only varying argument is unread must be constant, and
    #    must not be constant at the other setting.
    for block, const_idcmfp in CONSTANT_BLOCK_AT:
        rows = _idx(grid, block)
        for i, c in enumerate(calls):
            vals = cc[0, i, rows]
            constant = bool(np.all(vals == vals[0]))
            want = c["idcmfp"] == const_idcmfp
            if constant != want:
                raise SystemExit(
                    f"{block} block under {call_label(c)}: cc is "
                    f"{'constant' if constant else 'varying'}, expected "
                    f"{'constant' if want else 'varying'}"
                )
    report["constant_block_checks"] = len(CONSTANT_BLOCK_AT) * len(calls)

    # 8. `rp` and `t` are read at every setting.
    for block in ("rp", "t"):
        rows = _idx(grid, block)
        for i, c in enumerate(calls):
            vals = cc[0, i, rows]
            if np.all(vals == vals[0]):
                raise SystemExit(f"{block} block is constant under {call_label(c)}")

    # 9. `t` is unread at idcmfp=1: a row carrying tsqrt=SQRT(t_b) and t=999.0
    #    must be byte-equal to the consistent t_b row, and must not be at
    #    idcmfp=2.
    t_rows = _idx(grid, "t")
    d_rows = _idx(grid, "tsqrt_decoupled")
    if t_rows.size != d_rows.size:
        raise SystemExit("the t and tsqrt_decoupled blocks are not the same length")
    for i, c in enumerate(calls):
        same = np.array_equal(cc[0, i, t_rows], cc[0, i, d_rows])
        want = c["idcmfp"] == 1
        if same != want:
            raise SystemExit(
                f"tsqrt_decoupled under {call_label(c)}: rows "
                f"{'match' if same else 'differ from'} their consistent partners, "
                f"expected to {'match' if want else 'differ'}"
            )
    report["decoupled_rows"] = int(d_rows.size)

    # 10. `cc/sinkarr` is `term6*dcoff_cp*1e-6` on BOTH branches -- no `rp`, no
    #     `denom`, no `fkn`, no `akn`. So it is constant along the rp axis, and
    #     that is an algebraic identity of the source rather than a property of
    #     these numbers. Not byte-exact: the two expressions divide by the same
    #     `denom` in a different order.
    rp_rows = _idx(grid, "rp")
    worst = 0.0
    for i in range(len(calls)):
        ratio = cc[0, i, rp_rows] / sinkarr[0, i, rp_rows]
        spread = float(np.max(np.abs(ratio - ratio[0])) / abs(ratio[0]))
        worst = max(worst, spread)
    if worst > 1e-14:
        raise SystemExit(
            f"cc/sinkarr varies by {worst:.3e} along the rp axis; the source says "
            "it is term6*dcoff_cp*1e-6, which does not depend on rp"
        )
    report["cc_over_sinkarr_spread"] = worst

    return report


def capture(out_dir: Path, quiet: bool = False) -> Path:
    n_discriminating = verify_pmid_axis_discriminates()
    verify_source_literals()

    grid = build_grid()
    calls = build_calls()

    cc_by_setup, sink_by_setup = [], []
    paths: list[str] = []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"cond_coff setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as data:
                cc_by_setup.append(data["cc"])
                sink_by_setup.append(data["sinkarr"])
            if not quiet:
                print(f"  i_mode_setup = {setup}: {len(calls)} calls x {grid['rp'].size} rows")
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    cc = np.stack(cc_by_setup)
    sinkarr = np.stack(sink_by_setup)

    # One fingerprint per call, within one setup. The setup axis is compared
    # directly in verify_outputs, where a collision is the expected answer.
    check_varied(
        {
            call_label(c): {"cc": cc[0, i].tolist(), "sinkarr": sinkarr[0, i].tolist()}
            for i, c in enumerate(calls)
        },
        expected_identical=[
            (call_label({**c, "name": "h2so4"}), call_label(c))
            for c in calls
            for arg, other, inert in INERT_AT
            if c["name"] == other and c["idcmfp"] == inert
        ],
        what="cond_coff configurations",
    )

    report = verify_outputs(cc, sinkarr, grid, calls)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        cc=cc,
        sinkarr=sinkarr,
        setups=np.array(SETUPS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        block=grid["block"],
        mask=grid["mask"],
        rp=grid["rp"],
        t=grid["t"],
        tsqrt=grid["tsqrt"],
        airdm3=grid["airdm3"],
        rhoa=grid["rhoa"],
        pmid=grid["pmid"],
        call_name=np.array([c["name"] for c in calls], dtype="U24"),
        call_ifuchs=np.array([c["ifuchs"] for c in calls], dtype=np.int64),
        call_idcmfp=np.array([c["idcmfp"] for c in calls], dtype=np.int64),
        call_mmcg=np.array([c["mmcg"] for c in calls], dtype=np.float64),
        call_se=np.array([c["se"] for c in calls], dtype=np.float64),
        call_dmol=np.array([c["dmol"] for c in calls], dtype=np.float64),
        call_difvol=np.array([c["difvol"] for c in calls], dtype=np.float64),
        call_live=np.array([c["live"] for c in calls], dtype=np.int64),
        _pmid_discriminating=np.int64(n_discriminating),
    )
    if not quiet:
        print(f"  wrote {path.name}: {cc.shape} cc, {report['live_rows']} live rows per call")
        print(f"  pmid points that see the reciprocal rewrite: {n_discriminating}/{len(PMID_AXIS)}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grids and stop")
    args = parser.parse_args(argv)

    grid = build_grid()
    calls = build_calls()

    if args.dry_run:
        n = verify_pmid_axis_discriminates()
        verify_source_literals()
        print(f"leaf cond_coff sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['rp'].size:>6,}  ({int((grid['mask'] == 0).sum())} masked off)")
        for b in BLOCKS:
            print(f"    {b:<18}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(
            f"  calls       {len(calls):>6,}  = {len(SCALARS)} scalar sets"
            f" x {len(SWITCHES)} switch pairs"
        )
        print(f"  setups      {len(SETUPS):>6,}  {SETUPS}")
        print(f"  pmid points that discriminate p/c from p*(1/c): {n}/{len(PMID_AXIS)}")
        return 0

    print(f"sweeping leaf_cond_coff -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
