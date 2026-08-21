#!/usr/bin/env python3
"""Sweep `ukca_water_content_v` through `leaf_water_content` and capture it.

    python validation/capture_water_leaf.py --dry-run
    python validation/capture_water_leaf.py            # writes tests/goldens/*.npz

`ukca_water_content_v` evaluates ZSR binary molalities from Jacobson's Table
B.10 -- eight polynomial coefficients per (cation, anion) pair, twelve pairs --
to get the water content of each mode at a given relative humidity. It is one
of the five routines CLAUDE.md says needs a sequential `lax.scan` rather than a
`vmap`, because the pairing loop *depletes* each ion pair's concentration as it
goes: pair (1,-2) sees whatever pair (1,-4) left behind, not the ion's total.

This is a leaf-driver capture in the phase-D pattern (task 35, following
task 35a's `capture_vapour_leaf.py`-style siblings): call the real Fortran with
chosen inputs, in-process, through `validation/f2py/`. Everything shared with
the other three phase-D captures -- one subprocess per configuration, the
ereport bracket, the anti-collapse guard -- lives in `leaf_common.py` and is
used here, not reimplemented.

Three hazards specific to this routine, recorded in CLAUDE.md, issue #22 and
task 40's port (`src/glomap_jax/physics/water_content.py`):

* **`y(3,-4:-1,0:7)` has a NEGATIVE lower bound on its second axis.**
  `leaf_water_content` cannot express that to f2py, so `cl` and `ions` cross as
  plain `(n,8)` arrays and get remapped `+5` (Fortran 1-based) inside the
  wrapper (`validation/f2py/glomap_leaf_mod.F90:294-303`) -- `+4` in 0-based
  numpy terms, i.e. `col(species) = species + 4`. Getting this wrong does not
  raise: it returns a different, still-finite, still-plausible number with
  `ierr == 0` (measured below, and pinned by `tests/test_water_fixtures.py`).
* **`l_fix_ukca_water_content` patches `y(1,-3,6)` in place and never restores
  it, AND it is hardcoded `.TRUE.` before any namelist is read.**
  `glomap_box_config_mod.F90:322` sets the flag unconditionally, then
  `wrap_init` runs `init_state` -> `ukca_volume_mode` -> `ukca_water_content_v`,
  which patches the SAVEd, THREADPRIVATE (hence implicitly `SAVE`) `y` table on
  that very first call and never restores it. So calling `wrap_init` at all --
  regardless of what `wrap_set_fix_water_content` does afterwards -- makes the
  unpatched coefficient unreachable for the rest of the process (measured: flag
  read back as 0 post-init, routine still returns the patched answer). The only
  way to the unfixed arm is COLD: set the flag before `wrap_init` ever runs, and
  don't call `wrap_init` at all for this routine, which reads no `SAVE`d
  constant that init would populate -- only `ncation`/`nanion` (`PARAMETER`s),
  its own `DATA` tables, and the flag. Hence `run_child(..., init=False)` for
  BOTH arms below, and `wrap_set_fix_water_content` before either.
* **The two arms differ in a SECOND, unrelated way that has nothing to do with
  the patched coefficient**, and it is the bigger effect in this capture: the
  unfixed (`ELSE`) branch sets `aw` from `rh` ONCE, before the `ic`/`ia` loop,
  and then clamps it up to each pair's `rh_min` floor IN PLACE without ever
  resetting it -- so a pair visited late in the loop can see a floor left
  behind by an EARLIER pair, not its own. The fixed (`IF`) branch re-reads
  `rh` at the top of every iteration specifically to avoid this (the source
  comment says so: ":277 since they may get erroneously overwritten each time
  around the loop"). This threshold check runs for every one of the 12 pairs
  regardless of which ions are active, so it can move a row's answer even when
  that row never touches pair (1,-3) at all -- see `simulate` below, and the
  anti-collapse checks in `capture()`, which demonstrate the two effects on two
  different rows for exactly that reason.

What this capture is NOT: `leaf_water_content` returns `wc` only -- `mb` and
`clp`, the per-pair molality and ion-pair concentration, are local variables in
`ukca_water_content_v` and are not INTENT(OUT) anywhere on the path this
binding exposes. An earlier plan for this task described a second capture
block over `mb`/`clp`; that block cannot be built without changing
`validation/f2py/glomap_leaf_mod.F90`, which is outside this task's ownership,
so it is not attempted here. `simulate` below reconstructs `mb`/`clp`
analytically instead, and is cross-checked against every captured `wc` before
the golden is written.

Independent of the port's own tables
-------------------------------------
`src/glomap_jax/physics/water_tables.py` and `validation/extract_water_literals.py`
already hold Jacobson's Table B.10, extracted from this same source file, and
`src/glomap_jax/physics/water_content.py` (task 40) already ports the routine.
This module is deliberately a *second* extraction and a *second*
reimplementation, written independently and used only for grid design and
diagnostics (`parse_reference`, `simulate`, `divide_branch_hits`) -- never fed
into a Fortran call. The call always goes through the real `y` table inside
`leaf_water_content`; a bug in the parser or in `simulate` here could make a
diagnostic wrong, but cannot make the golden wrong.

Grids
-----
`rh`: 0.0 to 1.3 in steps of 0.005 (`0.005 * arange(261)`, integer times a
literal -- no `logspace`, no `pow`), plus the exact value and both
representable neighbours of every distinct `rh_min` the source declares
(`RH_THRESHOLDS_PCT`, cross-checked against the parsed source by
`_verify_thresholds` so the two cannot silently drift apart). `:281`/`:305` is
a strict `<`, so a value exactly on the floor takes the *false* arm (no clamp)
and the representable-predecessor takes the true arm -- the three-point
neighbourhood is what pins that a `<=` port would flip it. Sweeping to 1.3
rather than stopping at 0.9 is deliberate: `ukca_volume_mode.F90:306-307`
clamps its caller's `rh` to `[0.1, 0.9]` before this routine ever sees it, so
the whole reason a leaf driver exists here is to reach outside that box.

Compositions: every one of the 12 (cation, anion) pairs alone, at 4 magnitudes
spanning `1e-25` to `1e-12` mol/cc plus 2 deep-subnormal magnitudes on two
representative pairs (issue #15 -- XLA flushes subnormal *results*, and
`clp/mb` can be one); all 16 presence combinations over the 4 ions
`ukca_volume_mode` can actually populate (H+, Na+, Cl-, SO4--); all 8 presence
combinations over the 3 ions it never can (HSO4-, NO3-, NH4+) -- the *only*
way to reach the 8 pairs that involve them, since the caller never sets them;
and 3 depletion cases that pin the pairing loop's *order* (ic outer, ia inner):
H+ exhausted by Cl- before SO4-- is reached, SO4-- exhausted by H+ before Na+
is reached two `ic` passes later, and the caller's own charge-balance identity.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leaf_common import (
    CHILD_PREAMBLE,
    F2PY_DIR,
    NAMELISTS,
    REPO,
    check_varied,
    run_child,
)

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "water_content.f64.leaf.npz"

NANION = 4
NCATION = 3
# (cation, anion) in the source's own enumeration order: ic ascending, ia -1
# down to -4 -- the same order `ukca_water_content_v`'s nested loop visits.
PAIRS: tuple[tuple[int, int], ...] = tuple((ic, ia) for ic in (1, 2, 3) for ia in (-1, -2, -3, -4))

NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")
SETUP = 1  # setup-independent: ncation/nanion are PARAMETERs, not per-setup.

MAGNITUDES = tuple(float(s) for s in ("1e-25", "1e-20", "1e-16", "1e-12"))
SUBNORMAL_MAGNITUDES = tuple(float(s) for s in ("1e-310", "1e-320"))
BASELINE = float("1e-9")

RH_THRESHOLDS_PCT = tuple(
    float(s) for s in ("0.019", "0.065", "0.30", "0.37", "0.47", "0.58", "0.62")
)


def col(species: int) -> int:
    """Signed ion species (-4..3) to the 0-based column `leaf_water_content`
    expects, i.e. the inverse of `validation/f2py/glomap_leaf_mod.F90:294-297`.

    That wrapper does `ions(i,j) = ions_i(i, j+5)` for Fortran 1-based `j+5`;
    in 0-based numpy terms that is `species + 4`. Getting this offset wrong is
    the main correctness risk in this whole capture: it does not raise, it
    returns a different, finite, plausible number (measured in the capture
    report and pinned by `tests/test_water_fixtures.py`).
    """
    if not -NANION <= species <= NCATION:
        raise IndexError(f"ion species {species} outside -{NANION}..{NCATION}")
    return species + NANION


def _new_row(
    conc: dict[int, float], ions_on: set[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    cl_row = np.zeros(8, dtype=np.float64)
    ions_row = np.zeros(8, dtype=np.int32)
    for species, value in conc.items():
        cl_row[col(species)] = value
    active = ions_on if ions_on is not None else {s for s, v in conc.items() if v != 0.0}
    for species in active:
        ions_row[col(species)] = 1
    return cl_row, ions_row


def _rh_grid() -> np.ndarray:
    base = 0.005 * np.arange(261, dtype=np.float64)  # 0.0 .. 1.3, integer x literal
    extra: list[float] = []
    for t in RH_THRESHOLDS_PCT:
        extra += [t, np.nextafter(t, 0.0), np.nextafter(t, 1.0)]
    return np.unique(np.concatenate([base, np.array(extra, dtype=np.float64)]))


def build_main_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(rh, cl, ions, comp_label)` for the main sweep. Pure and deterministic
    -- called once in the parent for shapes/diagnostics and once in each child
    (imported fresh, not shared) to build exactly the same inputs."""
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []

    # Group 1: each of the 12 pairs alone, at 4 magnitudes.
    for ic, ia in PAIRS:
        for mag in MAGNITUDES:
            cl_row, ions_row = _new_row({ic: mag, ia: mag})
            rows.append((f"pair_{ic}_{ia}_mag_{mag:.0e}", cl_row, ions_row))
    # ... plus 2 deep-subnormal magnitudes on a z=1 pair and the z=2 (SO4) one,
    # so `cli/n` is subnormal for both the divide-by-1 and divide-by-2 cases.
    for ic, ia in ((1, -4), (1, -2)):
        for mag in SUBNORMAL_MAGNITUDES:
            cl_row, ions_row = _new_row({ic: mag, ia: mag})
            rows.append((f"pair_{ic}_{ia}_subnormal_{mag:.0e}", cl_row, ions_row))

    # Group 2: all 16 presence combinations over the ions the caller
    # (`ukca_volume_mode.F90:391-419`) can actually populate: H+, Na+, Cl-,
    # SO4--. This is the natural depletion pattern the box model itself drives.
    for h, na, cl4, so4 in itertools.product((0, 1), repeat=4):
        conc = {}
        if h:
            conc[1] = BASELINE
        if na:
            conc[3] = BASELINE
        if cl4:
            conc[-4] = BASELINE
        if so4:
            conc[-2] = BASELINE
        cl_row, ions_row = _new_row(conc)
        rows.append((f"combo4_h{h}_na{na}_cl{cl4}_so4{so4}", cl_row, ions_row))

    # Group 3: all 8 presence combinations over HSO4- (-1), NO3- (-3), NH4+
    # (2), the ions the caller never populates -- the only way to reach the 8
    # pairs involving them at all. H+ is always present as the pairing cation.
    for hso4, no3, nh4 in itertools.product((0, 1), repeat=3):
        conc = {1: BASELINE}
        if hso4:
            conc[-1] = BASELINE
        if no3:
            conc[-3] = BASELINE
        if nh4:
            conc[2] = BASELINE
        cl_row, ions_row = _new_row(conc)
        rows.append((f"combo3_hso4{hso4}_no3{no3}_nh4{nh4}", cl_row, ions_row))

    # Group 4: depletion cases pinning the pairing loop's order (ic outer, ia
    # inner: :250-268). A broadcast that computed all pairs from the *original*
    # cl and only subtracted afterwards would disagree with every one of these.
    #
    # (a) H+ fully consumed by (1,-4) before (1,-2) is reached, so cli(1) is
    # exactly 0 by the time the SO4 pair is processed.
    cl_row, ions_row = _new_row({1: 3e-10, -4: 3e-10, -2: 5e-10}, ions_on={1, -4, -2})
    rows.append(("depletion_a_H_exhausted_by_Cl_before_SO4", cl_row, ions_row))

    # (b) SO4-- fully consumed by (1,-2) before (3,-2) is reached -- two `ic`
    # passes later -- so Na+ gets nothing despite being present. cl(1) is set
    # to exactly 2x cl(-2) (n(1)=2, n(-2)=1 for this z=1/z=2 pair) so both
    # cli(1) and cli(-2) land on exactly 0, not merely small.
    cl_row, ions_row = _new_row({1: 8e-10, -2: 4e-10, 3: 6e-10}, ions_on={1, -2, 3})
    rows.append(("depletion_b_SO4_exhausted_by_H_starves_Na", cl_row, ions_row))

    # (c) The caller's own charge balance (`ukca_volume_mode.F90:416-417`,
    # restricted to the terms active here): cl(1) = 2*cl(-2) + cl(-4) - cl(3).
    so4_c, cl4_c, na_c = 2e-10, 3e-10, 1e-10
    h_c = max(2.0 * so4_c + cl4_c - na_c, 0.0)
    cl_row, ions_row = _new_row({1: h_c, -2: so4_c, -4: cl4_c, 3: na_c}, ions_on={1, -2, -4, 3})
    rows.append(("depletion_c_callers_charge_balance", cl_row, ions_row))

    rh = _rh_grid()
    labels = np.array([r[0] for r in rows])
    cl = np.stack([r[1] for r in rows])
    ions = np.stack([r[2] for r in rows])
    return rh, cl, ions, labels


def build_mask_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(mask_rh, mask_cl, mask_ions, patterns, pattern_label)` for the mask
    sweep, which pins the compaction at `ukca_water_content_v.F90:237-243`
    (`m`/`idx` built from `mask`, everything else keyed off `idx(:m)`)."""
    mask_rh = np.linspace(0.0, 1.3, 21)
    cl_row, ions_row = _new_row({1: BASELINE, -4: BASELINE})
    n = mask_rh.shape[0]
    all_true = np.ones(n, dtype=np.int32)
    all_false = np.zeros(n, dtype=np.int32)
    interleaved = np.array([i % 2 for i in range(n)], dtype=np.int32)
    patterns = np.stack([all_true, all_false, interleaved])
    pattern_label = np.array(["all_true", "all_false", "interleaved"])
    return mask_rh, cl_row, ions_row, patterns, pattern_label


# ---------------------------------------------------------------------------
# An independent second reading of the source, for grid design and
# diagnostics only. See the module docstring: never fed into a Fortran call,
# never imported from `water_tables.py` / `extract_water_literals.py`.
# ---------------------------------------------------------------------------

_SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_water_content_v.F90"

_Y_RE = re.compile(
    r"DATA\s*\(\s*y\(\s*(\d+)\s*,\s*(-\d+)\s*,\s*j\s*\)\s*,\s*j\s*=\s*0\s*,\s*7\s*\)\s*/(.*?)/",
    re.DOTALL,
)
_LIMITS_RE = re.compile(
    r"DATA\s+rh_min\(\s*(\d+)\s*,\s*(-\d+)\s*\)\s*,\s*molal_max\(\s*\1\s*,\s*\2\s*\)\s*/(.*?)/",
    re.DOTALL,
)
_Z_RE = re.compile(r"DATA\s*\(z\(i\),i=-nanion,ncation\)\s*/(.*?)/", re.IGNORECASE)
_PATCH_RE = re.compile(
    r"IF\s*\(\s*glomap_config%l_fix_ukca_water_content\s*\)\s*"
    r"y\(\s*(\d+)\s*,\s*(-\d+)\s*,\s*(\d+)\s*\)\s*=\s*([^\s!]+)"
)


def _fortran_numbers(blob: str) -> list[float]:
    cleaned = re.sub(r"&\s*\n\s*", "", blob)
    cleaned = re.sub(r"!.*", "", cleaned)
    return [float(t.strip()) for t in cleaned.split(",") if t.strip()]


def parse_reference() -> dict:
    """Independently re-derive `y`, `rh_min`, `molal_max`, `z` and the patch
    straight from the vendored source. Used only by `_verify_thresholds`,
    `simulate` and `divide_branch_hits` below -- diagnostics, not the
    call path."""
    text = _SOURCE.read_text(encoding="utf-8")

    coeffs: dict[tuple[int, int], list[float]] = {}
    for c, a, blob in _Y_RE.findall(text):
        coeffs[(int(c), int(a))] = _fortran_numbers(blob)

    limits: dict[tuple[int, int], tuple[float, float]] = {}
    for c, a, blob in _LIMITS_RE.findall(text):
        limits[(int(c), int(a))] = tuple(_fortran_numbers(blob))

    if set(coeffs) != set(PAIRS) or set(limits) != set(PAIRS):
        raise SystemExit("parsed (cation, anion) pairs do not match the 12 this module expects")

    zmatch = _Z_RE.search(text)
    if not zmatch:
        raise SystemExit("could not find the DATA (z(i), ...) statement in the source")
    zvals = _fortran_numbers(zmatch.group(1))
    if len(zvals) != 8:
        raise SystemExit(f"z DATA statement: expected 8 values, parsed {len(zvals)}")
    z = {i: zvals[i + NANION] for i in range(-NANION, NCATION + 1)}

    patches = _PATCH_RE.findall(text)
    if len(patches) != 1:
        raise SystemExit(
            f"expected exactly one l_fix_ukca_water_content patch, found {len(patches)}"
        )
    pc, pa, pidx, pval = patches[0]
    patch = ((int(pc), int(pa)), int(pidx), float(pval))

    return {"coeffs": coeffs, "limits": limits, "z": z, "patch": patch}


def _verify_thresholds(ref: dict) -> None:
    declared = {v[0] / 1.0e2 for v in ref["limits"].values() if v[0] != 0.0}
    swept = set(RH_THRESHOLDS_PCT)
    if declared != swept:
        raise SystemExit(
            f"rh threshold mismatch: source declares {sorted(declared)}, grid sweeps "
            f"{sorted(swept)} -- the rh grid no longer pins every rh_min floor"
        )


def pair_active_hits(ions: np.ndarray) -> dict[tuple[int, int], int]:
    """Composition rows (not row x rh cells) where a pair's `ions` are both
    set -- i.e. where `ukca_water_content_v`'s pairing WHERE-mask is true for
    that pair, for at least one row in the grid."""
    return {
        (ic, ia): int(np.sum((ions[:, col(ic)] == 1) & (ions[:, col(ia)] == 1))) for ic, ia in PAIRS
    }


def divide_branch_hits(ref: dict) -> dict[tuple[int, int], int]:
    """`:255-259`'s `n(ic)=n(ic)/z(ic)` branch, evaluated per pair from `z`
    alone (it does not depend on rh or concentration at all). Expected all
    zero for every (cation, anion) pair this source declares: it needs
    `z(ic) == z(ia)` (true only among the three cations, all z=1, which never
    pair with each other) simultaneously with `z(ia) != 1` (true only for the
    SO4 anion, z=2, which no cation matches) -- so the AND of the two is
    false for all 12 pairs. If this ever comes back nonzero the `z` table
    changed and every downstream claim about it needs re-checking.
    """
    z = ref["z"]
    eps = np.finfo(np.float64).eps
    return {(ic, ia): int(abs(z[ic] - z[ia]) < eps and abs(z[ia] - 1.0) > eps) for ic, ia in PAIRS}


_ORDER: tuple[int, ...] = (1, 2, 3)  # ic, ascending -- ukca_water_content_v.F90:250
_AORDER: tuple[int, ...] = (-4, -3, -2, -1)  # ia, ascending -- :251


def simulate(ref: dict, fix_water: bool, cl_row: np.ndarray, ions_row: np.ndarray, rh: float):
    """Reimplement `ukca_water_content_v` for one row and one rh value, in the
    Fortran's own iteration order, for both arms. Returns `(wc, clamped)`,
    where `clamped[(ic, ia)]` is True iff that pair's `mb` hit its
    `molal_max` ceiling on this call.

    Two loop-carried states, matching CLAUDE.md's "the scan is loop-carried
    twice" and task 40's finding:

    * `cli`, in BOTH arms -- the pairing loop depletes each ion's pool as it
      goes, so a pair visited late sees whatever an earlier pair left behind.
    * `aw`, in the UNFIXED arm only -- `rh` is copied into `aw` once, before
      the loop, and then only ever clamped UP to a pair's `rh_min` floor,
      never back down. So a pair with `rh_min = 0` visited after one with
      `rh_min = 0.62` sees `aw = 0.62`, not `rh`. The fixed arm re-reads `rh`
      at the top of every iteration and has no such carry-over. This is why
      the flag can move a row's `wc` even when the row never activates pair
      (1,-3) -- the only pair the coefficient patch touches.

    The `aw`-threshold check runs for EVERY (ic, ia) pair regardless of
    whether `ions` is set for it -- only the `mb` accumulation and the pairing
    assignment are `WHERE`-guarded. Skipping the threshold check for inactive
    pairs (an easy simplification to reach for) would silently drop the
    ratchet effect for exactly the rows built to demonstrate it.
    """
    coeffs = dict(ref["coeffs"])
    if fix_water:
        (pc, pa), pidx, pval = ref["patch"]
        coeffs[(pc, pa)] = list(coeffs[(pc, pa)])
        coeffs[(pc, pa)][pidx] = pval
    z = ref["z"]
    limits = ref["limits"]
    eps = np.finfo(np.float64).eps

    at = col
    cli = {s: float(cl_row[at(s)]) for s in range(-4, 4)}
    ions_on = {s: bool(ions_row[at(s)]) for s in range(-4, 4)}

    clp: dict[tuple[int, int], float] = {}
    for ic in _ORDER:
        for ia in _AORDER:
            n_ic, n_ia = z[ia], z[ic]
            if abs(z[ic] - z[ia]) < eps and abs(z[ia] - 1.0) > eps:
                n_ic, n_ia = n_ic / z[ic], n_ia / z[ia]
            if ions_on[ic] and ions_on[ia]:
                p = min(cli[ic] / n_ic, cli[ia] / n_ia)
                clp[(ic, ia)] = p
                cli[ic] -= n_ic * p
                cli[ia] -= n_ia * p
            else:
                clp[(ic, ia)] = 0.0

    mb: dict[tuple[int, int], float] = {}
    clamped: dict[tuple[int, int], bool] = {}
    aw = rh  # unfixed arm: set once, ratchets up, never reset (the bug)
    for ic in _ORDER:
        for ia in _AORDER:
            rh_min, molal_max = limits[(ic, ia)]
            if fix_water:
                aw_pair = rh if rh >= rh_min / 1.0e2 else rh_min / 1.0e2
            else:
                if aw < rh_min / 1.0e2:
                    aw = rh_min / 1.0e2
                aw_pair = aw
            if ions_on[ic] and ions_on[ia]:
                raw = sum(c * aw_pair**k for k, c in enumerate(coeffs[(ic, ia)]))
                clamped[(ic, ia)] = raw > molal_max
                mb[(ic, ia)] = min(raw, molal_max)
            else:
                clamped[(ic, ia)] = False
                mb[(ic, ia)] = 0.0

    total = 0.0
    for ic in _ORDER:
        for ia in _AORDER:
            if ions_on[ic] and ions_on[ia]:
                total += clp[(ic, ia)] / mb[(ic, ia)]
    return (1.0 / 18.0e-3) * total, clamped


# ---------------------------------------------------------------------------
# The child process. FALSE and TRUE each get their own (issue #22): the y
# table is DATA-initialised and THREADPRIVATE, hence implicitly SAVE, and the
# patch at :235 never restores it.
# ---------------------------------------------------------------------------

CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_water_leaf import build_main_grid, build_mask_grid\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "\n"
    "rh, cl, ions, labels = build_main_grid()\n"
    "C, R = cl.shape[0], rh.shape[0]\n"
    "cl_full = np.repeat(cl, R, axis=0)\n"
    "ions_full = np.repeat(ions.astype(np.int32), R, axis=0)\n"
    "rh_full = np.tile(rh, C)\n"
    "mask_full = np.ones(C * R, dtype=np.int32)\n"
    "wc, _ierr = call('leaf_water_content(main sweep)', g.leaf_water_content,\n"
    "                 mask_full, ions_full, cl_full, rh_full)\n"
    "wc = np.asarray(wc).reshape(C, R)\n"
    "\n"
    "mask_rh, mask_cl, mask_ions, patterns, pattern_label = build_mask_grid()\n"
    "Rm = mask_rh.shape[0]\n"
    "mask_cl_full = np.repeat(mask_cl[None, :], Rm, axis=0)\n"
    "mask_ions_full = np.repeat(mask_ions.astype(np.int32)[None, :], Rm, axis=0)\n"
    "mask_wc_rows = []\n"
    "for _p in range(patterns.shape[0]):\n"
    "    _m = patterns[_p].astype(np.int32)\n"
    "    _w, _ierr2 = call('leaf_water_content(mask pattern %d)' % _p, g.leaf_water_content,\n"
    "                      _m, mask_ions_full, mask_cl_full, mask_rh)\n"
    "    mask_wc_rows.append(np.asarray(_w))\n"
    "mask_wc = np.stack(mask_wc_rows)\n"
    "\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, wc=wc, mask_wc=mask_wc)\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'fix_water': int(_fix_water)}))\n"
)

# Exposed so `tests/test_capture_scripts.py`'s generic child-script checks
# (compile, setup-readback) cover this capture's child too, the same way they
# already cover `capture_modes.py`'s `_CHILD` and `leaf_common.CHILD_PREAMBLE`.
# `_do_init` is 0 for both arms here (see `capture()`), so the CHILD_PREAMBLE's
# own setup/flag readback block never runs -- `wrap_get_config_flags` itself
# requires `is_initialised`, which is exactly the state this routine cannot
# afford to reach for the unfixed arm. What confirms the flag took is not a
# readback but the physics: `simulate` cross-checks every captured `wc`
# against an independent computation of both arms before the golden is
# written, which is a strictly stronger guarantee than a boolean readback.
_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


def capture(out_dir: Path, quiet: bool = False) -> Path:
    ref = parse_reference()
    _verify_thresholds(ref)

    rh, cl, ions, labels = build_main_grid()
    mask_rh, mask_cl, mask_ions, patterns, pattern_label = build_mask_grid()
    C, R = cl.shape[0], rh.shape[0]

    npz_paths: dict[str, str] = {}
    # Both arms run COLD: init=False, so `leaf_water_content`'s own guard
    # (`is_initialised OR water_flag_set`) is satisfied by the flag alone, and
    # `wrap_init` never runs -- for either arm. Calling it for EITHER would
    # patch y(1,-3,6) permanently (see the module docstring) and make the
    # unfixed arm unreachable for the rest of that process. Order between the
    # two subprocesses does not matter -- each is a fresh process -- and is
    # kept FALSE-then-TRUE so a reader does not have to wonder.
    for name, fw in (("fix_water_off", 0), ("fix_water_on", 1)):
        record = run_child(
            CHILD_BODY,
            namelist_text=NAMELIST_TEXT,
            setup=SETUP,
            fix_water=fw,
            init=False,
            label=name,
        )
        npz_paths[name] = record["npz_path"]
        if not quiet:
            print(f"  {name:<16} fix_water={record['fix_water']} (cold, no wrap_init)")

    try:
        with (
            np.load(npz_paths["fix_water_off"]) as data_off,
            np.load(npz_paths["fix_water_on"]) as data_on,
        ):
            wc = np.stack([data_off["wc"], data_on["wc"]])
            mask_wc = np.stack([data_off["mask_wc"], data_on["mask_wc"]])
    finally:
        for p in npz_paths.values():
            Path(p).unlink(missing_ok=True)

    if wc.shape != (2, C, R):
        raise SystemExit(f"wc shape {wc.shape} != expected {(2, C, R)}")

    # -- Verification, all before np.savez_compressed. --

    # Mask compaction (`ukca_water_content_v.F90:237-243`): a masked-false row
    # must stay at the zero `leaf_water_content` pre-fills, and an
    # interleaved call must agree with the all-true call exactly at every
    # masked-true row (the compacted `idx(:m)` must not shift anything).
    idx_true, idx_false, idx_interleaved = 0, 1, 2
    for f in (0, 1):
        if not np.array_equal(mask_wc[f, idx_false], np.zeros(mask_rh.shape[0])):
            raise SystemExit(
                "mask=all-false left a nonzero wc -- the compaction guard is not what it says"
            )
        active = patterns[idx_interleaved].astype(bool)
        if not np.array_equal(mask_wc[f, idx_interleaved][active], mask_wc[f, idx_true][active]):
            raise SystemExit(
                "interleaved mask disagreed with the all-true call at a masked-true row"
            )
        if not np.array_equal(mask_wc[f, idx_interleaved][~active], np.zeros(int((~active).sum()))):
            raise SystemExit("interleaved mask wrote a nonzero value at a masked-false row")

    # Cross-check EVERY captured cell against `simulate`, the independent
    # reimplementation of both arms (including the aw-ratchet). This is the
    # main correctness pin for the whole archive: an offset bug in `col()`
    # (species landing in the wrong column), a depletion-order bug, or a
    # missed ratchet carry-over would all show up here as a disagreement
    # rather than as a plausible number -- see the capture report for the
    # measured max error.
    clamp_hits: dict[tuple[int, int], list[int]] = {p: [0, 0] for p in PAIRS}
    max_abs_err = 0.0
    for i in range(C):
        for j in range(R):
            rh_j = float(rh[j])
            for f, fw in ((0, False), (1, True)):
                predicted, clamped = simulate(ref, fw, cl[i], ions[i], rh_j)
                actual = float(wc[f, i, j])
                err = abs(predicted - actual)
                if err > max_abs_err:
                    max_abs_err = err
                if err > 1e-9 * max(1.0, abs(actual)):
                    raise SystemExit(
                        f"simulate() disagrees with the Fortran at row {labels[i]!r}, rh={rh_j!r}, "
                        f"fix_water={fw}: predicted {predicted!r}, captured {actual!r}"
                    )
                for pair, hit in clamped.items():
                    if hit:
                        clamp_hits[pair][f] += 1
    if not quiet:
        print(f"  simulate() cross-check: {C * R * 2:,} cells, max abs err {max_abs_err:.3e}")

    # Anti-collapse (issue #22 / task 40): the two effects are independent and
    # asserted on two DIFFERENT rows on purpose. A row that activates pair
    # (1,-3) demonstrates the coefficient patch; a row that never does (all
    # four caller-reachable ions, no (1,-3)) can only demonstrate the
    # aw-ratchet, since the patch cannot reach it. Collapsing this to one row
    # would not say which effect it showed.
    coef_rows = np.where((ions[:, col(1)] == 1) & (ions[:, col(-3)] == 1))[0]
    if not coef_rows.size:
        raise SystemExit("no row activates pair (1,-3) -- the coefficient-patch check is vacuous")
    if not np.any(wc[0][coef_rows, :] != wc[1][coef_rows, :]):
        raise SystemExit(
            "fix_water_off and fix_water_on agree on every (1,-3)-carrying row -- "
            "the coefficient patch had no effect (arm not reached cold, or the patch is dead)"
        )

    ratchet_label = "combo4_h1_na1_cl1_so41"
    ratchet_idx = np.where(labels == ratchet_label)[0]
    if not ratchet_idx.size:
        raise SystemExit(f"expected composition row {ratchet_label!r} is missing from group 2")
    ratchet_row = int(ratchet_idx[0])
    if ions[ratchet_row, col(1)] and ions[ratchet_row, col(-3)]:
        raise SystemExit(
            f"{ratchet_label} activates pair (1,-3) -- it no longer isolates the ratchet effect"
        )
    if not np.any(wc[0][ratchet_row, :] != wc[1][ratchet_row, :]):
        raise SystemExit(
            f"fix_water_off and fix_water_on agree at every rh for {ratchet_label!r}, which never "
            "activates pair (1,-3) -- the aw-ratchet effect is not demonstrated on its own"
        )
    low_rh = rh < 0.47
    if not low_rh.any():
        raise SystemExit(
            "rh grid has no point below 0.47 -- cannot check the low-rh ratchet divergence"
        )
    if not np.any(wc[0][ratchet_row, low_rh] != wc[1][ratchet_row, low_rh]):
        raise SystemExit(
            f"{ratchet_label}: fix_water_off and fix_water_on agree at every rh < 0.47 -- "
            "the ratchet effect is not demonstrated in the low-rh part of the grid"
        )

    check_varied(
        {
            "fix_water_off": {"wc": wc[0].tolist(), "mask_wc": mask_wc[0].tolist()},
            "fix_water_on": {"wc": wc[1].tolist(), "mask_wc": mask_wc[1].tolist()},
        },
        expected_identical=(),
        what="l_fix_ukca_water_content settings",
    )

    # Every pair must be activated by at least one composition row, or the
    # per-pair diagnostics below are describing a row that was never run.
    pair_hits = pair_active_hits(ions)
    dead = [p for p, n in pair_hits.items() if n == 0]
    if dead:
        raise SystemExit(f"composition grid never activates pair(s) {dead} -- extend group 1")

    # The z-only divide branch: expected dead for every pair (see
    # `divide_branch_hits`'s docstring). If this ever fires, the claim in the
    # capture report no longer holds and must not be written down unchecked.
    divide_hits = divide_branch_hits(ref)
    live = [p for p, v in divide_hits.items() if v]
    if live:
        raise SystemExit(
            f"the n(ic)/z(ic) divide branch at :255-259 fired for {live} -- re-derive the claim"
        )

    # molal_max clamp reachability, measured from `simulate` over the REAL
    # composition x rh grid -- not an isolated per-pair formula, which the
    # ratchet makes wrong for any row with more than one active pair (see
    # `simulate`'s docstring). Recorded once from this exact grid and this
    # exact composition set; either shrinking can only move a count towards
    # zero, never away from it, which is what makes this assertion able to
    # catch a deleted row or a narrowed rh range.
    never_off = {p for p, (off, _on) in clamp_hits.items() if off == 0}
    never_on = {p for p, (_off, on) in clamp_hits.items() if on == 0}
    EXPECTED_NEVER_CLAMPS_OFF = {(1, -1), (1, -2), (2, -1), (2, -3), (2, -4), (3, -1), (3, -2)}
    EXPECTED_NEVER_CLAMPS_ON = {(1, -1), (1, -2), (2, -1), (2, -3), (2, -4), (3, -1)}
    if never_off != EXPECTED_NEVER_CLAMPS_OFF:
        raise SystemExit(
            f"fix_water_off: molal_max reachability moved -- expected exactly "
            f"{sorted(EXPECTED_NEVER_CLAMPS_OFF)} never to clamp, got {sorted(never_off)}"
        )
    if never_on != EXPECTED_NEVER_CLAMPS_ON:
        raise SystemExit(
            f"fix_water_on: molal_max reachability moved -- expected exactly "
            f"{sorted(EXPECTED_NEVER_CLAMPS_ON)} never to clamp, got {sorted(never_on)}"
        )

    arrays = {
        "rh": rh,
        "cl": cl,
        "ions": ions.astype(np.int8),
        "comp_label": labels,
        "wc": wc,
        "fix_water_flags": np.array([0, 1], dtype=np.int64),
        "mask_rh": mask_rh,
        "mask_cl": mask_cl,
        "mask_ions": mask_ions.astype(np.int8),
        "mask_patterns": patterns.astype(np.int8),
        "mask_pattern_label": pattern_label,
        "mask_wc": mask_wc,
        "diag_pairs": np.array(PAIRS, dtype=np.int64),
        "diag_pair_active_hits": np.array([pair_hits[p] for p in PAIRS], dtype=np.int64),
        "diag_divide_branch_hits": np.array([divide_hits[p] for p in PAIRS], dtype=np.int64),
        "diag_molal_clamp_hits_fix_off": np.array(
            [clamp_hits[p][0] for p in PAIRS], dtype=np.int64
        ),
        "diag_molal_clamp_hits_fix_on": np.array([clamp_hits[p][1] for p in PAIRS], dtype=np.int64),
        "diag_rh_thresholds_pct": np.array(RH_THRESHOLDS_PCT, dtype=np.float64),
        "diag_simulate_max_abs_err": np.array(max_abs_err, dtype=np.float64),
        "_case": np.array("water_content"),
        "_mode": np.array("leaf"),
        "_variant": np.array("f64"),
        "_rows": np.array(C * R + patterns.shape[0] * mask_rh.shape[0]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"wrote {path.name}  {path.stat().st_size / 1e6:.3f} MB")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grids and stop")
    args = parser.parse_args(argv)

    rh, cl, ions, _labels = build_main_grid()
    mask_rh, _mask_cl, _mask_ions, patterns, _pattern_label = build_mask_grid()

    if args.dry_run:
        print(f"leaf water_content sweep -> {args.out / ARCHIVE}")
        print(f"  rh            {len(rh):>6,} points   [{rh.min():.3f}, {rh.max():.3f}]")
        print(f"  compositions  {len(cl):>6,} rows")
        print(f"  main cells    {len(cl) * len(rh):>6,} per fix_water_content setting")
        print(f"  mask rh       {len(mask_rh):>6,} points x {patterns.shape[0]} patterns")
        ref = parse_reference()
        _verify_thresholds(ref)
        hits = pair_active_hits(ions)
        print(f"  pairs active in >=1 row: {sum(1 for v in hits.values() if v)} / {len(PAIRS)}")
        return 0

    print(f"sweeping leaf_water_content -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
