"""Tasks 41-45 — `ukca_volume_mode`, byte-equal against the compiled routine.

Every comparison here is `assert_array_equal` against the *live* Fortran through
`leaf_volume_mode`, not against a tolerance and not against a committed archive.
One subprocess per `(setup, l_fix_ukca_water_content, l_fix_neg_pvol_wat)`:
`ukca_mode_setup` never deallocates, and `ukca_water_content_v` patches its own
`SAVE`d coefficient table in place and never restores it (issue #22).

`dvol` and `drydp` are fed from `leaf_drydiam` on the same rows, in the same
call, exactly as the model sequences them. Inventing them independently risks a
zero, and a single zero anywhere in a column trips the five-way guard at
`:704-708` -- which cannot survive its own `WRITE` (see
`test_the_five_way_guard_cannot_be_reached`), so it takes the process with it
and there is no comparison left to make.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
F2PY = REPO / "validation" / "f2py"
NAMELIST = REPO / "fortran" / "namelists" / "boundary_layer.nml"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_volume_mode.F90"

from glomap_jax.physics import modes, volume_mode  # noqa: E402

SETUPS = modes.supported_setups()

needs_binding = pytest.mark.skipif(
    not sorted(F2PY.glob("glomap_f2py*.so")),
    reason="binding not built; run validation/build_f2py.sh",
)

# ukca_volume_mode.F90:258. Strict `<`, so putls itself is tropospheric.
PUTLS = 1.5e4

# 0-based component slots. ncp is 6 in every supported setup.
CP_SU, CP_BC, CP_OC, CP_CL, CP_DU, CP_SO = range(6)

# The rh axis. 0.1 and 0.9 are the clamp bounds at :306-307 and both tests are
# strict, so the bounds themselves must come out UNCLAMPED and only 0.05/0.95
# fire. No shipped namelist exceeds 0.90, so nothing here has ever run.
RH_AXIS = (
    0.05,
    np.nextafter(0.1, 0.0),
    0.1,
    np.nextafter(0.1, 1.0),
    0.2,
    0.35,
    0.47,
    0.5,
    0.62,
    0.75,
    np.nextafter(0.9, 0.0),
    0.9,
    np.nextafter(0.9, 1.0),
    0.95,
)

# nd relative to num_eps. The test at :312 is strict, and the box seeds nd
# exactly at num_eps, so `eps_exact` must come out FALSE.
ND_KINDS = ("bulk", "eps_below", "eps_exact", "eps_above", "zero")

# Composition recipes, applied per mode over whichever components the setup
# gives it. A recipe a mode cannot carry falls back to `typical`.
VARIANTS = (
    "typical",
    "su_only",
    "cl_only",
    "cl_over_su",
    "oc_so",
    "insol_rich",
    "nosol",
    "negsu",
    "negsol",
    "tiny",
)

# Relative weights for `typical`, as decimal literals so the grid is
# reproducible bit for bit on any host (no libm in an abscissa).
TYPICAL_WEIGHTS = (0.50, 0.10, 0.20, 0.30, 0.40, 0.15)

T_REF, S_REF, RH_REF = 213.0, 1.0e-2, 0.6
P_TROP = 1.0e5
P_STRAT = 1.0e4

# The pmid axis. `putls` itself must be on it and must come out FALSE (`<`, not
# `<=`), and its two neighbouring doubles bracket it. Every value is a decimal
# literal or `np.nextafter`, both exact, so the axis is the same on any host.
PMID_AXIS = (
    1.0e4,
    1.4999e4,
    np.nextafter(PUTLS, 0.0),
    PUTLS,
    np.nextafter(PUTLS, np.inf),
    2.0e4,
    1.0e5,
)

# The t axis, swept at stratospheric pressure where t reaches an output. Chosen
# to walk `(NINT(wts/5))*5` across ukca_vapour's twelve `percent` entries and
# off the top of the table into the rhosol_strat = 1300.0 fall-through.
T_AXIS = (180.0, 204.0, 209.5, 218.0, 232.5, 255.5, 288.5, 306.0, 326.5)

# The s axis. bh2o = 1.609*s*pmid/p0 is clamped to [2e-8, 2e-6]
# (ukca_vapour.F90:141-142), and both ends of this axis are outside it.
S_AXIS = (1.0e-8, 2.0e-7, 2.0e-6, 1.0e-5, 1.0e-4, 1.0e-2)


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------

_CHILD = """
import json, sys
import numpy as np
sys.path.insert(0, {f2py!r})
sys.path.insert(0, {validation!r})
import glomap_f2py as g
import capture_modes as cm
import tempfile, pathlib

text = cm.render_namelist(pathlib.Path({namelist!r}).read_text(), {setup}, "default")
with tempfile.TemporaryDirectory() as tmp:
    nml = pathlib.Path(tmp) / "s.nml"
    nml.write_text(text)
    assert int(g.wrap_init(str(nml))) == 0
sizes = g.wrap_sizes()
assert int(sizes[7]) == {setup}, ("wrong setup: the namelist edit did not take",
                                  int(sizes[7]))
assert int(g.wrap_set_fix_water_content({fix_water})) == 0
assert int(g.wrap_set_fix_neg_pvol_wat({fix_neg})) == 0
fw, fn, os_, e = g.wrap_get_config_flags()
assert int(e) == 0 and int(fw) == {fix_water} and int(fn) == {fix_neg}, (
    "the flags did not take", int(fw), int(fn))

payload = json.loads(sys.stdin.read())
nd  = np.array(payload["nd"]);  md   = np.array(payload["md"])
mdt = np.array(payload["mdt"]); rh   = np.array(payload["rh"])
t   = np.array(payload["t"]);   pmid = np.array(payload["pmid"])
s   = np.array(payload["s"])

g.wrap_ereport_reset()
before = [int(v) for v in g.wrap_ereport_count()]
drydp, dvol, md_out, mdt_out, ierr = g.leaf_drydiam(nd, md, mdt)
assert int(ierr) == 0, ("leaf_drydiam", int(ierr))
mdwat, wvol, wetdp, rhopar, pvol, pvol_wat, ierr = g.leaf_volume_mode(
    nd, md_out, mdt_out, rh, dvol, drydp, t, pmid, s)
after = [int(v) for v in g.wrap_ereport_count()]

print("@@R@@" + json.dumps({{
    "ierr": int(ierr), "shim": [before, after],
    "drydp": drydp.tolist(), "dvol": dvol.tolist(),
    "md_out": md_out.tolist(), "mdt_out": mdt_out.tolist(),
    "mdwat": mdwat.tolist(), "wvol": wvol.tolist(), "wetdp": wetdp.tolist(),
    "rhopar": rhopar.tolist(), "pvol": pvol.tolist(),
    "pvol_wat": pvol_wat.tolist()}}))
"""


def _reference(setup: int, inputs: dict, *, fix_water: int = 1, fix_neg: int = 1) -> dict:
    """One subprocess, one configuration. Returns every output of both leaves."""
    script = _CHILD.format(
        f2py=str(F2PY),
        validation=str(REPO / "validation"),
        namelist=str(NAMELIST),
        setup=setup,
        fix_water=fix_water,
        fix_neg=fix_neg,
    )
    # Arrays cross as JSON on stdin: `repr` emits a bare `nan`, which is not
    # valid Python in the child, and the grid can carry one.
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps({k: np.asarray(v).tolist() for k, v in inputs.items()}),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    out = json.loads(proc.stdout[proc.stdout.rindex("@@R@@") + 5 :])
    assert out["ierr"] == 0, out["ierr"]
    assert out["shim"][0] == out["shim"][1], "the routine reached ereport; the comparison is void"
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def _filler(insol: list[int]) -> list[int]:
    """Insoluble components that do NOT feed `cl(-2)`.

    `cp_oc` and `cp_so` are insoluble in an insoluble mode and hygroscopic in a
    soluble one: `:372` and `:381` push both into the SO4 pool. So a recipe that
    needs `cl(-2)` exactly zero, or strictly negative, cannot use them as inert
    ballast -- which is what the first version of `cl_only` did, and it silently
    made `cl(-2)` positive.
    """
    return [c for c in insol if c not in (CP_OC, CP_SO)]


def _composition(tab, imode: int, variant: str) -> tuple[np.ndarray, str]:
    """One mode's `md` row, and the variant that was actually applied."""
    ncp = tab.ncp
    row = np.zeros(ncp)
    present = [c for c in range(ncp) if tab.component[imode, c]]
    if not present:
        return row, "empty"
    insol = [c for c in present if not tab.soluble[c]]
    sol = [c for c in present if tab.soluble[c]]
    weights = np.zeros(ncp)
    for c in present:
        weights[c] = TYPICAL_WEIGHTS[c]
    base = tab.mmid[imode] * weights / weights.sum()

    if variant == "typical":
        return base, variant
    if variant == "su_only" and CP_SU in present:
        row[CP_SU] = tab.mmid[imode]
        return row, variant
    if variant == "cl_only" and CP_CL in present:
        # No sulfate AND no organic: cl(-2) has to be exactly 0.0, and cp_oc
        # and cp_so feed it through the hygroscopic increments at :372/:381
        # just as cp_su does at :368. Then cl(3) and cl(-4) are the same double,
        # the charge balance is exactly 0.0 - 0.0, and ions(1) is FALSE.
        # Insoluble filler is bc/dust only, for the same reason.
        filler = _filler(insol)
        if not filler:
            return base, "typical"
        row[CP_CL] = tab.mmid[imode] * 0.8
        for c in filler:
            row[c] = tab.mmid[imode] * 0.2 / len(filler)
        return row, variant
    if variant == "cl_over_su" and CP_CL in present and CP_SU in present:
        # md(cl) >> md(su), so cl(1) ~ 2*cl(-2) is far below cl(-4) and the
        # FIRST pair the loop reaches, (1,-4), consumes all of H+. Pair (1,-2)
        # then still passes the presence mask -- built from the original cl --
        # and contributes exactly 0.0.
        filler = _filler(insol)
        row[CP_SU] = tab.mmid[imode] * 0.005
        row[CP_CL] = tab.mmid[imode] * (0.8 if filler else 0.995)
        for c in filler:
            row[c] = tab.mmid[imode] * 0.195 / len(filler)
        return row, variant
    if variant == "oc_so" and CP_OC in present and CP_SO in present:
        # Both hygroscopic-organic increments, so :372 and :381 both fire and
        # their ORDER is observable. Only setups 4 and 5 reach this.
        row[CP_OC] = tab.mmid[imode] * 0.55
        row[CP_SO] = tab.mmid[imode] * 0.35
        row[CP_SU] = tab.mmid[imode] * 0.10
        return row, variant
    if variant == "nosol" and insol and tab.modesol[imode] == 1:
        # mask_nosol: mdsol EXACTLY 0.0 with nd > num_eps. 0 hits in 2447
        # sampled golden points. Only a mode carrying an insoluble component can
        # supply it -- zeroing the whole mass of a purely soluble mode leaves
        # dvol = 0 and ukca_calc_drydiam trips its own guard first.
        for c in insol:
            row[c] = tab.mmid[imode] / len(insol)
        return row, variant
    if variant == "negsol" and CP_SU in present and insol and tab.modesol[imode] == 1:
        # mdsol < 0: the third state, neither mask_sol nor mask_nosol. The
        # insoluble mass has to dominate the DRY volume as well as the mass,
        # because a negative dvol would trip ukca_calc_drydiam and then the
        # five-way guard, which kills the process rather than reporting.
        for c in insol:
            row[c] = tab.mmid[imode] / len(insol)
        row[CP_SU] = -tab.mmid[imode] * 0.001
        return row, variant
    if variant == "negsu" and CP_SU in present and CP_CL in present and _filler(insol):
        # A NEGATIVE sulfate mass, so cl(-2) is negative and `ions` separates
        # `cl > 0.0` from `cl != 0.0`. Without this row a port that built the
        # presence mask from non-zeroness is byte-equal everywhere -- measured.
        # Insoluble filler keeps dvol, wvol and rhopar positive so the row does
        # not trip the five-way guard and take the process with it.
        filler = _filler(insol)
        for c in filler:
            row[c] = tab.mmid[imode] * 0.4 / len(filler)
        # cl(3) must be positive so pair (3,-2) passes the presence mask under
        # the mutation and does not under the faithful `cl > 0.0`.
        row[CP_CL] = tab.mmid[imode] * 0.6
        row[CP_SU] = -tab.mmid[imode] * 0.01
        return row, variant
    if variant == "insol_rich" and insol:
        row = base.copy()
        for c in sol:
            row[c] = base[c] * 0.01
        return row, variant
    if variant == "tiny":
        # Below ddplim0*0.1 for modes 1-3, so ukca_calc_drydiam's undersize
        # reset rewrites md and volume_mode is driven by the rewrite.
        return base * 1.0e-9, variant
    return base, "typical"


def _rows(tab, specs: list[dict]) -> dict:
    nmodes, ncp = modes.NMODES, tab.ncp
    n = len(specs)
    nd = np.zeros((n, nmodes))
    md = np.zeros((n, nmodes, ncp))
    applied = []
    for i, spec in enumerate(specs):
        kind = spec["nd"]
        for m in range(nmodes):
            eps = tab.num_eps[m]
            nd[i, m] = {
                "bulk": 1000.0,
                "zero": 0.0,
                "eps_exact": eps,
                "eps_below": np.nextafter(eps, 0.0),
                "eps_above": np.nextafter(eps, np.inf),
            }[kind]
        got = []
        for m in range(nmodes):
            md[i, m, :], which = _composition(tab, m, spec["variant"])
            got.append(which)
        applied.append(got)
    return {
        "nd": nd,
        "md": md,
        "mdt": md.sum(axis=2),
        "rh": np.array([s["rh"] for s in specs]),
        "t": np.array([s["t"] for s in specs]),
        "pmid": np.array([s["pmid"] for s in specs]),
        "s": np.array([s["s"] for s in specs]),
        "_variant_applied": applied,
    }


def _spec(rh=RH_REF, nd="bulk", variant="typical", t=T_REF, pmid=P_TROP, s=S_REF):
    return {"rh": rh, "nd": nd, "variant": variant, "t": t, "pmid": pmid, "s": s}


@functools.lru_cache(maxsize=16)
def _trop_grid(setup: int):
    """Tropospheric rows only: every branch task 41 owns, and no other.

    `pmid` is 1e5 throughout, so `:434`'s override never fires and `mdwat` is
    the ZSR result alone. Task 43 adds the stratospheric column.
    """
    tab = modes.build(setup)
    specs = [_spec(rh=rh) for rh in RH_AXIS]
    specs += [_spec(nd=kind) for kind in ND_KINDS]
    specs += [_spec(rh=rh, variant=v) for v in VARIANTS for rh in (0.2, 0.62, 0.9)]
    grid = _rows(tab, specs)
    return tab, grid


def _inputs(grid: dict) -> dict:
    return {k: v for k, v in grid.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Task 41 — the soluble branch's water content
# ---------------------------------------------------------------------------


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("fix_water", [0, 1], ids=["unfixed", "fixed"])
def test_mdwat_is_byte_equal_in_the_troposphere(setup, fix_water):
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid), fix_water=fix_water)
    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],  # the undersize reset rewrote md before volume_mode saw it
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=bool(fix_water),
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_the_water_content_flag_moves_mdwat_where_it_can(setup):
    """Both settings, and an assertion that they genuinely differ.

    `l_fix_ukca_water_content` patches a `SAVE`d table in place and never
    restores it, so a both-settings test run in ONE process compares the patched
    table against itself and passes. Each setting gets its own subprocess above;
    this is what makes the pair non-vacuous.

    The mechanism is not the patched coefficient. Pair (1,-3) is the only one
    the patch touches and `cl(-3)` is identically zero here (`ncp = 6`,
    `cp_no3 = 7`). What moves is `aw`: in the unfixed arm the per-pair `rh_min`
    floor ratchets cumulatively across the pair loop, so Na+/Cl- sees an `aw`
    already raised by NH4+/NO3-, whose ions are not present.
    """
    tab, grid = _trop_grid(setup)
    unfixed = _reference(setup, _inputs(grid), fix_water=0)["mdwat"]
    fixed = _reference(setup, _inputs(grid), fix_water=1)["mdwat"]
    soluble_modes = [m for m in range(modes.NMODES) if tab.mode[m] and tab.modesol[m] == 1]
    if not soluble_modes:
        assert np.array_equal(unfixed, fixed), (
            f"setup {setup} has no active soluble mode, so ukca_water_content_v "
            "is never called and the flag cannot move mdwat"
        )
        return
    assert not np.array_equal(unfixed, fixed), (
        f"setup {setup}: the flag moved nothing. Either the grid lost its "
        "low-humidity rows or the one-way latch fired before the flag was set"
    )


@needs_binding
@pytest.mark.fortran
def test_the_so4_increments_are_applied_in_source_order():
    """`cl(-2)` takes cp_su, then **cp_so**, then **cp_oc** -- indices 1, 6, 3.

    An ascending `icp` loop swaps the last two. Setups 4 and 5 are the only
    supported ones whose modes carry both, which is why this is not
    parametrized over all seven.
    """
    setup = 4
    tab = modes.build(setup)
    mixed = [
        m
        for m in range(modes.NMODES)
        if tab.mode[m] and tab.component[m, CP_OC] and tab.component[m, CP_SO]
    ]
    assert mixed, "setup 4 no longer has a mode carrying both cp_oc and cp_so"

    specs = [_spec(rh=rh, variant="oc_so") for rh in (0.2, 0.5, 0.9)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    # The faithful order.
    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])

    # The mutation, spelled out here rather than left to a reviewer's
    # imagination: ascending icp puts cp_oc (3) before cp_so (6).
    f_ao = volume_mode.aged_organic_moles()
    md = np.asarray(want["md_out"])
    avogadro = 6.022e23
    for m in mixed:
        faithful = md[:, m, CP_SU] / avogadro
        faithful = faithful + (0.65 / avogadro) * (md[:, m, CP_SO] / f_ao)
        faithful = faithful + (0.65 / avogadro) * (md[:, m, CP_OC] / f_ao)
        swapped = md[:, m, CP_SU] / avogadro
        swapped = swapped + (0.65 / avogadro) * (md[:, m, CP_OC] / f_ao)
        swapped = swapped + (0.65 / avogadro) * (md[:, m, CP_SO] / f_ao)
        if not np.array_equal(faithful, swapped):
            return
    pytest.fail(
        "the two orders agree on every row of this fixture, so the test cannot "
        "fail; re-choose the cp_oc/cp_so masses"
    )


def test_the_hygroscopic_increment_forms_two_quotients():
    """`(fhyg_aom/avogadro)*(md/f_ao)` (`:372`, `:381`), not one fraction.

    Compared as expressions, because the difference is in the last bit of a
    quantity that then goes through ZSR -- there is no input that isolates it.
    It has teeth only if the two spellings really are different doubles, which
    is asserted rather than assumed.
    """
    rng = np.random.default_rng(41)
    md = rng.uniform(1.0, 1e12, 200_000)
    f_ao = volume_mode.aged_organic_moles()
    avogadro = 6.022e23
    theirs = 0.65 * md / (avogadro * f_ao)
    ours = np.asarray(volume_mode._hygroscopic_increment(md, f_ao))
    differ = int((ours != theirs).sum())
    assert differ > 0, (
        "the two spellings agree on all 200,000 samples, so this test cannot "
        "fail; re-measure before relying on the grouping"
    )


def test_f_ao_is_recomputed_and_not_a_cached_constant():
    """`:332` sits inside the mode loop. CLAUDE.md forbids caching a derived
    quantity in `core/constants.py`, and this is the derived quantity most
    likely to be moved there by someone tidying."""
    from glomap_jax.core import constants

    assert volume_mode.aged_organic_moles() == volume_mode.MM_AGE_ORG / volume_mode.MM_POM
    assert not any("AGE_ORG" in n or "F_AO" in n or "POM" in n for n in constants.__all__)


def test_the_charge_balance_keeps_all_six_terms():
    """`:422`, left to right, inside `MAX(..., 0.0)`.

    **Dropping the three provably-zero terms is, on its own, a vacuous
    mutation** -- measured, 0 of 200,000 samples move, because `x + 0.0` is
    exact for every double this expression can hold. The live mutation is the
    *other* obvious simplification: `cl(-4)` and `cl(3)` are the same double by
    construction (`:396-398`, complete dissociation of NaCl), so a reader
    cancels them and writes `cl(1) = MAX(2*cl(-2), 0)`. That is a different
    number on **38.7% of 200,000 samples**, because `(2c + a) - a` is not `2c`
    in float64.

    Both spellings are exercised below so the vacuous one is recorded rather
    than mistaken for coverage.
    """
    rng = np.random.default_rng(422)
    n = 200_000
    cl = np.zeros((n, 8))
    cl[:, 2] = rng.uniform(0.0, 1e-10, n)  # -2, SO4
    cl[:, 0] = rng.uniform(0.0, 1e-10, n)  # -4, Cl
    cl[:, 7] = cl[:, 0]  # +3, Na -- complete dissociation, the same double

    ours = np.asarray(volume_mode.charge_balance(cl))

    # Vacuous: dropping only the exact zeros.
    zeros_dropped = np.maximum(((2.0 * cl[:, 2] + cl[:, 0]) - cl[:, 7]), 0.0)
    assert (ours == zeros_dropped).all(), (
        "dropping the exact-zero terms now changes the answer; re-derive"
    )

    # Live: cancelling the Na+/Cl- pair, which is algebraically exact and not
    # numerically exact.
    cancelled = np.maximum(2.0 * cl[:, 2], 0.0)
    differ = int((ours != cancelled).sum())
    assert differ > n // 10, (
        f"only {differ} of {n} samples distinguish the cancelled spelling; "
        "this test has lost its teeth"
    )

    # And the fold really is the source's, term by term in order.
    expected = 2.0 * cl[:, 2]
    for column in (3, 1, 0):
        expected = expected + cl[:, column]
    for column in (6, 7):
        expected = expected - cl[:, column]
    np.testing.assert_array_equal(ours, np.maximum(expected, 0.0))


def test_two_times_cl_is_bit_identical_to_c_plus_c():
    """Recorded because it looks like the same class of hazard as the ones that
    are real, and is not: `2.0*x` and `x + x` agree on all 200,000 samples.
    A mutation that swaps them is vacuous and proves nothing about the port."""
    rng = np.random.default_rng(4222)
    x = rng.uniform(0.0, 1e-10, 200_000)
    assert np.array_equal(2.0 * x, x + x)


def test_the_charge_balance_max_is_the_fortran_one():
    """`MAX(expr, 0.0)` with `expr` NaN returns NaN under this build, because
    the NaN is the FIRST argument and gfortran's MAX here behaves as
    `(b > a) ? b : a`. `jnp.maximum` agrees on this case and `numerics`'
    asymmetry is the point -- write the arguments in the Fortran's order."""
    cl = np.zeros((1, 8))
    cl[0, 2] = np.nan
    assert np.isnan(float(np.asarray(volume_mode.charge_balance(cl))[0]))


@needs_binding
@pytest.mark.fortran
def test_h_can_be_absent():
    """A sulfate-free sea-salt mode gives `cl(1) = 0` exactly, so `ions(1)` is
    FALSE and no H+ pair fires. Assuming H+ is present whenever an anion is
    inverts that, and the difference is visible in `mdwat`."""
    setup = 2
    tab = modes.build(setup)
    seasalt = [
        m
        for m in range(modes.NMODES)
        if tab.mode[m] and tab.modesol[m] == 1 and tab.component[m, CP_CL]
    ]
    assert seasalt, "setup 2 no longer has a soluble mode carrying cp_cl"

    specs = [_spec(rh=rh, variant="cl_only") for rh in (0.2, 0.5, 0.9)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    for m in seasalt:
        cl = np.asarray(volume_mode.ion_concentrations(tab, want["md_out"], m))
        from glomap_jax.physics import water_tables as wt

        assert np.array_equal(cl[:, wt.ion_slot(3)], cl[:, wt.ion_slot(-4)]), (
            "Na+ and Cl- are not the same double, so the balance cannot cancel"
        )
        assert (cl[:, wt.ion_slot(1)] == 0.0).all(), "H+ is not exactly zero"
        assert (cl[:, wt.ion_slot(-2)] == 0.0).all(), "the row is not sulfate-free"

    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])
    assert (np.asarray(got)[:, seasalt] > 0.0).all(), (
        "the sea-salt mode took up no water at all, so nothing was tested"
    )


@needs_binding
@pytest.mark.fortran
def test_ions_is_built_from_the_original_cl_not_the_depleted_pools():
    """`:425-427` snapshots `cl > 0.0` before ZSR draws the pools down.

    Constructed so H+ is fully consumed by pair (1,-2) and pair (1,-4) still
    passes the presence mask with `clp = 0.0`. Rebuilding the mask from the
    depleted `cli` would drop that pair.
    """
    setup = 2
    tab = modes.build(setup)
    target = next(
        m
        for m in range(modes.NMODES)
        if tab.mode[m]
        and tab.modesol[m] == 1
        and tab.component[m, CP_CL]
        and tab.component[m, CP_SU]
    )
    specs = [_spec(rh=rh, variant="cl_over_su") for rh in (0.2, 0.6)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    cl = np.asarray(volume_mode.ion_concentrations(tab, want["md_out"], target))
    from glomap_jax.physics import water_content as wcmod

    mask = np.ones(len(specs), dtype=bool)
    clp, present = wcmod._pair_concentrations(cl, cl > 0.0, mask)
    # (1,-4) runs FIRST -- the loop is cation-outer, anion-ascending from -4 --
    # and takes all of H+ because cl(1) ~ 2*cl(-2) << cl(-4).
    assert bool(present[(1, -2)].all()), "pair (1,-2) is not present; the row proves nothing"
    assert (np.asarray(clp[(1, -4)]) > 0.0).all(), "(1,-4) took nothing"
    exhausted = np.asarray(clp[(1, -2)])
    assert (exhausted == 0.0).all(), f"H+ was not fully consumed by (1,-4); got {exhausted}"
    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])


@needs_binding
@pytest.mark.fortran
def test_ions_is_a_strict_positivity_test_and_not_a_non_zero_test():
    """`:425-427` is `cl(:,i) > 0.0`. Only a NEGATIVE concentration separates
    that from `cl /= 0.0`, and only a negative `md` produces one.

    The grid's `negsu` row does: `md(cp_su) < 0` with `md(cp_cl) > 0`, so
    `cl(-2) < 0` and `cl(3) > 0`. Faithfully, pair (3,-2) never fires; read as
    non-zeroness it fires with a negative `clp` and moves `wc`. Without this row
    the substitution is byte-equal on every setup -- measured.

    `cl >= 0.0` is a different matter and is genuinely vacuous: it admits the
    identically-zero slots, whose pairs contribute `0.0/mb` to an ordered fold,
    which is exact.
    """
    setup = 2
    tab = modes.build(setup)
    specs = [_spec(rh=rh, variant="negsu") for rh in (0.2, 0.62, 0.9)]
    grid = _rows(tab, specs)
    applied = {v for row in grid["_variant_applied"] for v in row}
    assert "negsu" in applied, "no mode took the negsu recipe; the row proves nothing"

    want = _reference(setup, _inputs(grid))
    from glomap_jax.physics import water_tables as wt

    saw_negative = False
    for m in range(modes.NMODES):
        if not (tab.mode[m] and tab.modesol[m] == 1 and tab.component[m, CP_SU]):
            continue
        cl = np.asarray(volume_mode.ion_concentrations(tab, want["md_out"], m))
        if (cl[:, wt.ion_slot(-2)] < 0.0).any():
            saw_negative = True
            assert (cl[:, wt.ion_slot(3)] > 0.0).any(), "cl(3) is not positive on the same row"
    assert saw_negative, "no negative cl(-2) reached the port; the mutation cannot be seen"

    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])


def test_only_four_ion_pairs_are_reachable_in_the_box():
    """An invariant, not a behaviour. `cl(-1)` (HSO4) is never written by this
    routine at all, and `cl(2)` (NH4) and `cl(-3)` (NO3) sit behind `:402`'s
    `UBOUND(component,DIM=2) >= cp_no3` -- false because `ncp = 6 < 7` in every
    supported setup. So only H+, Na+, SO4(2-) and Cl- can ever be non-zero, and
    only pairs (1,-2), (1,-4), (3,-2), (3,-4) can fire."""
    source = SOURCE.read_text(encoding="utf-8")
    # cl(-1) appears exactly once, as a READ inside the charge balance. There is
    # no assignment to it anywhere, so HSO4- is identically zero.
    assert not re.search(r"cl\(:,\s*-1\s*\)\s*=", source), "cl(-1) is now assigned; re-derive"
    assert source.count("cl(:,-1)") == 1, "cl(-1) now appears more than once; re-derive"
    assert "IF (UBOUND(component,DIM=2) >= cp_no3) THEN" in source
    for setup in SETUPS:
        assert modes.build(setup).ncp == 6, f"setup {setup} no longer has ncp = 6"
    assert volume_mode.CP_NO3 == 7


def test_setup_eleven_is_not_constructible():
    """`:356-364` is dead here from both ends: `glomap_box_config_mod`'s
    `init_indices` has no `CASE` for setup 11 and ereports, and `modes.build`
    refuses it. The port raises rather than quietly taking the setup-1 path."""
    assert volume_mode.SETUP_SOLINSOL == 11
    assert 11 not in modes.supported_setups()
    with pytest.raises(NotImplementedError):
        modes.build(11)
    config = (REPO / "fortran" / "src" / "box" / "glomap_box_config_mod.F90").read_text()
    body = config[config.index("SELECT CASE (i_mode_setup)") :]
    body = body[: body.index("END SELECT")]
    assert "i_solinsol_6mode" not in body, "the box now dispatches setup 11; the port must too"
    assert "CASE DEFAULT" in body


def test_the_local_parameters_still_read_as_the_source_writes_them():
    """`fhyg_aom`, `mm_age_org`, `mm_pom` and `putls` are locals, so
    `tests/test_constants.py` cannot extract them by name. Re-parsed here
    instead, in the spirit of `test_inline_threshold_still_appears_in_its_routine`.
    """
    source = SOURCE.read_text(encoding="utf-8")
    wanted = {
        "fhyg_aom": volume_mode.FHYG_AOM,
        "mm_age_org": volume_mode.MM_AGE_ORG,
        "mm_pom": volume_mode.MM_POM,
    }
    for name, value in wanted.items():
        match = re.search(rf"REAL, PARAMETER :: {name}\s*=\s*([0-9.eE+-]+)", source)
        assert match, f"{name} is no longer a REAL, PARAMETER in {SOURCE.name}"
        assert float(match.group(1)) == value, f"{name}: {match.group(1)} != {value}"
    match = re.search(r"REAL, PARAMETER :: putls\s*=\s*([0-9.eE+-]+)", source)
    assert match and float(match.group(1)) == PUTLS


def test_both_humidity_clamps_fire_and_the_bounds_pass_through():
    """`:306-307`. Both clamps fire on this axis; no shipped namelist reaches
    either, the highest `rel_humid` being exactly 0.90.

    **The strictness of the two tests is not observable, and saying so is the
    point.** `WHERE (corrh > 0.9) corrh = 0.9` and `WHERE (corrh >= 0.9)
    corrh = 0.9` give the same double for every input, because the clamp target
    *is* the bound -- writing 0.9 over a value that already is 0.9 changes
    nothing. Measured: both mutations leave the whole file green. That is the
    opposite of `:312`'s `nd > num_eps`, where the tie selects a branch and the
    non-strict spelling reddens six setups.

    So this test asserts what can fail -- that the clamps fire, that the bounds
    survive, that the interior is untouched -- and records the rest as
    unobservable rather than pretending to cover it.
    """
    rh = np.array(RH_AXIS)
    got = np.asarray(volume_mode.corrected_humidity(rh))
    assert (rh > 0.9).any() and (rh < 0.1).any(), "the axis no longer leaves [0.1, 0.9]"
    assert (got[rh > 0.9] == 0.9).all()
    assert (got[rh < 0.1] == 0.1).all()
    inside = (rh >= 0.1) & (rh <= 0.9)
    np.testing.assert_array_equal(got[inside], rh[inside])

    # The unobservability, demonstrated rather than asserted from the source.
    strict = np.where(rh > 0.9, 0.9, rh)
    loose = np.where(rh >= 0.9, 0.9, rh)
    assert np.array_equal(strict, loose), (
        "the two spellings now differ; the strictness has become observable and "
        "this test should gain a real assertion about it"
    )


def test_the_charge_balance_max_cannot_be_told_from_jnp_maximum_here():
    """Recorded as a measured non-difference, not assumed.

    `numerics.fortran_max(x, 0.0)` and `jnp.maximum(x, 0.0)` differ on exactly
    one input, `x = -0.0`: the first keeps the sign, the second does not. That
    input is unreachable at `:422` -- `cl(-1)` is a positive zero and
    `-0.0 + 0.0` is `+0.0`, so the fold cannot carry a negative zero past its
    second term -- and it would be inert anyway, since `ions` is `cl > 0.0` and
    both zeros fail it. On `NaN` the two agree.

    Substituting `jnp.maximum` therefore leaves every byte-equality test in this
    file green (measured). `numerics.fortran_max` is used regardless, in the
    Fortran's argument order, because the next `MAX` may not be so forgiving.
    """
    from glomap_jax.core import numerics

    x = jnp.asarray([np.nan, 0.0, -0.0, -1.0, 1.0, np.inf, -np.inf])
    ours = np.asarray(numerics.fortran_max(x, 0.0))
    theirs = np.asarray(jnp.maximum(x, 0.0))
    differ = [i for i in range(len(ours)) if not _same_bits(ours[i], theirs[i])]
    assert differ == [2], f"expected only -0.0 to differ, got indices {differ}"


def _same_bits(a, b) -> bool:
    return np.float64(a).tobytes() == np.float64(b).tobytes() or (np.isnan(a) and np.isnan(b))


# ---------------------------------------------------------------------------
# Task 42 -- density, rhopar, pvol, pvol_wat
# ---------------------------------------------------------------------------


def _soluble_cols(tab) -> list[int]:
    return [m for m in range(modes.NMODES) if tab.mode[m] and tab.modesol[m] == 1]


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("fix_water", [0, 1], ids=["unfixed", "fixed"])
def test_the_soluble_volumes_are_byte_equal_in_the_troposphere(setup, fix_water):
    """`rhopar`, `pvol`, `pvol_wat` and `wvol` on every soluble column.

    `pvol` and `pvol_wat` appear in no state golden -- the box model does not
    dump them -- so the compiled routine is the only reference they have.
    """
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid), fix_water=fix_water)
    cols = _soluble_cols(tab)
    if not cols:
        pytest.skip(f"setup {setup} has no active soluble mode")

    mdwat, wvol, rhopar, pvol, pvol_wat = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=bool(fix_water),
        fix_neg_pvol_wat=True,
    )
    for name, got, ref in (
        ("mdwat", mdwat, want["mdwat"]),
        ("wvol", wvol, want["wvol"]),
        ("rhopar", rhopar, want["rhopar"]),
        ("pvol_wat", pvol_wat, want["pvol_wat"]),
    ):
        np.testing.assert_array_equal(
            np.asarray(got)[:, cols], np.asarray(ref)[:, cols], err_msg=name
        )
    np.testing.assert_array_equal(
        np.asarray(pvol)[:, cols, :], np.asarray(want["pvol"])[:, cols, :], err_msg="pvol"
    )


@needs_binding
@pytest.mark.fortran
def test_pvol_is_the_only_conditionally_written_output():
    """`:592-616` writes `pvol` only under `IF (component(imode,icp))`. Every
    other output is written on every mode, on every branch.

    The Fortran leaves a non-member `(imode, icp)` at whatever the caller had,
    and both callers pre-zero: `glomap_box_state_mod.F90:86` and
    `leaf_volume_mode` itself. This port returns the zero it started with, and
    that choice is recorded here rather than left implicit -- a caller that did
    not pre-zero would see the port and the Fortran disagree, and this is where
    to look.
    """
    setup = 2
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid))
    _, _, _, pvol, _ = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    outsiders = [
        (m, c) for m in _soluble_cols(tab) for c in range(tab.ncp) if not tab.component[m, c]
    ]
    assert outsiders, "every soluble mode carries every component; the test proves nothing"
    for m, c in outsiders:
        assert (np.asarray(want["pvol"])[:, m, c] == 0.0).all(), (
            f"the driver's pre-zero did not survive at mode {m + 1}, cpt {c + 1}"
        )
        assert (np.asarray(pvol)[:, m, c] == 0.0).all()


@needs_binding
@pytest.mark.fortran
def test_the_accumulation_seeds_water_first_then_ascending_icp():
    """`:439-475`: four folds, water first, components in index order.

    What actually distinguishes the faithful fold is measured here rather than
    taken from the hazard list, because one of the three obvious mutations
    turns out to be vacuous in JAX:

    * `water + SUM(terms)` -- **different**. The parenthesisation moves, and
      that is the whole content of "seed with water".
    * `(((t0+t1)+...)+water)`, water last -- **different**.
    * `jnp.sum` over a stack of `(water, t0, t1, ...)` -- **the same double**.
      XLA's reduce over an axis is left-associated: 0 of 20,000 differences for
      axis lengths 3 to 32, diverging only at 64 (69% of points), while
      `numpy`'s pairwise summation starts diverging at 8. With `ncp = 6` plus
      water, no fold in this routine is long enough for a reduction to
      associate differently -- so "use a reduction" is not the live hazard here,
      and a test claiming to catch it could not fail.
    """
    setup = 8
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid))
    scales = volume_mode._Scales(tab)
    md = np.asarray(want["md_out"])
    mdwat = np.asarray(want["mdwat"])

    saw_reduced, saw_water_last, saw_stack_identical = False, False, False
    for m in _soluble_cols(tab):
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        _, _, rhotmp2, _ = volume_mode._density_accumulators(
            tab, md, m, mask, jnp.asarray(mdwat[:, m]), scales
        )
        members = [c for c in range(tab.ncp) if tab.component[m, c]]
        if len(members) < 3:
            continue
        # Compare on the masked rows only: off the mask every term is replaced
        # by 0.0, which is the mask's business and not the fold's.
        sel = np.asarray(mask)
        water = (mdwat[:, m] * scales.mmwrhow)[sel]
        terms = [(md[:, m, c] * np.asarray(scales.mm_rhocp)[c])[sel] for c in members]

        running = terms[0].copy()
        for row in terms[1:]:
            running = running + row
        ours = np.asarray(rhotmp2)[sel]
        saw_reduced |= not np.array_equal(ours, water + running)
        saw_water_last |= not np.array_equal(ours, running + water)
        stacked = jnp.stack([jnp.asarray(water)] + [jnp.asarray(x) for x in terms])
        saw_stack_identical |= np.array_equal(ours, np.asarray(jnp.sum(stacked, axis=0)))

    assert saw_reduced, "`water + SUM(terms)` matches the fold everywhere; re-choose the setup"
    assert saw_water_last, "water-last matches the fold everywhere; re-choose the setup"
    assert saw_stack_identical, (
        "jnp.sum over the stack no longer reproduces the fold; XLA's reduction "
        "order has changed and the reduction hazard is now live -- re-measure"
    )


def test_a_short_jnp_sum_is_left_associated_and_a_long_one_is_not():
    """Recorded because the phase-D hazard list says reductions break an ordered
    fold, and at the lengths this routine uses they do not.

    Measured on this build: `jnp.sum` over an axis of length 7 or 32 gives the
    same double as the left-associated fold on every one of 20,000 samples, and
    diverges at 64. `ncp` is 6 in every supported setup, so the ordered folds in
    this port are written out for faithfulness and documentation -- not because
    a reduction would currently give a different answer.
    """
    rng = np.random.default_rng(14)
    for n, expect_same in ((7, True), (32, True), (64, False)):
        a = rng.uniform(1e-25, 1e-18, (n, 20_000))
        fold = a[0].copy()
        for row in a[1:]:
            fold = fold + row
        same = np.array_equal(np.asarray(jnp.sum(jnp.asarray(a), axis=0)), fold)
        assert same is expect_same, (
            f"jnp.sum over {n} terms is now "
            f"{'not left-associated' if expect_same else 'left-associated'}; "
            "the reduction hazard has moved"
        )


def test_the_where_construct_executes_in_order():
    """`:443` is `rhotmp2 = rhotmp`, one line after `:442` writes `rhotmp`.

    A `WHERE` construct's statements execute in sequence, so `rhotmp2` is seeded
    with the water term -- not with whatever `rhotmp` held before the construct,
    and not with zero. Read the other way, `rhopar` loses the water entirely.
    """
    tab = modes.build(1)
    scales = volume_mode._Scales(tab)
    mask = jnp.asarray([True, True])
    mdwat = jnp.asarray([1.0e3, 5.0e6])
    md = np.zeros((2, modes.NMODES, tab.ncp))
    _, _, rhotmp2, denom2 = volume_mode._density_accumulators(tab, md, 0, mask, mdwat, scales)
    # No component mass at all, so the folds are the seed alone.
    np.testing.assert_array_equal(np.asarray(rhotmp2), np.asarray(mdwat) * scales.mmwrhow)
    np.testing.assert_array_equal(np.asarray(denom2), np.asarray(mdwat) * 0.0180154)
    assert not np.array_equal(np.asarray(rhotmp2), np.zeros(2)), (
        "the seed is zero, so a port that skipped :443 would pass"
    )


def test_the_masked_divisions_keep_the_gradient_finite():
    """`:579`, `:593`, `:623` and `:627` all divide by something that is exactly
    0.0 off `mask_sol`, so every one of them goes through `numerics.safe_divide`.

    **The single-`where` form is not distinguishable here, and pretending
    otherwise would make this test vacuous.** Measured: substituting
    `jnp.where(mask_sol, rhotmp/denom, 0.0)` for `safe_divide` leaves the
    forward value *and* the gradient finite, and every byte-equality test in
    this file green. The reason is structural -- `rhotmp` and `denom` are
    themselves built by `jnp.where(mask, ., 0.0)`, whose VJP *selects* rather
    than multiplies, so the `NaN` cotangent produced by the `0/0` is discarded
    before it can reach an input.

    What this test does catch is the formulation that genuinely poisons, and
    which CLAUDE.md's "never multiply by the mask" rule is about: a
    multiplicative mask, `mask * (num/den)`. It is demonstrated inline so the
    test names its own failing counterpart rather than asserting a property no
    reachable mutation violates.
    """
    import jax

    tab = modes.build(1)
    nd = np.zeros((2, modes.NMODES))
    md = np.zeros((2, modes.NMODES, tab.ncp))
    nd[0, 0] = 1000.0  # mask true, mask_sol true
    nd[1, 0] = 0.0  # mask false: denom and rhotmp are both exactly 0.0 there
    md[0, 0, CP_SU] = tab.mmid[0]
    dvol = np.full((2, modes.NMODES), 1.0e-24)

    def total(masses):
        _, wvol, rhopar, _, pvol_wat = volume_mode.soluble_volumes(
            tab,
            nd,
            masses,
            np.array([0.6, 0.6]),
            dvol,
            np.array([213.0, 213.0]),
            np.array([1.0e5, 1.0e5]),
            np.array([1.0e-2, 1.0e-2]),
            fix_water_content=True,
            fix_neg_pvol_wat=True,
        )
        return jnp.sum(wvol) + jnp.sum(rhopar) + jnp.sum(pvol_wat)

    grad = np.asarray(jax.grad(total)(jnp.asarray(md)))
    assert np.isfinite(grad).all(), "a masked division poisoned the gradient"

    def poisoned(x):
        mask = x > 0.5
        den = jnp.where(mask, x, 0.0)
        return jnp.sum(mask.astype(jnp.float64) * (x / den))

    assert not np.isfinite(np.asarray(jax.grad(poisoned)(jnp.array([0.2, 0.8])))).all(), (
        "the multiplicative mask no longer poisons the gradient, so the "
        "safe_divide rule has lost its demonstration"
    )


def test_massh2so4kg_does_not_use_the_precomputed_mm_ovravc():
    """`:435` writes `md*mm(cp_su)/avogadro` although `mm_ovravc(cp_su)` is in
    scope and is exactly that quotient. The two are different doubles.

    Compared as expressions: the difference is in the last bit and there is no
    `md` that isolates it downstream. It is asserted here so the substitution
    cannot be made silently, and the assertion has teeth only because the two
    spellings really do differ -- which is measured, not assumed.
    """
    tab = modes.build(1)
    scales = volume_mode._Scales(tab)
    rng = np.random.default_rng(435)
    md = rng.uniform(1.0, 1e12, 200_000)
    faithful = (md * tab.mm[CP_SU]) / 6.022e23
    hoisted = md * float(np.asarray(scales.mm_ovravc)[CP_SU])
    assert (faithful != hoisted).sum() > 0, (
        "mm_ovravc and the inline quotient now agree on every sample; "
        "re-measure before relying on the distinction"
    )


def test_mm_ovravcrhocp_is_two_divisions():
    """`:296` writes `(mm/avogadro)/rhocomp`, not `mm/(avogadro*rhocomp)`."""
    distinguishable = False
    for setup in SETUPS:
        tab = modes.build(setup)
        scales = volume_mode._Scales(tab)
        mm = np.asarray(tab.mm[: tab.ncp])
        rhocomp = np.asarray(tab.rhocomp[: tab.ncp])
        np.testing.assert_array_equal(np.asarray(scales.mm_ovravcrhocp), (mm / 6.022e23) / rhocomp)
        distinguishable |= bool(((mm / 6.022e23) / rhocomp != mm / (6.022e23 * rhocomp)).any())
    assert distinguishable, (
        "the two spellings agree on every component of every setup, so this "
        "test cannot fail; re-measure before trusting the grouping"
    )


@needs_binding
@pytest.mark.fortran
def test_mask_sol_and_mask_nosol_do_not_partition_mask():
    """`mdsol < 0` is neither, and the difference is visible in `pvol`.

    On such a row the Fortran takes `:597`'s ELSE WHERE -- `pvol = dvol*mfrac_0`
    -- and `:601` does NOT zero it, while `pvol_wat = 0`, `wvol = dvol` and
    `rhopar = rho_so4` come from `:631`. Writing `mask_nosol = mask & ~mask_sol`
    zeroes that `pvol`.
    """
    setup = 2
    tab = modes.build(setup)
    specs = [_spec(rh=rh, variant="negsol") for rh in (0.2, 0.62)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    from glomap_jax.physics import volume_mode as vm

    md = jnp.asarray(want["md_out"])
    negative = []
    for m in _soluble_cols(tab):
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        mdsol = vm.soluble_mass(tab, md, m, mask)
        if bool(np.asarray(mdsol < 0.0).any()):
            negative.append(m)
    assert negative, (
        "no mode reached mdsol < 0; the `negsol` recipe no longer works and the "
        "third state is untested"
    )

    for m in negative:
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        mdsol = vm.soluble_mass(tab, md, m, mask)
        mask_sol, mask_nosol = vm.solubility_masks(mdsol, mask)
        naive = mask & ~mask_sol
        assert not np.array_equal(np.asarray(mask_nosol), np.asarray(naive)), (
            f"mode {m + 1}: mask_nosol and mask & ~mask_sol agree, so the "
            "distinction is untested here"
        )
        rows = np.asarray(mdsol < 0.0)
        for c in range(tab.ncp):
            if tab.component[m, c] and tab.soluble[c]:
                expected = np.asarray(want["dvol"])[rows, m] * tab.mfrac_0[m, c]
                np.testing.assert_array_equal(
                    np.asarray(want["pvol"])[rows, m, c],
                    expected,
                    err_msg=f"the Fortran did not leave the :597 default at mode {m + 1}",
                )
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[rows, m], np.asarray(want["dvol"])[rows, m]
        )
        np.testing.assert_array_equal(
            np.asarray(want["rhopar"])[rows, m], np.full(rows.sum(), 1769.0)
        )
        assert (np.asarray(want["pvol_wat"])[rows, m] == 0.0).all()

    _, _, _, pvol, _ = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    cols = _soluble_cols(tab)
    np.testing.assert_array_equal(
        np.asarray(pvol)[:, cols, :], np.asarray(want["pvol"])[:, cols, :]
    )


@needs_binding
@pytest.mark.fortran
def test_mask_nosol_is_reached_and_zeroes_the_soluble_pvol():
    """`:601`. Zero soluble mass with `nd > num_eps` -- 0 hits in 2447 sampled
    golden points, so it needs a constructed row. A mode with no insoluble
    component cannot supply one: zeroing its whole mass leaves `dvol = 0` and
    `ukca_calc_drydiam` trips first."""
    setup = 2
    tab = modes.build(setup)
    specs = [_spec(rh=rh, variant="nosol") for rh in (0.2, 0.62, 0.9)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    md = jnp.asarray(want["md_out"])
    hit = []
    for m in _soluble_cols(tab):
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        _, mask_nosol = volume_mode.solubility_masks(
            volume_mode.soluble_mass(tab, md, m, mask), mask
        )
        if bool(np.asarray(mask_nosol).any()):
            hit.append(m)
    assert hit, "no mode reached mask_nosol; the `nosol` recipe no longer works"

    for m in hit:
        rows = np.asarray(
            volume_mode.solubility_masks(
                volume_mode.soluble_mass(
                    tab, md, m, jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
                ),
                jnp.asarray(grid["nd"][:, m] > tab.num_eps[m]),
            )[1]
        )
        for c in range(tab.ncp):
            if tab.component[m, c] and tab.soluble[c]:
                assert (np.asarray(want["pvol"])[rows, m, c] == 0.0).all(), (
                    f"the Fortran did not zero the soluble pvol at mode {m + 1}"
                )
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[rows, m], np.asarray(want["dvol"])[rows, m]
        )

    cols = _soluble_cols(tab)
    _, wvol, rhopar, pvol, pvol_wat = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    for name, got, ref in (
        ("wvol", wvol, want["wvol"]),
        ("rhopar", rhopar, want["rhopar"]),
        ("pvol_wat", pvol_wat, want["pvol_wat"]),
    ):
        np.testing.assert_array_equal(
            np.asarray(got)[:, cols], np.asarray(ref)[:, cols], err_msg=name
        )
    np.testing.assert_array_equal(
        np.asarray(pvol)[:, cols, :], np.asarray(want["pvol"])[:, cols, :]
    )


def test_mfrac_0_is_zero_at_every_non_member_cell_of_an_active_mode():
    """Why `:597`/`:613`'s default cannot distinguish "written" from "not
    written" for a non-member component.

    Substituting `pvol = dvol*mfrac_0` for the port's "leave the caller's zero"
    at a non-member `(imode, icp)` leaves every byte-equality test green --
    measured -- and this is why: within an **active** mode, `mfrac_0` is exactly
    0.0 wherever `component` is false, so the default is zero and the caller's
    pre-zero is indistinguishable from it.

    The same is emphatically NOT true of an inactive mode: `mfrac_0` there
    carries 1.0 for dust in modes 6-8 of most setups, and 0.5/0.5 for bc/oc in
    mode 5 of setups 1, 3, 5 and 6 -- which is what makes `:686`'s default worth
    a test of its own (task 44).
    """
    live, dead = [], []
    for setup in SETUPS:
        tab = modes.build(setup)
        for m in range(modes.NMODES):
            for c in range(tab.ncp):
                if tab.component[m, c] or tab.mfrac_0[m, c] == 0.0:
                    continue
                (live if tab.mode[m] else dead).append((setup, m + 1, c + 1))
    assert not live, (
        f"an ACTIVE mode now has a non-zero mfrac_0 at a non-member cell: {live}. "
        "The pvol default is no longer indistinguishable from the caller's zero, "
        "and task 42's recorded choice needs re-deciding."
    )
    assert dead, "no inactive mode has a non-zero mfrac_0 at a non-member cell; re-derive"


@needs_binding
@pytest.mark.fortran
def test_the_wvol_accumulation_mask_is_only_observable_under_mask_sol():
    """Recorded because two plausible mutations of it are vacuous.

    `:596` accumulates under `mask_sol` and `:612` under `mask`. Swapping either
    for the other leaves every byte-equality test green, and the reason is
    `:631`: off `mask_sol` the whole accumulated `wvol` is thrown away and
    replaced by `dvol`. So the accumulation masks only ever matter where
    `mask_sol` is true, and there `mask` is true too.

    Demonstrated on a fixture that really does contain `mask_nosol` rows -- the
    place where the two masks differ and where the discard happens -- so the
    claim is measured rather than argued.
    """
    setup = 2
    tab = modes.build(setup)
    specs = [_spec(rh=rh, variant="nosol") for rh in (0.2, 0.9)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    md = jnp.asarray(want["md_out"])
    seen = False
    for m in _soluble_cols(tab):
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        _, mask_nosol = volume_mode.solubility_masks(
            volume_mode.soluble_mass(tab, md, m, mask), mask
        )
        rows = np.asarray(mask_nosol)
        if not rows.any():
            continue
        seen = True
        # The discard: wvol is dvol exactly, however much insoluble pvol was
        # accumulated into it first.
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[rows, m], np.asarray(want["dvol"])[rows, m]
        )
        insoluble_total = np.zeros(int(rows.sum()))
        for c in range(tab.ncp):
            if tab.component[m, c] and not tab.soluble[c]:
                insoluble_total = insoluble_total + np.asarray(want["pvol"])[rows, m, c]
        assert (insoluble_total > 0.0).all(), (
            "nothing was accumulated before the discard, so the row shows nothing"
        )
    assert seen, "no mask_nosol row; the discard is untested"


# ---------------------------------------------------------------------------
# Task 43 -- the stratospheric branch
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _strat_grid(setup: int):
    """One call, `nbox = 60+`, with a MIXED `pmid` column.

    That is the point of the fixture and not an optimisation. In the box model
    `pmid` is a run-level scalar (`glomap_box_env_mod.F90:75`, `nbox = 1`), so a
    one-pressure-per-call sweep cannot distinguish "the override is applied at
    the points where it should be" from "the override is applied to the whole
    call". A mixed column can.

    No `negsu`/`negsol` row appears here. A negative `md(cp_su)` below `putls`
    gives `mdwat = (100/wts - 1)*md_su*mm_su/mmw < 0` for any `wts <= 100`, and
    with `l_fix_neg_pvol_wat` on the `:882-898` guard then aborts the call --
    correctly, since a negative water mass is what it exists to catch. One such
    row would void the whole comparison.
    """
    tab = modes.build(setup)
    specs = [_spec(t=T_REF, pmid=pmid, s=S_REF, rh=rh) for pmid in PMID_AXIS for rh in (0.3, 0.9)]
    # The wts > 99 corner, where l_fix_neg_pvol_wat is the difference between a
    # positive and a negative stratospheric mdwat.
    specs += [_spec(t=288.0, pmid=pmid, s=1.0e-8, rh=RH_REF) for pmid in PMID_AXIS]
    specs += [_spec(t=t, pmid=P_STRAT, s=S_REF, rh=RH_REF) for t in T_AXIS]
    specs += [_spec(t=303.65, pmid=P_STRAT, s=s, rh=RH_REF) for s in S_AXIS]
    specs += [
        _spec(t=T_REF, pmid=pmid, s=S_REF, rh=rh, variant=v)
        for v in ("su_only", "cl_only", "nosol", "insol_rich", "tiny")
        for pmid in (P_STRAT, PUTLS)
        for rh in (0.3, 0.9)
    ]
    grid = _rows(tab, specs)
    pmids = np.asarray(grid["pmid"])
    assert PUTLS in pmids, "no row sits exactly on putls"
    assert (pmids < PUTLS).any() and (pmids > PUTLS).any(), (
        "the column is single-sided, so the override cannot be shown to be per point"
    )
    return tab, grid


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("fix_neg", [0, 1], ids=["unclamped", "clamped"])
def test_the_stratospheric_branch_is_byte_equal(setup, fix_neg):
    """`mdwat`, `rhopar`, `pvol`, `pvol_wat` and `wvol` across `putls`.

    Neither override has run in any validated trajectory -- the four shipped
    namelists run `pressure` in {1e5, 2e4, 1e5, 1e5} and `putls` is 1.5e4 -- so
    the compiled routine driven by a constructed pressure column is the only
    reference these branches have ever had.
    """
    tab, grid = _strat_grid(setup)
    want = _reference(setup, _inputs(grid), fix_neg=fix_neg)
    cols = _soluble_cols(tab)
    if not cols:
        pytest.skip(f"setup {setup} has no active soluble mode")

    mdwat, wvol, rhopar, pvol, pvol_wat = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=bool(fix_neg),
    )
    for name, got, ref in (
        ("mdwat", mdwat, want["mdwat"]),
        ("wvol", wvol, want["wvol"]),
        ("rhopar", rhopar, want["rhopar"]),
        ("pvol_wat", pvol_wat, want["pvol_wat"]),
    ):
        np.testing.assert_array_equal(
            np.asarray(got)[:, cols], np.asarray(ref)[:, cols], err_msg=name
        )
    np.testing.assert_array_equal(
        np.asarray(pvol)[:, cols, :], np.asarray(want["pvol"])[:, cols, :], err_msg="pvol"
    )


@needs_binding
@pytest.mark.fortran
def test_putls_itself_takes_the_tropospheric_arm():
    """`:434` and `:584` are `pmid < putls`, strict. `pmid == 1.5e4` exactly is
    tropospheric, and its two neighbouring doubles fall on opposite sides.

    Checked against the Fortran's own `mdwat`, not against the port: the row at
    `nextafter(putls, 0)` must be overridden and the row at `putls` must not.
    """
    setup = 1
    tab = modes.build(setup)
    below, at, above = np.nextafter(PUTLS, 0.0), PUTLS, np.nextafter(PUTLS, np.inf)
    specs = [_spec(t=T_REF, pmid=pm, s=S_REF, rh=RH_REF) for pm in (below, at, above)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    m = _soluble_cols(tab)[0]
    mdwat = np.asarray(want["mdwat"])[:, m]
    assert mdwat[1] == mdwat[2], "putls itself was overridden; the comparison at :434 is not strict"
    assert mdwat[0] != mdwat[1], (
        "the double below putls was NOT overridden, so the branch never fired "
        "and the strictness check proves nothing"
    )
    assert np.asarray(volume_mode.stratospheric(np.array([below, at, above]))).tolist() == [
        True,
        False,
        False,
    ]


@needs_binding
@pytest.mark.fortran
def test_the_strat_override_is_applied_per_point():
    """A mixed `pmid` column in ONE call, with the same composition on every row.

    If the override were applied per call rather than per point, every row would
    come back either overridden or not. This asserts both happen inside a single
    `leaf_volume_mode` invocation -- something the box model cannot show,
    because `pmid` there is a run-level scalar with `nbox = 1`.
    """
    setup = 1
    tab = modes.build(setup)
    specs = [_spec(t=T_REF, pmid=pm, s=S_REF, rh=RH_REF) for pm in PMID_AXIS]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))
    assert len(specs) >= 3

    m = _soluble_cols(tab)[0]
    mdwat = np.asarray(want["mdwat"])[:, m]
    strat = np.asarray(grid["pmid"]) < PUTLS
    assert strat.any() and (~strat).any(), "the column is single-sided"
    # Rows on the same side agree with each other; the two sides do not.
    assert len(set(mdwat[~strat].tolist())) == 1, (
        "tropospheric rows disagree although rh, t and composition are identical"
    )
    assert (mdwat[strat] != mdwat[~strat][0]).all(), (
        "a stratospheric row matched the tropospheric value; the override did not fire per point"
    )

    got = volume_mode.mdwat(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(got), want["mdwat"])


@needs_binding
@pytest.mark.fortran
def test_the_two_strat_overrides_use_different_masks():
    """`:434` is under `mask`, `:584` under `mask_sol`. At a `mask_nosol` point
    the water is overridden and the density is not.

    **Unifying them is asymmetric, and only one direction is observable.**
    Measured:

    * moving the `mdwat` override to `mask_sol` -- **reddens**. At a
      `mask_nosol` row the reference rebuilds `mdwat` from `wts` and the
      mutation leaves the ZSR value.
    * moving the `rhosol` override to `mask` -- **stays green**, on every
      setup. `rhosol` is *read* only at `:593` and `:623`, both inside
      `WHERE (mask_sol)`, so a value written at a `mask & ~mask_sol` point can
      never reach an output. In the Fortran that point holds the previous
      mode's uninitialised `rhosol`, which is the same statement.

    So the acceptance criterion "unify them, it must fail on a mask_nosol row"
    holds for `:434` and not for `:584`. This test asserts the masks are
    genuinely different sets on the fixture, checks the port against the Fortran
    on exactly those rows, and records the unobservable half rather than
    claiming coverage of it.
    """
    setup = 2
    tab = modes.build(setup)
    specs = [
        _spec(t=T_REF, pmid=pm, s=S_REF, rh=rh, variant="nosol")
        for pm in (P_STRAT, P_TROP)
        for rh in (0.3, 0.9)
    ]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    md = jnp.asarray(want["md_out"])
    strat = np.asarray(volume_mode.stratospheric(grid["pmid"]))
    seen = False
    for m in _soluble_cols(tab):
        mask = jnp.asarray(grid["nd"][:, m] > tab.num_eps[m])
        mask_sol, mask_nosol = volume_mode.solubility_masks(
            volume_mode.soluble_mass(tab, md, m, mask), mask
        )
        water_rows = np.asarray(mask) & strat
        density_rows = np.asarray(mask_sol) & strat
        if not (np.asarray(mask_nosol) & strat).any():
            continue
        seen = True
        assert not np.array_equal(water_rows, density_rows), (
            f"mode {m + 1}: the two override masks coincide on this fixture"
        )
    assert seen, "no mask_nosol row below putls; the two masks were never asked to differ"

    cols = _soluble_cols(tab)
    mdwat, _, rhopar, _, _ = volume_mode.soluble_volumes(
        tab,
        grid["nd"],
        want["md_out"],
        grid["rh"],
        want["dvol"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=True,
        fix_neg_pvol_wat=True,
    )
    np.testing.assert_array_equal(np.asarray(mdwat)[:, cols], np.asarray(want["mdwat"])[:, cols])
    np.testing.assert_array_equal(np.asarray(rhopar)[:, cols], np.asarray(want["rhopar"])[:, cols])


@needs_binding
@pytest.mark.fortran
def test_the_negative_water_case_is_reachable_only_with_the_flag_off():
    """`l_fix_neg_pvol_wat` changes the FAILURE MODE, not just a number.

    With it off, `ukca_vapour.F90:188` has no 99% ceiling and `wts` reaches
    103.8, so `(100.0/wts - 1.0)` is negative and the stratospheric `mdwat` with
    it. The same flag disables the `:882-898` abort that would have caught that,
    so the reference returns a negative water content in silence. **This port
    reproduces the silence**; the omitted guard is recorded in
    `test_the_diagnostic_ereport_blocks_are_omitted`.

    With the flag on, `wts` is clamped to 99 and `100/99 - 1 > 0`, so the same
    row is positive and the guard has nothing to fire on.
    """
    setup = 1
    tab = modes.build(setup)
    specs = [_spec(t=t, pmid=P_STRAT, s=1.0e-8, rh=RH_REF) for t in (288.0, 298.0, 303.65, 310.0)]
    grid = _rows(tab, specs)

    unclamped = _reference(setup, _inputs(grid), fix_neg=0)
    clamped = _reference(setup, _inputs(grid), fix_neg=1)
    cols = _soluble_cols(tab)
    assert (np.asarray(unclamped["mdwat"])[:, cols] < 0.0).any(), (
        "no row produced a negative mdwat with the flag off; wts never exceeded "
        "100 and the case is not reached"
    )
    assert (np.asarray(clamped["mdwat"])[:, cols] >= 0.0).all(), (
        "the clamped arm produced a negative mdwat, which the :884 guard should have aborted on"
    )
    assert not np.array_equal(
        np.asarray(unclamped["mdwat"])[:, cols], np.asarray(clamped["mdwat"])[:, cols]
    ), "the flag moved nothing; the both-settings comparison is vacuous"

    for flag, want in ((0, unclamped), (1, clamped)):
        got = volume_mode.mdwat(
            tab,
            grid["nd"],
            want["md_out"],
            grid["rh"],
            grid["t"],
            grid["pmid"],
            grid["s"],
            fix_water_content=True,
            fix_neg_pvol_wat=bool(flag),
        )
        np.testing.assert_array_equal(np.asarray(got), want["mdwat"], err_msg=f"flag={flag}")


def test_the_strat_water_forms_three_separate_statements():
    """`:435-437`. `(md*mm(cp_su))/avogadro`, then `(100/wts - 1)*that`, then
    `/mmwovravc` -- a division by a precomputed quotient, not a multiplication
    by its reciprocal.

    Compared as expressions, because each difference is in the last bit and no
    input isolates one from the others. Each alternative is asserted to be a
    genuinely different double first, so none of these checks can go vacuous.
    """
    tab = modes.build(1)
    scales = volume_mode._Scales(tab)
    rng = np.random.default_rng(437)
    md = rng.uniform(1.0, 1e12, 200_000)
    wts = rng.uniform(41.0, 103.0, 200_000)

    massh2so4kg = (md * tab.mm[CP_SU]) / 6.022e23
    masswaterkg = (100.0 / wts - 1.0) * massh2so4kg
    faithful = masswaterkg / float(scales.mmwovravc)

    alternatives = {
        "mm_ovravc substituted": (100.0 / wts - 1.0)
        * (md * float(np.asarray(scales.mm_ovravc)[CP_SU]))
        / float(scales.mmwovravc),
        "multiply by the reciprocal": masswaterkg * (1.0 / float(scales.mmwovravc)),
        "one fraction": (100.0 - wts) / wts * massh2so4kg / float(scales.mmwovravc),
    }
    for label, other in alternatives.items():
        assert (faithful != other).sum() > 0, (
            f"'{label}' agrees with the faithful form on all 200,000 samples; "
            "that check cannot fail"
        )
