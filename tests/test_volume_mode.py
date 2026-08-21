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

import jax
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

    mdwat, wvol, rhopar, pvol, pvol_wat = volume_mode.partial_volumes(
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
    _, _, _, pvol, _ = volume_mode.partial_volumes(
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
      XLA's reduce over the leading axis is left-associated at every length this
      routine uses: 0 of 20,000 differences for axis lengths 2 to 7 on both
      interpreters this port has been run against. Where it stops being so is
      version-dependent -- jax 0.9.2 departs at 64, jax 0.11.0 does not depart
      by 512 -- so `test_a_short_jnp_sum_reproduces_the_ordered_fold` asserts
      only the lengths in play and reports the crossover. With `ncp = 6` plus
      water, "use a reduction" is not the live hazard here, and a test claiming
      to catch it could not fail.
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


def test_a_short_jnp_sum_reproduces_the_ordered_fold():
    """Recorded because the phase-D hazard list says reductions break an ordered
    fold, and at the lengths this routine uses they do not.

    `ncp` is 6 in every supported setup, so the longest fold in
    `ukca_volume_mode` is six components plus water. Over an axis that short,
    `jnp.sum` gives the same double as the left-associated fold on every one of
    20,000 samples -- on both interpreters this port has been run against. The
    ordered folds in the port are therefore written out for faithfulness and
    documentation, not because a reduction would currently give a different
    answer.

    Where the two *do* part company is version-dependent, which is why nothing
    here pins a crossover. Measured over an axis of length 7 to 512, summing
    along axis 0:

    * jax 0.9.2 -- identical to 32, then 13,764 of
      20,000 differ at 64 and it grows with length.
    * jax 0.11.0 -- identical at every length tested, to 512.

    So a reduction is safe here by measurement on this interpreter, not by
    construction. The crossover is reported rather than asserted.
    """
    rng = np.random.default_rng(14)
    longest = max(modes.build(s).ncp for s in SETUPS) + 1
    assert longest <= 8, f"ncp has grown; the longest fold is now {longest} terms"
    for n in range(2, longest + 1):
        a = rng.uniform(1e-25, 1e-18, (n, 20_000))
        fold = a[0].copy()
        for row in a[1:]:
            fold = fold + row
        np.testing.assert_array_equal(
            np.asarray(jnp.sum(jnp.asarray(a), axis=0)),
            fold,
            err_msg=(
                f"jnp.sum over {n} terms is no longer left-associated on "
                f"jax {jax.__version__}; the reduction hazard has become live "
                "at a length this routine actually uses"
            ),
        )

    crossover = None
    for n in (16, 32, 64, 128, 256, 512):
        a = rng.uniform(1e-25, 1e-18, (n, 20_000))
        fold = a[0].copy()
        for row in a[1:]:
            fold = fold + row
        if not np.array_equal(np.asarray(jnp.sum(jnp.asarray(a), axis=0)), fold):
            crossover = n
            break
    print(
        f"\njax {jax.__version__}: jnp.sum over axis 0 first departs from the "
        f"ordered fold at n = {crossover if crossover else '>512 (never)'}"
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
        _, wvol, rhopar, _, pvol_wat = volume_mode.partial_volumes(
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

    _, _, _, pvol, _ = volume_mode.partial_volumes(
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
    _, wvol, rhopar, pvol, pvol_wat = volume_mode.partial_volumes(
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

    mdwat, wvol, rhopar, pvol, pvol_wat = volume_mode.partial_volumes(
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
    mdwat, _, rhopar, _, _ = volume_mode.partial_volumes(
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


# ---------------------------------------------------------------------------
# Task 44 -- the insoluble and inactive branches
# ---------------------------------------------------------------------------

# Setups with an active insoluble mode. Setups 1, 3 and 5 are four-mode,
# all-soluble, so `:638` is unreachable in them.
INSOLUBLE_SETUPS = tuple(
    s
    for s in SETUPS
    if any(modes.build(s).mode[m] and modes.build(s).modesol[m] != 1 for m in range(modes.NMODES))
)


def _insoluble_cols(tab) -> list[int]:
    return [m for m in range(modes.NMODES) if tab.mode[m] and tab.modesol[m] != 1]


def _inactive_cols(tab) -> list[int]:
    return [m for m in range(modes.NMODES) if not tab.mode[m]]


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("grid_name", ["trop", "strat"])
def test_every_mode_of_every_setup_is_byte_equal(setup, grid_name):
    """The whole mode loop, all three branches, all eight columns.

    Up to task 43 the comparisons were sliced to the soluble modes; this is the
    unsliced one, so an insoluble or absent column that was quietly wrong until
    now has nowhere left to hide.
    """
    tab, grid = (_trop_grid if grid_name == "trop" else _strat_grid)(setup)
    want = _reference(setup, _inputs(grid))
    mdwat, wvol, rhopar, pvol, pvol_wat = volume_mode.partial_volumes(
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
        ("mdwat", mdwat, want["mdwat"]),
        ("wvol", wvol, want["wvol"]),
        ("rhopar", rhopar, want["rhopar"]),
        ("pvol_wat", pvol_wat, want["pvol_wat"]),
        ("pvol", pvol, want["pvol"]),
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref), err_msg=name)


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", INSOLUBLE_SETUPS)
def test_the_insoluble_pvol_is_unmasked(setup):
    """`:647` has no `WHERE` and no `dvol*mfrac_0` default, unlike `:597`,
    `:613` and `:686`.

    Demonstrated on an EMPTY insoluble mode -- `nd` at or below `num_eps` --
    where the two spellings genuinely differ: the Fortran writes
    `md*mm_ovravcrhocp` and the symmetric default would write `dvol*mfrac_0`.
    """
    tab = modes.build(setup)
    cols = _insoluble_cols(tab)
    specs = [_spec(nd=kind) for kind in ("zero", "eps_below", "eps_exact", "eps_above")]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))

    scales = volume_mode._Scales(tab)
    md = np.asarray(want["md_out"])
    distinguishable = False
    for m in cols:
        empty = np.asarray(grid["nd"][:, m] <= tab.num_eps[m])
        assert empty.any(), "no unmasked row; the point of the test is gone"
        for c in range(tab.ncp):
            if not tab.component[m, c]:
                continue
            unmasked = md[empty, m, c] * np.asarray(scales.mm_ovravcrhocp)[c]
            default = np.asarray(want["dvol"])[empty, m] * tab.mfrac_0[m, c]
            np.testing.assert_array_equal(
                np.asarray(want["pvol"])[empty, m, c],
                unmasked,
                err_msg=f"setup {setup} mode {m + 1} cpt {c + 1}: :647 was masked",
            )
            distinguishable |= not np.array_equal(unmasked, default)
    assert distinguishable, (
        f"setup {setup}: the unmasked value equals the mfrac_0 default on every "
        "empty row, so adding the symmetric default would be invisible here"
    )


def test_the_third_guarded_division_uses_mask_not_mask_sol():
    """`:665` is guarded by `mask`; `:579` and `:627` by `mask_sol`.

    On an insoluble mode `mask_sol` is never computed at all -- there is no
    `mdsol` in that branch -- so reusing `:627`'s guard would mean inventing
    one. The distinction is asserted from the source text as well as from
    behaviour, because it is a *structural* claim about which masks exist.
    """
    source = SOURCE.read_text(encoding="utf-8")
    body = source[source.index("ELSE  ! if mode not soluble") :]
    body = body[: body.index("END IF ! if mode is soluble")]
    assert "rhopar(:,imode)=rhotmp2(:)/denom2(:)" in body, ":665 no longer reads as expected"
    assert "mask_sol" not in body, (
        "the insoluble branch now mentions mask_sol; :665's guard may have changed"
    )
    assert "WHERE (mask(:))" in body


@needs_binding
@pytest.mark.fortran
def test_the_insoluble_branch_has_no_denom2_diagnostic():
    """The `denom2 <= 0` `ereport` block at `:476-575` is inside the SOLUBLE arm
    only, so `:665` divides by whatever `denom2` holds.

    The gap is real and the port reproduces it -- `numerics.safe_divide` is
    guarded on `mask`, not on `denom2 != 0`, so a masked point with
    `denom2 == 0` gives an infinity here exactly as the reference does.

    It is also **unreachable**, and provably rather than empirically: no
    insoluble mode in any supported setup carries two different densities
    (mode 5 is bc + oc at 1500, modes 6 and 7 are dust at 2650). Write
    `x_c = md_c*mm_c`. Then `denom2 = sum x_c`, `rhotmp2 = rho*sum x_c` and
    `dvol = (1/avogadro)*sum x_c/rho` with a single `rho`, so `denom2 == 0`
    forces `dvol == 0`, and `ukca_calc_drydiam` -- or, past it, the five-way
    guard at `:704` -- stops the call before `:665` can be observed.

    Recorded as an invariant so that a future setup with a mixed-density
    insoluble mode fails this test rather than silently opening the gap.
    """
    for setup in INSOLUBLE_SETUPS:
        tab = modes.build(setup)
        for m in _insoluble_cols(tab):
            densities = {tab.rhocomp[c] for c in range(tab.ncp) if tab.component[m, c]}
            assert len(densities) == 1, (
                f"setup {setup} mode {m + 1} now carries components of {len(densities)} "
                "different densities, so denom2 == 0 with dvol > 0 has become "
                "constructible and :665 can divide by zero unreported"
            )

    source = SOURCE.read_text(encoding="utf-8")
    soluble_arm = source[
        source.index("mask_error(:)=") : source.index("ELSE  ! if mode not soluble")
    ]
    assert "DENOM2(I) <= 0.0" in soluble_arm, "the diagnostic moved"
    insoluble_arm = source[source.index("ELSE  ! if mode not soluble") :]
    insoluble_arm = insoluble_arm[: insoluble_arm.index("END IF ! if mode is soluble")]
    assert "ereport" not in insoluble_arm.lower(), "the insoluble branch gained a diagnostic"


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", INSOLUBLE_SETUPS)
def test_wvol_on_the_insoluble_branch_is_exactly_dvol(setup):
    """`:642`, set before the `pvol` loop and never accumulated into.

    So `wetdp` on an insoluble mode is a pure function of `dvol`: the partial
    volumes the branch goes on to compute do not enter it, however large they
    are. Asserted bit-for-bit against the Fortran's own `dvol`, on rows where
    the accumulated partial volumes are demonstrably non-zero.
    """
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid))
    for m in _insoluble_cols(tab):
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[:, m], np.asarray(want["dvol"])[:, m]
        )
        total = np.zeros(len(grid["rh"]))
        for c in range(tab.ncp):
            if tab.component[m, c]:
                total = total + np.asarray(want["pvol"])[:, m, c]
        assert (total > 0.0).any(), (
            f"mode {m + 1} accumulated no partial volume at all, so 'never "
            "accumulated into' is untested"
        )
        assert not np.array_equal(np.asarray(want["wvol"])[:, m], total), (
            f"mode {m + 1}: the sum of the partial volumes equals dvol, so a "
            "port that DID accumulate would be indistinguishable"
        )


def test_rhopar_on_the_insoluble_branch_cannot_be_shown_to_be_a_weighted_mean():
    """Recorded because the plan's acceptance criterion for it is unsatisfiable.

    It asks for "a setup-8 or setup-6 leaf row with mixed densities (dust 2650
    against BC/OC 1500)". No such row exists: setup 8's insoluble modes are 5
    (bc + oc, both 1500) and 6 and 7 (dust alone, 2650), and setup 6's are 6 and
    7 (dust alone). Every insoluble mode in every supported setup is
    single-density, so `rhotmp2/denom2` comes out at that density whatever the
    weights are, and no weighting error is observable on this branch.

    Where a genuine mass-weighted mean IS testable is the SOLUBLE branch, whose
    modes 3 and 4 carry su (1769), bc (1500), oc (1500) and cl (2165) together.
    That is covered by the byte-equality tests above, and the mutation "take
    rhopar from the soluble-only accumulators" reddens there.
    """
    mixed_soluble, mixed_insoluble = [], []
    for setup in SETUPS:
        tab = modes.build(setup)
        for m in range(modes.NMODES):
            if not tab.mode[m]:
                continue
            densities = {tab.rhocomp[c] for c in range(tab.ncp) if tab.component[m, c]}
            if len(densities) > 1:
                (mixed_soluble if tab.modesol[m] == 1 else mixed_insoluble).append((setup, m + 1))
    assert not mixed_insoluble, (
        f"an insoluble mode now has mixed densities: {mixed_insoluble}. The "
        "criterion has become satisfiable and deserves a real test."
    )
    assert mixed_soluble, "no soluble mode has mixed densities either; re-derive"


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_the_inactive_mode_pvol_default_is_dead_code(setup):
    """`:686` can never execute.

    `ukca_mode_setup.F90:693` sets `mode = (mode_choice > 0)` and `:700-702`
    makes `component(imode,icp)` true only when `mode_choice(imode) == 1`. So
    `component` implies `mode`, while `:686` is inside `ELSE` on `mode(imode)`.
    Checked from the tables rather than by reading the conditional, and the
    enclosing routine is checked too -- `ukca_mode_allcp_4mode` is dead code
    that contains a near-identical block.

    `pvol` for an absent mode is therefore never written by this routine at all,
    and the value the caller sees is the one it pre-zeroed. Asserted against the
    Fortran, not inferred.
    """
    tab = modes.build(setup)
    inactive = _inactive_cols(tab)
    if not inactive:
        pytest.skip(f"setup {setup} has all eight modes active")
    for m in inactive:
        assert not tab.component[m].any(), (
            f"setup {setup} mode {m + 1} is inactive and yet carries components; "
            ":686 has become live"
        )
        assert tab.mode_choice[m] != 1

    _, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid))
    for m in inactive:
        assert (np.asarray(want["pvol"])[:, m, :] == 0.0).all(), (
            "the Fortran wrote pvol for an absent mode; :686 is not dead"
        )
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[:, m], np.asarray(want["dvol"])[:, m]
        )
        assert (np.asarray(want["rhopar"])[:, m] == 1769.0).all()
        assert (np.asarray(want["mdwat"])[:, m] == 0.0).all()
        assert (np.asarray(want["pvol_wat"])[:, m] == 0.0).all()


def test_the_dead_default_would_be_visible_if_it_ever_ran():
    """Why `:686` is worth reproducing rather than deleting.

    Unlike the soluble branch, an inactive mode's `mfrac_0` is NOT zero at the
    cells `component` excludes -- dust carries 1.0 in modes 6-8 of most setups,
    and bc/oc carry 0.5 each in mode 5 of setups 1, 3, 5 and 6. So if a future
    table ever made `component` true for an inactive mode, `:686` would write a
    visibly non-zero `pvol` where the port currently leaves a zero.
    """
    nonzero = [
        (setup, m + 1, c + 1)
        for setup in SETUPS
        for m in range(modes.NMODES)
        for c in range(modes.build(setup).ncp)
        if not modes.build(setup).mode[m] and modes.build(setup).mfrac_0[m, c] != 0.0
    ]
    assert nonzero, "every inactive mode now has an all-zero mfrac_0; re-derive the argument"


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", INSOLUBLE_SETUPS)
def test_the_insoluble_accumulator_mask_is_not_observable(setup):
    """Recorded because dropping it is a vacuous mutation, and that is worth
    knowing before someone "tightens" the branch.

    `:658-668` accumulates `rhotmp2` and `denom2` under `WHERE (mask)`, and
    `:665-670` then writes `rhopar` under the same `WHERE` with `rho_so4` on the
    `ELSE`. So whatever the accumulators hold off the mask is discarded, and
    removing the inner mask leaves every byte-equality test in this file green
    -- measured.

    The mask that *is* observable in this branch is the one on `:665` itself:
    a masked-out row must come back as exactly `rho_so4`, not as a quotient.
    """
    tab, grid = _trop_grid(setup)
    want = _reference(setup, _inputs(grid))
    seen = False
    for m in _insoluble_cols(tab):
        empty = np.asarray(grid["nd"][:, m] <= tab.num_eps[m])
        if not empty.any():
            continue
        seen = True
        assert (np.asarray(want["rhopar"])[empty, m] == 1769.0).all(), (
            f"mode {m + 1}: a masked-out row is not rho_so4"
        )
        assert not (np.asarray(want["rhopar"])[~empty, m] == 1769.0).all(), (
            f"mode {m + 1}: every masked row is also 1769, so the :665 guard is untested here"
        )
    assert seen, "no masked-out insoluble row on this fixture"


# ---------------------------------------------------------------------------
# Task 45 -- wetdp, and the whole routine
# ---------------------------------------------------------------------------


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("grid_name", ["trop", "strat"])
@pytest.mark.parametrize("fix_water", [0, 1], ids=["unfixed", "fixed"])
@pytest.mark.parametrize("fix_neg", [0, 1], ids=["unclamped", "clamped"])
def test_the_whole_routine_is_byte_equal(setup, grid_name, fix_water, fix_neg):
    """All six outputs, all eight modes, both grids, both flags.

    This is the gate the four preceding tasks were building towards: one call to
    `volume_mode` against one call to `leaf_volume_mode`, compared with
    `assert_array_equal`.
    """
    tab, grid = (_trop_grid if grid_name == "trop" else _strat_grid)(setup)
    want = _reference(setup, _inputs(grid), fix_water=fix_water, fix_neg=fix_neg)
    mdwat, wvol, wetdp, rhopar, pvol, pvol_wat = volume_mode.volume_mode(
        tab,
        grid["nd"],
        want["md_out"],
        want["mdt_out"],
        grid["rh"],
        want["dvol"],
        want["drydp"],
        grid["t"],
        grid["pmid"],
        grid["s"],
        fix_water_content=bool(fix_water),
        fix_neg_pvol_wat=bool(fix_neg),
    )
    for name, got in (
        ("mdwat", mdwat),
        ("wvol", wvol),
        ("wetdp", wetdp),
        ("rhopar", rhopar),
        ("pvol", pvol),
        ("pvol_wat", pvol_wat),
    ):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want[name]), err_msg=name)


def test_wetdp_uses_the_volume_mode_spelling_of_sixovrpix():
    """`1.0/(x*(pi/6.0))` (`:292-293`), not `drydiam`'s `6.0/(pi*x)` (`:230`).

    The two are 2 ulp apart for two of the width parameters and the cube root
    amplifies that: measured here over 200,000 random `wvol` in [1e-19, 1e-15],
    the two spellings give different `wetdp` on more than half the samples for
    mode 4, which is active in six of the seven supported setups.

    A shared helper is the mutation, and it is the tempting one -- the two
    expressions look like the same constant.
    """
    from glomap_jax.physics import drydiam

    rng = np.random.default_rng(292)
    wvol = rng.uniform(1.0e-19, 1.0e-15, 200_000)
    amplified = {}
    for setup in SETUPS:
        tab = modes.build(setup)
        ours = np.asarray(volume_mode.six_over_pi_x(tab.x))
        theirs = np.asarray(drydiam.six_over_pi_x(tab.x))
        for m in range(modes.NMODES):
            if ours[m] == theirs[m]:
                continue
            a = (ours[m] * wvol) ** (1.0 / 3.0)
            b = (theirs[m] * wvol) ** (1.0 / 3.0)
            amplified[m + 1] = max(amplified.get(m + 1, 0.0), (a != b).mean())
    assert amplified, (
        "the two spellings now agree on every mode of every setup; re-measure "
        "before factoring them together"
    )
    assert max(amplified.values()) > 0.4, (
        f"the cube root no longer amplifies the 2 ulp gap: {amplified}"
    )
    assert 4 in amplified, "mode 4 no longer distinguishes the two spellings"


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_the_cube_root_runs_over_all_eight_modes_before_the_override(setup):
    """`:696` passes `nmodes*nbox` to `cubrt_v`, i.e. the whole array by sequence
    association, and `:698` then discards the inactive columns.

    So `wetdp` for an absent mode is `drydp` **bit for bit** -- not a cube root
    that happens to be close, and not zero. Asserted against the Fortran's own
    `drydp`, which makes it a check on the reference rather than on the port's
    own arithmetic.

    How much `:698` actually changes is worth stating precisely, because it is
    less than it looks. For an absent mode `wvol = dvol` exactly (`:681`), so
    the discarded cube root and `drydp` differ **only** through the two
    spellings of `sixovrpix` -- `1.0/(x*(pi/6))` here against `6.0/(pi*x)` in
    `ukca_calc_drydiam`. On the modes where those two agree bitwise, `:698` is a
    numerical no-op. The companion test below finds the inactive modes where it
    is not, across all seven setups, and requires at least one.
    """
    tab, grid = _trop_grid(setup)
    inactive = _inactive_cols(tab)
    if not inactive:
        pytest.skip(f"setup {setup} has all eight modes active")
    want = _reference(setup, _inputs(grid))
    for m in inactive:
        np.testing.assert_array_equal(
            np.asarray(want["wetdp"])[:, m],
            np.asarray(want["drydp"])[:, m],
            err_msg=f"mode {m + 1}",
        )
        # And wvol really is dvol there, which is what makes the two comparable.
        np.testing.assert_array_equal(
            np.asarray(want["wvol"])[:, m], np.asarray(want["dvol"])[:, m]
        )


@needs_binding
@pytest.mark.fortran
def test_the_inactive_override_is_not_everywhere_a_no_op():
    """At least one inactive mode where dropping `:698` would change `wetdp`.

    Without this, the test above passes on a port that never implements the
    override at all -- on the modes where the two `sixovrpix` spellings agree,
    the cube root reproduces `drydp` by itself.
    """
    from glomap_jax.physics import drydiam

    observable = []
    for setup in SETUPS:
        tab, grid = _trop_grid(setup)
        inactive = _inactive_cols(tab)
        if not inactive:
            continue
        want = _reference(setup, _inputs(grid))
        ours = np.asarray(volume_mode.six_over_pi_x(tab.x))
        theirs = np.asarray(drydiam.six_over_pi_x(tab.x))
        for m in inactive:
            if ours[m] == theirs[m]:
                continue
            discarded = (ours[m] * np.asarray(want["wvol"])[:, m]) ** (1.0 / 3.0)
            if not np.array_equal(discarded, np.asarray(want["drydp"])[:, m]):
                observable.append((setup, m + 1))
    assert observable, (
        "on every inactive mode of every setup the discarded cube root already "
        "equals drydp, so :698 cannot be shown to fire and the test above is "
        "vacuous"
    )


@needs_binding
@pytest.mark.fortran
def test_the_wvol_accumulation_order_reaches_wetdp():
    """`wvol` is zeroed at `:588`, accumulated over `icp` in index order, and
    `pvol_wat` is added **last** at `:625`; `wetdp` is its cube root.

    This is the path that reaches the live trajectory -- `wetdp` feeds
    `ukca_conden` (`:926`), `ukca_ageing` (`:1130`) and
    `ukca_calc_coag_kernel` (`:871`) -- so what is asserted is that adding the
    water first changes `wetdp`, not merely `wvol`.

    Aggregated over every setup and every soluble mode rather than asserted per
    mode, because it does not survive everywhere: the cube root compresses a
    relative difference by three, so a one-ulp gap in `wvol` frequently lands on
    the same double. Measured -- setup 8 mode 3 is one where it does not
    survive. What matters for the port is that the difference reaches `wetdp`
    somewhere reachable, and that the fixture shows where.
    """
    surviving, lost = [], []
    for setup in SETUPS:
        tab, grid = _trop_grid(setup)
        cols = _soluble_cols(tab)
        if not cols:
            continue
        want = _reference(setup, _inputs(grid))
        sixovrpix = np.asarray(volume_mode.six_over_pi_x(tab.x))
        for m in cols:
            members = [c for c in range(tab.ncp) if tab.component[m, c]]
            if len(members) < 2:
                continue
            parts = [np.asarray(want["pvol"])[:, m, c] for c in members]
            water = np.asarray(want["pvol_wat"])[:, m]

            last = parts[0].copy()
            for x in parts[1:]:
                last = last + x
            last = last + water

            first = water + parts[0]
            for x in parts[1:]:
                first = first + x

            if np.array_equal(last, first):
                continue
            a = (sixovrpix[m] * last) ** (1.0 / 3.0)
            b = (sixovrpix[m] * first) ** (1.0 / 3.0)
            (surviving if not np.array_equal(a, b) else lost).append((setup, m + 1))

    assert surviving, (
        "adding the water term first no longer changes wetdp on any soluble "
        f"mode of any setup (it changed wvol on {len(lost)} of them); the "
        "accumulation-order hazard is no longer reachable and this test would "
        "pass on a port that got the order wrong"
    )


@needs_binding
@pytest.mark.fortran
def test_mask_nosol_accumulates_then_discards():
    """At a `mask_nosol` point `:612` accumulates the insoluble partial volumes
    into `wvol` and `:633` then overwrites the whole thing with `dvol`.

    **And the overwrite is numerically invisible in every supported setup** --
    measured here, not assumed. `:605` builds the insoluble `pvol` from
    `mm_ovravcrhocp = (mm/avogadro)/rhocomp`, and
    `ukca_calc_drydiam.F90:196` builds `dvol` from
    `ratio1 = mm/(avogadro*rhocomp)`, in the same `icp` order from the same
    seed. Those two spellings differ only for `cp_su` and `cp_cl` -- both
    soluble, so neither reaches this fold -- and agree bitwise for `cp_bc`,
    `cp_oc`, `cp_du` and `cp_so`. So the accumulated volume already equals
    `dvol` to the last bit and `:633` changes nothing.

    That is why this test asserts the *equality* rather than a difference: a
    test demanding the discard be observable would be one that cannot pass. What
    it does pin is that `wetdp` at such a point is the cube root of
    `sixovrpix*dvol`, and that the fold really did run first.
    """
    setup = 2
    tab = modes.build(setup)
    specs = [_spec(rh=rh, variant="nosol") for rh in (0.2, 0.62, 0.9)]
    grid = _rows(tab, specs)
    want = _reference(setup, _inputs(grid))
    sixovrpix = np.asarray(volume_mode.six_over_pi_x(tab.x))

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
        accumulated = np.zeros(int(rows.sum()))
        for c in range(tab.ncp):
            if tab.component[m, c] and not tab.soluble[c]:
                accumulated = accumulated + np.asarray(want["pvol"])[rows, m, c]
        assert (accumulated > 0.0).all(), "nothing was accumulated before the discard"
        np.testing.assert_array_equal(
            np.asarray(want["wetdp"])[rows, m],
            (sixovrpix[m] * np.asarray(want["dvol"])[rows, m]) ** (1.0 / 3.0),
        )
        # And the discard is numerically invisible, which is the point worth
        # recording: the accumulated insoluble volume equals dvol bit for bit.
        np.testing.assert_array_equal(accumulated, np.asarray(want["dvol"])[rows, m])
    assert seen, "no mask_nosol row; the discard is untested"


def test_the_neg_pvol_wat_gate_is_always_active_in_the_box():
    """`:882` reads `l_fix_neg_pvol_wat .OR. l_glomap_clim_radaer`, and the
    second disjunct is a `LOGICAL, PARAMETER, .FALSE.`
    (`ukca_um_legacy_mod.F90:141`), so the gate reduces to the flag -- which
    `glomap_box_config_mod.F90:323` pins `.TRUE.`.

    Modelling the disjunct as a runtime flag would create a state the reference
    cannot reach, which is why the port has no knob for it. Asserted from the
    vendored source so an upstream change fails here rather than silently
    widening the configuration space.
    """
    legacy = (REPO / "fortran" / "src" / "ukca" / "ukca_um_legacy_mod.F90").read_text()
    assert re.search(r"LOGICAL,\s*PARAMETER\s*::\s*l_glomap_clim_radaer\s*=\s*\.FALSE\.", legacy), (
        "l_glomap_clim_radaer is no longer a PARAMETER .FALSE.; :882 has a second knob"
    )
    box = (REPO / "fortran" / "src" / "box" / "glomap_box_config_mod.F90").read_text()
    assert re.search(r"glomap_config%l_fix_neg_pvol_wat\s*=\s*\.TRUE\.", box), (
        "the box no longer pins l_fix_neg_pvol_wat true"
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "IF (glomap_config%l_fix_neg_pvol_wat .OR. l_glomap_clim_radaer) THEN" in source


def test_the_diagnostic_ereport_blocks_are_omitted():
    """The port's disposition of the three `ereport` blocks, stated where it can
    fail rather than only in a docstring.

    * `:476-575` (`denom <= 0`/`denom2 <= 0` under `mask_sol`) -- omitted. It
      prints and aborts and computes nothing that reaches an output.
    * `:703-880` (the five-way `MINVAL <= 0` guard) -- omitted, and
      **unreachable**: tripping it overruns the 256-character `cmessage` buffer
      in the `WRITE` at `:856-876` and gfortran raises "End of record" at
      `:876`, killing the process before `ereport` at `:877`. There is no
      reference behaviour to reproduce.
    * `:882-898` (`MINVAL(pvol_wat) < 0` or `MINVAL(mdwat) < 0`) -- omitted, and
      this one is **always active** in the box, so the port silently accepts a
      negative `mdwat` where the reference dies. That is a real divergence, not
      a tidy-up, and it is recorded here and in the module docstring rather than
      hidden.

    If the third is ever implemented it needs `nanmin` semantics: gfortran's
    `MINVAL` skips `NaN` and `jnp.min` propagates it.
    """
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("CALL ereport(RoutineName,errcode,cmessage)") == 4, (
        "the number of ereport sites has changed; re-decide the disposition. "
        "There are four: denom and denom2 inside :476-575, the five-way guard "
        "at :877, and the negative-water check at :895."
    )
    assert "errormessagelength" in source
    module = (REPO / "src" / "glomap_jax" / "physics" / "volume_mode.py").read_text(
        encoding="utf-8"
    )
    for phrase in (":476-575", ":703-880", ":882-898", "End of record"):
        assert phrase in module, f"the module docstring no longer records {phrase}"
    # And the port really does not raise on a negative mdwat.
    tab = modes.build(1)
    nd = np.full((1, modes.NMODES), 1000.0)
    md = np.zeros((1, modes.NMODES, tab.ncp))
    for m in range(modes.NMODES):
        for c in range(tab.ncp):
            if tab.component[m, c]:
                md[0, m, c] = tab.mmid[m] / max(1, int(tab.component[m].sum()))
    dvol = np.full((1, modes.NMODES), 1.0e-24)
    out = volume_mode.volume_mode(
        tab,
        nd,
        md,
        md.sum(axis=2),
        np.array([0.6]),
        dvol,
        dvol ** (1.0 / 3.0),
        np.array([303.65]),
        np.array([1.0e4]),
        np.array([1.0e-8]),
        fix_water_content=True,
        fix_neg_pvol_wat=False,
    )
    assert (np.asarray(out[0]) < 0.0).any(), (
        "the unclamped stratospheric row no longer produces a negative mdwat, "
        "so the omitted guard has nothing to be silent about"
    )


def test_mdt_reaches_no_output_of_the_port():
    """`mdt` is read at `:842` only, inside the block that cannot run.

    It stays in the signature so a caller cannot silently transpose arguments,
    and this asserts it is genuinely inert: garbage in, identical outputs.
    """
    tab = modes.build(2)
    nd = np.full((3, modes.NMODES), 1000.0)
    md = np.zeros((3, modes.NMODES, tab.ncp))
    for m in range(modes.NMODES):
        for c in range(tab.ncp):
            if tab.component[m, c]:
                md[:, m, c] = tab.mmid[m] / max(1, int(tab.component[m].sum()))
    dvol = np.full((3, modes.NMODES), 1.0e-24)
    args = (
        tab,
        nd,
        md,
    )
    tail = (
        np.array([0.3, 0.6, 0.9]),
        dvol,
        dvol ** (1.0 / 3.0),
        np.full(3, 213.0),
        np.array([1.0e5, 1.0e4, 1.0e5]),
        np.full(3, 1.0e-2),
    )
    clean = volume_mode.volume_mode(
        *args, md.sum(axis=2), *tail, fix_water_content=True, fix_neg_pvol_wat=True
    )
    for poison in (np.nan, -1.0, 1e300):
        dirty = volume_mode.volume_mode(
            *args,
            np.full((3, modes.NMODES), poison),
            *tail,
            fix_water_content=True,
            fix_neg_pvol_wat=True,
        )
        for a, b in zip(clean, dirty, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_mm_ovravcrhocp_is_unobservable_through_the_outputs():
    """A companion to `test_mm_ovravcrhocp_is_two_divisions`, and a caveat on it.

    `(mm/avogadro)/rhocomp` and `mm/(avogadro*rhocomp)` really are different
    doubles -- but only for `cp_su` and `cp_cl`. Every other component agrees
    bitwise, and `mm_ovravcrhocp` is read only at `:605` and `:647`, both of
    which are guarded `.NOT. soluble(icp)`. So the two spellings cannot be told
    apart from any output of this routine in any supported setup: substituting
    one for the other leaves every byte-equality test in this file green
    (measured).

    The faithful spelling is kept and the expression-level test is kept, because
    the components that differ are exactly the ones a future setup might make
    insoluble. This test records that the guard is currently expression-level
    only.
    """
    differing_all, differing_insoluble = set(), set()
    for setup in SETUPS:
        tab = modes.build(setup)
        mm = np.asarray(tab.mm[: tab.ncp])
        rho = np.asarray(tab.rhocomp[: tab.ncp])
        two, one = (mm / 6.022e23) / rho, mm / (6.022e23 * rho)
        for c in range(tab.ncp):
            if two[c] == one[c]:
                continue
            differing_all.add(c + 1)
            if not tab.soluble[c]:
                differing_insoluble.add((setup, c + 1))
    assert differing_all == {volume_mode.CP_SU, volume_mode.CP_CL}, (
        f"the set of components where the two spellings differ has changed: {sorted(differing_all)}"
    )
    assert not differing_insoluble, (
        f"an INSOLUBLE component now distinguishes the two spellings "
        f"({sorted(differing_insoluble)}), so mm_ovravcrhocp has become "
        "observable through pvol and deserves a byte-equality test"
    )


@needs_binding
@pytest.mark.fortran
def test_the_fixture_distinguishes_the_cube_root_spelling():
    """`wetdp` goes through `numerics.cbrt`, i.e. `x ** (1.0/3.0)` -- and this
    fixture is asked to show that it matters.

    Whether it *can* is interpreter-dependent, so this measures rather than
    assumes. `jnp.cbrt` and `x**(1.0/3.0)` are bit-identical on jax 0.9.2 over
    800,000 samples, and differ on jax 0.11.0 -- the same split
    `tests/test_numerics.py::test_how_far_jnp_cbrt_is_from_the_faithful_form_on_this_jax`
    records. On the canonical interpreter (`.venv`, `Makefile:18`) they differ,
    the fixture's own `sixovrpix*wvol` values reach a disagreement, and
    substituting `jnp.cbrt` in `wet_diameter` reddens the byte-equality gate --
    measured.

    Where they agree the substitution is vacuous, and the test says so instead
    of implying the green suite proved something.
    """
    differing = 0
    total = 0
    for setup in SETUPS:
        for grid_name in ("trop", "strat"):
            tab, grid = (_trop_grid if grid_name == "trop" else _strat_grid)(setup)
            want = _reference(setup, _inputs(grid))
            tmp1 = np.asarray(volume_mode.six_over_pi_x(tab.x))[None, :] * np.asarray(want["wvol"])
            faithful = np.asarray(jnp.asarray(tmp1) ** (1.0 / 3.0))
            other = np.asarray(jnp.cbrt(jnp.asarray(tmp1)))
            differing += int((faithful != other).sum())
            total += faithful.size

    generally_differ = not np.array_equal(
        np.asarray(jnp.asarray(np.exp(np.linspace(-70.0, -34.0, 20_000))) ** (1.0 / 3.0)),
        np.asarray(jnp.cbrt(jnp.asarray(np.exp(np.linspace(-70.0, -34.0, 20_000))))),
    )
    if not generally_differ:
        assert differing == 0
        pytest.skip(
            f"jax {jax.__version__}: jnp.cbrt is bit-identical to x**(1.0/3.0), so no "
            "fixture can distinguish them. The rule rests on tests/test_numerics.py."
        )
    assert differing > 0, (
        f"jax {jax.__version__}: jnp.cbrt differs from x**(1.0/3.0) in general but on "
        f"none of this fixture's {total} wetdp inputs, so a port using the forbidden "
        "cube root would pass the byte-equality gate. Widen the wvol range."
    )


# ---------------------------------------------------------------------------
# The divide-by-constant rewrite, which cost this module its byte equality once
# ---------------------------------------------------------------------------


def test_dividing_an_array_by_a_scalar_constant_is_a_true_divide():
    """`numerics.true_divide`, and why every array-by-constant division uses it.

    XLA rewrites `divide(x, broadcast(c))` into `multiply(x, broadcast(1/c))`
    for **any** scalar constant, not only powers of two, and `1/c` is inexact
    for every constant this routine divides by. Whether the rewrite happens
    eagerly depends on the JAX version, so this test asserts the invariant that
    holds on both -- `true_divide` reproduces a true division -- and *reports*
    whether the plain spelling does.

    This is not hypothetical. The module was first validated byte-equal on jax
    0.9.2, where `x / c` is a true divide, and produced **73 failures at 1-2
    ulp** on jax 0.11.0, where it is not. The canonical interpreter is `.venv`
    (`Makefile:18`).
    """
    from glomap_jax.core import numerics as num

    rng = np.random.default_rng(2049)
    y = rng.uniform(1.0e5, 1.0e15, 200_000)
    jy = jnp.asarray(y)
    divisors = {
        "f_ao": volume_mode.aged_organic_moles(),
        "avogadro": 6.022e23,
        "mmwovravc": 0.0180154 / 6.022e23,
    }
    rewritten = {}
    for name, c in divisors.items():
        want = y / c  # numpy: a true division, which is what gfortran does
        np.testing.assert_array_equal(
            np.asarray(num.true_divide(jy, c)),
            want,
            err_msg=f"true_divide is not a true divide for {name}",
        )
        rewritten[name] = int((np.asarray(jy / c) != want).sum())
        # And the reciprocal-multiply really is a different number, so the
        # helper is doing work rather than spelling the same thing twice.
        assert (np.asarray(jy * (1.0 / c)) != want).sum() > 0, (
            f"{name}: multiplying by the reciprocal now agrees with dividing, so "
            "this test cannot fail"
        )
    print(f"\njax {jax.__version__}: plain `x / c` departs from a true divide on {rewritten}")


def test_the_port_never_divides_an_array_by_a_bare_scalar_constant():
    """A source check, because the failure mode is invisible at review.

    `x / AVOGADRO` and `numerics.true_divide(x, AVOGADRO)` read identically and
    differ by 1 ulp on the canonical interpreter. Every array-by-constant site
    in the module -- `:368`, `:372`/`:381`, `:396-398`, `:435`, `:437` and
    `:294` -- must go through the helper, so this greps for the bare form.

    `1.0/(x*piovrsix)` and `100.0/wts` are exempt and are matched out: their
    numerator is the constant, so there is no constant divisor to fold.

    **Three of the seven sites are guarded by this grep alone.** Reverting
    `:368`, `:396-398` or `:294` to the bare form leaves every byte-equality
    test in this file green -- measured -- because the reciprocal-multiply and
    the true divide happen to agree on the `md` values this fixture carries
    (`avogadro`'s reciprocal differs on only ~3% of random doubles). Reverting
    `:372`/`:381`, `:435` or `:437` reddens the byte-equality gate directly.
    So this test is not belt-and-braces; for half the sites it is the only
    thing standing between the port and a silent 1 ulp.
    """
    module = (REPO / "src" / "glomap_jax" / "physics" / "volume_mode.py").read_text(
        encoding="utf-8"
    )
    body = module[module.index("CP_SU = 1") :]
    # Scalar-by-scalar arithmetic is exempt: it happens in Python, not XLA, so
    # there is no operand for the simplifier to fold. `MMW / AVOGADRO` at `:295`
    # is the only such site.
    exempt = ("self.mmwovravc = MMW / AVOGADRO",)
    offenders = [
        line.strip()
        for line in body.splitlines()
        if re.search(r"/\s*(AVOGADRO|f_ao|MMW|scales\.mmwovravc)\b", line)
        and "true_divide" not in line
        and not line.lstrip().startswith("#")
        and '"""' not in line
        and not line.lstrip().startswith("`")
        and line.strip() not in exempt
    ]
    assert not offenders, (
        "an array is divided by a bare scalar constant, which XLA rewrites into "
        f"a reciprocal multiply on the canonical interpreter: {offenders}"
    )
    assert body.count("numerics.true_divide(") == 7, (
        f"expected seven true_divide sites, found {body.count('numerics.true_divide(')}; "
        "if a division was added or removed, re-derive the list in the docstring"
    )
