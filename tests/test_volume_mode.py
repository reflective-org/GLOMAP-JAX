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
    "negsu",
    "tiny",
)

# Relative weights for `typical`, as decimal literals so the grid is
# reproducible bit for bit on any host (no libm in an abscissa).
TYPICAL_WEIGHTS = (0.50, 0.10, 0.20, 0.30, 0.40, 0.15)

T_REF, S_REF, RH_REF = 213.0, 1.0e-2, 0.6
P_TROP = 1.0e5


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
        fix_water_content=bool(fix_water),
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
    got = volume_mode.mdwat(tab, grid["nd"], want["md_out"], grid["rh"], fix_water_content=True)
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

    got = volume_mode.mdwat(tab, grid["nd"], want["md_out"], grid["rh"], fix_water_content=True)
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
    got = volume_mode.mdwat(tab, grid["nd"], want["md_out"], grid["rh"], fix_water_content=True)
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

    got = volume_mode.mdwat(tab, grid["nd"], want["md_out"], grid["rh"], fix_water_content=True)
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
