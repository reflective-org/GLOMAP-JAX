"""The parts of the capture scripts that must hold before a golden is written.

None of these need the f2py extension, and that is the point. A capture script
runs on a machine with gfortran, by hand, rarely — so the checks that stop it
writing a bad golden are the ones least likely to have been exercised. What is
testable in CI is the text rewriting, the anti-collapse guards and the grid
construction, so those are pulled out into functions and tested here.

The failure being guarded against is on the record: a capture script's namelist
substitution silently matched nothing, the golden held identical data for all
seven mode setups, and every byte-equality test passed against it.
"""

import ast
import inspect
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"

sys.path.insert(0, str(REPO / "validation"))

import capture_coag_mode as ccm  # noqa: E402
import capture_leaf as cl  # noqa: E402
import capture_modes as cm  # noqa: E402

NAMELIST = (REPO / "fortran" / "namelists" / "boundary_layer.nml").read_text(encoding="utf-8")


def _setup_in(text):
    return [int(v) for v in re.findall(r"^\s*i_mode_setup\s*=\s*(\d+)", text, flags=re.MULTILINE)]


# ---------------------------------------------------------------------------
# capture_modes: the namelist rewrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setup", cm.SETUPS)
@pytest.mark.parametrize("combo", list(cm.COMBOS))
def test_render_namelist_sets_the_setup_it_was_asked_for(setup, combo):
    """The substitution that silently no-op'd once already."""
    rendered = cm.render_namelist(NAMELIST, setup, combo)
    assert _setup_in(rendered) == [setup]


@pytest.mark.parametrize("combo,overrides", list(cm.COMBOS.items()))
def test_render_namelist_writes_every_switch_explicitly(combo, overrides):
    """All five switches are emitted whatever the combination overrides, so the
    capture cannot move because a namelist default moved."""
    rendered = cm.render_namelist(NAMELIST, 1, combo)
    group = rendered.split("&box_aerosol", 1)[1].split("\n/", 1)[0]
    for switch in cm.SWITCH_DEFAULTS:
        assert re.search(rf"^\s*{switch}\s*=\s*(\S+)", group, flags=re.MULTILINE), switch
    for switch, value in overrides.items():
        written = re.search(rf"^\s*{switch}\s*=\s*(\S+)", group, flags=re.MULTILINE).group(1)
        expected = str(value) if isinstance(value, int) and not isinstance(value, bool) else None
        assert written == (expected or (".TRUE." if value else ".FALSE."))


def test_render_namelist_refuses_a_namelist_with_no_setup_line():
    """The exact shipped failure: nothing to substitute, and a capture that
    carries on regardless produces seven copies of the default."""
    text = NAMELIST.replace("i_mode_setup = 1", "! i_mode_setup removed")
    with pytest.raises(SystemExit, match="i_mode_setup"):
        cm.render_namelist(text, 4, "default")


def test_render_namelist_refuses_a_namelist_with_two_setup_lines():
    """`count=1` would quietly edit the first and leave the second to win."""
    text = NAMELIST.replace("i_mode_setup = 1", "i_mode_setup = 1\n  i_mode_setup = 2")
    with pytest.raises(SystemExit, match="found 2"):
        cm.render_namelist(text, 4, "default")


def test_render_namelist_refuses_a_namelist_with_no_box_aerosol_group():
    text = NAMELIST.replace("&box_aerosol", "&box_aerosol_renamed")
    with pytest.raises(SystemExit, match="box_aerosol"):
        cm.render_namelist(text, 4, "default")


def test_render_namelist_refuses_an_unknown_combination():
    with pytest.raises(SystemExit, match="unknown switch combination"):
        cm.render_namelist(NAMELIST, 1, "no_such_combo")


# ---------------------------------------------------------------------------
# capture_modes: the anti-collapse guards
# ---------------------------------------------------------------------------


def _records_from_golden():
    """The committed archive, back in the shape `capture_one` returns.

    Testing the guards against the real capture rather than against invented
    data: if `check_capture_varied` would reject the golden this repo already
    has, it is the guard that is wrong.
    """
    data = np.load(GOLDENS / "modes.f64.tables.npz", allow_pickle=False)
    fields = sorted({k[len("s1_") :] for k in data.files if k.startswith("s1_")})
    records = {}
    for combo in cm.COMBOS:
        records[combo] = {}
        for setup in cm.SETUPS:
            prefix = f"s{setup}_" if combo == "default" else f"v_{combo}_s{setup}_"
            rec = {f: data[prefix + f].tolist() for f in fields}
            rec["setup"] = setup
            records[combo][setup] = rec
    return records


@pytest.fixture(scope="module")
def golden_records():
    return _records_from_golden()


def test_the_guards_accept_the_committed_capture(golden_records):
    cm.check_capture_varied(golden_records)


def test_a_capture_that_ran_one_setup_seven_times_is_refused(golden_records):
    collapsed = {
        combo: {setup: dict(recs[cm.SETUPS[0]], setup=setup) for setup in cm.SETUPS}
        for combo, recs in golden_records.items()
    }
    with pytest.raises(SystemExit, match="identical tables"):
        cm.check_capture_varied(collapsed)


def test_a_capture_in_which_two_setups_collide_is_refused(golden_records):
    """Not only total collapse: one setup that did not take is enough."""
    records = {c: dict(r) for c, r in golden_records.items()}
    records["default"][8] = dict(records["default"][2], setup=8)
    with pytest.raises(SystemExit, match=r"setups 2==8"):
        cm.check_setups_differ(records["default"], "default")


def test_switch_combinations_that_did_nothing_are_refused(golden_records):
    default = golden_records["default"][1]
    by_combo = {combo: dict(default) for combo in cm.COMBOS}
    with pytest.raises(SystemExit, match="collided as"):
        cm.check_combos_differ(by_combo, 1)


def test_the_documented_i_tune_bc_fall_through_is_required_not_merely_tolerated(golden_records):
    """`bc_oob` must stay identical to `default`. If it stops being identical
    the unnamed-CASE fall-through has changed, which is a finding about the
    Fortran, not a licence to write the archive anyway."""
    by_combo = {c: golden_records[c][1] for c in cm.COMBOS}
    by_combo["bc_oob"] = dict(by_combo["bc_oob"], rhocomp=[0.0] * 6)
    with pytest.raises(SystemExit, match="collided as \\[none\\]"):
        cm.check_combos_differ(by_combo, 1)


def test_a_missing_combination_is_refused(golden_records):
    by_combo = {c: golden_records[c][1] for c in cm.COMBOS if c != "dust_ageing"}
    with pytest.raises(SystemExit, match="no record for combination"):
        cm.check_combos_differ(by_combo, 1)


# ---------------------------------------------------------------------------
# Both scripts: the children, and the trailing ierr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", [cm._CHILD, ccm._CHILD], ids=["modes", "coag_mode"])
def test_the_child_scripts_compile(script):
    """They run seven subprocesses deep on a machine with a toolchain; a syntax
    error in one would surface there and nowhere else."""
    compile(script, "<child>", "exec")


@pytest.mark.parametrize("script", [cm._CHILD, ccm._CHILD], ids=["modes", "coag_mode"])
def test_the_child_scripts_read_the_setup_back_out_of_the_fortran(script):
    """Every check on the namelist text is a check on the text. This is the one
    that fails when the text was right and the setup still did not take."""
    assert "wrap_sizes()" in script
    assert re.search(r"assert int\((sizes\[7\]|i_mode_setup)\) == ", script), script


@pytest.mark.fortran
@pytest.mark.skipif(
    not sorted((REPO / "validation" / "f2py").glob("glomap_f2py*.so")),
    reason="binding not built; run validation/build_f2py.sh",
)
def test_the_readback_actually_fires_against_the_fortran(tmp_path):
    """The guard above checks that the line is there; this one checks that it
    works. The child is handed a namelist for setup 2 and told it asked for
    setup 6 — exactly what a silently no-op'd substitution looks like from the
    Fortran's side — and must die rather than return a record."""
    nml = tmp_path / "setup.nml"
    nml.write_text(cm.render_namelist(NAMELIST, 2, "default"))
    proc = subprocess.run(
        [sys.executable, "-c", cm._CHILD, str(nml), "6"], capture_output=True, text=True
    )
    assert proc.returncode != 0, proc.stdout
    assert "@@RESULT@@" not in proc.stdout
    assert "wrong setup" in proc.stderr, proc.stderr


CAPTURES = sorted((REPO / "validation").glob("capture_*.py"))


@pytest.mark.parametrize("path", CAPTURES, ids=lambda p: p.name)
def test_no_capture_script_subscripts_a_wrap_call(path):
    """`int(g.wrap_topmode()[0])` drops the trailing `ierr` that every
    neighbouring call asserts is 0 — and `wrap_topmode` returns `out = 0`
    alongside `ierr = 4` when it is uninitialised, so the discarded code is the
    only thing separating "topmode is 5" from "the tables were never built".
    Bind the tuple and assert the last element instead.
    """
    source = path.read_text(encoding="utf-8")
    offenders = re.findall(r"\bg\.wrap_\w+\([^()]*\)\s*\[", source)
    assert not offenders, f"{path.name}: {offenders}"


# ---------------------------------------------------------------------------
# capture_leaf: reproducible abscissae
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(cl.LOG_GRIDS))
def test_every_log_grid_point_is_an_exact_short_decimal(name):
    """`np.logspace` is `10.0 ** linspace(...)`, i.e. one libm `pow` per point,
    and it is not correctly rounded: 4 of the 1801 points of the old `cubrt`
    grid came back 1 ulp off on the capture platform. That makes the *sample
    points* platform-dependent, not just the results. Every point must now be a
    decimal literal, whose conversion IEEE 754 requires to be correctly rounded.
    """
    grid = cl._decade_grid(*cl.LOG_GRIDS[name])
    strays = [v for v in grid if float(f"{v:.3e}") != v]
    assert not strays, f"{name}: {len(strays)} point(s) are not exact short decimals"


@pytest.mark.parametrize("name", list(cl.LOG_GRIDS))
def test_the_swept_grids_are_built_from_those_points(name):
    """The reproducible construction has to be what `grids()` actually uses."""
    built = cl._decade_grid(*cl.LOG_GRIDS[name])
    swept = cl.grids()[name]
    missing = np.setdiff1d(built, np.abs(swept))
    assert missing.size == 0, f"{name}: {missing.size} decade points are not in the sweep"


def test_the_exact_cubes_are_the_64_integer_cubes():
    """The point of this part of the grid is that an honest cube root returns an
    integer, so the inputs must be exactly the integer cubes and nothing near
    them. (That `float(k**3)` replaced `float(k) ** 3`, dropping a libm `pow`,
    is not visible in the values — no libm gets those 64 wrong — so it is a
    source-level change and `test_no_grid_construction_reaches_libm` is what
    holds the line.)"""
    cubes = cl._exact_cubes()
    assert cubes.tolist() == (np.arange(1, 65, dtype=np.int64) ** 3).astype(np.float64).tolist()
    assert (cubes == np.trunc(cubes)).all()


@pytest.mark.parametrize("name", ["cubrt", "log", "oneover"])
def test_the_grid_rewrite_kept_every_sample_count(name):
    """The committed golden was captured with the `logspace` abscissae and must
    be regenerated. What must NOT change with it is the coverage: same number of
    points per grid, each within 0.4% of the point it replaces.
    """
    golden = np.load(GOLDENS / "numerics.f64.leaf.npz", allow_pickle=False)
    grid, old = cl.grids()[name], golden[f"{name}_x"]
    assert len(grid) == len(old)
    assert np.max(np.abs(grid / old - 1.0)) < 4e-3


#: numpy entry points whose float64 path is a libm call, so an abscissa built
#: with one is not reproducible across platforms. `linspace`, `arange`,
#: `nextafter` and `finfo` are exact or correctly-rounded and stay allowed.
LIBM_BACKED = {"logspace", "geomspace", "power", "float_power", "exp", "exp2", "log", "log10"}


def test_no_grid_construction_reaches_libm():
    """Checked on the *code* of the grid builders, through the AST, so that the
    docstrings that discuss `logspace` on purpose do not count as a use."""
    called = set()
    for fn in (cl._decade_grid, cl._exact_cubes, cl._dense_through_zero, cl.grids):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                called.add(node.attr)
            elif isinstance(node, ast.Name):
                called.add(node.id)
    assert not (called & LIBM_BACKED), sorted(called & LIBM_BACKED)


@pytest.mark.parametrize("name", ["erf", "cubrt", "exp", "log", "oneover", "nint", "vapour_round"])
def test_every_grid_is_finite_and_strictly_increasing(name):
    grid = cl.grids()[name]
    assert np.isfinite(grid).all()
    assert (np.diff(grid) > 0).all()


# ---------------------------------------------------------------------------
# capture_leaf: the ereport rule
# ---------------------------------------------------------------------------


def test_check_no_ereport_passes_when_the_counters_did_not_move():
    cl.check_no_ereport((0, 0, 0), (0, 0, 0), "leaf_erf")
    cl.check_no_ereport((2, 1, 0), (2, 1, 0), "leaf_erf")


@pytest.mark.parametrize(
    "after,expected",
    [((1, 0, 0), "fatal"), ((0, 1, 0), "warning"), ((0, 0, 1), "info")],
)
def test_check_no_ereport_refuses_a_sweep_that_reached_the_shim(after, expected):
    """The shim returns where the real `ereport` would `STOP 1`, so the driver
    comes back with a number that means nothing. docs/harness.md makes the check
    unconditional; these leaves cannot reach it today, which is why the check
    has to be in place before one that can is added."""
    with pytest.raises(SystemExit, match=expected):
        cl.check_no_ereport((0, 0, 0), after, "leaf_erf", (7, b"routine", b"message"))


def test_the_capture_checks_the_shim_around_every_driver_call():
    """Every `leaf_*` call in the capture goes through the wrapper that counts.
    A new sweep added without it is the omission this test exists to catch."""
    source = (REPO / "validation" / "capture_leaf.py").read_text(encoding="utf-8")
    body = source.split("def capture(", 1)[1]
    for call in re.findall(r"g\.leaf_\w+|driver\(", body):
        line = next(ln for ln in body.splitlines() if call in ln and "def " not in ln)
        assert "call(" in line, f"{call} is not wrapped by the ereport check: {line.strip()}"
