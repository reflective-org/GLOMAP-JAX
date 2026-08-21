"""Tasks 36 and 37 — `ukca_calc_drydiam`, byte-equal against the compiled routine.

One worktree, two concerns, because they cannot be separated honestly: the rows
that prove 36's inverted mask (`nd` straddling `num_eps` with small `md`) are
exactly the rows that fire 37's reset. Split, 36 would have to exclude reset
rows and 37 re-enable them — a test weakened and then re-strengthened, which is
the shape of a test that cannot fail.

Byte equality, not `RTOL_ALGEBRAIC`. That tolerance is 1e-13 and `jnp.cbrt`
differs from `x**(1.0/3.0)` by at most 1.3e-14, so a port using the *forbidden*
cube root would pass it — while producing a different `drydp`, which gates mode
merging.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
F2PY = REPO / "validation" / "f2py"
NAMELIST = REPO / "fortran" / "namelists" / "boundary_layer.nml"

from glomap_jax.physics import drydiam, modes  # noqa: E402

SETUPS = modes.supported_setups()

needs_binding = pytest.mark.skipif(
    not sorted(F2PY.glob("glomap_f2py*.so")),
    reason="binding not built; run validation/build_f2py.sh",
)


def _reference(setup: int, nd, md, mdt) -> dict:
    """One subprocess per setup: `ukca_mode_setup` never deallocates."""
    script = f"""
import json, sys
import numpy as np
sys.path.insert(0, {str(F2PY)!r})
sys.path.insert(0, {str(REPO / "validation")!r})
import glomap_f2py as g
import capture_modes as cm
import tempfile, pathlib
text = cm.render_namelist(pathlib.Path({str(NAMELIST)!r}).read_text(), {setup}, "default")
with tempfile.TemporaryDirectory() as tmp:
    nml = pathlib.Path(tmp) / "s.nml"
    nml.write_text(text)
    assert int(g.wrap_init(str(nml))) == 0
sizes = g.wrap_sizes()
assert int(sizes[7]) == {setup}, "wrong setup: the namelist edit did not take"
before = [int(v) for v in g.wrap_ereport_count()]
drydp, dvol, md_out, mdt_out, ierr = g.leaf_drydiam(
    np.array(json.loads({json.dumps(nd.tolist())!r})),
    np.array(json.loads({json.dumps(md.tolist())!r})),
    np.array(json.loads({json.dumps(mdt.tolist())!r})))
after = [int(v) for v in g.wrap_ereport_count()]
print("@@R@@" + json.dumps({{
    "ierr": int(ierr), "shim": [before, after],
    "drydp": np.asarray(drydp).tolist(), "dvol": np.asarray(dvol).tolist(),
    "md": np.asarray(md_out).tolist(), "mdt": np.asarray(mdt_out).tolist()}}))
"""
    # Arrays cross as JSON, not repr: repr emits a bare `nan`, which is not
    # valid Python in the child, and the mdt-garbage test feeds exactly that.
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    out = json.loads(proc.stdout[proc.stdout.rindex("@@R@@") + 5 :])
    assert out["ierr"] == 0, out["ierr"]
    assert out["shim"][0] == out["shim"][1], "the routine reached ereport; the comparison is void"
    return out


@functools.lru_cache(maxsize=8)
def _grid(setup: int):
    """Inputs chosen to reach every branch, derived from the mode tables.

    The undersize thresholds are computed from `modes.build()` rather than
    transcribed: the plan's quoted `mmid`/`mlo` were the numpy-`pow` spelling
    and differ from the Fortran's `d*d*d` in the last ulp, which is exactly
    enough to miss the edge the row exists to sit on.
    """
    tab = modes.build(setup)
    ncp = tab.ncp
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for imode in range(modes.NMODES):
        eps = tab.num_eps[imode]
        # The mask boundary, including the exact tie -- `>` is strict, so
        # nd == num_eps takes the mmid branch.
        for value in (0.0, np.nextafter(eps, 0.0), eps, np.nextafter(eps, np.inf), 1e-6, 1e3):
            nd = np.zeros((1, modes.NMODES))
            md = np.zeros((1, modes.NMODES, ncp))
            nd[0, imode] = value
            for icp in range(ncp):
                if tab.component[imode, icp]:
                    md[0, imode, icp] = tab.mmid[imode] / max(1, int(tab.component[imode].sum()))
            rows.append((nd, md, md.sum(axis=2)))

        # The undersize reset: modes 1-3 only, and unreachable from any shipped
        # namelist. Solve for the md total that puts drydp on ddplim0*0.1.
        if imode < 3 and tab.mode[imode]:
            sixovrpix = 6.0 / (np.pi * tab.x[imode])
            dvol_thresh = (tab.ddplim0[imode] * 0.1) ** 3 / sixovrpix
            members = [c for c in range(ncp) if tab.component[imode, c]]
            ratio = tab.mm[members[0]] / (6.022e23 * tab.rhocomp[members[0]])
            for scale in (0.9, 1.0, 1.1):
                nd = np.zeros((1, modes.NMODES))
                md = np.zeros((1, modes.NMODES, ncp))
                nd[0, imode] = 1e-6
                md[0, imode, members[0]] = dvol_thresh * scale / ratio
                rows.append((nd, md, md.sum(axis=2)))
            # md all zero: dvol = 0, drydp = 0, reset fires and restores it.
            nd = np.zeros((1, modes.NMODES))
            nd[0, imode] = 1e-6
            md = np.zeros((1, modes.NMODES, ncp))
            rows.append((nd, md, md.sum(axis=2)))

    nd = np.concatenate([r[0] for r in rows])
    md = np.concatenate([r[1] for r in rows])
    mdt = np.concatenate([r[2] for r in rows])
    return tab, nd, md, mdt


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_the_port_is_byte_equal_to_the_compiled_routine(setup):
    tab, nd, md, mdt = _grid(setup)
    want = _reference(setup, nd, md, mdt)
    drydp, dvol, md_out, mdt_out = drydiam.calc_drydiam(tab, nd, md, mdt)
    for name, got in (
        ("drydp", drydp),
        ("dvol", dvol),
        ("md", md_out),
        ("mdt", mdt_out),
    ):
        np.testing.assert_array_equal(np.asarray(got), np.array(want[name]), err_msg=name)


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_the_undersize_reset_actually_fires(setup):
    """Task 37's acceptance. Without this the byte-equality test above passes
    on a port that never implements the reset at all, because no shipped
    namelist reaches it -- 0 of 2160 records in the branch dump."""
    tab, nd, md, mdt = _grid(setup)
    want = _reference(setup, nd, md, mdt)
    fired = ~np.isclose(np.array(want["md"]), md, rtol=0, atol=0)
    active_low = [m for m in range(3) if tab.mode[m]]
    if not active_low:
        pytest.skip(f"setup {setup} has no active mode in 1-3; the reset is dead by mode()")
    assert fired.any(), "no row fired the reset; the fixture no longer reaches task 37's branch"
    # And the port fired it in the same places.
    _, _, md_out, _ = drydiam.calc_drydiam(tab, nd, md, mdt)
    np.testing.assert_array_equal(np.asarray(md_out) != md, fired)


@needs_binding
@pytest.mark.fortran
def test_a_mass_in_a_non_member_component_survives_the_reset():
    """`:250` rewrites `md` only where `component(imode,icp)`. A port that
    rewrote the whole row would zero a mass the Fortran leaves alone."""
    tab = modes.build(2)
    ncp = tab.ncp
    outsider = next(c for c in range(ncp) if not tab.component[0, c])
    nd = np.zeros((1, modes.NMODES))
    md = np.zeros((1, modes.NMODES, ncp))
    nd[0, 0] = 1e-6
    md[0, 0, outsider] = 12345.0

    want = _reference(2, nd, md, md.sum(axis=2))
    assert np.array(want["md"])[0, 0, outsider] == 12345.0, "the Fortran overwrote a non-member"
    _, _, md_out, _ = drydiam.calc_drydiam(tab, nd, md, md.sum(axis=2))
    np.testing.assert_array_equal(np.asarray(md_out), np.array(want["md"]))


@needs_binding
@pytest.mark.fortran
def test_mdt_is_written_and_never_read():
    """`mdt` appears at `:40, :135, :243, :256` only -- an output dressed as an
    in-out. Feeding garbage and requiring every other output to be unchanged is
    what establishes that, and it is cheap."""
    _, nd, md, mdt = _grid(1)
    clean = _reference(1, nd, md, mdt)
    for poison in (np.nan, -1.0, 1e300):
        garbage = np.full_like(mdt, poison)
        dirty = _reference(1, nd, md, garbage)
        for name in ("drydp", "dvol", "md"):
            np.testing.assert_array_equal(
                np.array(dirty[name]), np.array(clean[name]), err_msg=f"{name} moved at {poison}"
            )


def test_six_over_pi_x_is_not_the_other_spelling():
    """2 ulp on two width parameters, and the cube root downstream turns that
    into a different double on more than half of a random volume sweep."""
    x = np.array([1.4, 1.59, 1.8, 2.0]) ** 2
    theirs = 1.0 / (x * (np.pi / 6.0))
    ours = np.asarray(drydiam.six_over_pi_x(x))
    assert not np.array_equal(ours, theirs), "the two spellings agree; re-measure before relying"


def test_the_threshold_is_a_product_not_a_division():
    """`:251` writes `ddplim0(imode)*0.1`. `ddplim0/10.0` is a different double
    for one of setup 1's modes.

    Tested as an expression rather than through an input, because the two
    spellings differ by one ulp in a *threshold* -- there is no `md` that puts
    `drydp` between them, so no fixture row can tell them apart. This is the
    one place in this file where comparing against the source's spelling is the
    only available check, and it has teeth precisely because the alternative
    spelling really is a different number.
    """
    distinguishable = False
    for setup in SETUPS:
        ddplim0 = modes.build(setup).ddplim0
        np.testing.assert_array_equal(
            np.asarray(drydiam.undersize_threshold(ddplim0)), ddplim0 * 0.1
        )
        distinguishable |= bool((ddplim0 * 0.1 != ddplim0 / 10.0).any())
    assert distinguishable, (
        "x*0.1 and x/10.0 now agree on every ddplim0, so this test cannot fail; "
        "re-measure before trusting the spelling"
    )


@needs_binding
@pytest.mark.fortran
@pytest.mark.parametrize("setup", SETUPS)
def test_modes_above_three_are_not_reset_even_when_undersize(setup):
    """The loop bound, tested by inputs rather than by reading it.

    An active mode 4-8 whose `drydp` is well below its own `ddplim0*0.1` must
    come back untouched. Without these rows, widening the loop to all eight
    modes leaves every other test in this file green -- verified.

    `dvol` is kept strictly positive: a mode above three with `dvol = 0` has no
    reset to rescue it and trips the `MINVAL` abort at `:266` instead.
    """
    tab = modes.build(setup)
    high = [m for m in range(3, modes.NMODES) if tab.mode[m]]
    if not high:
        pytest.skip(f"setup {setup} has no active mode above 3")

    ncp = tab.ncp
    nd = np.zeros((len(high), modes.NMODES))
    md = np.zeros((len(high), modes.NMODES, ncp))
    for row, imode in enumerate(high):
        nd[row, imode] = 1e-6
        member = next(c for c in range(ncp) if tab.component[imode, c])
        sixovrpix = 6.0 / (np.pi * tab.x[imode])
        target = (0.5 * tab.ddplim0[imode] * 0.1) ** 3 / sixovrpix
        ratio = tab.mm[member] / (6.022e23 * tab.rhocomp[member])
        md[row, imode, member] = target / ratio
    mdt = md.sum(axis=2)

    want = _reference(setup, nd, md, mdt)
    for row, imode in enumerate(high):
        assert np.array(want["drydp"])[row, imode] < tab.ddplim0[imode] * 0.1, (
            f"mode {imode + 1} is not actually undersize; the row proves nothing"
        )
        assert np.array(want["dvol"])[row, imode] > 0.0
    np.testing.assert_array_equal(np.array(want["md"]), md, err_msg="the Fortran reset a high mode")

    drydp, _, md_out, _ = drydiam.calc_drydiam(tab, nd, md, mdt)
    np.testing.assert_array_equal(np.asarray(md_out), np.array(want["md"]))
    np.testing.assert_array_equal(np.asarray(drydp), np.array(want["drydp"]))


def test_the_reset_spans_modes_one_to_three():
    """Not all eight, and not "the soluble modes" -- mode 4 is soluble and
    excluded. `:245` reads `DO imode = mode_nuc_sol, mode_acc_sol`."""
    assert (drydiam.MODE_NUC_SOL, drydiam.MODE_ACC_SOL) == (1, 3)
    source = (REPO / "fortran" / "src" / "ukca" / "ukca_calc_drydiam.F90").read_text()
    assert "DO imode=mode_nuc_sol,mode_acc_sol" in source.replace(" ", "").replace(
        "DOimode=", "DO imode="
    ).replace("DO imode=", "DO imode=")
