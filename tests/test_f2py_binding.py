"""Task 20: the in-process binding (validation gate A).

Gate A is the only mechanism here that reaches machine precision. Everything
else — trajectory goldens at `RTOL_TRAJECTORY`, per-substep dumps at
`RTOL_STEP` — goes through a text file and compares a whole call. This calls
the Fortran in-process, so a routine can be driven with chosen inputs and read
back at full double precision.

**Every test runs the binding in a subprocess.** That is not tidiness. UKCA's
mode setup allocates under `IF (.NOT. ALLOCATED)` and never deallocates, and
the 283 `nmas*` budget indices have no initialiser, so a second
`init_ukca_for_box` in one process leaves stale indices — and since `nbudaer`
also changes (8 vs 138) a stale index can be out of bounds. One process per
setup is the only safe arrangement, and running the tests any other way would
be testing a configuration that cannot exist. The helper below is the seed of
task 20b's harness.

`ereport` doing `STOP 1` in-process is the other half of 20b and is not yet
handled. A subprocess turns that from "pytest dies with no traceback" into a
non-zero exit code, which is a large part of why the tests are shaped this way.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"

pytestmark = pytest.mark.fortran


def _extension() -> Path | None:
    return next(iter(sorted(F2PY_DIR.glob("glomap_f2py*.so"))), None)


needs_binding = pytest.mark.skipif(
    _extension() is None, reason="binding not built; run validation/build_f2py.sh"
)


def run_in_subprocess(body: str) -> dict:
    """Execute `body` against a freshly imported binding; return its `result`.

    The snippet gets `g` (the extension) and must assign a JSON-serialisable
    `result`. Anything the Fortran does to process-global state dies with the
    subprocess, which is the entire point.
    """
    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(F2PY_DIR)!r})
        import numpy as np
        import glomap_f2py as g
        NAMELISTS = {str(NAMELISTS)!r}
        result = None
        {textwrap.indent(textwrap.dedent(body), " " * 8).lstrip()}
        print("@@RESULT@@" + json.dumps(result))
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    marker = proc.stdout.rindex("@@RESULT@@") + len("@@RESULT@@")
    return json.loads(proc.stdout[marker:])


@needs_binding
def test_the_extension_exposes_the_expected_entry_points():
    result = run_in_subprocess("result = sorted(n for n in dir(g) if n.startswith('wrap'))")
    assert result == [
        "wrap_get_2d",
        "wrap_get_budgets",
        "wrap_get_md",
        "wrap_get_s0g",
        "wrap_init",
        "wrap_set_2d",
        "wrap_set_md",
        "wrap_sizes",
        "wrap_step",
    ]


@needs_binding
def test_init_then_sizes_reports_the_runtime_module_scalars():
    """The acceptance criterion's first half: `init_ukca_for_box()` runs, and
    `ncp` comes back explicitly rather than being inferred."""
    result = run_in_subprocess("""
        ierr = g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        sizes = g.wrap_sizes()
        result = {'ierr': int(ierr), 'sizes': [int(x) for x in sizes]}
    """)
    assert result["ierr"] == 0
    nbox, nmodes, ncp, nchemg, nadvg, nbudaer, nsteps, setup, ierr = result["sizes"]
    assert ierr == 0
    assert (nbox, nmodes, ncp, setup) == (1, 8, 6, 2)
    assert nsteps == 48, "nsteps comes from the namelist, not a default"
    assert nadvg == nchemg + 2, "nadvg = 2 + nchemg"
    assert nbudaer == 107, "one of the seven distinct nbudaer values (8/46/89/104/107/123/138)"


@needs_binding
def test_sizes_before_init_reports_not_initialised():
    """Returning zeros silently would let a caller allocate empty arrays and
    compare them against a golden of zeros."""
    result = run_in_subprocess("result = [int(x) for x in g.wrap_sizes()]")
    assert result[-1] == 4
    assert result[:-1] == [0] * 8


@needs_binding
def test_one_step_reproduces_the_committed_golden_exactly():
    """The acceptance criterion's second half, and the strongest check in the
    suite so far.

    The binding is built from the plain vendored tree; the goldens came from
    the patched stage with all four instrumentation overlays applied. Agreement
    here is therefore three statements at once: the wrapper's transcription of
    the driver's `ukca_aero_step` call is faithful, the `ES24.16` overlay
    round-trips float64 without loss, and the overlays really are
    instrumentation rather than science.

    Asserted as **bit-identical**, not at a tolerance. It is the same
    arithmetic on the same machine, so anything else is a defect and rounding
    it away would hide exactly what gate A exists to find.
    """
    result = run_in_subprocess("""
        g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        nbox, nmodes, ncp = g.wrap_sizes()[:3]
        g.wrap_step()
        nd, _ = g.wrap_get_2d('nd', nbox, nmodes)
        drydp, _ = g.wrap_get_2d('drydp', nbox, nmodes)
        rhopar, _ = g.wrap_get_2d('rhopar', nbox, nmodes)
        result = {'nd': nd[0].tolist(), 'drydp': drydp[0].tolist(),
                  'rhopar': rhopar[0].tolist()}
    """)

    golden = np.load(REPO / "tests" / "goldens" / "marine_bcoc.f64.trajectory.npz")
    columns = list(golden["columns"])
    row = golden["values"][1]  # after one chemistry step

    modes = ["nucsol", "aitsol", "accsol", "corsol", "aitins"]
    for i, mode in enumerate(modes):
        for field, key, scale in (
            ("N", "nd", 1.0),
            ("Ddry", "drydp", 1e9),  # the CSV reports nm, the state is m
            ("rhop", "rhopar", 1.0),
        ):
            unit = {"N": "cm3", "Ddry": "nm", "rhop": "kgm3"}[field]
            column = f"{field}_{mode}_{unit}"
            assert result[key][i] * scale == row[columns.index(column)], column


@needs_binding
def test_budget_slot_zero_is_reachable_and_empty():
    """`bud_aer_mas` is dimensioned `(nbox, 0:nbudaer)`, so the accessor wants
    `nbudaer + 1` columns. Getting that wrong by one is the easiest possible
    error and would silently drop the last budget."""
    result = run_in_subprocess("""
        g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        nbudaer = g.wrap_sizes()[5]
        g.wrap_step()
        bud, ierr = g.wrap_get_budgets(1, nbudaer + 1)
        off_by_one = g.wrap_get_budgets(1, nbudaer)[1]
        result = {'ierr': int(ierr), 'width': bud.shape[1], 'slot0': bud[0, 0],
                  'nonzero': int((bud != 0).sum()), 'off_by_one': int(off_by_one)}
    """)
    assert result["ierr"] == 0
    assert result["width"] == 108
    assert result["slot0"] == 0.0, "slot 0 is a hole, never written"
    assert result["nonzero"] > 0, "no budget was populated at all"
    assert result["off_by_one"] == 2, "the shape guard did not catch nbudaer vs nbudaer+1"


@needs_binding
def test_state_can_be_driven_from_python():
    """Gate A's whole purpose: compare chosen inputs, not whatever the
    trajectory happened to reach."""
    result = run_in_subprocess("""
        g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        nbox, nmodes = g.wrap_sizes()[:2]
        nd, _ = g.wrap_get_2d('nd', nbox, nmodes)
        nd[0, 1] = 1234.5
        # Setter signature is (field, values): f2py infers the sizes from the
        # array. See the wrapper header on why the asymmetry is left in place.
        set_ierr = g.wrap_set_2d('nd', nd)
        back, _ = g.wrap_get_2d('nd', nbox, nmodes)
        wrong = int(g.wrap_set_2d('nd', np.zeros((nbox, nmodes + 1))))
        result = {'set_ierr': int(set_ierr), 'roundtrip': back[0, 1],
                  'wrong_shape': wrong}
    """)
    assert result["set_ierr"] == 0
    assert result["roundtrip"] == 1234.5
    assert result["wrong_shape"] == 2, (
        "a wrong-width array must be refused by the Fortran-side check. f2py cannot "
        "catch this: it derives n1/n2 from the array, so they always agree with each "
        "other and it knows nothing about nmodes"
    )


@needs_binding
def test_shape_and_field_guards_reject_rather_than_read_out_of_bounds():
    """f2py infers dimensions when it can, and an inferred dimension that
    disagrees with the Fortran's is a silent out-of-bounds read."""
    result = run_in_subprocess("""
        g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        nbox, nmodes, ncp = g.wrap_sizes()[:3]
        result = {
            'wrong_nmodes': int(g.wrap_get_2d('nd', nbox, nmodes + 1)[1]),
            'wrong_nbox':   int(g.wrap_get_2d('nd', nbox + 1, nmodes)[1]),
            'wrong_ncp':    int(g.wrap_get_md(nbox, nmodes, ncp + 1)[1]),
            'unknown':      int(g.wrap_get_2d('not_a_field', nbox, nmodes)[1]),
            'ok':           int(g.wrap_get_2d('nd', nbox, nmodes)[1]),
        }
    """)
    assert result == {
        "wrong_nmodes": 2,
        "wrong_nbox": 2,
        "wrong_ncp": 2,
        "unknown": 3,
        "ok": 0,
    }


@needs_binding
def test_reinit_with_a_different_setup_is_refused_and_poisons_the_process():
    """The once-per-process constraint, made safe.

    `read_box_namelist` has to run before the setup is knowable, so by the time
    `wrap_init` can refuse it has already replaced every config scalar. What is
    left is the new namelist's switches paired with the old mode setup — a
    configuration that never existed. Continuing from there would produce
    plausible numbers, so every entry point refuses afterwards.
    """
    result = run_in_subprocess("""
        first = g.wrap_init(NAMELISTS + '/marine_bcoc.nml')       # setup 2
        second = g.wrap_init(NAMELISTS + '/boundary_layer.nml')   # setup 1
        result = {
            'first': int(first),
            'second': int(second),
            'step_after': int(g.wrap_step()),
            'sizes_after': int(g.wrap_sizes()[-1]),
            'get_after': int(g.wrap_get_2d('nd', 1, 8)[1]),
            'init_again': int(g.wrap_init(NAMELISTS + '/marine_bcoc.nml')),
        }
    """)
    assert result["first"] == 0
    assert result["second"] == 1
    assert result["step_after"] == 1, "a poisoned process must not run a step"
    assert result["sizes_after"] == 1
    assert result["get_after"] == 1
    assert result["init_again"] == 1, "poisoning must not be clearable in-process"


@needs_binding
def test_reinit_with_the_same_setup_is_allowed_and_resets_the_state():
    """Re-running one setup is safe — the mode tables and indices are already
    correct — and is how a per-routine driver will reset between cases."""
    result = run_in_subprocess("""
        g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        nbox, nmodes = g.wrap_sizes()[:2]
        g.wrap_step()
        stepped, _ = g.wrap_get_2d('nd', nbox, nmodes)
        ierr = g.wrap_init(NAMELISTS + '/marine_bcoc.nml')
        reset, _ = g.wrap_get_2d('nd', nbox, nmodes)
        result = {'ierr': int(ierr), 'stepped': stepped[0, 1], 'reset': reset[0, 1]}
    """)
    assert result["ierr"] == 0
    assert result["stepped"] != result["reset"]
    assert result["reset"] == 300.0, "init_state should restore the namelist's nd_init"


@needs_binding
@pytest.mark.parametrize(
    ("namelist", "setup", "nbudaer"), [("boundary_layer", 1, 46), ("marine_bcoc", 2, 107)]
)
def test_each_setup_runs_in_its_own_process(namelist, setup, nbudaer):
    """Proves the one-process-per-setup arrangement actually works, and pins
    two of the seven `nbudaer` values while doing it."""
    result = run_in_subprocess(f"""
        ierr = g.wrap_init(NAMELISTS + '/{namelist}.nml')
        sizes = g.wrap_sizes()
        step = g.wrap_step()
        nd, _ = g.wrap_get_2d('nd', sizes[0], sizes[1])
        result = {{'ierr': int(ierr), 'step': int(step), 'setup': int(sizes[7]),
                   'nbudaer': int(sizes[5]), 'finite': bool(np.isfinite(nd).all())}}
    """)
    assert result == {
        "ierr": 0,
        "step": 0,
        "setup": setup,
        "nbudaer": nbudaer,
        "finite": True,
    }
