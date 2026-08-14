"""Tasks 11-13: the Fortran reference builds in both precisions, and the
single-vs-double precision floor is measured rather than assumed.

Everything here is marked ``fortran`` and skips without a toolchain, because CI
has no gfortran. The pure-Python port must stay verifiable on a bare runner.

The measured floor is the headline result. My plan asserted that a float64 JAX
port compared against the shipped float32 Fortran would disagree at "~1e-6 for
reasons that are not bugs". The measurement says **3.7e-4** over a 48-step run —
roughly 370x larger. Two consequences:

* ``ref-f32`` is useless as a validation target for a float64 port. The floor
  sits four orders of magnitude above any tolerance worth gating on, so
  ``ref-f64`` is the only meaningful reference. ``ref-f32`` survives purely as
  the number that explains why.
* It independently confirms that gating a 24-hour trajectory at 1e-9 was never
  achievable, which is why that run is now a soak at ``RTOL_SOAK``.

Note the CSV output resolves ~1e-7 relative (``ES14.6``, 7 significant digits),
so a 3.7e-4 signal is real and not a truncation artefact. That headroom is
adequate for *this* measurement and nowhere near adequate for validating the
port, which is why task 11b adds a high-precision dump.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORTRAN = REPO / "fortran"
BUILD_SCRIPT = REPO / "validation" / "build_reference.sh"

pytestmark = pytest.mark.fortran

HAVE_GFORTRAN = shutil.which("gfortran") is not None
needs_gfortran = pytest.mark.skipif(not HAVE_GFORTRAN, reason="gfortran not available")

# Measured on the pinned toolchain; see fortran/TOOLCHAIN.txt. Asserted as a
# band, not a point: the exact value is platform-dependent, but it must stay far
# above any port tolerance (or ref-f32 would be worth gating on, and it is not)
# and far below O(1) (or single precision would be diverging, not just noisy).
PRECISION_FLOOR_MIN = 1.0e-6
PRECISION_FLOOR_MAX = 1.0e-2


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not HAVE_GFORTRAN:
        pytest.skip("gfortran not available")
    subprocess.run([str(BUILD_SCRIPT), "both"], check=True, capture_output=True)
    return FORTRAN


def _run(exe: Path, out_csv: Path, template: Path) -> None:
    nml = out_csv.with_suffix(".nml")
    text = template.read_text(encoding="utf-8")
    # Redirect output without editing the vendored namelist.
    nml.write_text(text.replace("out/boundary_layer.csv", str(out_csv)), encoding="utf-8")
    subprocess.run([str(exe), str(nml)], check=True, capture_output=True)


def _load(path: Path):
    rows = list(csv.reader(path.open()))
    return rows[0], [[float(x) for x in r] for r in rows[1:]]


def _max_rel_diff(a_rows, b_rows, floor=1e-30):
    worst = 0.0
    for ra, rb in zip(a_rows, b_rows):
        for x, y in zip(ra, rb):
            if abs(y) > floor:
                worst = max(worst, abs(x - y) / abs(y))
    return worst


@needs_gfortran
@pytest.mark.parametrize("variant", ["f32", "f64"])
def test_variant_builds_and_runs(built, tmp_path, variant):
    exe = built / f"bin-ref-{variant}" / "glomap_box"
    assert exe.is_file(), f"ref-{variant} produced no executable"
    out = tmp_path / f"bl_{variant}.csv"
    _run(exe, out, built / "namelists" / "boundary_layer.nml")
    header, rows = _load(out)
    assert len(rows) == 49, "expected 48 steps plus the initial state"
    assert all(v == v for r in rows for v in r), "non-finite value in reference output"


@needs_gfortran
def test_building_does_not_modify_the_vendored_tree(built):
    """The build script asserts this itself; asserted again here so the property
    is covered by the suite rather than only by the script that could change."""
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", "--", "fortran/"],
        capture_output=True,
    )
    assert diff.returncode == 0, "building the reference modified fortran/"


@needs_gfortran
def test_f64_actually_differs_from_f32(built, tmp_path):
    """If -fdefault-real-8 were silently ineffective, the two would be identical
    and we would have a double-precision reference in name only."""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    tmpl = built / "namelists" / "boundary_layer.nml"
    _run(built / "bin-ref-f32" / "glomap_box", a, tmpl)
    _run(built / "bin-ref-f64" / "glomap_box", b, tmpl)
    _, rows_a = _load(a)
    _, rows_b = _load(b)
    assert _max_rel_diff(rows_a, rows_b) > 0.0, "-fdefault-real-8 had no effect"


@needs_gfortran
def test_precision_floor_is_in_the_expected_band(built, tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    tmpl = built / "namelists" / "boundary_layer.nml"
    _run(built / "bin-ref-f32" / "glomap_box", a, tmpl)
    _run(built / "bin-ref-f64" / "glomap_box", b, tmpl)
    _, rows_a = _load(a)
    _, rows_b = _load(b)
    floor = _max_rel_diff(rows_a, rows_b)
    assert PRECISION_FLOOR_MIN < floor < PRECISION_FLOOR_MAX, (
        f"single-vs-double floor is {floor:.3e}, outside the expected band. "
        f"Below {PRECISION_FLOOR_MIN:.0e} would mean -fdefault-real-8 is barely "
        f"doing anything; above {PRECISION_FLOOR_MAX:.0e} would mean single "
        f"precision is diverging rather than merely noisy."
    )


@needs_gfortran
def test_toolchain_is_recorded(built):
    """Goldens are meaningless without the toolchain that produced them."""
    text = (built / "TOOLCHAIN.txt").read_text(encoding="utf-8")
    assert "gfortran:" in text
    assert "-ffp-contract=off" in text
