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

The shipped CSV writer emits ``ES14.6`` — 7 significant digits, resolving ~1e-7
relative. Adequate for measuring a 3.7e-4 floor, and four to seven orders of
magnitude short of the tolerances the port is gated at (``RTOL_STEP = 1e-11``,
``RTOL_ALGEBRAIC = 1e-13``). Task 11b raises it to ``ES24.16`` via a build-time
overlay, so the reference can actually carry the precision it computes in.
"""

import csv
import math
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
    _, rows = _load(out)
    assert len(rows) == 49, "expected 48 steps plus the initial state"
    # `v == v` rejects NaN but passes Infinity, which gfortran writes as
    # `Infinity` and Python parses to `inf`. A reference that blew up would have
    # sailed through.
    assert all(math.isfinite(v) for r in rows for v in r), "non-finite value in reference output"


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


@needs_gfortran
def test_reference_output_carries_full_double_precision(built, tmp_path):
    """Task 11b: the overlay must deliver ~17 significant digits, not 7.

    Without this, a -fdefault-real-8 reference is truncated at output to the
    same 7 digits as the single-precision one, and Gate C cannot be met at any
    tolerance the policy actually gates on.

    Asserted on an evolved field rather than the initial state: a value like
    1.0000000000000000E-08 is wide but carries no information, so field width
    alone would not distinguish a real precision gain from padding.
    """
    out = tmp_path / "hp.csv"
    _run(built / "bin-ref-f64" / "glomap_box", out, built / "namelists" / "boundary_layer.nml")
    rows = list(csv.reader(out.open()))
    header = [h.strip() for h in rows[0]]
    col = next(i for i, h in enumerate(header) if h.startswith("Ddry_aitsol"))

    last = rows[-1][col].strip()
    mantissa = last.split("E")[0].replace("-", "").replace(".", "")
    significant = len(mantissa.rstrip("0"))
    assert significant >= 15, (
        f"reference output carries only {significant} significant digits "
        f"({last!r}); the high-precision overlay is not applied. ES14.6 gives 7."
    )


@needs_gfortran
def test_high_precision_value_round_trips_through_python(built, tmp_path):
    """A written value must survive parse -> repr at full double precision."""
    out = tmp_path / "rt.csv"
    _run(built / "bin-ref-f64" / "glomap_box", out, built / "namelists" / "boundary_layer.nml")
    rows = list(csv.reader(out.open()))
    header = [h.strip() for h in rows[0]]
    col = next(i for i, h in enumerate(header) if h.startswith("Ddry_aitsol"))
    text = rows[-1][col].strip()
    value = float(text)
    # The phase B review found this test asserted `float(f"{value:.16E}") ==
    # value` and `abs(value - float(text)) == 0.0` -- 17 significant digits
    # round-trip EVERY double, and the second compares a value to itself.
    # Reverting the overlay to ES14.6 did not fail it.
    #
    # The real property: the file's own digits must survive parse -> re-render
    # at the file's own format. That fails immediately if the reference emits
    # fewer digits than a double needs.
    rendered = f"{value:24.16E}".strip()
    assert rendered == text, (
        f"the reference wrote {text!r}, which does not re-render to itself at "
        f"ES24.16 ({rendered!r}) -- the write lost precision"
    )
    assert abs(value - float(text)) == 0.0
