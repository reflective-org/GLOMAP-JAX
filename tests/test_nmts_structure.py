"""Task 12b: the nested nmts/nzts structure is actually exercised.

Every shipped namelist -- all three production cases and all four
process-isolation cases -- uses ``nmts = 1``. So the outer loop body runs
exactly once per chemistry step in every existing fixture, and the entire
outer/inner structure the port must reproduce is **unvalidated** by them:

    DO imts = 1, nmts
        s0g -> gc, snapshot gcold
        calc_coag_kernel            <- kernels frozen for all nzts
        DO izts = 1, nzts
            conden -> calcnucrate -> coagwithnucl -> ageing
        drydiam -> volume_mode -> REMODE -> drydiam -> volume_mode
        gc -> s0g, one lumped delta + clamp

The comparison is chosen so the substep size is identical and only the
structure differs: ``nmts=1, nzts=15`` and ``nmts=3, nzts=5`` both give
``dtz = dt_chem / 15``. If the outer loop were a no-op the two would agree, and
this golden would be worthless.

They differ by 5.8e-3, dominated by H2SO4 -- which is exactly right, because the
gas-phase reconciliation and its clamp happen once per ``nmts``, not once per
``nzts``. So a port that flattened the nesting, or recomputed kernels in the
wrong loop, would fail conspicuously rather than subtly.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORTRAN = REPO / "fortran"

pytestmark = pytest.mark.fortran
needs_gfortran = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="gfortran not available"
)

# Measured on the pinned toolchain. Asserted as a lower bound: the point is that
# the structure is observable, and a regression that made it *less* observable
# would mean the nesting had been flattened somewhere.
MIN_STRUCTURAL_DIFFERENCE = 1.0e-4


def _run(exe: Path, template: Path, out_csv: Path, marker: str) -> None:
    nml = out_csv.with_suffix(".nml")
    nml.write_text(
        template.read_text(encoding="utf-8").replace(marker, str(out_csv)), encoding="utf-8"
    )
    subprocess.run([str(exe), str(nml)], check=True, capture_output=True)


def _load(path: Path):
    rows = list(csv.reader(path.open()))
    return [h.strip() for h in rows[0]], [[float(x) for x in r] for r in rows[1:]]


@needs_gfortran
def test_nmts_outer_loop_is_observable(tmp_path):
    exe = FORTRAN / "bin-ref-f64" / "glomap_box"
    if not exe.is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")

    a, b = tmp_path / "n1.csv", tmp_path / "n3.csv"
    _run(exe, FORTRAN / "namelists" / "boundary_layer.nml", a, "out/boundary_layer.csv")
    _run(exe, REPO / "validation" / "namelists" / "bl_nmts3.nml", b, "out/bl_nmts3.csv")

    header, rows_a = _load(a)
    _, rows_b = _load(b)
    assert len(rows_a) == len(rows_b)

    worst, where = 0.0, None
    for ra, rb in zip(rows_a, rows_b):
        for i, (x, y) in enumerate(zip(ra, rb)):
            if abs(y) > 1e-30:
                e = abs(x - y) / abs(y)
                if e > worst:
                    worst, where = e, header[i]

    assert worst > MIN_STRUCTURAL_DIFFERENCE, (
        f"nmts=1/nzts=15 and nmts=3/nzts=5 agree to {worst:.3e} despite having "
        f"the same dtz. Either the outer loop has become a no-op, or this "
        f"golden no longer validates the nested structure."
    )
    # The gas reconciliation runs once per nmts, so it should dominate.
    assert where is not None and "H2SO4" in where, (
        f"largest structural difference is in {where!r}, not H2SO4. Expected the "
        f"once-per-nmts gc->s0g reconciliation to dominate; investigate before "
        f"updating this assertion."
    )


@needs_gfortran
def test_nmts3_case_runs_and_is_finite(tmp_path):
    exe = FORTRAN / "bin-ref-f64" / "glomap_box"
    if not exe.is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")
    out = tmp_path / "n3.csv"
    _run(exe, REPO / "validation" / "namelists" / "bl_nmts3.nml", out, "out/bl_nmts3.csv")
    _, rows = _load(out)
    assert len(rows) == 49
    assert all(v == v for r in rows for v in r)
