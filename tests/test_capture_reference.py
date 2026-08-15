"""Task 16: `validation/capture_reference.py` and its `--mode` dispatch.

The dry-run tests carry no `fortran` marker on purpose. Whether the capture
matrix has the right *shape* is a design question, not a Fortran question, and
it is exactly the kind of thing that silently drifts once fixtures exist. CI has
no gfortran, so a matrix test that skipped there would never run at all.

The capture tests do need a reference binary and are marked accordingly.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "validation" / "capture_reference.py"

sys.path.insert(0, str(REPO / "validation"))
import capture_reference as cr  # noqa: E402

_needs_gfortran = pytest.mark.skipif(
    shutil.which("gfortran") is None, reason="gfortran not available"
)


def needs_reference(fn):
    """Requires a built reference. Also tagged `fortran` so `-m fortran` selects
    it, matching the other harness tests -- the dry-run tests above deliberately
    carry neither, so they still run in CI."""
    return pytest.mark.fortran(_needs_gfortran(fn))


CASES = sorted(cr.discover_cases())


def test_every_shipped_and_added_namelist_is_a_case():
    """A namelist that exists but is never captured is a silent coverage hole."""
    assert set(CASES) >= {"boundary_layer", "free_troposphere", "marine_bcoc", "bl_nmts3"}


def test_dry_run_prints_the_matrix_and_writes_nothing(tmp_path, capsys):
    rc = cr.main(["--dry-run", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert not list(tmp_path.iterdir())
    for case in CASES:
        for mode in cr.MODES:
            assert f"{case}.f64.{mode}.npz" in out or mode == "trajectory"
    assert f"{len(CASES) * 5} capture(s)" in out


def test_dry_run_runs_as_a_script_without_a_reference_build(tmp_path):
    """It has to be usable to inspect the plan before anything is built."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "case" in result.stdout and "variant" in result.stdout


def test_the_matrix_is_not_a_cross_product():
    """ref-f32 exists to measure the precision floor, which is a property of the
    trajectory. Capturing per-substep dumps in single precision would produce
    large fixtures no test can use, so only the trajectory is captured twice."""
    jobs = cr.build_matrix(CASES, list(cr.MODES), ["f32", "f64"], None)
    variants = {}
    for job in jobs:
        variants.setdefault(job.mode, set()).add(job.variant)
    assert variants == {
        "trajectory": {"f32", "f64"},
        "budgets": {"f64"},
        "state": {"f64"},
        "branches": {"f64"},
    }


def test_selecting_a_mode_narrows_the_matrix():
    jobs = cr.build_matrix(["marine_bcoc"], ["branches"], ["f32", "f64"], None)
    assert [j.stem for j in jobs] == ["marine_bcoc.f64.branches"]


def test_namelist_rewrite_silences_the_streams_not_being_captured(tmp_path):
    """Capturing branches must not also write a 300k-row state dump. Only the
    trajectory is exempt: the driver always opens output_file."""
    source = cr.discover_cases()["marine_bcoc"]
    text = cr._rewrite_namelist(source, "branch_file", tmp_path / "s.csv", steps=3)
    assert f"branch_file = '{tmp_path / 's.csv'}'" in text
    assert "state_file = ''" in text
    assert "budget_file = ''" in text
    assert "output_file = ''" not in text
    assert "nsteps       = 3" in text


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    if not (REPO / "fortran" / "bin-ref-f64" / "glomap_box").is_file():
        pytest.skip("reference not built; run validation/build_reference.sh")
    out = tmp_path_factory.mktemp("cap")
    cr.main(["--case", "marine_bcoc", "--steps", "2", "--out", str(out)])
    return out


@needs_reference
def test_smoke_capture_writes_one_archive_per_job(smoke):
    written = sorted(p.name for p in smoke.glob("*.npz"))
    assert written == [
        "marine_bcoc.f32.trajectory.npz",
        "marine_bcoc.f64.branches.npz",
        "marine_bcoc.f64.budgets.npz",
        "marine_bcoc.f64.state.npz",
        "marine_bcoc.f64.trajectory.npz",
    ]


@needs_reference
@pytest.mark.parametrize("mode", ["trajectory", "budgets"])
def test_wide_streams_round_trip_as_a_named_table(smoke, mode):
    data = np.load(smoke / f"marine_bcoc.f64.{mode}.npz")
    assert data["values"].dtype == np.float64
    assert data["values"].shape[1] == len(data["columns"])
    assert np.isfinite(data["values"]).all()


def test_the_codebook_reconstructs_the_original_labels():
    """The string columns are factorised to keep the archives small. That is only
    safe if the levels actually reconstruct the original labels.

    The phase B review found the previous version of this test never touched the
    original labels: it asserted `len(levels[codes]) == rows`,
    `set(levels) == set(levels[codes])` and `codes.max() < len(levels)`, all
    true of ANY factorisation. Reversing the level table so every code decoded
    to the wrong label left all fifteen tests in this file passing.

    So decode and compare against the input, on data with the properties that
    break naive factorisation: repeats, a singleton, and a label that first
    appears last."""
    header = ["step", "imts", "izts", "site", "i1", "i2", "tag", "ibox", "value"]
    labels = ["conden", "conden", "remode", "conden", "ageing", "remode", "zzz"]
    rows = [["1", "1", "1", s, "0", "0", f"t{i}", "1", "0"] for i, s in enumerate(labels)]

    packed = cr._pack_long("branches", header, rows)
    decoded = list(packed["site_levels"][packed["site"]])
    assert decoded == labels, f"codebook does not round-trip: {decoded} != {labels}"

    # First-appearance order, not sorted -- so a diff between two archives is
    # readable rather than reshuffled.
    assert list(packed["site_levels"]) == ["conden", "remode", "ageing", "zzz"]


@needs_reference
def test_branch_values_are_stored_as_int8(smoke):
    """Masks and branch codes are 0/1 and 0-7. Storing them as float64 would cost
    eight times as much for no information."""
    data = np.load(smoke / "marine_bcoc.f64.branches.npz")
    assert data["value"].dtype == np.int8
    assert data["value"].min() >= 0


@needs_reference
def test_every_archive_carries_its_provenance(smoke):
    """A golden without the namelist that produced it cannot be regenerated, and
    task 17's drift gate has nothing to compare against."""
    for path in smoke.glob("*.npz"):
        data = np.load(path)
        assert str(data["_case"]) == "marine_bcoc"
        assert str(data["_variant"]) in ("f32", "f64")
        assert len(str(data["_namelist_sha256"])) == 64
        assert int(data["_rows"]) > 0


@needs_reference
def test_capture_matches_what_the_reference_actually_wrote(smoke, tmp_path):
    """End-to-end: run the binary by hand and compare, so a packing bug cannot
    hide behind the packer's own round-trip."""
    exe = REPO / "fortran" / "bin-ref-f64" / "glomap_box"
    source = cr.discover_cases()["marine_bcoc"]
    target = tmp_path / "stream.csv"
    nml = tmp_path / "m.nml"
    nml.write_text(cr._rewrite_namelist(source, "output_file", target, 2), encoding="utf-8")
    subprocess.run([str(exe), str(nml)], cwd=tmp_path, check=True, capture_output=True)

    header, rows = cr._read_csv(target)
    data = np.load(smoke / "marine_bcoc.f64.trajectory.npz")
    assert list(data["columns"]) == header
    expected = np.array([[float(c) for c in r] for r in rows])
    np.testing.assert_array_equal(data["values"], expected)


@needs_reference
def test_a_missing_reference_build_fails_loudly(tmp_path):
    """Silently producing no fixture would look like success."""
    job = cr.Job("marine_bcoc", "trajectory", "f99", None)
    with pytest.raises(SystemExit, match=re.escape("build_reference.sh")):
        cr.capture(job, tmp_path)
