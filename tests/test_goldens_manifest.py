"""Task 17: the drift / orphan gate over `tests/goldens/`.

A golden that changes silently is worse than no golden — the suite keeps passing
and the reference it passes against is no longer the one anybody reviewed. This
gate lands *before* the fixtures do, so most of what it must do is prove it can
fail. Every failure mode below is exercised by mutating a real archive and
asserting the gate names what moved; a manifest test that only ever sees intact
data is a test that would not notice if the hashing were a no-op.

No `fortran` marker anywhere here. The gate has to run in CI, which has no
gfortran, and once the fixtures land (task 19) CI is exactly where drift has to
be caught.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
import goldens_manifest as gm  # noqa: E402


def make_archive(path: Path, **overrides):
    """A miniature stand-in for a captured golden, with the same shape of keys."""
    arrays = {
        "columns": np.array(["time_s", "N_nucsol_cm3"], dtype=np.str_),
        "values": np.array([[0.0, 1.5], [1800.0, 2.5]], dtype=np.float64),
        "_case": np.array("marine_bcoc"),
        "_mode": np.array("trajectory"),
        "_variant": np.array("f64"),
        "_rows": np.array(2, dtype=np.int64),
        "_namelist_sha256": np.array("0" * 64),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)


@pytest.fixture
def goldens(tmp_path):
    d = tmp_path / "goldens"
    d.mkdir()
    return d


def check(goldens):
    return gm.verify(goldens, goldens / "MANIFEST.json")


# --------------------------------------------------------------------------
# The state this gate lands in: nothing captured yet.
# --------------------------------------------------------------------------


def test_passes_with_zero_fixtures_and_no_manifest(goldens):
    """Task 17 precedes task 19 on purpose — the gate must exist before the
    thing it guards, which means an empty directory is a valid passing state."""
    assert check(goldens) == []


def test_passes_with_zero_fixtures_and_an_empty_manifest(goldens):
    gm.write(goldens, goldens / "MANIFEST.json")
    assert check(goldens) == []


def test_the_repository_goldens_are_currently_intact():
    """Runs against the real directory, whatever is in it today."""
    assert gm.verify() == []


# --------------------------------------------------------------------------
# Drift: a listed archive's contents changed.
# --------------------------------------------------------------------------


def test_intact_archives_pass(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    assert check(goldens) == []


def test_regenerating_the_same_data_does_not_report_drift(goldens):
    """np.savez_compressed writes a zip, whose entries carry timestamps, so the
    file bytes differ on every write. Hashing array contents rather than the
    file is what makes the gate usable at all."""
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    (goldens / "a.npz").unlink()
    make_archive(goldens / "a.npz")
    assert check(goldens) == []


def test_a_changed_value_is_caught_and_named(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "a.npz", values=np.array([[0.0, 1.5], [1800.0, 2.5000001]]))
    problems = check(goldens)
    assert problems == ["drift: a.npz[values] values changed"]


def test_a_widened_dtype_is_caught_even_though_the_numbers_agree(goldens):
    """int32 quietly becoming int64 on another platform is a real hazard for the
    long-format dumps, and `allclose` would never see it."""
    make_archive(goldens / "a.npz", step=np.array([1, 2], dtype=np.int32))
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "a.npz", step=np.array([1, 2], dtype=np.int64))
    assert check(goldens) == ["drift: a.npz[step] dtype <i4 -> <i8"]


def test_a_reshaped_array_is_caught(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "a.npz", values=np.array([[0.0, 1.5, 3.0], [1800.0, 2.5, 4.0]]))
    assert check(goldens) == ["drift: a.npz[values] shape [2, 2] -> [2, 3]"]


def test_a_renamed_array_reports_both_the_loss_and_the_gain(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    data = {"vals" if k == "values" else k: v for k, v in np.load(goldens / "a.npz").items()}
    np.savez_compressed(goldens / "a.npz", **data)
    assert set(check(goldens)) == {
        "drift: a.npz gained array 'vals'",
        "drift: a.npz lost array 'values'",
    }


def test_a_golden_regenerated_from_a_different_namelist_says_so(goldens):
    """The most confusing failure to debug without provenance: the numbers are
    all different and nothing says why."""
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "a.npz", _namelist_sha256=np.array("1" * 64))
    problems = check(goldens)
    assert any("regenerated from a different namelist" in p for p in problems)


# --------------------------------------------------------------------------
# Orphan and missing.
# --------------------------------------------------------------------------


def test_an_unlisted_archive_is_an_orphan(goldens):
    """Someone captured a fixture and forgot to update the manifest. Left
    unchecked, a test would load it and nothing would guard its contents."""
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "stray.npz")
    problems = check(goldens)
    assert len(problems) == 1
    assert problems[0].startswith("orphan: stray.npz")
    assert "--write" in problems[0]


def test_a_listed_archive_that_vanished_is_reported(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    (goldens / "a.npz").unlink()
    assert check(goldens) == [f"missing: MANIFEST.json lists a.npz but it is not in {goldens}"]


def test_orphan_and_drift_are_reported_together(goldens):
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    make_archive(goldens / "a.npz", values=np.array([[9.0, 9.0], [9.0, 9.0]]))
    make_archive(goldens / "stray.npz")
    problems = check(goldens)
    assert len(problems) == 2


# --------------------------------------------------------------------------
# What the manifest records.
# --------------------------------------------------------------------------


def test_the_manifest_records_name_dtype_shape_and_content_per_array(goldens):
    make_archive(goldens / "a.npz")
    data = gm.write(goldens, goldens / "MANIFEST.json")
    entry = data["goldens"]["a.npz"]["arrays"]["values"]
    assert entry["dtype"] == "<f8"
    assert entry["shape"] == [2, 2]
    assert len(entry["sha256"]) == 64


def test_the_manifest_records_the_toolchain_that_built_the_reference(goldens):
    """fortran/TOOLCHAIN.txt is a gitignored build product, so without copying it
    the committed goldens would carry no record of what produced them — and they
    are not portable across compilers or platforms."""
    data = gm.write(goldens, goldens / "MANIFEST.json")
    if not gm.TOOLCHAIN.is_file():
        pytest.skip("reference not built, so there is no toolchain record to copy")
    assert "gfortran" in data["toolchain"]
    assert "-ffp-contract=off" in data["toolchain"]["flags"]


def test_the_manifest_is_stable_across_regeneration(goldens):
    """Otherwise every regeneration produces a diff and the gate becomes noise."""
    make_archive(goldens / "a.npz")
    first = json.dumps(gm.write(goldens, goldens / "MANIFEST.json"), sort_keys=True)
    second = json.dumps(gm.write(goldens, goldens / "MANIFEST.json"), sort_keys=True)
    assert first == second


# --------------------------------------------------------------------------
# The CLI, which is what CI and a developer actually run.
# --------------------------------------------------------------------------


def test_cli_exits_nonzero_on_drift_and_zero_when_clean(goldens, capsys):
    make_archive(goldens / "a.npz")
    gm.main(["--write", "--goldens", str(goldens)])
    assert gm.main(["--goldens", str(goldens)]) == 0

    make_archive(goldens / "a.npz", values=np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert gm.main(["--goldens", str(goldens)]) == 1
    assert "values changed" in capsys.readouterr().out
