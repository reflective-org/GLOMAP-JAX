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
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE_BUILD_MD = REPO / "docs" / "REFERENCE_BUILD.md"
sys.path.insert(0, str(REPO / "validation"))
import capture_reference as cr  # noqa: E402
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


# ADR-007: goldens are committed as plain files, no Git LFS. These are the
# tripwires that re-open that decision, and they are deliberately loose -- they
# exist to catch an order-of-magnitude change (a multi-box capture, say), not to
# police a few hundred kilobytes. The exact figures are deliberately NOT written
# down here: they have gone stale twice, and
# test_the_sizes_this_page_states_are_the_sizes_on_disk re-derives the ones
# docs/REFERENCE_BUILD.md quotes rather than trusting a comment.
MAX_ARCHIVE_BYTES = 5_000_000
MAX_TOTAL_BYTES = 25_000_000


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
    """A regeneration that changed nothing must report nothing, or the gate is
    noise and gets loosened."""
    make_archive(goldens / "a.npz")
    gm.write(goldens, goldens / "MANIFEST.json")
    (goldens / "a.npz").unlink()
    make_archive(goldens / "a.npz")
    assert check(goldens) == []


def test_an_npz_of_the_same_data_is_byte_identical(goldens):
    """The reason the module docstring used to give for hashing arrays rather
    than the file — that zip entries carry timestamps, so the same data written
    twice gives different bytes — is false, and this is what says so.

    numpy stamps every member with the DOS epoch (1980, 1, 1). The design is
    still right for two other reasons (a file hash cannot name *which* array
    moved, and it hashes numpy's container rather than the data), but a stated
    reason that does not survive checking is worse than none: it is the kind of
    thing a later reviewer builds on.
    """
    make_archive(goldens / "a.npz")
    first = (goldens / "a.npz").read_bytes()
    (goldens / "a.npz").unlink()
    make_archive(goldens / "a.npz")
    assert (goldens / "a.npz").read_bytes() == first
    with zipfile.ZipFile(goldens / "a.npz") as z:
        assert {i.date_time for i in z.infolist()} == {(1980, 1, 1, 0, 0, 0)}


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


def test_regenerating_without_a_toolchain_refuses_rather_than_blanking_it(goldens, monkeypatch):
    """The failure this replaces was silent, which is why it needs its own test.

    `TOOLCHAIN.txt` is a gitignored build product. Regenerating the manifest in
    a tree that has not built the reference — capture one new golden from an
    already-built extension, refresh the manifest — used to overwrite a
    populated toolchain block with `{}`. No error, and a diff that reads as a
    routine regeneration. Found while porting `coag_mode`, where only the
    assertion above caught it.
    """
    manifest = goldens / "MANIFEST.json"
    make_archive(goldens / "a.npz")
    monkeypatch.setattr(gm, "TOOLCHAIN", goldens / "present.txt")
    gm.TOOLCHAIN.write_text("gfortran: 14.2.0\nflags: -ffp-contract=off\n", encoding="utf-8")
    assert gm.write(goldens, manifest)["toolchain"]

    monkeypatch.setattr(gm, "TOOLCHAIN", goldens / "absent.txt")
    with pytest.raises(SystemExit, match="provenance"):
        gm.write(goldens, manifest)
    # And the block on disk is still there, not half-written.
    assert json.loads(manifest.read_text())["toolchain"]["gfortran"] == "14.2.0"


def test_regenerating_an_empty_toolchain_over_an_empty_one_is_allowed(goldens, monkeypatch):
    """A fresh clone and CI have no reference build, and must still be able to
    regenerate. The refusal is about losing a record, not about lacking one."""
    monkeypatch.setattr(gm, "TOOLCHAIN", goldens / "absent.txt")
    make_archive(goldens / "a.npz")
    assert gm.write(goldens, goldens / "MANIFEST.json")["toolchain"] == {}


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


# --------------------------------------------------------------------------
# Size budget (ADR-007). Vacuous until the fixtures land at task 19, and live
# from then on.
# --------------------------------------------------------------------------


def test_no_single_golden_exceeds_the_per_file_budget():
    oversized = {
        p.name: p.stat().st_size
        for p in gm.GOLDENS.glob("*.npz")
        if p.stat().st_size > MAX_ARCHIVE_BYTES
    }
    assert not oversized, (
        f"{oversized} exceed {MAX_ARCHIVE_BYTES / 1e6:.0f} MB. ADR-007 says goldens are "
        f"committed as plain files; re-open it rather than raising this number."
    )


def test_the_golden_set_stays_within_the_lfs_free_budget():
    total = sum(p.stat().st_size for p in gm.GOLDENS.glob("*.npz"))
    assert total <= MAX_TOTAL_BYTES, (
        f"goldens total {total / 1e6:.1f} MB, budget {MAX_TOTAL_BYTES / 1e6:.0f} MB. "
        f"See ADR-007 -- the likely cause is a multi-box capture, where every stream "
        f"scales with nbox."
    )


# --------------------------------------------------------------------------
# The CLI spelling two documents have always told people to use.
# --------------------------------------------------------------------------


def test_the_documented_check_flag_exists(goldens, capsys):
    """`--check` is what this module's docstring and docs/REFERENCE_BUILD.md
    both document. Before it existed, the documented command exited 2 with
    "unrecognized arguments: --check" -- a gate nobody could run as written."""
    make_archive(goldens / "a.npz")
    gm.main(["--write", "--goldens", str(goldens)])
    capsys.readouterr()

    assert gm.main(["--check", "--goldens", str(goldens)]) == 0
    assert "intact" in capsys.readouterr().out

    make_archive(goldens / "a.npz", values=np.array([[0.0, 0.0], [0.0, 0.0]]))
    assert gm.main(["--check", "--goldens", str(goldens)]) == 1
    assert "values changed" in capsys.readouterr().out


def test_check_verifies_rather_than_writing(goldens):
    """It must be the *check*, not a second spelling of --write: an orphan is
    still an orphan after it runs."""
    make_archive(goldens / "a.npz")
    gm.main(["--write", "--goldens", str(goldens)])
    make_archive(goldens / "stray.npz")
    before = (goldens / "MANIFEST.json").read_bytes()

    assert gm.main(["--check", "--goldens", str(goldens)]) == 1
    assert (goldens / "MANIFEST.json").read_bytes() == before
    assert check(goldens), "--check blessed the orphan it was supposed to report"


def test_check_and_write_together_are_refused(goldens):
    """`--check --write` has no meaning. Silently preferring one of them is how
    someone blesses a golden while believing they verified it."""
    with pytest.raises(SystemExit) as excinfo:
        gm.main(["--check", "--write", "--goldens", str(goldens)])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# `make goldens` must not auto-bless. docs/harness.md states this as a property
# of the harness ("goldens_manifest.py --write is always an explicit act,
# because auto-blessing would make the drift gate silent the one time it
# matters") and every capture script prints the same advice -- while the
# Makefile target that regenerates every golden used to end in `--write`.
# --------------------------------------------------------------------------


def _recipe(target: str) -> str:
    """The commands `make` would run for a target, prerequisites included."""
    result = subprocess.run(["make", "-n", "-C", str(REPO), target], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_make_goldens_does_not_bless_what_it_regenerates():
    recipe = _recipe("goldens")
    assert "capture_reference.py" in recipe, "the target no longer regenerates anything"
    assert "goldens_manifest.py --write" not in recipe, (
        "`make goldens` re-blesses every golden it just regenerated, which is "
        "exactly the auto-blessing docs/harness.md says the harness does not do."
    )
    assert "goldens_manifest.py --check" in recipe, (
        "`make goldens` must report what moved, or a regeneration says nothing at all"
    )


def test_blessing_is_its_own_target():
    """Splitting the drift report out of `make goldens` only helps if there is
    still one obvious command that records an intended change."""
    assert "goldens_manifest.py --write" in _recipe("goldens-bless")


# --------------------------------------------------------------------------
# The sizes docs/REFERENCE_BUILD.md states, re-derived from the archives.
# --------------------------------------------------------------------------


def _doc_claim(pattern: str) -> tuple[int, re.Match]:
    """Find a stated number and return it with its line number, so a failure
    names the place to edit.

    Searched over the whole text rather than line by line: these are sentences
    in prose, and a claim that happens to wrap across a line break is the same
    claim.
    """
    text = REFERENCE_BUILD_MD.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(
            f"docs/REFERENCE_BUILD.md no longer states anything matching {pattern!r}. "
            f"If the sentence was reworded, reword this pattern with it -- do not "
            f"delete the check."
        )
    return text.count("\n", 0, match.start()) + 1, match


def _box_archives() -> list[Path]:
    """The four cases x four modes, plus the f32 trajectories: the set the page
    calls the golden set. The leaf sweep and the index tables are separate
    fixtures, covered by the ADR-007 budgets above rather than here."""
    stems = {
        f"{case}.{variant}.{mode}"
        for case in cr.discover_cases()
        for mode, (_, variants) in cr.MODES.items()
        for variant in variants
    }
    return sorted(p for p in gm.GOLDENS.glob("*.npz") if p.stem in stems)


def test_the_sizes_this_page_states_are_the_sizes_on_disk():
    """Prose that states a measurement must go red when the measurement moves.

    This paragraph has drifted twice: overlay 0005 regrew the state dumps by
    making their records uniquely keyed, and nobody re-derived the figures.
    Three documents then carried three different values for one measurable
    quantity. A number in prose is only worth having if something checks it.
    """
    archives = _box_archives()
    assert len(archives) == 20, f"expected 20 box archives, found {len(archives)}"

    total = sum(p.stat().st_size for p in archives)
    lineno, match = _doc_claim(r"golden set is ([\d,]+) bytes\*\*, i\.e\. ([\d.]+) MB")
    assert int(match.group(1).replace(",", "")) == total, (
        f"docs/REFERENCE_BUILD.md:{lineno} says the golden set is "
        f"{match.group(1)} bytes; it is {total:,}"
    )
    assert float(match.group(2)) == round(total / 1e6, 2), (
        f"docs/REFERENCE_BUILD.md:{lineno} says {match.group(2)} MB; it is {total / 1e6:.2f} MB"
    )

    rows: dict[int, list[str]] = {}
    for path in archives:
        if not path.name.endswith(".f64.state.npz"):
            continue
        with np.load(path) as data:
            rows.setdefault(int(data["_rows"]), []).append(path.name)
    common = max(rows, key=lambda n: len(rows[n]))
    lineno, match = _doc_claim(r"state dump's \*\*([\d,]+)\*\* rows per case")
    assert int(match.group(1).replace(",", "")) == common, (
        f"docs/REFERENCE_BUILD.md:{lineno} says {match.group(1)} state rows per "
        f"case; the measured counts are {dict(sorted(rows.items()))}"
    )
    lineno, match = _doc_claim(r"\(\*\*([\d,]+)\*\* for `bl_nmts3`")
    odd = sorted(n for n in rows if n != common)
    assert [int(match.group(1).replace(",", ""))] == odd, (
        f"docs/REFERENCE_BUILD.md:{lineno} says bl_nmts3 has {match.group(1)} "
        f"state rows; the counts that differ from the common one are {odd}"
    )

    state_mb = sorted(p.stat().st_size / 1e6 for p in archives if p.name.endswith(".f64.state.npz"))
    lineno, match = _doc_claim(r"compress to\s+([\d.]+)\u2013([\d.]+) MB")
    assert (float(match.group(1)), float(match.group(2))) == (
        round(state_mb[0], 2),
        round(state_mb[-1], 2),
    ), (
        f"docs/REFERENCE_BUILD.md:{lineno} says the state archives are "
        f"{match.group(1)}-{match.group(2)} MB; they are "
        f"{state_mb[0]:.2f}-{state_mb[-1]:.2f} MB"
    )
