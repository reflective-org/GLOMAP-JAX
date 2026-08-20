"""The generated literal tables must match the vendored Fortran.

`src/glomap_jax/physics/_mode_literals.py` is machine-extracted, and this
re-runs the extraction and compares. Two things it prevents: the file drifting
if the vendored tree is ever updated, and someone hand-editing a number in it.

Seven setups times ten tables is several hundred numbers. `core/constants.py`
follows the same convention for the same reason — a mistyped digit produces
tables that look plausible and a model that is quietly wrong.

No `fortran` marker: reads source text, needs no toolchain.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
import extract_mode_literals as extractor  # noqa: E402

from glomap_jax.physics._mode_literals import SETUP_LITERALS  # noqa: E402


@pytest.fixture(scope="module")
def freshly_extracted():
    return extractor.extract()


def test_the_committed_file_is_not_stale(freshly_extracted):
    assert SETUP_LITERALS == freshly_extracted, (
        "src/glomap_jax/physics/_mode_literals.py disagrees with the Fortran. "
        "Regenerate: python validation/extract_mode_literals.py"
    )


def test_the_check_mode_agrees(capsys):
    """`--check` is what a future CI job would run. It compares the DATA, not
    the bytes: `ruff format` reformats the generated file after it is written,
    so a byte comparison would report every formatted file as stale."""
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize("setup", sorted(extractor.ROUTINES))
def test_each_setup_extracted_every_table(freshly_extracted, setup):
    rec = freshly_extracted[setup]
    for field in extractor.VECTORS:
        assert len(rec[field]) == 8, f"{field} is per-mode and must be full width"
    for field in extractor.CP_VECTORS:
        assert len(rec[field]) == rec["ncp"]
    for field in extractor.MATRICES:
        assert len(rec[field]) == 8
        assert all(len(row) == rec["ncp"] for row in rec[field])
    assert set(rec["no_ions"]) == {"both", "hygro_only", "default"}
    assert len(rec["component_names"]) == rec["ncp"]


def test_continuations_are_joined_before_parsing(freshly_extracted):
    """`ddplim0` and `ddplim1` span two source lines. A line-based parser
    silently truncates them to whatever fitted on the first, which would give
    a short array rather than an error."""
    for setup, rec in freshly_extracted.items():
        assert len(rec["ddplim0"]) == 8, f"setup {setup}"
        assert len(rec["ddplim1"]) == 8, f"setup {setup}"


def test_unsupported_setups_are_not_extracted(freshly_extracted):
    """10-13 are dispatched by UKCA but rejected by the box model, so they have
    no reference to validate against."""
    assert set(freshly_extracted) & {10, 11, 12, 13} == set()
