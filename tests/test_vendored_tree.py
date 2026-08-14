"""Task 6 acceptance: `fortran/src/ukca/` cannot be edited in place.

The whole validation strategy rests on the vendored Fortran being upstream UKCA
plus exactly the patches in `fortran/patches/`. If someone "just fixes" a science
routine here, every golden silently starts describing a different model and no
test would notice — the JAX port would agree with a Fortran that is no longer
UKCA.

So: a content hash over the read-only subtree, checked in CI. This is a stronger
guarantee than `glomap-box`'s `make verify-vendor`, which needs a UKCA checkout
to compare against; this one is self-contained and runs on a bare runner.

Legitimate changes go through `fortran/patches/` and update `MANIFEST.sha256`
in the same commit, with a rationale in `PROVENANCE.md`.
"""

import hashlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
UKCA_DIR = REPO / "fortran" / "src" / "ukca"
MANIFEST = REPO / "fortran" / "MANIFEST.sha256"

# The vendored UKCA closure: the transitive USE closure of ukca_aero_step plus
# common_mode_setup_interface_mod. A change in this count means a file was added
# or dropped, which is as significant as a file being edited.
EXPECTED_FILE_COUNT = 46


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _actual() -> dict[str, str]:
    return {p.name: _digest(p) for p in sorted(UKCA_DIR.glob("*.F90"))}


def _expected() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def test_manifest_exists():
    assert MANIFEST.is_file(), "fortran/MANIFEST.sha256 is missing"


def test_file_count_is_the_full_closure():
    assert len(_actual()) == EXPECTED_FILE_COUNT


def test_no_file_was_added_or_removed():
    actual, expected = set(_actual()), set(_expected())
    unlisted = sorted(actual - expected)
    assert not unlisted, f"unlisted files in fortran/src/ukca: {unlisted}"
    missing = sorted(expected - actual)
    assert not missing, f"listed but missing from fortran/src/ukca: {missing}"


@pytest.mark.parametrize("name", sorted(_expected()) if MANIFEST.is_file() else [])
def test_file_is_unmodified(name):
    """Parametrised per file so a failure names the culprit, not just 'the tree'."""
    assert _actual()[name] == _expected()[name], (
        f"fortran/src/ukca/{name} was edited in place. This subtree is read-only: "
        f"changes belong in fortran/patches/ with a rationale in PROVENANCE.md. "
        f"If the edit is intended, regenerate MANIFEST.sha256 in the same commit."
    )
