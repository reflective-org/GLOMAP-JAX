"""Task 4 acceptance: licensing and attribution cannot silently regress.

This repository is public and vendors Crown Copyright code, so BSD-3 clause 1
(retain the notice) and clause 3 (no endorsement) are obligations, not
formalities. Asserting them in the suite means a careless edit fails CI rather
than quietly shipping.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# BSD-3 clause 3. The exact string is asserted because paraphrasing it away is
# precisely the regression this guards against.
DISCLAIMER = "Not affiliated with, endorsed by, or an official product of"


def _read(name: str) -> str:
    path = REPO / name
    assert path.is_file(), f"{name} is missing"
    return path.read_text(encoding="utf-8")


def test_root_licence_is_apache_2():
    """Fails if the outbound licence is reverted to BSD-3 or truncated."""
    text = _read("LICENCE")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    # The last section of the terms -- gone if the text is truncated.
    assert "9. Accepting Warranty or Additional Liability" in text


def test_fortran_licence_is_bsd_3_clause_with_crown_copyright():
    """Fails if a re-vendor drops the BSD notice that clause 1 obliges us to keep."""
    text = _read("fortran/LICENCE")
    assert "BSD 3-Clause" in text
    assert "Crown Copyright (c) Met Office" in text
    # Clause 3 must be present verbatim -- it is what forbids implying endorsement.
    assert "Neither the name of the copyright holder" in text


def test_readme_carries_the_no_endorsement_disclaimer():
    assert DISCLAIMER in _read("README.md")


def test_notice_carries_bsd_attribution_verbatim():
    """Fails if NOTICE is deleted, or its BSD reproduction drifts from
    fortran/LICENCE -- clause 2 requires the notice, conditions and
    disclaimer verbatim, not a paraphrase."""
    text = _read("NOTICE")
    assert "Copyright (c) 2026 Reflective" in text
    assert DISCLAIMER in text
    # The whole BSD licence must appear, byte-for-byte as vendored.
    assert _read("fortran/LICENCE").strip() in text


def test_copyright_file_separates_vendored_from_new_code():
    text = _read("COPYRIGHT.md")
    assert "Crown Copyright (c) Met Office" in text
    assert "University of Leeds" in text
    assert "Reflective" in text
    assert DISCLAIMER in text


def test_provenance_records_the_upstream_commit():
    text = _read("PROVENANCE.md")
    # The full hash, not a prefix: goldens are only meaningful against a known
    # upstream, and an abbreviated hash is ambiguous across forks.
    assert "387c5bb0f1166e67f029930ba624bf159bc68627" in text
    assert "MetOffice/ukca" in text


@pytest.mark.parametrize(
    "phrase",
    [
        "official UKCA",
        "endorsed by the Met Office",
        "Met Office product",
    ],
)
def test_no_endorsement_framing_anywhere_in_docs(phrase):
    """Clause 3 forbids implying endorsement, so scan the prose for it.

    COPYRIGHT.md is excluded: it quotes these phrases in order to prohibit them.
    """
    for path in [REPO / "README.md", *(REPO / "docs").glob("*.md")]:
        if not path.is_file():
            continue
        assert phrase.lower() not in path.read_text(encoding="utf-8").lower(), (
            f"{path.name} contains endorsement framing: {phrase!r}"
        )
