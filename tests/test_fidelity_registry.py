"""Task 9 acceptance: the fidelity registry cannot drift out of date.

Three properties, each of which has a specific failure mode it prevents:

1. Every flag defaults to reproducing the Fortran. A default set the wrong way
   round silently changes results while every test still passes -- the worst
   possible failure for a port.
2. Every flag has a docs/fidelity.md section, and every section has a flag.
   Orphans in either direction mean the rationale and the code have diverged.
3. Every flag is referenced somewhere in src/. A flag nothing reads is a
   decision that looks recorded and is not.
"""

import re
from dataclasses import fields
from pathlib import Path

import pytest

from glomap_jax.config import FidelityConfig

REPO = Path(__file__).resolve().parents[1]
FIDELITY_DOC = REPO / "docs" / "fidelity.md"

# The Fortran-reproducing value for each flag. Written out literally rather than
# read from the dataclass, so that changing a default requires changing this
# table too -- and thereby noticing.
FORTRAN_BEHAVIOUR = {
    "coag_intra_factor3": True,
    "ageing_totage_rescale_noop": True,
    "s_cond_s_zero_when_cond_off": True,
    "conden_insol_num_eps_by_sol_mode": True,
    "drydiam_undersize_reset": True,
    "l_fix_ukca_water_content": True,
    "l_fix_neg_pvol_wat": True,
    "l_fix_ukca_hygroscopicities": True,
    "checkmd_nd": False,
    "iextra_checks": 0,
    "cbrt_exact": False,
    "l_fix_nacl_density": True,
}


def _flag_names():
    return {f.name for f in fields(FidelityConfig)}


def _documented_names():
    text = FIDELITY_DOC.read_text(encoding="utf-8")
    return set(re.findall(r"^## `([a-z0-9_]+)`", text, flags=re.MULTILINE))


# Flags no ported code reads yet, with the phase that will. This list must
# SHRINK; it is not a suppression mechanism. Every entry is a promise.
NOT_YET_CONSUMED = {
    "coag_intra_factor3",  # phase H, task 66
    "ageing_totage_rescale_noop",  # phase I, task 74
    "s_cond_s_zero_when_cond_off",  # phase F, task 57
    "conden_insol_num_eps_by_sol_mode",  # phase G, task 62
    "drydiam_undersize_reset",  # phase D, task 37
    "l_fix_ukca_water_content",  # phase D, task 40
    "l_fix_neg_pvol_wat",  # phase D, task 38
    "checkmd_nd",  # phase I, task 79
    "iextra_checks",  # phase H, task 71
    "cbrt_exact",  # phase D, task 36 -- numerics.cbrt takes it as an argument
}


def _mentions_in_code(path, name):
    """True if `name` appears as actual code, not in a comment or a docstring.

    Tokenising rather than string-matching, because prose is where flags get
    mentioned most. `core/numerics.py` documents `FidelityConfig.cbrt_exact` at
    length in its module docstring and does not read it — the flag is passed to
    `cbrt(exact=...)` by a caller that does not exist yet. A substring match
    would call that consumed and the test would be measuring documentation.
    """
    import tokenize

    with path.open("rb") as fh:
        try:
            tokens = list(tokenize.tokenize(fh.readline))
        except tokenize.TokenError:  # pragma: no cover - malformed source
            return name in path.read_text(encoding="utf-8")
    return any(
        name in tok.string for tok in tokens if tok.type not in (tokenize.COMMENT, tokenize.STRING)
    )


def test_every_flag_is_in_the_expected_behaviour_table():
    assert _flag_names() == set(FORTRAN_BEHAVIOUR), (
        "FidelityConfig and FORTRAN_BEHAVIOUR disagree; a new flag needs an "
        "explicit statement of what the Fortran does."
    )


@pytest.mark.parametrize("name", sorted(FORTRAN_BEHAVIOUR))
def test_default_reproduces_the_fortran(name):
    default = getattr(FidelityConfig(), name)
    assert default == FORTRAN_BEHAVIOUR[name], (
        f"{name} defaults to {default!r}, which is NOT what the Fortran does "
        f"({FORTRAN_BEHAVIOUR[name]!r}). Order-1 defaults must reproduce the "
        f"reference, however wrong it is."
    )


def test_no_undocumented_flags():
    undocumented = sorted(_flag_names() - _documented_names())
    assert not undocumented, f"flags with no docs/fidelity.md section: {undocumented}"


def test_no_orphan_documentation():
    orphans = sorted(_documented_names() - _flag_names())
    assert not orphans, f"docs/fidelity.md documents non-existent flags: {orphans}"


@pytest.mark.parametrize("name", sorted(FORTRAN_BEHAVIOUR))
def test_every_flag_section_states_its_default(name):
    text = FIDELITY_DOC.read_text(encoding="utf-8")
    section = text.split(f"## `{name}`", 1)[1].split("\n## ", 1)[0]
    assert "**Default" in section, f"{name} section does not state its default"


@pytest.mark.parametrize("name", sorted(FORTRAN_BEHAVIOUR))
def test_every_flag_is_referenced_in_src(name):
    """A flag nothing reads is a decision that looks recorded and is not.

    This test had NO ASSERTION until the phase B review: it built `consumers`,
    skipped if empty, and otherwise fell off the end. Both paths passed, and all
    ten parametrisations skipped, so the property the module docstring
    advertises was never checked at all.

    The honest version: no physics is ported yet, so every flag is legitimately
    unconsumed. That is tracked explicitly in NOT_YET_CONSUMED rather than by a
    skip, so the list has to shrink as each phase lands, and a flag that is
    neither consumed nor declared pending fails.

    The config subpackage is excluded -- defining the field is not using it.
    So are comments: a flag named only in prose is not read by anything.
    """
    consumers = sorted(
        p.name
        for p in (REPO / "src").rglob("*.py")
        if "config" not in p.parts and _mentions_in_code(p, name)
    )
    if name in NOT_YET_CONSUMED:
        assert not consumers, (
            f"{name} is consumed by {consumers} but is still listed in "
            f"NOT_YET_CONSUMED; remove it from the list."
        )
    else:
        assert consumers, (
            f"{name} is read by nothing under src/. Either consume it or add it "
            f"to NOT_YET_CONSUMED with the phase that will."
        )
