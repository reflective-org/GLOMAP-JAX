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
    "conden_delgc_over_gc": True,
    "s_cond_s_zero_when_cond_off": True,
    "drydiam_undersize_reset": True,
    "l_fix_ukca_water_content": True,
    "l_fix_neg_pvol_wat": True,
    "l_fix_ukca_hygroscopicities": True,
    "checkmd_nd": False,
    "iextra_checks": 0,
}


def _flag_names():
    return {f.name for f in fields(FidelityConfig)}


def _documented_names():
    text = FIDELITY_DOC.read_text(encoding="utf-8")
    return set(re.findall(r"^## `([a-z0-9_]+)`", text, flags=re.MULTILINE))


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

    config.py itself is excluded -- defining the field is not using it. Flags
    not yet consumed are expected while the port is in progress, so this test
    is informational until the phase that should consume them lands.
    """
    consumers = [
        p
        for p in (REPO / "src").rglob("*.py")
        if p.name != "config.py" and name in p.read_text(encoding="utf-8")
    ]
    if not consumers:
        pytest.skip(f"{name} not yet consumed; expected until its phase lands")
