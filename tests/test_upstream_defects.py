"""Task 22: every upstream defect has a disposition, and the disposition is real.

`docs/UPSTREAM_DEFECTS.md` records ten defects found in `MetOffice/ukca` at
`387c5bb`. Recording them is not the hard part — keeping the record honest as
the port grows is. A defect that says "reproduced behind a fidelity flag" and
has no such flag, or a flag that cites a defect that was renumbered, reads as
diligence and is fiction.

So each defect declares a **disposition** and this file enforces it. Five kinds,
because "add a fidelity flag" is the right answer for only four of the ten:

    fidelity-flag: X   the port must CHOOSE to reproduce it, so it is a flag
    invariant-test     unreachable; nothing to choose, so assert that instead
    not-implemented    no correct reference exists, so the port refuses
    harness-patch: F   the reference itself is unusable without a fix
    documentation-only a comment disagrees with the code; the code is right

The distinction is not bookkeeping. A flag whose two settings are
bit-identical can never have the both-settings test `docs/fidelity.md`
requires of every flag, so it would sit in `FidelityConfig` forever as an
untestable decision. That is exactly what happened to UP-4, and removing it is
what this file's arrival is for.

No `fortran` marker: the branch-dump goldens are committed, so the UP-4
invariant runs in CI.
"""

import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from glomap_jax.config import FidelityConfig

REPO = Path(__file__).resolve().parents[1]
DEFECTS = REPO / "docs" / "UPSTREAM_DEFECTS.md"
FIDELITY = REPO / "docs" / "fidelity.md"
UNSUPPORTED = REPO / "docs" / "unsupported.md"
PATCHES = REPO / "fortran" / "patches"
GOLDENS = REPO / "tests" / "goldens"

ROW = re.compile(r"^\|\s*(UP-\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*`(.+?)`\s*\|$", re.MULTILINE)


def dispositions() -> dict[str, str]:
    rows = ROW.findall(DEFECTS.read_text(encoding="utf-8"))
    assert rows, "the disposition table could not be parsed; has its format changed?"
    return {defect: disposition for defect, _, _, disposition in rows}


def locations() -> dict[str, str]:
    rows = ROW.findall(DEFECTS.read_text(encoding="utf-8"))
    return {defect: location for defect, location, _, _ in rows}


DISPOSITIONS = dispositions()
DEFECT_IDS = sorted(DISPOSITIONS, key=lambda d: int(d.split("-")[1]))


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_the_defect_ids_are_contiguous_from_one():
    """A gap means a defect was deleted rather than superseded, and a
    superseded one still needs its number kept so the write-ups Ali files
    upstream stay referenceable."""
    numbers = sorted(int(d.split("-")[1]) for d in DEFECT_IDS)
    assert numbers == list(range(1, len(numbers) + 1)), numbers


@pytest.mark.parametrize("defect", DEFECT_IDS)
def test_every_defect_has_a_prose_section(defect):
    text = DEFECTS.read_text(encoding="utf-8")
    assert f"## {defect} —" in text, f"{defect} is in the table with no section below it"


@pytest.mark.parametrize("defect", DEFECT_IDS)
def test_every_defect_names_a_file_and_line(defect):
    """A defect without a location is not a defect report, it is an opinion.
    These get filed upstream, where the first question is always 'where'."""
    assert re.search(r"`\w+\.F90:[\d\-/]+`", locations()[defect]), locations()[defect]


@pytest.mark.parametrize("defect", DEFECT_IDS)
def test_every_disposition_is_one_of_the_five_kinds(defect):
    disposition = DISPOSITIONS[defect]
    kind = disposition.split(":")[0].strip()
    assert kind in {
        "fidelity-flag",
        "invariant-test",
        "not-implemented",
        "harness-patch",
        "documentation-only",
    }, f"{defect}: unknown disposition {disposition!r}"


# --------------------------------------------------------------------------
# Each disposition is actually realised
# --------------------------------------------------------------------------


def _flagged() -> dict[str, str]:
    return {
        defect: d.split(":", 1)[1].strip()
        for defect, d in DISPOSITIONS.items()
        if d.startswith("fidelity-flag")
    }


@pytest.mark.parametrize("defect", sorted(_flagged()))
def test_a_claimed_fidelity_flag_exists_and_cites_its_defect(defect):
    """Both directions. The flag must exist, and its documentation must name
    the defect — otherwise renumbering a defect silently orphans the flag."""
    flag = _flagged()[defect]
    assert flag in {f.name for f in fields(FidelityConfig)}, (
        f"{defect} claims fidelity flag {flag!r}, which is not a FidelityConfig field"
    )
    text = FIDELITY.read_text(encoding="utf-8")
    assert f"## `{flag}`" in text, f"{flag} has no docs/fidelity.md section"
    section = text.split(f"## `{flag}`", 1)[1].split("\n## ", 1)[0]
    assert defect in section, f"the {flag} section does not cite {defect}"


# Every flag section that belongs to a defect declares it with this phrase.
# Casual cross-references ("see UP-6", "until UP-6 is fixed") read differently
# and are deliberately not matched -- a section may mention a neighbouring
# defect without claiming it.
DECLARES = re.compile(r"[Uu]pstream\s+defect\s+(UP-\d+)")


def test_no_fidelity_flag_claims_a_defect_that_is_handled_some_other_way():
    """The reverse orphan, and the one that actually happened.

    UP-4 spent a session with both a fidelity flag and an UPSTREAM_DEFECTS
    entry saying it gets "an invariant test rather than a fidelity flag". Two
    documents, each internally consistent, contradicting each other. Nothing
    caught it because nothing compared them.

    So: if a flag's section declares itself to be for UP-n, then UP-n's
    disposition must name that exact flag."""
    text = FIDELITY.read_text(encoding="utf-8")
    checked = 0
    for section in text.split("\n## `")[1:]:
        flag = section.split("`", 1)[0]
        declared = DECLARES.search(section)
        if declared is None:
            continue
        defect = declared.group(1)
        checked += 1
        assert DISPOSITIONS.get(defect) == f"fidelity-flag: {flag}", (
            f"docs/fidelity.md says {flag} is for {defect}, but {defect}'s "
            f"disposition is {DISPOSITIONS.get(defect)!r}"
        )
    assert checked == len(_flagged()), (
        f"{checked} flag sections declare a defect but {len(_flagged())} defects "
        f"claim a flag; one side has an entry the other does not"
    )


@pytest.mark.parametrize(
    "defect", [d for d, x in DISPOSITIONS.items() if x.startswith("harness-patch")]
)
def test_a_claimed_harness_patch_exists(defect):
    name = DISPOSITIONS[defect].split(":", 1)[1].strip()
    assert (PATCHES / name).is_file(), f"{defect} claims {name}, which is not in {PATCHES}"


@pytest.mark.parametrize("defect", [d for d, x in DISPOSITIONS.items() if x == "not-implemented"])
def test_a_not_implemented_defect_is_recorded_as_unsupported(defect):
    """Otherwise a user discovers the refusal at runtime, having assumed parity
    with UM GLOMAP."""
    text = UNSUPPORTED.read_text(encoding="utf-8")
    subject = locations()[defect].split(".F90")[0].strip("`").split("/")[-1]
    assert subject in text or "icoag" in text, (
        f"{defect} is not-implemented but docs/unsupported.md does not mention it"
    )


@pytest.mark.parametrize(
    "defect", [d for d, x in DISPOSITIONS.items() if x == "documentation-only"]
)
def test_a_documentation_only_defect_says_which_source_to_trust(defect):
    """'The header and the code disagree' is useless to a porter without the
    next sentence."""
    text = DEFECTS.read_text(encoding="utf-8")
    section = text.split(f"## {defect} —", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"code is correct|code is right|Port from the code|versus", section), (
        f"{defect} does not say which of the two sources to believe"
    )


# --------------------------------------------------------------------------
# UP-4's invariant, which is what it has instead of a flag
# --------------------------------------------------------------------------


def test_up4_guard_never_fires_in_any_committed_golden():
    """UP-4: `ukca_conden.F90:353-354` clamps with `delgc_cond = delgc_cond/gc`
    where `= gc` was intended. Three lines above,
    `delgc_cond = gc*(1 - exp(-x))` with `x >= 0` bounds it in `[0, gc]`, so the
    `> gc` guard cannot fire.

    That is the argument. This is the observation: the branch dump records the
    predicate explicitly, and it is false in every record of every committed
    golden. Which is why UP-4 has no fidelity flag — there are not two
    behaviours to choose between, so a flag could never be tested and would sit
    in `FidelityConfig` as an untestable decision.

    If this ever fails, UP-4 has become reachable and needs a real decision."""
    archives = sorted(GOLDENS.glob("*.f64.branches.npz"))
    assert archives, "no branch-dump goldens committed; run `make goldens`"

    total = 0
    for path in archives:
        data = np.load(path, allow_pickle=False)
        tag_is_guard = np.array(data["tag_levels"]) == "up4_guard"
        rows = tag_is_guard[data["tag"]]
        assert rows.any(), f"{path.name} carries no up4_guard records"
        assert (data["value"][rows] == 0).all(), (
            f"{path.name}: the UP-4 guard fired. It is no longer unreachable, and "
            f"the port now has a real choice to make about it."
        )
        total += int(rows.sum())
    assert total > 1000, f"only {total} guard records across {len(archives)} goldens"
