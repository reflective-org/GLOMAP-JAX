#!/usr/bin/env python3
"""Extract the gas-phase index tables from `ukca_setup_indices.F90` (task 31).

    python validation/extract_gas_literals.py           # regenerate
    python validation/extract_gas_literals.py --check   # exit 1 if stale

Writes `src/glomap_jax/physics/_gas_literals.py`. Generated, committed, and
verified by `tests/test_gas_indices.py`, which re-runs the extraction and
compares — so the file cannot drift from the vendored tree, and cannot be
hand-edited without the test noticing. Same convention as
`validation/extract_mode_literals.py` and `core/constants.py`.

**Machine-extracted, never retyped.** Each of the four gas-phase routines
assigns the *same* 178 module variables — 174 integer scalars and four arrays
of 50 — so a hand transcription is roughly 900 numbers with nothing but eyes
between a typo and a plausible-looking wrong index. Byte equality against the
capture would catch it, but only after somebody spent a morning on it.

What is NOT extracted, because it is derived and the port recomputes it:

* `nadvg = 2 + nchemg` and `ntrag = nadvg + noffox`;
* `condensable = (condensable_choice > 0)`.

Those three are the only non-literal right-hand sides in any of the four
routines, and this script asserts that.

Four routines, not seven. `glomap_box_config_mod`'s `init_indices` pairs a
gas-phase routine with a mode routine per setup, and the gas side collapses:

    setup 1        -> ukca_indices_sv1
    setups 2, 3, 8 -> ukca_indices_orgv1_soto3
    setups 4, 5    -> ukca_indices_orgv1_soto6
    setup 6        -> ukca_indices_nochem

So three setups share one gas table exactly, and two more share another. Any
"the tables differ across setups" check has to be written per *routine group*
or it is asserting something false.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_setup_indices.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_gas_literals.py"

# i_mode_setup -> gas-phase routine. From `init_indices`'s SELECT CASE,
# glomap_box_config_mod.F90:369-395. Setups 10-13 are rejected there, so they
# have no reference and are not extracted.
SETUP_ROUTINE = {
    1: "ukca_indices_sv1",
    2: "ukca_indices_orgv1_soto3",
    3: "ukca_indices_orgv1_soto3",
    4: "ukca_indices_orgv1_soto6",
    5: "ukca_indices_orgv1_soto6",
    6: "ukca_indices_nochem",
    8: "ukca_indices_orgv1_soto3",
}

ARRAYS = ("mm_gas", "condensable_choice", "dimen")
DERIVED = ("nadvg", "ntrag", "condensable")

# Group boundaries, as (name of first member, name of last member, expected
# size). The sizes are asserted: they are how a silently truncated parse — the
# failure mode a regex extractor actually has — turns into an error instead of
# a short table. Note the counts disagree with the source's own section
# comments, which say "40 tropospheric chemistry species" (there are 41) and
# "60 indices for tropospheric chemistry species" (there are 61). Ported from
# the code, not the comments.
GROUPS = (
    ("count", "nchemg", "ngasbudget", 6),
    ("s0", "mox", "mpt", 55),
    ("st", "no", "npt", 77),
    ("budget", "ndmsemoc", "nsorg_wdep", 26),
    ("reaction", "iohdms1", "icosoh", 8),
)

NCHEMGMAX = 50  # `ukca_setup_indices.F90:608`, a PARAMETER.


def _routine_text(source: str, name: str) -> str:
    """One routine's body, continuations joined and comments stripped.

    Continuations first: `mm_gas` spans six lines, and a line-based parser
    silently truncates it to the eight values that fitted on the first.
    """
    match = re.search(
        rf"^SUBROUTINE {name}\b.*?^END SUBROUTINE {name}\b",
        source,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    assert match, f"routine {name} not found"
    body = re.sub(r"&\s*\n\s*", "", match.group(0))
    return re.sub(r"!.*$", "", body, flags=re.MULTILINE)


def _assignments(body: str) -> list[tuple[str, str]]:
    """Every `name = rhs` in source order, lowercased. Fortran is
    case-insensitive and this module spells the same variable `MMeSMe` in its
    declaration and `MMeSMe`/`mmesme` in its bodies."""
    out = []
    for line in body.split("\n"):
        match = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", line)
        if not match:
            continue
        name, rhs = match.group(1).lower(), match.group(2)
        if name in ("routinename", "zhook_in", "zhook_out"):
            continue  # PARAMETER declarations, not table entries
        out.append((name, rhs))
    return out


def _numbers(text: str) -> list:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        out.append(int(tok) if re.fullmatch(r"[+-]?\d+", tok) else float(tok))
    return out


def _array(rhs: str, field: str, routine: str) -> list:
    match = re.fullmatch(r"\[(.*)\]", rhs)
    assert match, f"{routine}: {field} is not a bracketed literal: {rhs[:60]}"
    values = _numbers(match.group(1))
    assert len(values) == NCHEMGMAX, f"{routine}: {field} has {len(values)}, want {NCHEMGMAX}"
    return values


def _grouped(names: list[str]) -> dict[str, list[str]]:
    """Slice the scalar names into the five sections, by sentinel."""
    groups = {}
    for label, first, last, size in GROUPS:
        assert first in names, f"{label}: {first} not assigned"
        assert last in names, f"{label}: {last} not assigned"
        lo, hi = names.index(first), names.index(last)
        assert lo <= hi, f"{label}: {first} comes after {last}"
        members = names[lo : hi + 1]
        assert len(members) == size, f"{label}: {len(members)} members, expected {size}"
        groups[label] = members
    covered = [n for g in groups.values() for n in g]
    assert len(covered) == len(set(covered)), "groups overlap"
    missing = sorted(set(names) - set(covered) - set(DERIVED))
    assert not missing, f"scalars in no group: {missing}"
    return groups


def extract() -> dict:
    source = SOURCE.read_text(encoding="utf-8")

    parameter = re.search(r"INTEGER,\s*PARAMETER\s*::\s*nchemgmax\s*=\s*(\d+)", source)
    assert parameter, "nchemgmax parameter not found"
    assert int(parameter.group(1)) == NCHEMGMAX, "nchemgmax moved"

    routines: dict[str, dict] = {}
    order: list[str] | None = None
    for routine in sorted(set(SETUP_ROUTINE.values())):
        body = _routine_text(source, routine)
        pairs = _assignments(body)

        names = [n for n, _ in pairs]
        assert len(names) == len(set(names)), f"{routine}: a name is assigned twice"
        if order is None:
            order = names
        else:
            # Not cosmetic: the group slicing below is positional, so a routine
            # that assigned the same names in a different order would be sliced
            # into the wrong sections without this.
            assert names == order, f"{routine}: assignment order differs from the first routine"

        rec: dict = {"scalars": {}, "arrays": {}}
        for name, rhs in pairs:
            if name in DERIVED:
                continue  # recomputed in physics/gas_indices.py
            if name in ARRAYS:
                rec["arrays"][name] = _array(rhs, name, routine)
            else:
                assert re.fullmatch(r"[+-]?\d+", rhs), f"{routine}: {name} = {rhs[:40]} not literal"
                rec["scalars"][name] = int(rhs)
        assert set(rec["arrays"]) == set(ARRAYS), f"{routine}: arrays {set(rec['arrays'])}"
        routines[routine] = rec

    assert order is not None
    scalar_order = [n for n in order if n not in ARRAYS and n != "condensable"]
    groups = _grouped(scalar_order)

    return {
        "nchemgmax": NCHEMGMAX,
        "setup_routine": dict(SETUP_ROUTINE),
        "groups": groups,
        "derived": list(DERIVED),
        "routines": routines,
    }


def render(data: dict) -> str:
    import pprint

    body = pprint.pformat(data, width=88, sort_dicts=True)
    return (
        '"""Gas-phase index tables, extracted from `ukca_setup_indices.F90`.\n'
        "\n"
        "GENERATED -- do not edit. Regenerate with::\n"
        "\n"
        "    python validation/extract_gas_literals.py\n"
        "\n"
        "`tests/test_gas_indices.py` re-runs the extraction and compares, so an edit\n"
        "here fails rather than silently diverging from the vendored Fortran.\n"
        "\n"
        "Indices are as the Fortran writes them: **1-based, with 0 meaning absent**.\n"
        "`physics/gas_indices.py` converts to 0-based and maps absent to -1.\n"
        "\n"
        "Literals only. `nadvg`, `ntrag` and `condensable` are derived and are\n"
        "recomputed there, because recomputing them is what makes this a port.\n"
        '"""\n\n'
        f"GAS_LITERALS = {body}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fresh = extract()
    if args.check:
        # Compares the DATA, not the bytes: `ruff format` rewrites the
        # generated file after it is written, so a byte comparison would report
        # every formatted file as stale.
        if not TARGET.is_file():
            print(f"{TARGET.relative_to(REPO)} does not exist; generate it")
            return 1
        namespace: dict = {}
        exec(compile(TARGET.read_text(encoding="utf-8"), str(TARGET), "exec"), namespace)
        if namespace.get("GAS_LITERALS") != fresh:
            print(f"{TARGET.relative_to(REPO)} disagrees with the Fortran; regenerate it")
            return 1
        print("up to date")
        return 0

    TARGET.write_text(render(fresh), encoding="utf-8")
    n = len(fresh["routines"])
    print(f"wrote {TARGET.relative_to(REPO)} ({n} gas routines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
