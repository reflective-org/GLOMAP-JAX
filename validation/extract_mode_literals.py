#!/usr/bin/env python3
"""Extract the per-setup literal tables from `ukca_mode_setup.F90`.

    python validation/extract_mode_literals.py           # regenerate
    python validation/extract_mode_literals.py --check   # exit 1 if stale

Writes `src/glomap_jax/physics/_mode_literals.py`. Generated, committed, and
verified by `tests/test_mode_literals.py`, which re-runs the extraction and
compares — so the file cannot drift from the vendored tree, and cannot be
hand-edited without the test noticing.

**Machine-extracted, never retyped.** Seven setups times roughly ten tables is
several hundred numbers. `mam4-jax` established the convention for exactly this
shape of problem and `core/constants.py` already follows it; the argument is
the same and stronger here, because a single mistyped digit in `mfrac_0` or
`rhocomp` produces tables that look plausible and a model that is quietly
wrong. The byte-equality tests would catch it — but only after someone spent a
morning on the mismatch.

What is NOT extracted: everything derived. `ddpmid`, `x`, `mmid`/`mlo`/`mhi`,
`mode`, `component` and `soluble` are computed by `physics/modes.py` from these
literals, because recomputing them is the point of the port. The derivation is
identical across all seven routines — verified by diffing their tails, where
the only difference is `fracbcem`/`fracocem`, which are literals.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_mode_setup.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_mode_literals.py"

# i_mode_setup -> the routine that builds its tables. From the SELECT CASE in
# common_mode_setup_interface_mod.F90:111-165. Setups 10-13 are dispatched
# there too but glomap_box_config_mod's init_indices rejects them, so they have
# no reference and are not extracted.
ROUTINES = {
    1: "ukca_mode_suss_4mode",
    2: "ukca_mode_sussbcoc_5mode",
    3: "ukca_mode_sussbcoc_4mode",
    4: "ukca_mode_sussbcocso_5mode",
    5: "ukca_mode_sussbcocso_4mode",
    6: "ukca_mode_duonly_2mode",
    8: "ukca_mode_sussbcocdu_7mode",
}

VECTORS = (
    "mode_choice",
    "modesol",
    "ddplim0",
    "ddplim1",
    "sigmag",
    "num_eps",
    "fracbcem",
    "fracocem",
)
CP_VECTORS = ("component_choice", "soluble_choice", "mm", "rhocomp")
MATRICES = ("component_mode", "mfrac_0")


def _routine_text(source: str, name: str) -> str:
    """The body of one routine, with Fortran continuations joined.

    Continuations are joined before anything else is parsed: an array literal
    routinely spans two lines, and a line-based parser silently truncates it to
    whatever fitted on the first.
    """
    match = re.search(
        rf"^SUBROUTINE {name}\b.*?^END SUBROUTINE {name}\b",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"routine {name} not found"
    body = match.group(0)
    body = re.sub(r"&\s*\n\s*", "", body)  # join continuations
    return re.sub(r"^\s*!.*$", "", body, flags=re.MULTILINE)  # drop comments


def _numbers(text: str) -> list:
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        out.append(int(tok) if re.fullmatch(r"[+-]?\d+", tok) else float(tok))
    return out


def _vector(body: str, field: str, indexed: bool) -> list:
    suffix = r"\(1:ncp\)" if indexed else ""
    m = re.search(rf"%{field}\s*{suffix}\s*=\s*\[([^\]]*)\]", body)
    assert m, f"{field} not found"
    return _numbers(m.group(1))


def _matrix(body: str, field: str, nrows: int) -> list:
    rows = []
    for i in range(1, nrows + 1):
        m = re.search(rf"%{field}\s*\(\s*{i}\s*,\s*1:ncp\s*\)\s*=\s*\[([^\]]*)\]", body)
        assert m, f"{field}({i},:) not found"
        rows.append(_numbers(m.group(1)))
    return rows


def _no_ions(body: str) -> dict:
    """Three switch-dependent branches at `:157-165`.

    Keyed by `(l_fix_ukca_hygroscopicities, l_fix_nacl_density)` in source
    order: both true, hygroscopicities only, else the default.
    """
    found = re.findall(r"%no_ions\s*\(1:ncp\)\s*=\s*\[([^\]]*)\]", body)
    assert len(found) == 3, f"expected 3 no_ions branches, found {len(found)}"
    return {
        "both": _numbers(found[0]),
        "hygro_only": _numbers(found[1]),
        "default": _numbers(found[2]),
    }


def extract() -> dict:
    source = SOURCE.read_text(encoding="utf-8")
    out = {}
    for setup, routine in ROUTINES.items():
        body = _routine_text(source, routine)
        ncp_match = re.search(r"%ncp\s*=\s*(\d+)", body)
        assert ncp_match, f"{routine}: ncp not found"
        ncp = int(ncp_match.group(1))

        names = re.search(r"%component_names\s*\(1:ncp\)\s*=\s*\[([^\]]*)\]", body)
        assert names, f"{routine}: component_names not found"

        rec = {
            "routine": routine,
            "ncp": ncp,
            "component_names": [s.strip().strip("'").strip() for s in names.group(1).split(",")],
            "no_ions": _no_ions(body),
        }
        for field in VECTORS:
            rec[field] = _vector(body, field, indexed=False)
        for field in CP_VECTORS:
            rec[field] = _vector(body, field, indexed=True)
        for field in MATRICES:
            rec[field] = _matrix(body, field, nrows=8)
        out[setup] = rec
    return out


def render(data: dict) -> str:
    import pprint

    body = pprint.pformat(data, width=88, sort_dicts=True)
    return (
        '"""Per-setup literal tables, extracted from `ukca_mode_setup.F90`.\n'
        "\n"
        "GENERATED -- do not edit. Regenerate with::\n"
        "\n"
        "    python validation/extract_mode_literals.py\n"
        "\n"
        "`tests/test_mode_literals.py` re-runs the extraction and compares, so an\n"
        "edit here fails rather than silently diverging from the vendored Fortran.\n"
        "\n"
        "Literals only. Everything derived -- `ddpmid`, `x`, the mode masses, `mode`,\n"
        "`component`, `soluble` -- is computed in `physics/modes.py`, because\n"
        "recomputing it is what makes this a port rather than a copy.\n"
        '"""\n\n'
        f"SETUP_LITERALS = {body}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fresh = extract()
    if args.check:
        # Compares the DATA, not the bytes. `ruff format` reformats the
        # generated file after it is written, so a byte comparison reports
        # every formatted file as stale -- which is a property of the
        # formatter, not of the tables.
        if not TARGET.is_file():
            print(f"{TARGET.relative_to(REPO)} does not exist; generate it")
            return 1
        namespace: dict = {}
        exec(compile(TARGET.read_text(encoding="utf-8"), str(TARGET), "exec"), namespace)
        if namespace.get("SETUP_LITERALS") != fresh:
            print(f"{TARGET.relative_to(REPO)} disagrees with the Fortran; regenerate it")
            return 1
        print("up to date")
        return 0

    TARGET.write_text(render(fresh), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO)} ({len(fresh)} setups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
