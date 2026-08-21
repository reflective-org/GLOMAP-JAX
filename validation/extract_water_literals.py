#!/usr/bin/env python3
"""Extract the ZSR ion tables from `ukca_water_content_v.F90`.

    python validation/extract_water_literals.py           # regenerate
    python validation/extract_water_literals.py --check   # exit 1 if stale

Writes `src/glomap_jax/physics/_water_literals.py`. Generated, committed, and
verified by `tests/test_water_tables.py`, which re-runs the extraction and
compares -- so the file cannot drift from the vendored tree, and cannot be
hand-edited without the test noticing.

**Machine-extracted, never retyped.** Twelve electrolyte pairs times eight
polynomial coefficients, plus two validity limits each, is 120 numbers with
eleven significant digits apiece. A mistyped digit gives a plausible water
content and a quietly wrong model.

## Two tables, differing in one number

`ukca_water_content_v.F90:235` reads:

    IF (glomap_config%l_fix_ukca_water_content) y(1,-3,6) = -1.220611402e3

and the source's own comment above the `(1,-3)` block says "One of the y
co-efficients is incorrect here (j=6)". So there are two tables: the DATA
literals, and the DATA literals with that one entry replaced. Both are
extracted here and both are emitted, because the port holds them as two
immutable constants rather than reproducing the patch.

It reproduces them as constants deliberately. In the Fortran `y` is
DATA-initialised and `!$OMP THREADPRIVATE` (`:140`), hence implicitly `SAVE`,
and `:235` writes it **in place with no restore** -- so once a process has seen
the flag true the patched coefficient persists even after the flag goes back to
false. That one-way latch (issue #22) is a property of mutable module state and
has no analogue in a pair of frozen arrays, which is the point.

## The index base

`y` is declared `REAL :: y(3,-4:-1,0:7)` at `:122`: cation 1..3, anion -4..-1,
coefficient 0..7. Anion indices are **negative**, and the emitted tables are
keyed by the Fortran `(cation, anion)` pair verbatim rather than rebased, so
that a reader comparing against the source is comparing like with like. The
0-based remap belongs in `physics/water_tables.py`, in one place, with a test.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_water_content_v.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_water_literals.py"

NCATION, NANION = 3, 4
# Source order: cation ascending, anion -1 down to -4, so a reader comparing
# the emitted table against the DATA statements goes down both at once.
PAIRS = [(c, a) for c in range(1, NCATION + 1) for a in range(-1, -NANION - 1, -1)]

# `DATA (y(1,-1,j),j=0,7)/ a, b, ... /` across continuation lines.
_Y = re.compile(
    r"DATA\s*\(\s*y\(\s*(\d+)\s*,\s*(-\d+)\s*,\s*j\s*\)\s*,\s*j\s*=\s*0\s*,\s*7\s*\)\s*/(.*?)/",
    re.DOTALL,
)
# `DATA rh_min(1,-1),molal_max(1,-1)/0.0e0,30.4e0/`
_LIMITS = re.compile(
    r"DATA\s+rh_min\(\s*(\d+)\s*,\s*(-\d+)\s*\)\s*,\s*molal_max\(\s*\1\s*,\s*\2\s*\)\s*/(.*?)/",
    re.DOTALL,
)
# The single in-place patch the fidelity switch applies.
_PATCH = re.compile(
    r"IF\s*\(\s*glomap_config%l_fix_ukca_water_content\s*\)\s*"
    r"y\(\s*(\d+)\s*,\s*(-\d+)\s*,\s*(\d+)\s*\)\s*=\s*([^\s!]+)"
)


def _numbers(blob: str) -> list[float]:
    """Fortran reals from one DATA value list, continuations and all."""
    cleaned = re.sub(r"&\s*\n\s*", "", blob)
    cleaned = re.sub(r"!.*", "", cleaned)
    out = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token.replace("e", "E").replace("E", "e")))
    return out


def extract(text: str | None = None) -> dict:
    text = SOURCE.read_text(encoding="utf-8") if text is None else text

    base: dict[tuple[int, int], list[float]] = {}
    for cation, anion, blob in _Y.findall(text):
        key = (int(cation), int(anion))
        values = _numbers(blob)
        if len(values) != 8:
            raise SystemExit(f"y{key}: expected 8 coefficients, parsed {len(values)}")
        if key in base:
            raise SystemExit(f"y{key} declared twice")
        base[key] = values

    limits: dict[tuple[int, int], tuple[float, float]] = {}
    for cation, anion, blob in _LIMITS.findall(text):
        key = (int(cation), int(anion))
        values = _numbers(blob)
        if len(values) != 2:
            raise SystemExit(f"limits{key}: expected rh_min and molal_max, parsed {len(values)}")
        limits[key] = (values[0], values[1])

    missing = [p for p in PAIRS if p not in base or p not in limits]
    if missing:
        raise SystemExit(f"no DATA for pair(s) {missing}")
    extra = sorted(set(base) - set(PAIRS))
    if extra:
        raise SystemExit(f"DATA for pair(s) outside the declared bounds: {extra}")

    patches = _PATCH.findall(text)
    if len(patches) != 1:
        raise SystemExit(
            f"expected exactly one l_fix_ukca_water_content patch, found {len(patches)}. "
            "If the Fortran now patches more than one coefficient, the two-table model "
            "here is no longer the right shape."
        )
    cation, anion, index, value = patches[0]
    patch_key = (int(cation), int(anion))
    patch_index = int(index)
    patch_value = float(value.replace("e", "E").replace("E", "e"))

    fixed = {k: list(v) for k, v in base.items()}
    if fixed[patch_key][patch_index] == patch_value:
        raise SystemExit(
            f"the patch at y{patch_key}[{patch_index}] sets the value the DATA statement "
            "already holds, so the two tables would be identical and every both-settings "
            "test would be vacuous"
        )
    fixed[patch_key][patch_index] = patch_value

    return {
        "base": base,
        "fixed": fixed,
        "limits": limits,
        "patch": (patch_key, patch_index, patch_value),
    }


def _entry(key: tuple[int, int], values) -> list[str]:
    """One dict entry, wrapped. Eight eleven-digit coefficients do not fit in
    100 columns, and the generated file is linted like any other."""
    out = [f"    {key}: ("]
    for value in values:
        out.append(f"        {value!r},")
    out.append("    ),")
    return out


def render(data: dict) -> str:
    (pc, pa), pi, pv = data["patch"]
    lines = [
        '"""Generated by `validation/extract_water_literals.py`. Do not edit.',
        "",
        "The ZSR water-activity coefficients from `ukca_water_content_v.F90`,",
        "Jacobson Table B.10. Keys are the Fortran `(cation, anion)` pair with the",
        "anion index left negative, exactly as declared at `:122` -- so a reader",
        "can compare a row against the source without doing arithmetic first.",
        "",
        f"`FIXED` differs from `BASE` in exactly one entry: y({pc},{pa})[{pi}],",
        f"which `:235` replaces with {pv!r} when `l_fix_ukca_water_content` is on.",
        "That is a factor of ten, not a rounding: the source comment above the",
        "block says so.",
        "",
        "Note that (1,-1) and (1,-2) carry byte-identical coefficients in the",
        "Fortran -- H+ HSO4- and 2H+ SO42- share a fit. That is upstream's doing,",
        "not an extraction bug, so a test asserting all twelve pairs differ would",
        "be wrong to add.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f"PATCHED_ENTRY = (({pc}, {pa}), {pi}, {pv!r})",
        "",
        "BASE: dict[tuple[int, int], tuple[float, ...]] = {",
    ]
    for key in PAIRS:
        lines += _entry(key, data["base"][key])
    lines += ["}", "", "FIXED: dict[tuple[int, int], tuple[float, ...]] = {"]
    for key in PAIRS:
        lines += _entry(key, data["fixed"][key])
    lines += [
        "}",
        "",
        "# (rh_min, molal_max) -- validity limits, in percent and molal.",
        "LIMITS: dict[tuple[int, int], tuple[float, float]] = {",
    ]
    for key in PAIRS:
        lines.append(f"    {key}: {data['limits'][key]!r},")
    lines += ["}", ""]
    return "\n".join(lines)


def _display(path: Path) -> str:
    """Repo-relative when it can be; absolute otherwise, so a monkeypatched
    path in a test does not turn the failure into a ValueError from pathlib."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args(argv)

    fresh = extract()
    if args.check:
        if not TARGET.is_file():
            print(f"{_display(TARGET)} does not exist; generate it")
            return 1
        if TARGET.read_text(encoding="utf-8") != render(fresh):
            print(f"{_display(TARGET)} disagrees with the Fortran; regenerate it")
            return 1
        print(f"{_display(TARGET)} is up to date")
        return 0

    TARGET.write_text(render(fresh), encoding="utf-8")
    print(f"wrote {_display(TARGET)} ({len(fresh['base'])} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
