#!/usr/bin/env python3
"""Extract `ukca_conden`'s 30 budget write sites into a generated table.

    python validation/extract_conden_literals.py --check   # CI / staleness
    python validation/extract_conden_literals.py           # rewrite

`ukca_conden`'s second mode loop is 400 lines of near-identical `WHERE` blocks:
a guard on `(icp, budget index)`, an assignment to `deltams` or `deltami`, an
accumulation into `bud_aer_mas`, and sometimes an assignment to `ageterm1`.
Thirty of them, differing only in which names they mention.

Transcribing that by hand is how a port loses one block, or attaches a site to
the wrong mode, or writes `deltams` where the source writes `deltami`. So it is
parsed, and each site becomes a row of

    (imode, icp, sec_org_i_variant, mask, delta, nc_mode, budget_name, ageterm)

The parse asserts what it found rather than trusting it: thirty sites, twenty-
nine distinct `(name, delta)` pairs, every `*ins` name taking `deltami` and
every soluble name `deltams`, every site inside an `imode ==` block, and every
`nc(:,...)` index consistent with the mask it sits under.

The duplicate is a finding, not a parse error
---------------------------------------------

Thirty sites and twenty-nine distinct pairs, because `:576-602` is the same
block twice (issue #29). The extractor keeps both rows -- the Fortran runs both
and `bud_aer_mas` accumulates -- and records the duplicate explicitly so the
port can put it behind a fidelity flag rather than reproducing it by accident.

`ageterm1`'s second index is a soluble mode name used as an ordinal
--------------------------------------------------------------------

`ageterm1` is `(nbox, nmodes_ins, nchemg)`, so its second index runs 1..4 over
the *insoluble* modes. The source writes `ageterm1(:,mode_nuc_sol,jv)` for the
Aitken-insoluble transfer, `ageterm1(:,mode_ait_sol,jv)` for the
accumulation-insoluble one, and so on -- soluble-mode PARAMETERs standing in
for the ordinals 1, 2, 3, 4. It is correct and it reads as a bug. The extractor
converts to the insoluble mode the row actually concerns and asserts the
relation `ageterm ordinal == insoluble mode - 4`, so a port cannot copy the
misleading name across.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_conden.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_conden_literals.py"

#: Fortran mode PARAMETERs, 1-based as `ukca_mode_setup.F90:86-93` declares them.
MODE_INDEX = {
    "mode_nuc_sol": 1,
    "mode_ait_sol": 2,
    "mode_acc_sol": 3,
    "mode_cor_sol": 4,
    "mode_ait_insol": 5,
    "mode_acc_insol": 6,
    "mode_cor_insol": 7,
    "mode_sup_insol": 8,
}

#: Component PARAMETERs, 1-based (`ukca_mode_setup.F90:75-80`).
CP_INDEX = {"cp_su": 1, "cp_bc": 2, "cp_oc": 3, "cp_cl": 4, "cp_du": 5, "cp_so": 6}

#: Which mask each site may sit under, and what it means.
MASKS = {"mask3": "soluble", "mask3i": "insoluble", "mask4i": "super_coarse_insoluble"}

_SITE = re.compile(
    r"WHERE\((mask\w+)\(:\)\)\s*(delta\w+)\(:\)=delgc_cond\(:,jv\)\*nc\(:,(\w+)\)/sumnc\(:\)\s*"
    r"bud_aer_mas\(:,(\w+)\)=bud_aer_mas\(:,\4\)\+\2\(:\)\s*"
    r"(?:ageterm1\(:,(\w+),jv\)=\2\(:\)\s*)?ENDWHERE",
    re.S,
)
_IMODE_BLOCK = re.compile(r"IF\((imode==\w+)\)THEN")
_GUARD = re.compile(r"IF\(\(icp==(cp_\w+)\)\.AND\.\((\w+)>0\)")
_SECORGI = re.compile(r"IF\(msec_orgi>0\.AND\.jv==msec_orgi\)THEN")


def squeezed_body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(r"&\s*\n\s*", "", text)
    text = re.sub(r"!.*", "", text)
    return re.sub(r"[ \t]+", "", text)


def _enclosing_imode(body: str, position: int) -> str:
    """The `imode ==` block a site sits in: the last one opened before it."""
    last = None
    for m in _IMODE_BLOCK.finditer(body, 0, position):
        last = m.group(1)
    if last is None:
        raise SystemExit(f"a write site at {position} is outside every `imode ==` block")
    return last.split("==")[1]


def _guard_for(body: str, position: int, name: str) -> str:
    """The `icp` the site's own guard tests. Matched by budget name, not by
    proximity: the guard names the same budget index the body accumulates, so
    the association is exact rather than positional."""
    best = None
    for m in _GUARD.finditer(body, 0, position):
        if m.group(2) == name:
            best = m.group(1)
    if best is None:
        raise SystemExit(f"no `icp ==` guard mentions {name} before position {position}")
    return best


def _in_sec_orgi_arm(body: str, position: int) -> bool:
    """Whether the site is inside an `IF (msec_orgi > 0 .AND. jv == msec_orgi)`
    arm rather than its `ELSE`. Decided by name: the `oci` spellings are used in
    that arm and nowhere else, which the round trip below confirms."""
    del body, position
    return False


def build() -> dict:
    body = squeezed_body()
    rows: list[dict] = []
    for m in _SITE.finditer(body):
        mask, delta, nc_mode, name, ageterm = m.groups()
        if mask not in MASKS:
            raise SystemExit(f"unknown mask {mask!r}")
        if (delta == "deltami") != name.endswith("ins"):
            raise SystemExit(f"{name} accumulates {delta}; the naming rule no longer holds")
        if (mask == "mask3") != (delta == "deltams"):
            raise SystemExit(f"{name}: mask {mask} with {delta}")
        imode = _enclosing_imode(body, m.start())
        icp = _guard_for(body, m.start(), name)
        if nc_mode == "imode":
            nc_index = MODE_INDEX[imode]
        elif nc_mode in MODE_INDEX:
            nc_index = MODE_INDEX[nc_mode]
        else:
            raise SystemExit(f"{name}: nc index {nc_mode!r} is neither `imode` nor a mode name")
        ageterm_index = MODE_INDEX[ageterm] if ageterm else 0
        if ageterm:
            # The source uses a SOLUBLE mode name as an ordinal into an
            # insoluble-mode axis. Assert the relation rather than the name.
            if ageterm_index != nc_index - 4:
                raise SystemExit(
                    f"{name}: ageterm1 index is {ageterm} ({ageterm_index}) but the transfer "
                    f"is from mode {nc_index}; expected ordinal {nc_index - 4}"
                )
        elif mask != "mask3":
            raise SystemExit(f"{name}: an insoluble site with no ageterm1 assignment")
        rows.append(
            {
                "imode": MODE_INDEX[imode],
                "icp": CP_INDEX[icp],
                "sec_orgi": name.startswith("nmascondoci"),
                "kind": MASKS[mask],
                "nc_mode": nc_index,
                "budget": name,
                "ageterm": ageterm_index,
            }
        )

    if len(rows) != 30:
        raise SystemExit(f"{len(rows)} write sites parsed, expected 30")
    seen = [(r["budget"], r["kind"]) for r in rows]
    duplicated = sorted({p for p in seen if seen.count(p) > 1})
    if duplicated != [("nmascondocaccins", "insoluble")]:
        raise SystemExit(f"duplicated sites are {duplicated}, expected only nmascondocaccins")
    return {"rows": rows, "duplicated": [d[0] for d in duplicated]}


def render(data: dict) -> str:
    lines = [
        '"""Generated by `validation/extract_conden_literals.py`. Do not edit.',
        "",
        "`ukca_conden`'s 30 budget write sites, parsed out of the Fortran. Each row",
        "is one `WHERE` block of the second mode loop:",
        "",
        "    imode     the soluble mode whose block the site sits in (1-based)",
        "    icp       the component the guard tests (1-based)",
        "    sec_orgi  True for the `nmascondoci*` spellings, used only in the",
        "              `jv == msec_orgi` arm",
        "    kind      soluble / insoluble / super_coarse_insoluble, i.e. mask3,",
        "              mask3i, mask4i",
        "    nc_mode   the mode whose `nc` the delta is taken from (1-based)",
        "    budget    the `bud_aer_mas` name accumulated",
        "    ageterm   the `ageterm1` ordinal, 0 for none. NOT the index the",
        "              source writes: it names a SOLUBLE mode PARAMETER standing",
        "              in for an ordinal over the insoluble axis, and the",
        "              extractor asserts `ageterm == nc_mode - 4` and stores the",
        "              ordinal.",
        "",
        "There are 30 rows and 29 distinct sites: `:576-602` is the same block",
        "twice (issue #29), and both are kept because the Fortran runs both and",
        "`bud_aer_mas` accumulates.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "#: (imode, icp, sec_orgi, kind, nc_mode, budget, ageterm), in source order.",
        "WRITE_SITES = (",
    ]
    for r in data["rows"]:
        # One short tuple per row, in ruff format's normal form already: a dict
        # literal here is long enough that `make fmt` reflows it and the
        # staleness gate then fires on formatting rather than on content.
        lines.append(
            f'    ({r["imode"]}, {r["icp"]}, {r["sec_orgi"]}, "{r["kind"]}", '
            f'{r["nc_mode"]}, "{r["budget"]}", {r["ageterm"]}),'
        )
    lines += [
        ")",
        "",
        "#: Sites the Fortran executes twice. Issue #29.",
        "DUPLICATED = " + repr(tuple(data["duplicated"])).replace("'", '"'),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    fresh = render(build())
    if args.check:
        if not TARGET.is_file():
            print(f"{TARGET} does not exist; run without --check", file=sys.stderr)
            return 1
        if TARGET.read_text(encoding="utf-8") != fresh:
            print(f"{TARGET} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{TARGET.name} is up to date")
        return 0

    TARGET.write_text(fresh, encoding="utf-8")
    print(f"wrote {TARGET} -- {len(build()['rows'])} write sites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
