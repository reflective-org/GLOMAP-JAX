#!/usr/bin/env python3
"""Extract `ukca_coagwithnucl`'s budget write sites into a generated table.

    python validation/extract_coagwithnucl_literals.py --check
    python validation/extract_coagwithnucl_literals.py

The last 280 lines of `ukca_coagwithnucl` are one nested triple loop over
`(imode, icp, jmode)` containing 53 near-identical guarded accumulations::

    IF ((imode == mode_nuc_sol) .AND. (jmode == mode_ait_sol)) THEN
      IF ((icp == cp_su) .AND. (nmascoagsuintr12 > 0)) THEN
        WHERE (mask3(:))
          bud_aer_mas(:,nmascoagsuintr12) =
          bud_aer_mas(:,nmascoagsuintr12) + mtran(:,cp_su,imode,jmode)
      END IF

Each row of the generated table is `(imode, jmode, icp, budget_name)`, and the
parse asserts three things rather than trusting them:

* the name encodes the same `(imode, jmode)` its enclosing guard tests -- the
  `intr12` suffix is `1 -> 2`, and a site whose guard says otherwise is a
  transcription error upstream or here;
* the `mtran` component subscript is the *same* `cp_XX` the `icp` guard tests,
  not `icp`. The Fortran writes `mtran(:,cp_su,imode,jmode)` inside an
  `icp == cp_su` guard, so the two always agree -- but they are written
  independently, and a site where they diverged would silently bank one
  component's transfer against another's diagnostic;
* every name matches `nmascoag<cp>intr<i><j>`, so nothing is dropped by a
  parser that did not recognise it.

Three tags are not prefixes of their component names
-----------------------------------------------------

Sea salt is `ss` against `cp_cl`, ammonium `nh` against `cp_nh4`, nitrate `nt`
against `cp_no3`. Deriving the tag from the component name looked obviously
right and failed on `nmascoagntintr23`, so the map is tabulated and asserted to
cover every site the parse finds.

`cp_mp` and `cp_nn` are in the table and in no supported setup
--------------------------------------------------------------

Several sites guard on `cp_mp` -- the mineral-dust "mixed particle" component,
which exists only in the `ncp = 9`/`10` configurations that `docs/unsupported.md`
records as rejected. The rows are kept: the port skips them because
`component(imode, icp)` is false, which is the same reason the Fortran skips
them, and dropping them here would make the table describe a different routine.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_coagwithnucl.F90"
TARGET = REPO / "src" / "glomap_jax" / "physics" / "_coagwithnucl_literals.py"

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
CP_INDEX = {
    "cp_su": 1,
    "cp_bc": 2,
    "cp_oc": 3,
    "cp_cl": 4,
    "cp_du": 5,
    "cp_so": 6,
    "cp_nh4": 7,
    "cp_no3": 8,
    "cp_nn": 9,
    "cp_mp": 10,
}

#: Budget-name tag to component PARAMETER. **Not** a prefix of the component
#: name in three cases: sea salt is `ss` against `cp_cl`, ammonium `nh` against
#: `cp_nh4`, and nitrate `nt` against `cp_no3`. Deriving the tag from the
#: component name instead of tabulating it looked right and failed on
#: `nmascoagntintr23` -- which is the whole reason this map is written out and
#: asserted to cover every site the parse finds.
TAG_TO_CP = {
    "su": "cp_su",
    "bc": "cp_bc",
    "oc": "cp_oc",
    "ss": "cp_cl",
    "du": "cp_du",
    "so": "cp_so",
    "nh": "cp_nh4",
    "nt": "cp_no3",
    "nn": "cp_nn",
    "mp": "cp_mp",
}

_PAIR = re.compile(r"IF\(\(imode==(\w+)\)\.AND\.\(jmode==(\w+)\)\)THEN")
_SITE = re.compile(
    r"IF\(\(icp==(cp_\w+)\)\.AND\.\((nmascoag\w+)>0\)\)THEN"
    r"WHERE\(mask3\(:\)\)"
    r"bud_aer_mas\(:,\2\)=bud_aer_mas\(:,\2\)\+mtran\(:,(cp_\w+),imode,jmode\)"
)
_NAME = re.compile(r"^nmascoag([a-z0-9]+?)intr(\d)(\d)$")


def squeezed_body() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    text = re.sub(r"&\s*\n\s*", "", text)
    text = re.sub(r"!.*", "", text)
    return re.sub(r"\s+", "", text)


def build() -> dict:
    body = squeezed_body()
    pairs = [
        (m.start(), MODE_INDEX[m.group(1)], MODE_INDEX[m.group(2)]) for m in _PAIR.finditer(body)
    ]
    if not pairs:
        raise SystemExit("no `imode ==` / `jmode ==` guard found")

    rows: list[tuple[int, int, int, str]] = []
    for m in _SITE.finditer(body):
        icp_name, budget, mtran_cp = m.groups()
        if icp_name != mtran_cp:
            raise SystemExit(
                f"{budget}: guarded on {icp_name} but banks mtran(:,{mtran_cp},...) -- "
                "one component's transfer against another's diagnostic"
            )
        enclosing = [p for p in pairs if p[0] < m.start()]
        if not enclosing:
            raise SystemExit(f"{budget} is outside every (imode, jmode) guard")
        _pos, imode, jmode = enclosing[-1]
        name_match = _NAME.match(budget)
        if not name_match:
            raise SystemExit(f"{budget} does not match nmascoag<cp>intr<i><j>")
        tag, ni, nj = name_match.group(1), int(name_match.group(2)), int(name_match.group(3))
        if (ni, nj) != (imode, jmode):
            raise SystemExit(f"{budget} encodes {ni}->{nj} but sits in the {imode}->{jmode} guard")
        if tag not in TAG_TO_CP:
            raise SystemExit(f"{budget} carries an unknown component tag {tag!r}")
        if TAG_TO_CP[tag] != icp_name:
            raise SystemExit(
                f"{budget} encodes component {tag!r} = {TAG_TO_CP[tag]} but guards on {icp_name}"
            )
        rows.append((imode, jmode, CP_INDEX[icp_name], budget))

    if len({r[3] for r in rows}) != len(rows):
        raise SystemExit("a budget name appears at more than one site")
    return {"rows": rows}


def render(data: dict) -> str:
    lines = [
        '"""Generated by `validation/extract_coagwithnucl_literals.py`. Do not edit.',
        "",
        "`ukca_coagwithnucl`'s budget write sites: one row per guarded",
        "accumulation of `mtran` into `bud_aer_mas`, parsed out of the Fortran.",
        "",
        "    (imode, jmode, icp, budget)   all 1-based, as the Fortran declares them",
        "",
        "The extractor asserts that each name's `intr<i><j>` suffix and component",
        "tag agree with the guard it sits inside, and that the `mtran` component",
        "subscript is the same `cp_XX` the `icp` guard tests -- the two are written",
        "independently in the source and a divergence would bank one component's",
        "transfer against another's diagnostic.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "#: (imode, jmode, icp, budget), in source order.",
        "COAG_BUDGET_SITES = (",
    ]
    for imode, jmode, icp, budget in data["rows"]:
        lines.append(f'    ({imode}, {jmode}, {icp}, "{budget}"),')
    lines += [")", ""]
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
