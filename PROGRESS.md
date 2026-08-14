# Progress

Order 1 of 3 (faithful port). Task numbering follows the plan.

## Phase A — scaffolding: **complete (10/10)**

| # | Task | Commit |
|---|---|---|
| 1 | pyproject, package skeleton, single x64 enable | `1e2c3d8` |
| 2 | Makefile venv wrapper | `91f3fa8` |
| 3 | CI: lint, test 3.11/3.12, macOS, weekly soak | `e634187` |
| 4 | COPYRIGHT, PROVENANCE, licensing regression tests | `e8ed13e` |
| 5 | CLAUDE.md binding rules | `a95812a` |
| 6 | Vendor the Fortran + tamper test | `7c0c40f` |
| 7 | Tolerance policy and golden loader | `13efed2` |
| 8 | State NamedTuples, static config dataclasses | `27c6ec0` |
| 9 | Fidelity registry with Fortran-reproducing defaults | `d2f6e3c` |
| 10 | Docs skeleton, mkdocs, ADR seeds | this commit |

## Phase B — reference harness: **not started (0/18)**

Must complete before any physics commit. Tasks 11–23 plus 11b, 11c, 12b, 15b,
20b. The additions came out of adversarial review:

* **11b** a high-precision state dump is a *prerequisite*, not polish — the
  Fortran driver only emits `ES14.6`, seven significant digits, so a
  double-precision reference truncated to that is worth no more than a
  single-precision one.
* **15b** the branch-mask dump (Gate 0) is the highest-value gate in the plan.
  This code diverges by flipped predicates, not precision drift.
* **20b** an `ereport` shim, because a fatal `ereport` does `STOP 1` in-process
  and would kill the pytest interpreter.

## Phases C–K — physics: not started (0/82)

## Orders 2 and 3: not started

## Verified along the way

| check | result |
|---|---|
| CI | green on lint, 3.11, 3.12, macOS |
| tamper test | fails and names the file when a vendored source is edited |
| fidelity registry | fails on a flipped default and on an undocumented flag |
| tolerance floor | permits 0-vs-1e-300, still catches a 10% discrepancy |
| f2py mechanism | `SAVE`d module state and `INTENT(IN OUT)` verified end-to-end |
