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
| 10 | Docs skeleton, mkdocs, ADR seeds | `0d3edd2` |

### Phase A review

Closed with an adversarial agent review, per the working practice. It found 20
issues; the serious ones are fixed, the rest are filed (#6-#11).

Fixed in `3dd6b0f`:

* **The float64 justification was factually wrong**, repeated in five places
  including a test that could not fail. 1e-20 is a *normal* float32 number and
  1e-40 is subnormal but non-zero; the reference Fortran runs in single
  precision and branches correctly. Replaced with the real reasons.
* **`s0g` was sized by the wrong axis** — it is `nadvg` (advected tracers), not
  `nchemg`. Large enough today by luck.
* **"nbudaer takes eight distinct values"** — seven.
* **UP-10 confirmed and was missing entirely**: `ukca_conden.F90:372-387` gates
  insoluble-mode condensation with `num_eps` indexed by the *soluble* mode,
  wrong by 1e6 and results-changing on setup 8.
* **`docs/UPSTREAM_DEFECTS.md` was an empty stub** that shipped code cited, and
  `PROVENANCE.md` claimed the defects were "reported upstream" when they are
  drafted and unfiled.
* **Two tests could not fail**, one ending in `or True`.
* **`ibln`/`icondiam`/`imerge`/`ifuchs`/`idcmfp` were unvalidated** despite
  `ModelConfig` citing the very ereport that covers `ibln`.

## Phase B — reference harness: **in progress (9/18)**

| # | Task | Commit |
|---|---|---|
| 11 | `ref-f32` build script + `fortran` marker | `4cd6690` |
| 12 | `-fdefault-real-8` (`ref-f64`) variant | `4cd6690` |
| 13 | Quantify the f32-vs-f64 precision floor | `4cd6690` |
| 11b | High-precision output overlay (`ES24.16`) | `6377d23` |
| 11c | Pinned toolchain, `-ffp-contract=off`, `TOOLCHAIN.txt` | `6377d23` |
| 12b | `nmts > 1` case | `db0dfad` |
| 14 | `--dump-budgets` overlay | `474af8f` |
| 15 | Per-process state-snapshot overlay | `888922b` |
| 15b | Branch-mask dump overlay (gate 0) | this commit |

**Measured precision floor: 3.7e-4** over a 48-step run — not the ~1e-6 the plan
assumed, roughly 370x larger. So `ref-f32` is useless as a validation target for
a float64 port and `ref-f64` is the only meaningful reference. It also
independently confirms that gating a 24-hour trajectory at 1e-9 was never
achievable. Now recorded in `docs/porting-notes.md`, which task 13's acceptance
criterion asked for and which was missed at the time.

**Gate 0 findings (task 15b).** The branch dump is the first instrumentation
that says anything the trajectory cannot, and three results change later work:

* **UP-1 is more reachable than its write-up claimed.** The factor-3 branch
  fires every substep of every shipped namelist, for the top *soluble* mode, in
  the default 4-mode setup — not only for the insoluble modes. Its fidelity flag
  must default to reproducing the defect.
* **UP-4 is confirmed unreachable by observation**, not only by argument. It
  gets an invariant test, not a flag.
* **The shipped fixtures reach only 4 of `ukca_solvecoagnucl_v`'s 8 branch
  codes**, and never reach the `MDCPNEW < 0` reset, the undersize diameter
  reset, or any mode merge at all. Those cannot be validated from a trajectory
  fixture and need constructed inputs — task 64 for coagulation, and an open gap
  for remode ahead of phase I.

Also found: `ukca_calc_drydiam` runs **five** times per chemistry step, not the
four in the splitting diagram — `glomap_box_state_mod`'s `update_size` calls it
once more from the driver.

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

Remaining: 16-23 and 20b.

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
