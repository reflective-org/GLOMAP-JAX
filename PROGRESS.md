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

## Phase B — reference harness: **in progress (17/18)**

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
| 15b | Branch-mask dump overlay (gate 0) | `06d5699` |
| 16 | `capture_reference.py` with `--mode` dispatch | `16dcb0f` |
| 17 | Golden manifest drift/orphan gate | `76c87a6` |
| 18 | Fixture size / Git-LFS ADR | `77f9f5d` |
| 19 | Commit the reference fixtures | `687d0b3` |
| 20 | f2py wrapper + in-process binding | `1d060b8` |
| 21 | Leaf reference-driver pattern + numerics driver | `efd13e7` |
| 22 | Document the harness; record all upstream defects | `a1c3859` |
| 23 | Draft the upstream write-ups for Ali to file | this commit |

**Measured precision floor: 3.7e-4** over a 48-step run — not the ~1e-6 the plan
assumed, roughly 370x larger. So `ref-f32` is useless as a validation target for
a float64 port and `ref-f64` is the only meaningful reference. It also
independently confirms that gating a 24-hour trajectory at 1e-9 was never
achievable. Now recorded in `docs/porting-notes.md`, which task 13's acceptance
criterion asked for and which was missed at the time.

**But 3.7e-4 is a setup-1 number, not a global one** (found at task 19, issue
#14). Re-derived from the committed fixtures it is 3.7e-4 / 1.0e-3 / 2.9e-4 for
the three `i_mode_setup = 1` cases and **0.80** for `marine_bcoc`, where ageing
depletes the Aitken insoluble mode over four orders of magnitude and f32 loses
the residual: `Ddry_aitins` collapses from 30 nm to 5.8 nm and `N_aitins` stops
decaying and turns back upward. The branch dump shows this is cancellation and
not a flipped predicate — only 60 of 107,664 branch records differ, all from
step 45, while the trajectory diverges continuously from step 20. First use of
gate 0 to *exclude* a branch explanation, which is worth as much as confirming
one. The f64 reference is well behaved throughout, so the port is unaffected.

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

**Gate A reaches bit-identity (task 20).** The in-process binding, built from
the *plain* vendored tree, reproduces the committed goldens — captured from the
*fully patched* stage — to 0.0e+00 relative difference on every field. That is
three confirmations in one: the wrapper's transcription of the driver is
faithful, the `ES24.16` overlay round-trips float64 losslessly, and the four
overlays really are instrumentation and not science. The meson/ninja blocker is
gone; all four of the plan's f2py blockers turned out to be real and are
documented in `docs/REFERENCE_BUILD.md`.

**The sleeper risk is dead (task 21).** The numerics leaf sweep — 15,382 points
through the Fortran itself — finds `erf` **bit-identical** between gfortran and
JAX, so the merge/no-merge flip the plan feared in `ukca_remode` cannot happen
via erf. `log` and `1/x` are bit-identical too; `exp` differs by one ulp on 14%
of points, inside tolerance. Task 34 shrinks to three specific rules, all now
asserted: write the cube root as `x ** (1.0/3.0)` (`np.cbrt` disagrees on 94% of
the grid by up to 1.3e-14), never use `jnp.round` (every `NINT` tie disagrees,
and the live consumer indexes a lookup table), and `powr_v` takes a scalar
exponent. Plus one hazard the plan did not have: **XLA flushes subnormal
arithmetic results to zero** while gfortran does not (issue #15, latent).

**The defect record is now mechanical (task 22).** Each of UP-1…UP-10 declares a
disposition — `fidelity-flag: X`, `invariant-test`, `not-implemented`,
`harness-patch: F` or `documentation-only` — and
`tests/test_upstream_defects.py` enforces every row against the code. It
immediately found that UP-4 had *both* a fidelity flag and an
UPSTREAM_DEFECTS entry saying it gets an invariant test instead: two documents,
each internally consistent, contradicting each other, with nothing comparing
them. The flag is removed (its two settings were bit-identical, so no
both-settings test could ever have existed) and replaced by an invariant
asserted over the committed branch-dump goldens.

`docs/harness.md` maps the four gates, what each one catches, and — the part
that is easy to leave implicit — what each one cannot.

**Fixture size (task 16, and most of task 18's answer).** The complete golden
set — 4 cases x 4 modes, at the namelists' own 48 steps — is **0.80 MB** as
compressed `.npz`, against roughly 70 MB of CSV from the reference. The state
dump is the bulk of it at 318k rows per case, and compresses to ~0.15 MB once
its `site`/`field` labels are integer codes rather than repeated strings. Git
LFS is not warranted — recorded as **ADR-007**, with the per-file (5 MB) and
whole-set (25 MB) budgets asserted in `tests/test_goldens_manifest.py` so the
decision is re-opened by a failing test rather than by someone noticing. The
likely trigger is a multi-box capture, where every stream scales with `nbox`.

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

Remaining: 20b, then the phase-B review.

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
