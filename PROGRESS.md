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

Fixed in `b1d8f54`:

* **The float64 justification was factually wrong**, repeated in five places
  including a test that could not fail. 1e-20 is a *normal* float32 number and
  1e-40 is subnormal but non-zero; the reference Fortran runs in single
  precision and branches correctly. Replaced with the real reasons.
* **`s0g` was sized by the wrong axis** — it is `nadvg` (advected tracers), not
  `nchemg`. Large enough today by luck.
* **"nbudaer takes eight distinct values"** — seven.
* **UP-10 confirmed and was missing entirely**: `ukca_conden.F90:372-387` gates
  insoluble-mode condensation with `num_eps` indexed by the *soluble* mode,
  wrong by 1e6 and results-changing on setup 8. (The phase B review later
  corrected the mechanism: the live line is `:377`, not `:387`, and setup 8 does
  not have `mode_sup_insol` active at all.)
* **`docs/UPSTREAM_DEFECTS.md` was an empty stub** that shipped code cited, and
  `PROVENANCE.md` claimed the defects were "reported upstream" when they are
  drafted and unfiled.
* **Two tests could not fail**, one ending in `or True`.
* **`ibln`/`icondiam`/`imerge`/`ifuchs`/`idcmfp` were unvalidated** despite
  `ModelConfig` citing the very ereport that covers `ibln`.

## Phase B — reference harness: **complete (18/18)**

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
| 23 | Draft the upstream write-ups for Ali to file | `9ed31fa` |
| 20b | `ereport` shim + one-process-per-setup harness | `4c84059` |
| — | **Phase B adversarial review**, findings applied | `b53b871`…`2f8c1d4` |

**Measured precision floor: 3.7e-4** over a 48-step run — not the ~1e-6 the plan
assumed, roughly 370x larger. So `ref-f32` is useless as a validation target for
a float64 port and `ref-f64` is the only meaningful reference. It also
independently confirms that gating a 24-hour trajectory at 1e-9 was never
achievable. Now recorded in `docs/porting-notes.md`, which task 13's acceptance
criterion asked for and which was missed at the time.

**But 3.7e-4 is a setup-1 number, not a global one** (found at task 19, issue
#14). Re-derived from the committed fixtures it is 3.7e-4 / 1.0e-3 / 2.9e-4 for
the three `i_mode_setup = 1` cases and **0.80** for `marine_bcoc`, where ageing
depletes the Aitken insoluble mode over seven orders of magnitude and f32 loses
the residual: `Ddry_aitins` collapses from 30 nm to 5.8 nm and `N_aitins` stops
decaying and turns back upward. The branch dump shows this is cancellation and
not a flipped predicate — only 60 of 108,432 branch records differ, all from
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
*fully patched* stage — to 0.0e+00 relative difference on every field compared,
which is 15 of the trajectory's 39 columns on one row of one case. That is
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
the grid by up to 1.3e-14), never use `jnp.round` (64 of the 129 `NINT` ties disagree,
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
set — 4 cases x 4 modes, at the namelists' own 48 steps — is **0.94 MB** as
compressed `.npz`, against roughly 70 MB of CSV from the reference. The state
dump is the bulk of it at 321k rows per case (367k for `bl_nmts3`), and
compresses to 0.16-0.28 MB once
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

All 18 tasks committed, and the adversarial review of the phase diff against
the Fortran is done — findings applied in `b53b871`…`2f8c1d4`, the rest filed
as issues #6–#11 and #16–#18.

## Phase C — mode tables and indices: **complete (10/10)**

| # | Task | Commit |
|---|---|---|
| 24 | Capture mode tables, all 7 setups | `6599091` |
| 25 | Port `modes.py` for setup 1 | `6442f24` |
| 26 | Setups 2, 3 | `4bdd755` |
| 27 | Setups 4, 5 | `4bdd755` |
| 28 | Setup 6 (dust-only) | `4bdd755` |
| 29 | Setup 8 | `4bdd755` |
| 30 | Density/hygroscopicity switches, both settings | `d2d3fbf` |
| 31 | Gas-phase index tables | `bf7eb5c` |
| 32 | Budget index map + ADR-008 | `bed9160` |
| 33 | `coag_mode` table + mask carriers | `3c5aaf4` |

**182/182 field comparisons byte-equal** across all seven setups —
`array_equal`, not `allclose`.

Literals are **machine-extracted** from `ukca_mode_setup.F90`, never retyped:
seven setups times eighteen tables is 1,351 numbers, and a mistyped digit
gives plausible tables and a quietly wrong model. Same convention as
`core/constants.py`. Everything derived is recomputed, which is what makes it a
port rather than a copy.

Four ways algebraically-identical code gave a different answer, each found by
byte equality and each now pinned as its own test:

* `d**3` vs `d*d*d` — gfortran expands an integer literal exponent to repeated
  multiplication, numpy calls `pow()`. One ulp apart on two of eight modes.
* Factor order in the mass products — factoring out the shared
  `(pi/6)·(rhommav·avogadro)·x` reassociates and breaks all three.
* Switch ordering — `rhocomp` is patched by `l_fix_nacl_density` *before* the
  masses derive from it; applying it after leaves the coarse soluble mode 35%
  out.
* `no_ions` needs **both** switches — `l_fix_nacl_density` only reaches it when
  `l_fix_ukca_hygroscopicities` is also on. Reading it as an independent knob
  selects the default branch and gets all seven setups wrong identically.

Tasks 31–33 ran as three agents in separate git worktrees and merged with
one-line conflicts in `Makefile`, `build_f2py.sh` and `test_goldens.py`. All
three came back byte-equal on the first comparison. **1125 tests pass.**

### The index tables are byte-equal, but two of them should not have been captured at all

Two agents independently hit the same class of defect, now issue #19: **module
integers that no dispatched routine ever assigns, read by live guards.**

The 38 `nmas*mp*` budget indices are assigned only by
`ukca_indices_sussbcocdump_8mode`, which the box dispatch at
`glomap_box_config_mod.F90:371-390` never calls — while its *mode*-setup
namesake `ukca_mode_sussbcocdump_8mode` **is** dispatched, which is what makes
it easy to miss. 51 live `IF (nmas...mp... > 0)` sites read them. On the gas
side, `budget`, `nbudget`, `traqu`, `ntraqu` come only from
`ukca_indices_traqu38`/`traqu9`, called from nowhere, and `idustdep`,
`ndustdep`, `nbudaertot` are assigned nowhere at all.

gfortran zeroes `.bss`, so the guards are false and the code works. The
standard promises nothing. **Task 31's acceptance criterion named `budget` as a
table to capture** — doing so would have committed uninitialised memory as a
reference golden, stable enough across runs to look correct.

### What each table turned out to be

* **Gas indices collapse to four routines, not seven** — 1→`sv1`, 2/3/8→
  `orgv1_soto3`, 4/5→`orgv1_soto6`, 6→`nochem`. "The tables vary across setups"
  is false, so a variation check written that way is vacuous; `soto3` and
  `soto6` differ in exactly one number in the entire table. Also: `msotwo = 1`
  in Fortran, so index **0** after conversion — carrying the `IF (mxxx > 0)`
  idiom across silently drops SO2. And four live indices (`msec_orgi`, `mh2o2`,
  `mhno3`, `mnh3`) are 0 in every supported setup, so the isoprene-SOA block,
  `ukca_aero_step`'s `IF (mh2o2 > 0)` branch and both nitrate modules have **no
  reference in any validatable configuration**.
* **Budget indices**: seven setups, seven distinct `nbudaer` (8, 46, 89, 104,
  107, 123, 138). All 344 `bud_aer_mas` write sites are guarded on their own
  index, so slot 0 is a hole — asserted at source level, not just empirically,
  because no golden reaches every site. **4 of the 344 overwrite rather than
  accumulate**; routing those through an accumulate helper would turn a
  per-step flux into a running total.
* **`coag_mode` is setup-independent**, measured rather than grepped: read with
  no init at all, then once per setup before and after `wrap_init`, all
  byte-equal. It is also **symmetric on all 64 entries**, so a transposed
  transcription would be byte-equal to the correct one — neither the source
  parse nor the capture can catch it, and the `(imode, jmode)` order rests on
  the call site's subscripts alone. Recorded in a test named for the gap.

### ADR-008 — budget indices are traced

Measured, not argued (CPU, float64, 344 sites, `nbudaer = 138`): static
sequential 13.66 ms at `nbox = 1024` against traced sequential 0.38 ms; static
grouped-and-stacked 0.145 ms against traced fused 0.371 ms. All four
bit-identical, all four leave slot 0 zero. So static *is* faster — 2.6–3.7x —
but only in the grouped form, which is exactly the form that needs the index
map at trace time, i.e. the recompile-per-setup ADR-002 forbids. Seven static
compilations cost 4.06 s; the gap is 0.23 ms/step on a diagnostic. Traced.

The sentinel is `NOT_CARRIED = 0`, not `-1`: `.at[-1].add()` **wraps to the
last element under every scatter mode** including `drop` and `clip`, so a `-1`
sentinel would silently accumulate every uncarried flux into the highest budget
slot. Out-of-range is the benign case; in-range-but-wrong is not.

**Task 30** captures eight switch combinations per setup — 56 golden records —
and every one is byte-equal. The switches split two ways, which is worth
stating because issue #10 lumped them together: `l_fix_nacl_density` selects
between a wrong number and its correction, so it is a fidelity flag;
`l_radaer`, `i_tune_bc` and `l_dust_mp_ageing` are model configuration and
belong to `ModelConfig`.

Two behaviours pinned there. `i_tune_bc` has no `CASE DEFAULT`, so an
out-of-range value silently leaves `rhocomp(cp_bc)` at its literal instead of
failing — reproduced, not corrected, and captured as the `bc_oob` golden. And
`i_tune_bc` is inert unless `l_radaer` is on, which the box model defaults off,
so BC density tuning is unreachable by default.

### Phase C review

Four adversarial agents, one per dimension: the ported data against a
from-scratch re-derivation, tests that cannot fail, every factual claim, and
the capture harness. Roughly 60 findings; the fixes ran as four more agents in
parallel worktrees. **1330 tests pass**, CI green on ubuntu x86_64 and macOS
arm64.

**The data itself is right.** An independent re-derivation — its own parsers,
not this repo's extractors — found zero mismatches on 7 setups x 18 mode-literal
tables, 4 gas routines x 175 fields, 7 setups x 283 budget names, and
`coag_mode`. Nothing derived is copied: `_mode_literals.py` holds only what is
a literal in the Fortran, and no module under `src/` opens a `.npz`. Float
ordering, switch ordering and index-base conversion all clean.

What the review was for is everything around it.

**One finding would have changed a number.** `coag_mode.py` said up to eight
`mtran` terms reach `mtrantoi[:, 3, icp]` but "four in any configuration the
box model can actually run", and a test pinned that. `l_dust_mp_ageing` is a
`box_aerosol` namelist variable that `validate_config` does not constrain, so
eight is reachable — and the docstring listed that very row four paragraphs
above, under "reachable configurations". Unrolling four adds instead of eight
drops four terms. Third wrong reachability claim in this project, same root
cause each time: asserting what is reachable without measuring it.

**Verification that could not fail.** Mutation testing, 34 mutations, each
reverted: three staleness gates passed when stubbed to compare nothing, and
they guard 4,155 lines of generated literals. The gas anti-collapse guard —
"the check that catches a capture which silently ran one setup seven times" —
keyed on `ntraer` and `nbudaer`, the only two of 176 scalars that differ
between `soto3` and `soto6`, and both mode-side. The four tests pinned "so a
simplification fails with a reason" never called the function they named, so
all four simplifications left them green while reddening 56-77 others. All
fixed, each proven by watching the mutation go red.

**A third evasion of the additive-only gate.** The awk arms itself on a `+++`
header, which exists only in unified diffs, and `patch` was invoked with no
format flag. A context diff and a normal diff each deleted a science line from
`src/ukca/` with the gate exiting 0. The gate now refuses anything that is not
a unified diff, `patch -u` stops auto-detection, and — this is the part that
had been missing through three rounds of hardening — the gate has tests, ten
of them, unmarked so they run in CI.

**Byte equality is a property of a platform pair.** Six leaf comparisons failed
on x86_64 against goldens captured on arm64: `erf` by up to 4 ulp, the powers
by 1. The job that found it has no gfortran, so it was comparing this
platform's JAX against another platform's Fortran — which
`goldens_manifest.py`'s own docstring has called invalid since it was written,
while `build_reference.sh` recorded `uname` all along and nothing read either.
Worse, a same-machine gate is not reproducible either: `linux-reference`
passed, failed and passed again across three runs of identical code, because
XLA-CPU lowers f64 `erf` and `pow` to the host libm and glibc's path depends on
the runner's CPU. Bit equality is now required where the goldens were captured
and bounded elsewhere, with the bounds measured per primitive, and the Linux
job reports the achieved agreement instead of asserting one.

**The leaf sweep's own inputs went through libm.** The grids were built with
`np.logspace`, which is `10.0 ** linspace(...)`, so four abscissae were a ulp
off the correctly-rounded value — the *sample points* were platform-dependent,
not just the results. Decimal literals and integer cubes now.

**Documentation.** Thirteen wrong claims, eight stale, six cross-document
contradictions. The float32 underflow argument that ADR-001 has recorded as
false since phase A survived its **fourth** review, in `CLAUDE.md`. "`np.cbrt`
disagrees by a hundred times `RTOL_ALGEBRAIC`" was in four places and is 0.13
times it — the rule stands on the branch flip, not the magnitude. `CLAUDE.md`
credited two guarantees to tests that do not exist. The UP-defect counts
disagreed three ways; they are derived from the table now, and prose that
drifts fails.

Filed rather than fixed: #19 (uninitialised index variables), #20 (dead
`ukca_mode_allcp_4mode`, where citations drift), #21 (ADR-008's benchmark is
not committed, so its table cannot be re-derived).

## Phase D — size, water and volume: **9/12**

| # | Task | Commit |
|---|---|---|
| 34 | Transcendental/rounding compat layer | satisfied by task 21 |
| 35a | Four physics leaf drivers + shared capture machinery | `98847f0`, `a9c124a` |
| 35b | Vapour fixture | `e93738e` |
| 35d | Drydiam fixture | `fb13a08` |
| 35e | `volume_mode` fixture | `task-35e` |
| 36 | `calc_drydiam` core | `6e19c0d` |
| 37 | Undersize reset, modes 1–3 | `6e19c0d` |
| 38 | `ukca_vapour` | `7c02d33` |
| 39 | The two ZSR ion tables | `3b858ec` |
| 40 | `water_content_v` | `9fd9cbb` |

Remaining: 35c (water fixture, in flight), and 41–45 (`volume_mode`, strictly
serial on one file).

**Every port is byte-equal to the compiled routine**, not to a tolerance. Three
of the plan's acceptance criteria were unfailable as written and were tightened
before any code was accepted against them: task 36's `RTOL_ALGEBRAIC` is 1e-13
while `jnp.cbrt` differs from `x**(1.0/3.0)` by at most 1.3e-14, so a port using
the forbidden cube root passed it; task 38's `RTOL_TRANSCENDENTAL` covered a
live path containing only `LOG` and `SQRT`; task 40's covered a routine with no
transcendental at all.

### Three findings that changed the work

**`jit` is not byte-equal to the reference.** XLA contracts `a*b + c` into an
FMA; the Fortran is built `-ffp-contract=off`. 23.4% of random triples differ,
and every differing value is the correctly-rounded FMA — checked exactly with
`Fraction`, so this is contraction and not fast-math reassociation. On the ZSR
polynomials the gap reaches 4.8e-11 against `RTOL_JIT_VS_EAGER = 1e-14`, a
constant that was a plausible tightness rather than a measurement. Order 1 is
unaffected because every gate runs eager; order 2 cannot claim jit parity until
#23 is settled.

**`l_fix_ukca_water_content` is off-limits after `init`.**
`glomap_box_config_mod.F90:322` hardcodes it `.TRUE.` — it is not a namelist
variable — and `init_ukca_for_box` then runs `init_state` → `volume_mode` →
`water_content_v`, which patches its own `SAVE`d table in place and never
restores it. The latch has fired before `wrap_init` returns, so setting the
flag afterwards changes the flag and not the table: measured, the flag reads
back 0 and the routine still returns the fixed answer. Task 40's both-settings
acceptance was unsatisfiable by the obvious route and would have passed
vacuously. Reachable only by not calling `wrap_init` at all, which that routine
uniquely permits. Issue #22.

**gfortran's `MAX` does not propagate `NaN` here, and that is load-bearing.**
At the one temperature where `ukca_vapour`'s Ayers denominator is *exactly*
zero, `xsb` is `0/0` and everything downstream is `NaN` until
`MAX(41.0, ws*100)` — which returns 41.0 where `jnp.maximum` gives `NaN`. Not a
language property: measured 41.0 at `-O0` through `-O3` under this project's
flags, but a probe without `-fdefault-real-8` returned `NaN` at `-O2`. What
`numerics.fortran_max` reproduces is `TOOLCHAIN.txt`.

**UP-11 is a crash, not a truncation.** Writing the five-way negative-size
diagnostic into its 256-character buffer gives "Fortran runtime error: End of
record" and exit 2 at `:876`, so `ereport` at `:877` never runs. Reproduced
standalone and in the model. The block exists to say which mode went
non-positive and where; it aborts naming neither.

### Coverage the fixtures bought

Three `volume_mode` branches had never executed in any validated run and now
have reference data: the stratospheric override (all four namelists run above
`putls`), `mask_nosol` (0 hits in 2447 sampled golden points), and the
relative-humidity clamps. The undersize reset is 0 of 3456 `undersize` records
across all four branch dumps and is now reached by constructed inputs.

## Phases E–K — physics: not started

## Orders 2 and 3: not started

## Verified along the way

| check | result |
|---|---|
| CI | green on lint, 3.11, 3.12, macOS |
| tamper test | fails and names the file when a vendored source is edited |
| fidelity registry | fails on a flipped default and on an undocumented flag |
| tolerance floor | permits 0-vs-1e-300, still catches a 10% discrepancy |
| f2py mechanism | `SAVE`d module state and `INTENT(IN OUT)` verified end-to-end |
