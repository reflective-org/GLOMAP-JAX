# Key decisions (ADRs)

Decisions with a rationale, so they are not re-litigated. Each states what was
decided, why, and what evidence would change it.

## ADR-001 — float64 only; float32 is not a supported trajectory mode

**Decided.** `jax_enable_x64` is set once, in `src/glomap_jax/__init__.py`.

This is correctness, not accuracy — but **not for the reason first given
here**. The original justification claimed float32 flushes `eps_d = 1e-40` and
`num_eps = 1e-20` to zero so a different branch executes in
`ukca_solvecoagnucl_v`. That is false: `1e-20` is a normal float32 number and
`1e-40` is subnormal but non-zero, and running `ref-f32` and `ref-f64` against
the branch dump gives **identical** form-code histograms. The phase A review
recorded this claim as wrong; it survived here until the phase B review found
it again.

The real reasons are measured rather than argued:

* **The f32 reference is a different trajectory, not a noisier one.** For
  `marine_bcoc` the column-scaled f32-vs-f64 gap is **0.80** — ageing depletes
  the Aitken insoluble mode over seven orders of magnitude, f32 loses the
  residual, and `Ddry_aitins` collapses from 30 nm to 5.8 nm while `N_aitins`
  stops decaying and turns back upward. Issue #14.
* **The measured floor is 3.7e-4** even away from that, four orders of
  magnitude above any tolerance worth gating on.
* `carma-jax` reached the same conclusion independently for its own
  `SMALL_PC = 1e-50`, where the underflow argument *does* hold.

float32 is permitted only as an explicitly labelled benchmark mode.

## ADR-002 — one compiled kernel for all seven mode setups

**Decided**, with a caveat to resolve in phase C.

`nmodes = 8` and `ncp_max = 10` are Fortran PARAMETERs, and all seven supported
setups have `ncp = 6`, so arrays are allocated at full extent and masked. What
varies is `nchemg`, `nadvg` and `nbudaer`; those are padded to their maxima.

**Settled in ADR-008: traced.** The 283 `nmas*` budget index scalars are
per-setup integers, so they were either static config — recompiling per setup,
contradicting the above — or traced, making every budget write a dynamic
scatter. Measured in phase C. The recommendation above turned out to be right
for the wrong reason: static is not slower, it is 2.6–3.7x *faster*, but only
in the grouped-and-stacked form, which is precisely the form that needs the
index map at trace time. See ADR-008 for the numbers.

## ADR-003 — `jax-metal` is out of scope

**Decided.** Incomplete op coverage and weak-to-absent float64, which ADR-001
rules out on its own. Order 3 targets NVIDIA CUDA on cloud or cluster hardware,
and specifically datacenter parts: float64 is ½ FP32 rate on A100/H100/V100 but
**1/32 on consumer RTX 30xx/40xx**, so consumer-card numbers would be
meaningless. Tracked as issue #4.

## ADR-004 — diffrax is order 2 only, scoped to the `nzts` interior

**Decided.** The faithful path uses no ODE library.

`ukca_solvecoagnucl_v` is an *exact closed-form integral*, not a numerical step,
so a solver would be slower and less accurate. `aer3d` and `vbs-jax`, the two
newest faithful ports in this family, use no ODE library at all; `tomas-jax`
abandoned `Tsit5` for forward Euler because "higher-order methods amplify N²
coagulation rates", which is exactly this system; and `mam4-jax` had to relax
its acceptance bar from `1e-6` to "3% over 24h" to adopt diffrax.

Where diffrax *does* fit: inside the `nzts` loop, sizes and coagulation kernels
are never updated, so that block is a genuine constant-coefficient ODE. It
cannot absorb the `nmts` boundary, where `ukca_remode` is a discontinuous remap
and the `deltas0g` reconciliation is a projection.

**Would change this:** a diffrax configuration that matches the faithful path to
`RTOL_STEP` on the shipped cases.

## ADR-005 — goldens are generated once on a pinned toolchain

**Decided.** `make goldens` is never run in CI.

The Fortran reference is not bit-reproducible across platforms: the vendored
Makefile leaves `-ffp-contract` at gfortran's default `fast`, so FMA contraction
differs by architecture and optimisation level, and `EXP`/`LOG`/`ERF` come from
the platform libm. A CI job regenerating goldens would fail the drift gate on
the first PR from another machine, and the gate would then get loosened — which
the tolerance policy forbids.

Goldens are committed with the compiler version, flags (including
`-ffp-contract=off`), OS and architecture recorded in the manifest.

## ADR-006 — consistency routines are mostly not ported

**Decided.** `ukca_check_md_nd` declares its state arguments `INTENT(IN)` and
only prints, so omitting it is exact. `ukca_mode_check_artefacts` has no caller
in the box model. `ukca_mode_check_mdt` is gated on `iextra_checks > 1`, which
the box model leaves at 0, and it mutates state, so it raises rather than being
silently skipped.

The one always-active state mutation, `ukca_calc_drydiam`'s undersize reset, IS
ported — see `drydiam_undersize_reset` in `fidelity.md`.

## ADR-007 — goldens are committed as plain files; no Git LFS

**Decided.** `tests/goldens/*.npz` go into the repository directly.

The concern was real before it was measured: 283 budgets × 48 steps × several
namelists × 13 hook points is hundreds of megabytes captured naively, and the
per-substep state dump is the largest stream by an order of magnitude. So the
question was settled with numbers rather than a guess.

Measured, at the namelists' own 48 steps:

| | one case, as CSV | full set, as `.npz` |
|---|---|---|
| trajectory | 0.05 MB | |
| budgets | 0.13 MB | |
| state | 15.90 MB | |
| branches | 3.82 MB | |
| **total** | **19.89 MB** for `marine_bcoc` | **0.78 MB** for all four cases and all four modes |

A factor of about 99 measured across all four cases (76.8 MB of CSV against
0.78 MB of `.npz`); the single-case row above is `marine_bcoc`, the largest.
Three things do the work, and all three are properties of
this data rather than of compression in general: the long-format dumps repeat a
small vocabulary of `site`, `field` and `tag` labels over hundreds of thousands
of rows, so factorising them into integer codes removes most of the bytes before
zlib sees them; branch values are 0/1 and 0–7 and are stored as `int8`; and the
state dump's float64 values contain long runs of exact zeros for inactive modes
and components.

LFS would buy nothing at this size and cost a great deal: contributors need a
`git lfs install` step, CI needs LFS quota, and a shallow clone without LFS
gives pointer files that fail as *corrupt fixtures* rather than as missing ones
— the worst possible failure mode for a validation gate.

**Would change this:** any single archive above 5 MB, or the set above 25 MB.
Both are asserted in `tests/test_goldens_manifest.py`, so the decision is
re-opened by a failing test rather than by someone noticing. The likely trigger
is not more namelists — it is capturing a multi-box case, where every stream
scales with `nbox`.

## ADR-008 — budget indices are traced data, not static config

**Decided.** The 283 `nmas*` slot indices are carried as an `int32` array and
every budget write is a masked scatter into a fixed-width array. They are not
Python integers baked into the trace. This settles the "Open" item in ADR-002.

The port's map is `physics/budget_indices.py`, machine-extracted from
`ukca_setup_indices.F90` and byte-compared against
`tests/goldens/budidx.f64.tables.npz`, which was read out of the compiled
Fortran one subprocess per setup.

**What is actually being decided.** `bud_aer_mas(nbox, 0:nbudaer)` is written
at 344 sites across 13 source files. (684 is the count of `bud_aer_mas(...)`
index expressions in executable code — 344 on the left of an assignment and 340
reading the slot back to accumulate into it — which is what
`test_budget_slot_zero_is_never_written` calls "~684 writes".) Each site names
one of the 283 scalars, whose value is a
function of `i_mode_setup`. Static means seven traces of every process routine
that touches a budget; traced means one, with the map as an argument.

Measured on this machine (CPU, float64, `nbudaer = 138`, all 344 sites, 20-50
reps after a warm-up call):

| form | nbox=1 | nbox=1024 | nbox=16384 | compile (nbox=1024) |
|---|---|---|---|---|
| static, 344 sequential `.at[int].add` | 0.122 ms | 13.66 ms | — | 0.61 s |
| traced, 344 sequential `.at[idx].add` | 0.098 ms | 0.38 ms | — | 1.08 s |
| static, grouped by slot + one stack | 0.040 ms | 0.145 ms | 1.88 ms | 0.73 s |
| traced, one fused scatter | 0.006 ms | 0.371 ms | 6.92 ms | 0.02 s |

All four agree **bit for bit**, and all four leave slot 0 exactly zero.
Under `vmap` over 1024 boxes the same split holds: 15.99 ms static against
0.49 ms traced for the sequential form. Seven static compilations cost 4.06 s.

Three things that table says, none of them the one the plan assumed:

* **"Static is marginally faster" is not what was measured.** In the form a
  port would naturally write — one guarded update per site, mirroring the
  Fortran — traced is **36× faster** at nbox=1024. Why is not established:
  both forms lower to 344 scatters, and the plausible explanation is that XLA
  fuses the dynamic ones better, but that is a hypothesis and not a
  measurement.
* **Static wins only in its best form**, grouping the 344 deltas by slot and
  building the column stack in one go: 2.6× at nbox=1024 and 3.7× at
  nbox=16384 over the fused traced scatter. That form needs the map at trace
  time, so it is exactly the form that costs a recompile per setup.
* **The absolute numbers are small and this is a diagnostic.** The gap between
  the best static and the best traced form is 0.23 ms per step at nbox=1024.

**What could not be measured, and is not claimed.** No process routine is
ported yet, so the budget scatter cannot be quoted as a *fraction* of a step.
Everything above is CPU; XLA's scatter on CUDA is a different implementation
and this has to be re-measured there.

**The sentinel, which is not a performance question.** The Fortran dimension is
`0:nbudaer`, so the slot number and the 0-based column index are the same
integer and no rebasing is needed. That matters because the alternatives are
unsafe rather than merely awkward: `jnp.zeros(5).at[-1].add(1.0)` **wraps to
the last element**, so a −1 sentinel would silently accumulate every uncarried
flux into the highest budget slot. Measured, that wrap happens under *every*
scatter mode — `drop`, `clip` and the default `promise_in_bounds` alike — so it
is not something a mode flag fixes; an out-of-*range* index is the benign
case, dropped by default and only clamped into a real slot under `clip`.
`NOT_CARRIED = 0` keeps the sentinel inside the array, pointed at the one
column the reference never writes, and the mask does the rest: uncarried sites
scatter a bit-exact `0.0` into slot 0. Masked with
`jnp.where`, never `mask * delta` — an uncarried delta is exactly where an
`inf` can sit, and `0.0 * inf` is `NaN`.

**A defect this forced into the open.** Each `ukca_indices_*` routine assigns
245 of the 283 scalars. The other 38 are the `nmas*mp*` family, assigned only
by `ukca_indices_sussbcocdump_8mode`, which no supported setup dispatches to —
so in all seven they are **read without ever having been assigned**, 34 of them
from a live `IF (nmasxxx > 0)` guard. Module scalars have static storage, so
gfortran hands back a `.bss` zero and the guard is false; the standard promises
nothing. The capture measured 0 for all 38 in all seven setups, and the port
defines them as 0 explicitly rather than inheriting the compiler's answer.

**Would change this:** a measurement showing the budget scatter above ~5% of a
step at the `nbox` the port actually runs, once there is a step to measure it
against; or a CUDA measurement where the fused scatter is more than ~5× the
grouped-static form. Either would justify a per-setup specialisation of the
budget writes *only* — the process routines would stay shared, because 7×
recompiling those for a diagnostic is what this ADR is refusing. A run pinned
to a single setup can already take the grouped form without changing anything
here, since the map is available at trace time whenever the caller wants it.
