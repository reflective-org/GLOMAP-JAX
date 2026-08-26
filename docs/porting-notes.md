# porting-notes

Facts measured against the reference, in the order they were established. Each
one is here because assuming otherwise would have cost time later.

## The f32-vs-f64 precision floor is 3.7e-4, not 1e-6

The shipped Fortran runs in **single** precision — default `REAL` is kind=4 and
the Makefile sets no `-fdefault-real-8`. Running the same 48-step case in both
precisions separates "the port is wrong" from "the reference is single
precision", and the gap over 48 steps of 1800 s is **3.7e-4**, roughly 370×
larger than the ~1e-6 first assumed.

Two consequences. `ref-f32` is useless as a validation target for a float64
port, so `ref-f64` is the only meaningful reference. And gating a 24-hour
trajectory at `RTOL_TRAJECTORY = 1e-9` was never achievable — that run is a soak
at `RTOL_SOAK`, and the primary gate is a bounded number of steps from a golden
state.

The floor is only interpretable **away from branch boundaries**. Near any of the
predicates below the f32/f64 gap is O(1), not 1e-4.

### …and 3.7e-4 is a setup-1 number, not a global one

Re-derived from the committed fixtures, the column-scaled f32-vs-f64 gap is
3.7e-4 for `boundary_layer`, 1.0e-3 for `free_troposphere` and 2.9e-4 for
`bl_nmts3` — all `i_mode_setup = 1`. For `marine_bcoc`, the only shipped case
with an insoluble mode, it is **0.80**.

Ageing depletes the Aitken insoluble mode over seven orders of magnitude across
the run. In f64, number and mass leave in proportion and the mean dry diameter
stays pinned near 30 nm while the number decays to 2e-5 cm⁻³. In f32 the
residual loses significance, mass leaves faster than number, `Ddry_aitins`
collapses from 30 nm to 5.8 nm, and `N_aitins` stops decaying and turns back
upward.

**It is cancellation, not a flipped predicate**, and the branch dump is what
makes that checkable rather than a hypothesis: of 108,432 branch records exactly
60 differ between the variants, all `mask4` at `coag_insol_insol`, all from step
45 onward — while the trajectory divergence is already visible at step 20 and is
continuous. The flips are a late symptom of an already-diverged state.

The f64 reference is well behaved throughout, so nothing here threatens the
port; `ref-f32` is diagnostic only by ADR-001. What it does mean is that for a
configuration with a depleting mode, the f32 run is not a slightly-worse version
of the f64 run — it is a different trajectory. Issue #14.

## What the branch dump shows (gate 0)

`validation/patches/0004-dump-branches.patch` records every predicate the
science selects a closed form on, per box and per substep. The numbers below are
from all three shipped namelists at their full 48 steps. The gaps matter more
than the hits.

### `ukca_solvecoagnucl_v` — half the branches are never reached

The routine selects among five closed forms on `|A|`, `sign(D)` and `|B|`; the
dump assigns each element an integer code (0 masked out, 1–7 the forms, with 2
being "A≠0, D<0 but `TERM3`/`TERM4` degenerate, so `NDNEW` is left at `ND`").
Observed across every shipped case:

| code | form | records |
|---|---|---|
| 0 | masked out | 1440 |
| 1 | `A≠0, D<0` | 5759 |
| 5 | `A≠0, D=0, B=0` — the factor-3 branch | 2160 |
| 7 | `A=0, B=0` | **1** |

Codes 2, 3 (`TAN`), 4 (the `ereport` branch) and 6 never run, and neither does
the `SQD·Δt > 50` clamp nor a `TAN` argument past π/2 (0 hits in 9360 records
each). They cannot be reached from a trajectory fixture, because a trajectory
fixture never visits them. Validating those forms needs the synthetic branch
sweep of task 64, driving the routine directly over `(A, B, C)` space.

Code 7's single occurrence is worth keeping: it is the nucleation mode at the
very first substep of `free_troposphere`, when nucleation has produced a source
term (`C > 0`) but the mode is still empty, so `A` and `B` are both zero and the
solution is the linear `ND + C·Δt`. It happens exactly once in 24 simulated
hours and never again. A port that got that one substep wrong would show up only
as a small constant offset in the whole trajectory.

### UP-1 is more reachable than the defect note claimed

Code 5 is `1/(1/N − 3·A·Δt)`, the branch with the spurious factor 3. The defect
note argued it fires for the top *insoluble* modes, where inter-modal
coagulation is skipped. The dump shows it also fires for the top **soluble**
mode, on **every substep of every shipped case**, including the default 4-mode
setup 1: for `mode_cor_sol` there is no larger soluble mode to coagulate with
and no nucleation source, so `B` and `C` are exactly zero and so is the
discriminant.

The fidelity flag for UP-1 must therefore default to reproducing the defect. A
"corrected" default would not be a subtle improvement; it would break every
trajectory gate in the suite.

### UP-4 is unreachable, now observed rather than argued

The `delgc_cond > gc` guard in `ukca_conden` was argued dead because
`delgc_cond = gc·(1 − exp(−x))` with `x ≥ 0` is bounded in `[0, gc]`. The dump
carries that predicate explicitly and it is 0 in all 2880 records. That is why
UP-4 gets an invariant test rather than a fidelity flag — there is nothing to
select between.

### Rare paths no shipped fixture exercises

Never observed, and therefore not validated by any trajectory golden:

Counts are over **all four** committed cases. Several were first written for
the three shipped namelists only, before `bl_nmts3` was added, and the smaller
denominators survived — the undersize reset was recorded as 0/2160, which is
3 x 720 and omits `bl_nmts3`'s 1296. Recomputed from the committed archives:

| predicate | tag | routine | records |
|---|---|---|---|
| `MDCPNEW < 0` → `ND = 0`, `mask1` falsified mid-`ICP` | `mdcp_neg` | `ukca_coagwithnucl` | 0 / 23760 |
| the undersize reset `DP < DDPLIM0·0.1` | `undersize` | `ukca_calc_drydiam` | 0 / **3456** |
| any mode merge at all | `merge_trigger` | `ukca_remode` | 0 / 1440 |
| the coagulation kernel's Fuchs corrections | `mask4`, `mask4i` | `ukca_coagwithnucl` | 0 / 19440, 0 / 14400 |
| UP-4's `delgc_cond > gc` guard | `up4_guard` | `ukca_conden` | 0 / 3600 |
| the `SQD·Δt > 50` clamp and the `TAN` pole | `sqd_clamp`, `tan_pole` | `ukca_solvecoagnucl_v` | 0 / 12240 each |
| `NTOT < 4`, and `T < 195.15` → flat `J = 1e5` | `ntot_lt_4`, `t_lt_195` | `ukca_binapara` | 0 / 2880 each |
| `l2` | `l2` | `ukca_calcnucrate` | 0 / 2880 |

The remode entries are the sharpest gap: mode merging is the highest-variance
part of the port, its mode loop is loop-carried, and no shipped case merges at
all. It needs constructed fixtures **before** phase I, not after.

## The transcendental compat layer, measured (task 21 → task 34)

The plan called this the sleeper risk, on the grounds that an `erf`
discrepancy becomes a merge/no-merge flip in `ukca_remode`.
`validation/capture_leaf.py` sweeps each primitive through the Fortran itself
over grids that land *on* each hazard, 15,382 points in total.

**First, a correction to that framing.** `erf` does not gate merging.
`ukca_remode.F90:234` decides whether to merge with
`IF ((dp > dp_thresh1) .OR. (imerge == 3))` — a bare comparison on `drydp`, no
`erf` anywhere. `erf` enters only *after* that decision, at `:245` and `:258`,
to compute how much number and mass to transfer. Its two thresholds,
`frac_n < 0.5` and `frac_m < 0.001`, are clamps that are **continuous at the
boundary** — either side gives essentially the same value — so a one-ulp `erf`
difference perturbs the result by one ulp, not by O(1). The one genuinely
discontinuous consumer is `newn > num_eps` at `:271`, and since
`newn = nd·frac_n` with `frac_n ≥ 0.5`, reaching it needs `nd` within a factor
of two of `num_eps` ≈ 1e-20, where the outer gate has only just opened.

So the merge/no-merge flip is real, but it belongs to **`drydp`**, which comes
from `cubrt_v` — and that reassigns the risk from `erf` to the cube root, which
is exactly where the measurements below say it should sit.

Results against JAX on the CPU backend, float64:

| primitive | agreement | verdict |
|---|---|---|
| `erf` (via `umErf`) | **bit-identical**, 4330/4330 (arm64) | no shim needed |
| `log`, `1/x` | bit-identical | safe |
| `x ** (1.0/3.0)` | bit-identical (arm64) | **this** is what `cubrt_v` computes |
| `exp` | 456/3199 differ, max 2.1e-16 | one ulp; inside tolerance but real |
| `np.cbrt` | 1763/1865 differ (arm64; 1793 on x86_64), max 1.3e-14 | **must not be used** |
| `NINT` vs `round` | 64/642 differ | **must not use `jnp.round`** |

Three rules follow, and all three are asserted in
`tests/test_numerics_reference.py` so they cannot be quietly broken later.

**Write the cube root as `x ** (1.0/3.0)`.** `cubrt_v` is literally that
expression; it is not a cube-root function. `np.cbrt` disagrees on 94% of the
grid by up to 1.3e-14, which is 0.13 times `RTOL_ALGEBRAIC` and not the
hundred times claimed here for three reviews. This is the one that
carries the branch risk the plan mis-attributed to `erf`: `cubrt_v` produces
`drydp`, and `drydp` is compared directly against `dp_thresh1` (merge or not)
and against `ddplim0·0.1` (rewrite `md`/`mdt` or not). Both are step changes,
so a parcel sitting within 1.3e-14 of either threshold goes one way in the
reference and the other in the port. The two also
disagree about negatives: `x ** (1.0/3.0)` is NaN, `np.cbrt` returns the real
root. Unreachable today (`dvol >= 0` wherever `cubrt_v` is called), but it is
the failure a `cbrt` port would produce the first time it wasn't.

**Do not use `jnp.round`.** Fortran `NINT` rounds half away from zero; numpy
and JAX round half to even. The grid holds 129 ties; **64 of them disagree and
65 agree**, because the two rules coincide whenever rounding away from zero
already lands on an even number (`-63.5 → -64` either way). Away from ties they
agree exactly, so a targeted shim suffices. The live consumer is
`ukca_vapour.F90:226`, `(NINT(wts/5))*5`, whose result *indexes a table*: at
`wts ∈ {42.5, 52.5, 62.5, 72.5, 82.5, 92.5}` the naive version selects a
different table entry, not a slightly different number.

**`powr_v` takes a scalar exponent.** It raises a whole array to one power; it
does not do elementwise pairs. An elementwise port would compile, run, and be a
different routine.

### A hazard the plan did not have: JAX flushes subnormals

XLA flushes the *result of any arithmetic operation* to zero when it would be
subnormal — eager and under `jit`, even for `x + 0.0`. gfortran and numpy
compute it. A subnormal *constant* survives conversion untouched, which is what
makes it easy to miss: the value is representable, it just cannot be produced.

Latent rather than live — `num_eps` bottoms out at 1e-20 and
`eps_d = eps_ab² = 1e-40`, both comfortably normal in float64. Note in float32
`1e-20` is *also* normal and `1e-40` is subnormal but non-zero — neither is
flushed, so this is **not** an argument for ADR-001, whatever earlier drafts
said. Recorded because the failure it
would cause is a zero where the reference has a small positive number, feeding a
`> eps` comparison, which separates trajectories by O(1) and would present as a
gate-0 disagreement with no arithmetic explanation. Issue #15; re-measure on
CUDA at task 3.7, where denormal handling differs again.

## `ukca_calc_drydiam` runs five times per step, not four

The splitting diagram lists four calls inside `ukca_aero_step`. The branch dump
counts five: `glomap_box_state_mod`'s `update_size` calls it once more per step
from the driver, outside `ukca_aero_step` entirely. Those records are tagged
`imts = izts = -1` so they are not mistaken for the tail of the step that just
ran. The port's driver has to reproduce that call.

## Four ways identical algebra gives a different double (phase C)

Porting the mode tables to byte equality took four corrections. None was a
mistake about the physics; all four are the same category of error, and all
four would have passed silently under a tolerance.

**`d**3` is not `d*d*d`.** gfortran expands an integer literal exponent into
repeated multiplication; numpy's `**` calls `pow()`. They disagree by one ulp
on two of the eight modes here. This was the last thing checked.

**Factor order is load-bearing.** The Fortran writes
`(pi/6) * d**3 * (rhommav*avogadro) * x`, left-associated. All three mode
masses share the `(pi/6) * (rhommav*avogadro) * x` sub-product, and factoring
it out — the obvious optimisation — reassociates and breaks all three.
Multiplication is commutative in the reals and not in float64.

**Switch order is load-bearing too.** `rhocomp` is laid down as a literal, then
patched by `l_fix_nacl_density` (1600 → 2165 for sea salt), and only *then* do
the masses derive from it. Applying the switch afterwards leaves the masses
built from the uncorrected density, silently, and by 35% on any mode carrying
sea salt.

**A nested `.AND.` is not two independent knobs.**
`ukca_mode_setup.F90:678-679` tests
`l_fix_ukca_hygroscopicities .AND. l_fix_nacl_density` before assigning
`no_ions`, so NaCl density only reaches that table when hygroscopicities is
also on. Reading them as independent selects the default branch and gets all
seven setups wrong *in the same way* — which presents as a systematic bug
rather than a misread conditional.

Also written faithfully rather than idiomatically, for the same reason:
`ddpmid` as `EXP(0.5*(LOG a + LOG b))` not `sqrt(a·b)`, `x` as
`EXP(4.5·LOG(sg)·LOG(sg))` not a square, and `rhommav` accumulated in index
order rather than as a vector `sum` — a pairwise reduction associates
differently.

## `topmode` is not the highest active mode

It reads exactly as though it should be. `ukca_mode_setup.F90:418-422` sets it
to `nmodes` when `l_dust_mp_ageing` and to `mode_ait_insol` (**5**) otherwise,
*regardless of `mode_choice`*.

The box model defaults that switch off, so `topmode` is 5 in every supported
setup — including setup 8, where modes 6 and 7 are active. Loops written
`DO imode = 1, topmode` (`ukca_conden.F90:299`) or
`DO imode = mode_ait_insol, topmode` (`ukca_ageing.F90:219`) therefore stop at
5 and never reach them. Verified against the binding: flipping the switch gives
8.

This is what made UP-10's impact claim wrong twice — three of its four lines
are gated on `topmode > mode_ait_insol`, which is false by default.

## `ukca_mode_allcp_4mode` is dead code, and citations drift into it

`ukca_mode_setup.F90:305-509` defines `ukca_mode_allcp_4mode`. Nothing calls
it — it appears nowhere else in `fortran/src/`, and
`common_mode_setup_interface_mod`'s `SELECT CASE` has no branch for it. It is a
200-line near-duplicate of `ukca_mode_suss_4mode` (`:511-714`), the live
setup-1 routine.

That matters for more than tidiness. Every switch block exists twice with
identical text, so a citation found by grepping for the *content* lands in the
dead copy about half the time and looks right on inspection. It has now
happened twice here: `:168` was corrected to `:474-475` after confirming the
line said exactly what the citation claimed — and `:474` is inside the dead
routine. The live line is `:678-679`.

**Check the enclosing routine, not just the line text.** Pinned by
`tests/test_modes.py::test_the_dead_routine_really_is_dead` and the
machine-checked `CITATIONS` table beside it.

## `component_mode` is a permission table, not a presence table

`component_mode` is which components are *allowed* in each mode — the source
comments say "allowed in nuc_sol". `component` is which are actually *present*
for this setup: the three-way intersection of allowed, chosen, and mode-is-on.

Assuming they are the same fails on **all seven** supported setups — 18
differing cells on setup 1, 22 on setup 6. `component_mode` is in fact the same
table in every setup, being the full permission list; `component` is the
intersection and never equals it. (This said "passes on setup 1 and fails on
five of the other six", which is wrong in both halves.) The invariant is
containment, and the test also asserts the two genuinely differ, so the
containment check cannot quietly go vacuous.

## `i_tune_bc` has no `CASE DEFAULT`

`ukca_mode_setup.F90:425-430` selects on `i_tune_bc` with exactly two named
cases, 1 (tuned, 1900 kg m⁻³) and 2 (Mie-mixture, 1800). There is no
`CASE DEFAULT`, so any other value silently leaves `rhocomp(cp_bc)` at its
literal 1500 rather than failing.

Reproduced rather than corrected — the port matches the reference including its
silences — and captured as the `bc_oob` golden so the silence is recorded
rather than assumed. Same shape as UP-5's unchecked `icoag`, and worth
mentioning alongside it if that is ever filed upstream.

`i_tune_bc` is also inert unless `l_radaer` is on, which the box model defaults
off, so BC density tuning is unreachable in the default configuration.

## Bit-identity is a property of a platform pair, not of the port

The measurements above were all taken on one machine — gfortran 16.1.0 on
Darwin arm64 — and written up as though they were properties of `erf` and
`**`. They are not. Ubuntu x86_64 CI, running the same tests against the same
committed goldens, disagrees:

| primitive | points differing | worst gap |
|---|---|---|
| `erf` | 1521 / 4330 (35%) | **4 ulp**, at `erf(x) = 0.4928` |
| `x ** (1.0/3.0)` | 86 / 1865 (4.6%) | 1 ulp |
| `x ** p` (`powr_v`) | 1 / 1865 | 1 ulp |
| `log`, `1/x`, `nint`, `vapour_round` | 0 | — |

`erf` is the outlier and only just: exactly two of its 4330 points exceed 2
ulp, at 4 and 3, and both sit mid-range rather than near zero — so this is
glibc's `erf` against Apple's, not a cancellation artefact at a small value.
The bounds in `conftest.CROSS_PLATFORM_ULP_BY_PRIMITIVE` are those
measurements. Exceeding one means look, not bump.

Everything that differs is a libm transcendental; everything exact stays exact
everywhere, which is the expected shape and a useful check that the failures
are what they look like.

**The CI job that found this could not have interpreted it.** The `test` job
installs no gfortran — deliberately, so the pure-Python port stays verifiable
without a Fortran toolchain — so it was comparing *this* platform's JAX against
*another* platform's Fortran. That is not a comparison between the port and the
reference; it is a comparison between two machines. `goldens_manifest.py` has
said "goldens are NOT portable across compilers or platforms" in its docstring
since it was written, and `build_reference.sh` has recorded `uname -srm` in
`TOOLCHAIN.txt` all along. Nothing read either.

Two changes. `tests/conftest.py:assert_matches_reference` requires bit equality
when the running platform matches the `uname` the manifest recorded, and a
bounded 2 ulp otherwise, naming both platforms when it fails — the relaxed
window still catches every structural porting error, which are orders of
magnitude out rather than two ulp. Quantities that are integer-valued off a
comparison (`nint`, `vapour_round`) stay exact everywhere, because there is no
rounding for a platform to disagree about. And the `linux-reference` CI job
builds gfortran on x86_64, re-captures the sweep there, re-stamps the manifest
so the strict path is selected, and then demands bit equality — so the strong
claim is re-established per platform instead of assumed to travel.

What this means for order 1: **byte-equal trajectories are achievable against a
reference built on the same machine, and are not a cross-platform property of
this port.** Any gate that compares against a committed golden inherits the
capture platform. Worth knowing before phase I, where the trajectory goldens
start branching on computed floats — a 2 ulp difference in `drydp` is normally
2 ulp in the answer, but at a merge threshold it is a different mode structure.

## `jit` is not byte-equal, and the cost has a number now

XLA-CPU contracts `a*b + c` into a true FMA — one rounding where the reference
does two, since every reference variant is built `-ffp-contract=off`
(`build_reference.sh:37`, `build_f2py.sh:42`). Eager JAX does not contract.

Measured on this arm64 build with jax 0.11.0: **23.4% of 200,000 random triples
differ**, and every differing jitted value is the correctly-rounded FMA,
checked exactly with `Fraction`. So this is contraction, not fast-math
reassociation — which matters, because reassociation would be far harder to
reason about.

On the ZSR polynomials `ukca_water_content_v` evaluates over the box-live range
(`aw = max(rh, rh_min/100)`, `rh` in [0.1, 0.9]):

| pair | points differing | max relative |
|---|---|---|
| (1,−2) H₂SO₄ | 179,687 / 200,001 | 4.2e-13 |
| (1,−4) HCl | 153,315 / 200,001 | 8.6e-14 |
| (3,−2) | 122,371 / 200,001 | 2.4e-14 |
| **(3,−4)** | 105,907 / 200,001 | **4.8e-11** |

`RTOL_JIT_VS_EAGER = 1e-14`. Water content under `jit` misses it by three
orders. That constant was a plausible tightness rather than a measurement, and
it cannot be met for anything FMA-shaped.

Ineffective: `--xla_cpu_enable_fast_math=false`,
`--xla_allow_excess_precision=false`, `lax.optimization_barrier` on the
product, a `bitcast_convert_type` round trip. Contraction looks unconditional
in XLA's CPU lowering.

Nothing in order 1 changes — `CLAUDE.md` already requires porting eager first
and keeping the eager driver permanently, and every byte-equality gate runs
eager. What changes is that order 2 cannot claim jit parity with the goldens
without settling this. Issue #23, pinned by
`test_xla_contracts_multiply_add_under_jit_and_gfortran_does_not`.

## Two spellings of `sixovrpix`, and the cube root amplifies them

`1.0/(x*(pi/6))` and `6.0/(pi*x)` are the same number in exact arithmetic and
differ by 2 ulp in float64 for σg = 2.0 and 1.8. That would be unremarkable if
it stopped there. It does not: `drydp` takes the cube root of a product
involving it, and over 200,000 random `wvol` in [1e-19, 1e-15] the two
spellings give different doubles on **53.2% (mode 4), 53.3% (mode 7) and 64.5%
(mode 8)** of points. Mode 4 is active in setups 1–5 and 8.

So the spelling is load-bearing, and the answer to "does a 2 ulp constant
matter here" is yes, on half the domain.

## `l_fix_neg_pvol_wat` cannot be tested through `rhosol_strat`

The flag looks like it protects the stratospheric density, and
`docs/fidelity.md` said so. It does not, and the argument is short enough to
check: the two arms differ only where `ws*100 > 99`, the clamped arm then gives
`wts = 99` and the unclamped arm more, and `(NINT(wts/5))*5` sends **both** to
100 or above — while `percent` (`ukca_vapour.F90:90`) stops at 95. Neither
matches, both fall through to `rhosol_strat = 1300.0`.

`rhosol_strat` is therefore bit-identical at both settings at every input, and
a both-settings test written against it could not fail. What the flag moves is
`wts`, and through it `mdwat` at `ukca_volume_mode.F90:436`.

## Setup coverage of the shipped namelists

`boundary_layer` and `free_troposphere` are both `i_mode_setup = 1` — four
soluble modes — so neither enters `ukca_coagwithnucl`'s insoluble blocks,
`ukca_ageing`, or the insoluble condensation path. `marine_bcoc` is setup 2 and
is the smallest shipped case that reaches every instrumented site. Any test
meant to cover insoluble-mode behaviour has to use it.

## `ukca_cond_coff_v`'s switches select which *arguments* are read (task 46)

Not just which formula. Measured over the leaf sweep, and asserted as byte
equalities between calls that differ in one argument alone:

| argument | `idcmfp = 1` | `idcmfp = 2` |
|---|---|---|
| `tsqrt` | read (`vel_cp` `:171`, `dcoff_cp` `:174`) | read (`vel_cp`, then `mfp_cp` `:181`) |
| `rp` | read | read |
| `rhoa` | read (`:174`) | **not read** |
| `airdm3` | read (`:177`) | **not read** |
| `t` | **not read** | read (`:179`) |
| `pmid` | **not read** | read (`:179`) |
| `dmol` | read (`term2` `:154`, `term3` `:157`) | **not read** |
| `difvol` | **not read** | read (`term8` `:167`) |

Every shipped namelist and every scenario in `inputs/` runs `ifuchs = 1` and
`idcmfp = 1`, so the branch dump covers one corner of a 2x2 and three of the
four closed forms have no trajectory reference at all.

`tsqrt` and `t` are separate arguments and the callee never checks that they
agree. `ukca_conden.F90:281` passes `SQRT(t)`, but a leaf call can pass
`tsqrt = SQRT(t_b)` with `t = 999.0` — and at `idcmfp = 1` the result is
byte-equal to the consistent `t_b` row. That is "`t` is unread", not "`t` did
not matter much on this grid", and it is the kind of claim only a leaf driver
can make.

## `se = 1.0` switches off the term that defines the Fuchs-Sutugin branch

`akn = 1.0/(1.0 + 1.33*kn*fkn*(1.0/se - 1.0))` (`ukca_cond_coff_v.F90:210`) is
the only place `se` reaches `ifuchs = 2`. `ukca_conden.F90:235-237` sets
`se_sol` and `se_ins` both to `1.0`, so `1.0/se - 1.0` is **exactly** `0.0` and
`akn` is exactly `1.0` on every row of every configuration. `cc` reduces to
`term6*dcoff_cp*rp*fkn`, and the correction for limitations in interfacial mass
transport does nothing.

Both the routine's own header (`:60`) and `ukca_conden`'s (`:53`) document `se`
as `0.3`, where `1.0/se - 1.0 = 2.333...` and `akn` depends on
`kn = mfp_cp/rp` — so it varies along the size distribution rather than
scaling it. Over the sweep's rp axis (1e-10 to 2e-5 m) the `se = 0.3` answer is
**0.3002 to 0.9954** of the `se = 1.0` answer, a factor spanning 3.32 across
radii.

The lower end is not a coincidence and is worth stating as a check on the
reading: as `kn -> inf`, `fkn -> 1/(1.33*kn)` (`:208`), so `1.33*kn*fkn -> 1`
and `akn -> 1/(1 + (1/se - 1)) = se`. The smallest particles lose exactly the
factor `se`; the largest lose nothing. UP-9 records the header discrepancy as
documentation-only and does not say any of this; issue #27.

## The `pmid` axis is chosen to be able to see the reciprocal rewrite

`ukca_cond_coff_v.F90:179` divides an array by a scalar literal,
`pmid(:)/101325.0` — the site XLA rewrites into a multiply by `1/101325.0`.
Of twelve plausible pressures only **two**, `9.0e4` and `1.05e5`, give a
different double under the rewrite. So a grid of round pressures passes against
a port that has it, which is the same shape as the phase-D failures: a six-value
humidity grid that missed the powi chain on 481 of 1301 points.

The axis carries seven discriminating values, and the count is re-derived by
`tests/test_cond_coff_fixtures.py` rather than trusted. With the guard removed,
12 of the 12 `idcmfp = 2` calls lose byte equality and none of the 12
`idcmfp = 1` calls move — they read no pressure at all.

## The double-`where` idiom does not save a poisoned numerator

`numerics.safe_divide` substitutes a safe *denominator*, which is what makes a
masked division differentiable. It does nothing about a numerator that is
already `inf` or `NaN`.

At `ifuchs = 1` the numerator of `cc` is `term6*dcoff_cp*rp`. On a masked row
`dcoff_cp` is `0.0` (`safe_divide` zeroes it) and the fixture's `rp` is `inf`,
so the product is `NaN`, and `jax.grad` returns a non-finite cotangent on 4 of
the 7 masked rows. At `ifuchs = 2` the same product is formed without a
division around it and all 7 are finite. Every **live** row is finite at all
four switch settings.

No model state can produce those inputs — `rp` is a wet radius — so this is a
property of the fixture's deliberate poison, not of the physics. It is asserted
at exactly 4 so that a later phase needing clean cotangents through an
arbitrary masked box knows the cost is masking the inputs, not the divisions.

## `ukca_dcoff_par_av_k` and `ukca_vgrav_av_k` have no reference

They are coefficient kernels of the same shape as `cond_coff` and `coag_coff` —
moment-averaged over a log-normal, both 129 lines, both closed-form — and
neither is in phase E. Their only callers are `ukca_ddepaer_mod` and
`ukca_ddepaer_incl_sedi_mod`, and `glomap_box.F90:144` passes `ddepaer = 0` and
`sedi = 0`. There is no validated trajectory that enters either, so porting
them would be a physics commit with nothing to check it against.

A leaf fixture could be built for them without dry deposition — they take no
`glomap_variables` — but the routines they feed cannot be validated at all, so
the coefficient would be a leaf with no tree. Left out until dry deposition is
in scope; see `docs/unsupported.md`.

## `jnp.exp` is not the platform libm, and byte equality ends where `EXP` begins

Task 50, and the most consequential measurement of phase E.

42 of `ukca_calc_coag_kernel`'s 3,690 non-zero golden outputs differ from the
compiled routine: 20 at `icoag = 1` (max 1 ulp), 22 at `icoag = 2` (max 3 ulp),
0 at `icoag = 3`, 0 at `coag_on = 0`. Substituting `numpy.exp` for `jnp.exp`
inside `coag_coff` makes **all 42 disappear** — asserted by
`test_the_whole_gap_is_the_exponential`, so the attribution is a test and not
an argument.

The one call responsible is `EXP(-1.1/kn)` in the Cunningham slip correction,
`ukca_coag_coff_v.F90:266`. `icoag = 3` replaces that correction with the
constant `1.591` and reaches no exponential at all; it is byte-equal on every
setup, which is the control.

`numpy.exp` and gfortran's `EXP` both reach the platform libm and agree bit for
bit on every argument probed through `leaf_exp`. `jnp.exp` is XLA's own
evaluation and does not: on the `coag_coff` grid it disagrees on 5 of the 25
distinct arguments method 1 evaluates and 22 of the 95 method 2 does.

**This is the first ported routine with `EXP` in a live path.** `ukca_vapour`
reaches only `LOG` and `SQRT` there, `ukca_water_content_v` has no
transcendental, and neither `calc_drydiam` nor `volume_mode` reaches `EXP`
anywhere a golden compares. Four phases of byte-equality gating never tested
what CLAUDE.md's numerics table has recorded since phase B.

### Task 49's byte equality was cancellation, not agreement

`coag_coff` is byte-equal on all 280 rows of its own fixture *despite* those
disagreements, because `0.4 * ulp(0.37)` is about a tenth of `ulp(1.4)` and
vanishes in the `1.257 + 0.4*EXP(...)` addition — usually. On the kernel's grid
it survives on 42 elements.

That is the two phase-D failures one level up. Those were grids too coarse for
the defect; this is a grid on which the defect **cancels**, which no amount of
widening the same axes would have found. What found it was a second routine
consuming the first.

`test_the_byte_equality_here_is_cancellation_not_agreement` now asserts the
disagreement counts on the `coag_coff` grid, so a future grid edit that hides
the hazard fails rather than looking cleaner.

### What is gated, and what is not

`tests/test_coag_kernel.py` asserts `ulp = 0` for the paths with no
exponential and a **same-platform** 4-ulp window for the two that have one,
plus the exact affected counts so the allowance cannot grow silently. The
window is deliberately not `assert_matches_reference`'s cross-platform
allowance: that helper takes the exact branch whenever it runs where the
goldens were captured, and this gap is on the capture platform itself.

Every remaining routine with an exponential inherits this — `ukca_calcnucrate`,
`ukca_binapara_mod`, `ukca_conden`'s `1.0-EXP(-sumnc*dtz)`, `ukca_remode`.
Issue #28.

## `kij_arr` is not one triangle, and nothing below the driver fixes its order

`ukca_calc_coag_kernel` fills three families of slot:

* `:243-262` soluble `imode`, soluble `jmode > imode` — upper triangle of 1..4
* `:276-291` soluble `imode`, insoluble `jmode >= imode+4` — upper, across the split
* `:308-322` **insoluble** `imode`, soluble `jmode >= imode-2` — **lower**

19 slots with every mode active, none filled in both orders. Three
soluble/insoluble pairs are filled in neither — `(2,5)`, `(3,6)`, `(4,7)`
1-based, the same-size-class pairs — and that is consistent rather than a gap:
`ukca_coagwithnucl.F90:307`, `:345` and `:467` read with the same three bounds
and never ask for them. It reads *less*, in fact: its insoluble inner loop
stops at `topmode`, which is `mode_ait_insol` unless `l_dust_mp_ageing` is on,
so nine of the slots the kernel fills are unconsumed by default.

**No value comparison can check any of this.** The kernel is byte-symmetric
under an `(i, j)` swap (task 48) and `coag_mode` is symmetric on all 64 entries
(phase C), so a transposed convention writes the right number into the wrong
slot. Transposing the port's `kij_arr` gives an array whose non-zero multiset
is identical to the golden's — a check that sorted, summed, or symmetrised
would pass it. Only the zero/non-zero pattern discriminates, which is what
`test_the_port_fills_the_slots_the_fortran_filled` compares.

## The driver passes an insoluble mode's dry size as `j` and its wet size as `i`

`ukca_calc_coag_kernel.F90:281-286` selects the partner's size by `modesol` —
soluble partners enter at `wetdp`/`wvol`, insoluble ones at `drydp`/`dvol`.
The insoluble-`imode` loop at `:297-299` takes `wetdp`/`wvol` unconditionally.
So the same insoluble mode enters wet when it is `i` and dry when it is `j`.

In the model the two coincide, because an insoluble mode carries no water — and
that coincidence is exactly what would hide the routing. The leaf grid
therefore feeds `drydp = 0.7*wetdp` for **every** mode and the capture calls
`ukca_coag_coff_v` directly under both conventions to see which one each filled
slot matches: 8 dry, 43 wet, over the seven setups. Measured, not read.
