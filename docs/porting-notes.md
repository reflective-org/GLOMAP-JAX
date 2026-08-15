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

| predicate | routine | records |
|---|---|---|
| `MDCPNEW < 0` → `ND = 0`, `mask1` falsified mid-`ICP` | `ukca_coagwithnucl` | 0 / 19440 |
| the undersize reset `DP < DDPLIM0·0.1` | `ukca_calc_drydiam` | 0 / 2160 |
| any mode merge at all | `ukca_remode` | 0 / 864 |
| the `FRAC_N < 0.5` / `FRAC_M < 0.001` clamps | `ukca_remode` | never entered |
| the Kulmala and boundary-layer nucleation gates | `ukca_calcnucrate` | never entered |
| `NTOT < 4` with `T < 195.15` → flat `J = 1e5` | `ukca_binapara` | 0 / 2160 |

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
| `erf` (via `umErf`) | **bit-identical**, 4330/4330 | no shim needed |
| `log`, `1/x` | bit-identical | safe |
| `x ** (1.0/3.0)` | bit-identical | **this** is what `cubrt_v` computes |
| `exp` | 456/3199 differ, max 2.1e-16 | one ulp; inside tolerance but real |
| `np.cbrt` | 1756/1865 differ, max 1.3e-14 | **must not be used** |
| `NINT` vs `round` | 64/642 differ | **must not use `jnp.round`** |

Three rules follow, and all three are asserted in
`tests/test_numerics_reference.py` so they cannot be quietly broken later.

**Write the cube root as `x ** (1.0/3.0)`.** `cubrt_v` is literally that
expression; it is not a cube-root function. `np.cbrt` disagrees on 94% of the
grid by up to 1.3e-14 — a hundred times `RTOL_ALGEBRAIC`. This is the one that
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

## Setup coverage of the shipped namelists

`boundary_layer` and `free_troposphere` are both `i_mode_setup = 1` — four
soluble modes — so neither enters `ukca_coagwithnucl`'s insoluble blocks,
`ukca_ageing`, or the insoluble condensation path. `marine_bcoc` is setup 2 and
is the smallest shipped case that reaches every instrumented site. Any test
meant to cover insoluble-mode behaviour has to use it.
