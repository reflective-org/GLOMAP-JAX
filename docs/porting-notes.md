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
