# Upstream UKCA defects

Ten defects found in `MetOffice/ukca` @ `387c5bb` while building the reference
box model and planning this port.

**Status: drafted, not yet filed.** By decision, Reflective files these
upstream, not the port. Issue
[#1](https://github.com/reflective-org/GLOMAP-JAX/issues/1) tracks it. Two are
already carried as patches here (`fortran/patches/`); the rest are reproduced
faithfully behind fidelity flags — see [fidelity.md](fidelity.md).

Each entry states **reachability** explicitly, because six of the ten are
latent, config-gated or diagnostic-only, and conflating those with
"changes results" misdirects both upstream and the port.

| ID | Location | Reachability |
|---|---|---|
| UP-1 | `ukca_solvecoagnucl_v.F90:259` | **changes results, every substep** |
| UP-2 | `ukca_solvecoagnucl_v.F90:60-68` | documentation only |
| UP-3 | `ukca_ageing.F90:296-298` | diagnostics only |
| UP-4 | `ukca_conden.F90:353-354` | unreachable, latent |
| UP-5 | `ukca_coag_coff_v.F90:339-340` | feature unusable (`icoag=4`) |
| UP-6 | `ukca_aero_step.F90:504/1034` | direct callers only, not the UM |
| UP-7 | `ukca_aero_step.F90:1022-1023` | fatal under bounds checking |
| UP-8 | `ereport_mod.F90:50` | standalone harnesses only |
| UP-9 | `ukca_conden.F90:52-53` | documentation only |
| UP-10 | `ukca_conden.F90:372-387` | **changes results on setup 8** |

## UP-1 — spurious factor 3 in the `dN/dt = A·N²` branch

`ukca_solvecoagnucl_v.F90:259` computes `1/(1/N − 3·A·Δt)`; the exact integral
has no factor 3. The header at `:77` repeats it, so code and comment agree with
each other and both disagree with the mathematics.

**Reachable every substep.** `ukca_coagwithnucl.F90:462` skips inter-modal
coagulation for the top insoluble modes, so `B ≡ 0`, `C ≡ 0` and the
discriminant is exactly zero. Intra-modal number decays three times too fast. It
never blows up (`A < 0`), which is why it survived. KGO-changing upstream.

**Measured, not only argued.** The branch dump (task 15b) records which closed
form each element takes. The factor-3 branch runs on **every substep of every
shipped namelist**, including the default 4-mode `i_mode_setup = 1` — and for
the top *soluble* mode, not only the insoluble ones this note originally named.
`mode_cor_sol` has no larger soluble mode to coagulate with and no nucleation
source, so `B` and `C` are exactly zero there too. Every supported configuration
has a largest soluble mode, so every supported configuration hits this.

## UP-2 — header swaps arctan and log relative to the code

`:60-68` assigns arctan to `D < 0` and the logarithmic form to `D > 0`; the code
does the reverse and the code is correct. Port from the code.

## UP-3 — `naged` overwritten before use as divisor

`ukca_ageing.F90:296-298` — the `totage` rescale evaluates to exactly 1.0.
Affects `bud_aer_mas` and the `SUM(totage) > 0` gate only; `md` uses `totage1`.
The naive fix would lose mass, since `ukca_conden` records insoluble condensate
only in `ageterm1`.

## UP-4 — `delgc_cond = delgc_cond/gc` where `= gc` was intended

`ukca_conden.F90:353-354`. Unreachable: `delgc_cond = gc·(1−exp(−x))` at `:349`
with `x ≥ 0` bounds it in `[0, gc]`. The branch dump carries the guard
explicitly and it is false in every record of every shipped namelist, so this is
an observation and not only an argument — which is why UP-4 gets an invariant
test in the port rather than a fidelity flag.

## UP-5 — `icoag=4` reads unassigned `mfppi`/`mfppj`

`ukca_coag_coff_v.F90:340`; both assigned only at `:262`/`:274` inside the
mutually exclusive `IF (icoag == 1)` at `:252`. Always reads undefined memory.
Secondary: `icoag` is never range-checked and `kij` is pre-zeroed, so an
out-of-range value silently disables coagulation.

## UP-6 — `s_cond_s` read unassigned when `cond_on=0, nucl_on=1`

Declared `ukca_aero_step.F90:504`, written only under `cond_on == 1`, read at
`:1034`. Not reachable from the UM (`ukca_setup_mod.F90:1615` hard-sets
`l_mode_bhn_on = .FALSE.` when chemistry is off) but immediate for any direct
caller. Vehkamäki guards `s_cond_s > 0`; the BLN path at
`ukca_calcnucrate.F90:413` does not.

## UP-7 — `msec_org` used as a subscript before its own guard

`ukca_aero_step.F90:1022-1023` reads `condensable_choice(msec_org)` and
`mm_gas(msec_org)` three lines before `IF (msec_org > 0)`. `msec_org = 0` is
legitimate for every setup without a secondary-organic component, including the
default. Fatal under bounds checking, making debug builds unusable for the
default configuration. **Carried as `fortran/patches/0001`.**

## UP-8 — `ereport` exits 0 on fatal errors

`ereport_mod.F90:50` uses a bare `STOP`, which exits with status 0, so every
UKCA abort looks like success to the shell. **Carried as
`fortran/patches/0002`.**

## UP-9 — header documents `SE_INS = 0.3`, live value is 1.0

`ukca_conden.F90:53` versus `:237`.

## UP-10 — insoluble-mode `num_eps` indexed by the soluble mode

`ukca_conden.F90:372-387`:

```fortran
372:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_ait_insol) > num_eps(imode) )
377:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_acc_insol) > num_eps(imode) )
382:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_cor_insol) > num_eps(imode) )
387:  mask4i(:) = mask2(:) .AND. ( nd(:,mode_sup_insol) > num_eps(imode) )
```

`imode` is the **soluble** mode of the enclosing loop; the threshold should
belong to the insoluble mode being tested. `num_eps` spans twelve orders of
magnitude across modes (`[1e-8, 1e-8, 1e-8, 1e-14, 1e-8, 1e-14, 1e-14, 1e-20]`,
`ukca_mode_setup.F90:395`), so `imode = mode_cor_sol` (1e-14) gates
`mode_sup_insol` (1e-20) — a threshold wrong by 10⁶.

Contrast `:366`, which correctly uses `num_eps(imode)` for `nd(:,imode)`.

**Changes results on `i_mode_setup = 8`**, the only supported setup with
`mode_sup_insol` active. Found during the phase A review; reproduced behind
`conden_insol_num_eps_by_sol_mode`.
