# Upstream UKCA defects

Eleven defects found in `MetOffice/ukca` @ `387c5bb` while building the
reference box model and planning this port.

**Status: drafted and submission-ready, not yet filed.** By decision,
Reflective files these upstream, not the port. Issue
[#1](https://github.com/reflective-org/GLOMAP-JAX/issues/1) tracks it.

Every entry below carries a title, a `file:line`, its reachability, its impact,
and a **suggested patch** — enough to paste into a tracker without
reconstructing the analysis.

**Path note for whoever files these.** The diffs below use this repository's
vendored layout, `src/ukca/<file>.F90`. Upstream the same files live at

| here | `MetOffice/ukca` |
|---|---|
| `src/ukca/ukca_*.F90` | `src/science/core/aerosols/glomap/ukca_*.F90` |
| `src/ukca/ereport_mod.F90` | `src/control/legacy/ereport_mod.F90` |

so the diffs are illustrative of the change, not directly appliable upstream.
They also elide surrounding comment lines in places. Re-generate against an
upstream checkout before submitting anything as a patch rather than as a
description. `tests/test_upstream_defects.py` fails if any of
those five is missing, so a half-drafted entry cannot ship looking finished.
Two of the suggested patches are already applied here, under
`fortran/patches/`, because the reference is unusable without them; the rest are
proposals, and **UP-3's is deliberately argued against** in its own entry. Two are
carried as patches to the vendored tree because the reference is unusable
without them; four are reproduced faithfully behind fidelity flags (see
[fidelity.md](fidelity.md)); the remaining four need no code action, for
reasons the disposition table below makes explicit.

Each entry states **reachability** explicitly, because six of the ten are
latent, config-gated or diagnostic-only, and conflating those with
"changes results" misdirects both upstream and the port.

Each also states a **disposition**: what the port does about it. Not every
defect earns a fidelity flag — a flag is for a defect the port must *choose* to
reproduce, and where there is nothing to choose the honest answer is something
else. `tests/test_upstream_defects.py` enforces every row of this table, so a
disposition cannot be claimed here and quietly not exist in the code.

| ID | Location | Reachability | Disposition |
|---|---|---|---|
| UP-1 | `ukca_solvecoagnucl_v.F90:259` | **changes results, every substep** | `fidelity-flag: coag_intra_factor3` |
| UP-2 | `ukca_solvecoagnucl_v.F90:60-68` | documentation only | `documentation-only` |
| UP-3 | `ukca_ageing.F90:296-298` | diagnostics only | `fidelity-flag: ageing_totage_rescale_noop` |
| UP-4 | `ukca_conden.F90:353-354` | unreachable, latent | `invariant-test` |
| UP-5 | `ukca_coag_coff_v.F90:339-340` | feature unusable (`icoag=4`) | `not-implemented` |
| UP-6 | `ukca_aero_step.F90:504/1034` | direct callers only, not the UM | `fidelity-flag: s_cond_s_zero_when_cond_off` |
| UP-7 | `ukca_aero_step.F90:1022-1023` | fatal under bounds checking | `harness-patch: 0001-guard-msec_org-zero-index.patch` |
| UP-8 | `ereport_mod.F90:50` | standalone harnesses only | `harness-patch: 0002-ereport-nonzero-exit-status.patch` |
| UP-9 | `ukca_conden.F90:52-53` | documentation only | `documentation-only` |
| UP-10 | `ukca_conden.F90:372-387` | **changes results on setup 8** | `fidelity-flag: conden_insol_num_eps_by_sol_mode` |
| UP-11 | `ukca_volume_mode.F90:856-877` | diagnostic unusable when it fires | `diagnostic-only` |

What each disposition means, and what the test checks:

| disposition | meaning | checked by |
|---|---|---|
| `fidelity-flag: X` | the port reproduces the defect by default and can be told not to | `X` is a real `FidelityConfig` field, has a `docs/fidelity.md` section, and that section cites this defect |
| `invariant-test` | nothing to choose — the defect is unreachable, and that is asserted rather than assumed | a test in this file names the defect and checks the invariant |
| `not-implemented` | the affected feature has no correct reference to validate against, so the port refuses it | recorded in `docs/unsupported.md` |
| `harness-patch: F` | the reference itself is unusable without a fix; carried as a patch to the vendored tree | `fortran/patches/F` exists |
| `documentation-only` | a comment or header disagrees with the code; the code is right | no code action; the entry says which to trust |
| `diagnostic-only` | affects only a print/abort path the port does not reproduce (ADR-006) | no code action; the entry says why the port is unaffected |

## UP-1 — spurious factor 3 in the `dN/dt = A·N²` branch

`ukca_solvecoagnucl_v.F90:259` computes `1/(1/N − 3·A·Δt)`; the exact integral
has no factor 3. The header at `:77` repeats it, so code and comment agree with
each other and both disagree with the mathematics.

**Reachable every substep.** `ukca_coagwithnucl.F90:462` skips inter-modal
coagulation for the top insoluble modes, so `B ≡ 0`, `C ≡ 0` and the
discriminant is exactly zero. Intra-modal number decays three times too fast. It
never blows up (`A < 0`), which is why it survived. KGO-changing upstream.

**Measured for the soluble modes, argued for the insoluble ones.** The branch
dump (task 15b) records which closed form each element takes. The factor-3
branch runs on **every substep of every shipped namelist**, including the
default 4-mode `i_mode_setup = 1` — and for the top *soluble* mode, which this
note originally did not mention at all. `mode_cor_sol` has no larger soluble
mode to coagulate with and no nucleation source, so `B` and `C` are exactly zero
there too. Every supported configuration has a largest soluble mode, so every
supported configuration hits this.

The insoluble half of the original claim remains **argued, not measured**: the
only shipped case with an insoluble mode is `marine_bcoc`, whose single
insoluble mode is `mode_ait_insol`, and `ukca_coagwithnucl.F90:462` is
`IF (imode < mode_cor_insol)` — so it *does* take part in inter-modal
coagulation and all 720 of its records are code 1, never code 5. Reaching code 5
on an insoluble mode needs a setup with `mode_cor_insol` or `mode_sup_insol`
active, which means `i_mode_setup = 8` and a fixture that does not exist yet.

**Impact.** Intra-modal number decays three times too fast in every affected
mode, on every substep. Measured with per-substep branch instrumentation: the
branch fires for the top soluble mode in **every** shipped configuration,
including the default `suss_4mode`, because that mode has no larger soluble mode
to coagulate with and no nucleation source, so `B` and `C` are exactly zero.
KGO-changing. It never blows up (`A < 0`, so the denominator only grows), which
is presumably why it has survived.

**Suggested patch.**

```diff
--- a/src/ukca/ukca_solvecoagnucl_v.F90
+++ b/src/ukca/ukca_solvecoagnucl_v.F90
 ! Below is for case where A /= 0, D = 0 and B=0
-WHERE (logic1cb(:)) ndnew(:)=1.0/(1.0/nd(:)-3.0*a(:)*dtz)
+WHERE (logic1cb(:)) ndnew(:)=1.0/(1.0/nd(:)-a(:)*dtz)
```

The header at `:77` states the same expression and needs the same correction.

## UP-2 — header swaps arctan and log relative to the code

`:60-68` assigns arctan to `D < 0` and the logarithmic form to `D > 0`; the code
does the reverse.

**Trust the code.** The header is wrong; `:199-212` and `:249-252` are right.

**Impact.** Documentation only; no numerical effect. Reported because anyone
porting or reviewing this routine from its header — which is the natural thing
to do, the header being unusually complete — derives the wrong solution for both
branches.

**Suggested patch.** Exchange the two descriptions in the header block at
`:60-68` so they match the code at `:199-212` and `:249-252`. The code is
correct and should not change.

## UP-3 — `naged` overwritten before use as divisor

`ukca_ageing.F90:296-298` — the `totage` rescale evaluates to exactly 1.0.
Affects `bud_aer_mas` and the `SUM(totage) > 0` gate only; `md` uses `totage1`.
The naive fix would lose mass, since `ukca_conden` records insoluble condensate
only in `ageterm1`.

**Impact.** Diagnostics only, and the fix is not obvious.

`totage` is read in exactly two places: the `IF (SUM(totage) > 0.0)` gate at
`:308` and the `bud_aer_mas` accumulation; the prognostic `md` update uses
`totage1`. So the rescale as written affects nothing at all.

**Suggested patch.** The minimal change that makes the line do what it reads as
doing is to use the original `naged` before it is overwritten:

```diff
--- a/src/ukca/ukca_ageing.F90
+++ b/src/ukca/ukca_ageing.F90
           IF (naged > nd(jl,imode)) THEN
+            totage(:)=totage(:)*nd(jl,imode)/naged
             naged=nd(jl,imode) ! limit so no -ves
-            totage(:)=totage(:)*nd(jl,imode)/naged
             ! above reduces ageing if limited by insoluble particles
           END IF
```

**This is KGO-changing and should not be applied without a decision.** Scaling
`totage` down can close the `SUM(totage) > 0.0` gate at `:308`, which skips the
entire component-transfer block — so a change that looks like a diagnostics fix
can suppress a real transfer. It also cannot be pushed into the prognostic path:
`ukca_conden` records condensate onto insoluble modes only in `ageterm1` and
never adds it to insoluble `md`, so ageing must move all of it or mass is lost.

The alternative, and possibly the better report, is that the rescale was never
needed and the line should be deleted with a comment saying why.

## UP-4 — `delgc_cond = delgc_cond/gc` where `= gc` was intended

`ukca_conden.F90:353-354`. Unreachable: `delgc_cond = gc·(1−exp(−x))` at `:349`
with `x ≥ 0` bounds it in `[0, gc]`. The branch dump carries the guard
explicitly and it is false in every record of every shipped namelist, so this is
an observation and not only an argument — which is why UP-4 gets an invariant
test in the port rather than a fidelity flag.

**Impact.** None currently — the guard cannot fire. Reported as latent: the
expression is dimensionally meaningless (a concentration divided by a
concentration, then subtracted from a concentration), so if a future change to
`:349` ever lifts the bound, the result is silently wrong rather than
obviously so.

**Suggested patch.**

```diff
--- a/src/ukca/ukca_conden.F90
+++ b/src/ukca/ukca_conden.F90
     WHERE (mask2(:) .AND. delgc_cond(:,jv) > gc(:,jv))
-      delgc_cond(:,jv)=delgc_cond(:,jv)/gc(:,jv) ! make sure no -ves
+      delgc_cond(:,jv)=gc(:,jv) ! make sure no -ves
     END WHERE
```

Bit-identical today, by construction. Deleting the guard outright is equally
defensible and would document the bound instead.

## UP-5 — `icoag=4` reads unassigned `mfppi`/`mfppj`

`ukca_coag_coff_v.F90:340`; both assigned only at `:262`/`:274` inside the
mutually exclusive `IF (icoag == 1)` at `:252`. Always reads undefined memory.
Secondary: `icoag` is never range-checked and `kij` is pre-zeroed, so an
out-of-range value silently disables coagulation.

**Impact.** `icoag = 4` is unusable: it reads two arrays that are never
assigned on that path, so the coagulation kernel is computed from whatever was
left in memory. Under `-finit-real=snan` it traps immediately.

The block's own header at `:321-323` states the intent exactly — *"values of
CCI,CCJ,MFPPI,MFPPJ are computed rather than just setting MFPPI=MFPPJ=MFPA and
CCI=CCJ=1.591"* — and the code computes only `CCI` and `CCJ`.

**Suggested patch.** Add the two omitted computations, following `icoag == 1`
at `:256-262` and `:268-274`:

```diff
--- a/src/ukca/ukca_coag_coff_v.F90
+++ b/src/ukca/ukca_coag_coff_v.F90
 IF (icoag == 4) THEN
   WHERE (mask(:))
     kni(:)=mfpa(:)/ri(:)
     cci(:)=1.0+kni(:)*(1.257+0.4*EXP(-1.1/kni(:)))
+    veli(:)=SQRT(term1*t(:)/(rhoi(:)*vi(:)))
+    dcoefi(:)=term2*cci(:)*t(:)/(ri(:)*dvisc(:))
+    mfppi(:)=term3*dcoefi(:)/veli(:)
     knj(:)=mfpa(:)/rj(:)
     ccj(:)=1.0+knj(:)*(1.257+0.4*EXP(-1.1/knj(:)))
+    velj(:)=SQRT(term1*t(:)/(rhoj(:)*vj(:)))
+    dcoefj(:)=term2*ccj(:)*t(:)/(rj(:)*dvisc(:))
+    mfppj(:)=term3*dcoefj(:)/velj(:)
     termv3(:)=(2.0e6*boltzmann*t(:)/3.0/dvisc(:))
```

**Secondary, and worth fixing in the same change.** `icoag` is never
range-checked, and `kij` is pre-zeroed at `:237`, so an out-of-range value
silently disables coagulation entirely rather than failing. An `ereport` for
`icoag` outside `1..4` would turn a silent no-op into an error.

## UP-6 — `s_cond_s` read unassigned when `cond_on=0, nucl_on=1`

Declared `ukca_aero_step.F90:504`, written only under `cond_on == 1`, read at
`:1034`. Not reachable from the UM (`ukca_setup_mod.F90:1615` hard-sets
`l_mode_bhn_on = .FALSE.` when chemistry is off) but immediate for any direct
caller. Vehkamäki guards `s_cond_s > 0`; the BLN path at
`ukca_calcnucrate.F90:413` does not.

**Impact.** Reads an unassigned local. Not reachable from the UM, where
`ukca_setup_mod.F90:1615` hard-sets `l_mode_bhn_on = .FALSE.` when chemistry is
off, but immediate for any direct caller — a box model, or a process-isolation
test. The Vehkamäki path guards `s_cond_s > 0`, so garbage silently zeroes
nucleation; the BLN path at `ukca_calcnucrate.F90:413` feeds it straight into
`EXP` with no guard.

The practical consequence is that **no nucleation-only reference case can
exist**, which is why `glomap-box` ships `cond_only`, `coag_only` and `all_off`
namelists but no `nucl_only`.

**Suggested patch.** Initialise at the point of declaration, so the value is
defined on every path:

```diff
--- a/src/ukca/ukca_aero_step.F90
+++ b/src/ukca/ukca_aero_step.F90
+! S_COND_S is consumed by UKCA_CALCNUCRATE when NUCL_ON=1 but is only assigned
+! when COND_ON=1, so it must be defined before the substep loop.
+s_cond_s(:) = 0.0
```

placed immediately before the `DO imts = 1, nmts` loop. Zero is the value that
makes the Vehkamäki guard fail and the BLN enhancement factor collapse to
`exp(0) = 1`, i.e. the behaviour a reader would expect from "no condensation
sink".

## UP-7 — `msec_org` used as a subscript before its own guard

`ukca_aero_step.F90:1022-1023` reads `condensable_choice(msec_org)` and
`mm_gas(msec_org)` three lines before `IF (msec_org > 0)`. `msec_org = 0` is
legitimate for every setup without a secondary-organic component, including the
default. Fatal under bounds checking, making debug builds unusable for the
default configuration. **Carried as `fortran/patches/0001`.**

**Impact.** Latent in an optimised build — the garbage value is computed and
then discarded, because the `ELSE` branch is the one that executes when
`msec_org == 0`. It becomes **fatal the moment bounds checking is enabled**,
which makes a debug build unusable for the default configuration: exactly the
build wanted when investigating anything else.

**Suggested patch.** Carried in this repository as
`fortran/patches/0001-guard-msec_org-zero-index.patch`. Move the two reads
inside the existing guard; no logic change. Verified bit-identical in release
builds across all three shipped namelists, confirming the discarded value never
influenced results.

**Related, deliberately not patched.** The same shape appears at
`ukca_coarse_no3_mod.F90:211` and `ukca_fine_no3_mod.F90:207-208`. Those are
reachable only when the nitrate/ammonium switches are on, and in the setups that
enable them the indices are non-zero — so they are reported rather than changed
speculatively.

## UP-8 — `ereport` exits 0 on fatal errors

`ereport_mod.F90:50` uses a bare `STOP`, which exits with status 0, so every
UKCA abort looks like success to the shell. **Carried as
`fortran/patches/0002`.**

**Impact.** Severe for any standalone use. Every script, `make` target or CI job
that checks an exit status treats a crashed run as a passing one. It defeated
this repository's own test suite, which reported `passed: 8 failed: 0` against a
binary that never ran.

Harmless inside the UM, which aborts through its own MPI-aware error path and
never relies on `ereport`'s exit status — presumably why it was never noticed.
`ereport_mod.F90` is a UKCA-supplied standalone shim rather than UM code, so the
exit status is squarely its responsibility.

**Suggested patch.** Carried as
`fortran/patches/0002-ereport-nonzero-exit-status.patch`: `STOP 1` in place of
the bare `STOP`, preceded by `FLUSH(6)` so the diagnostic is not lost when
output is redirected. Reachable only on the abort path, so no numerical effect.

## UP-9 — header documents `SE_INS = 0.3`, live value is 1.0

`ukca_conden.F90:53` versus `:237`.

**Trust the code.** The live value is `1.0` at `:237`; the header's `0.3` is
stale.

**Impact.** Documentation only. Reported because `se_ins` is a sticking
coefficient a user might reasonably tune, and the header is where they would
look for its current value.

**Suggested patch.** Correct the header at `:52-53` to state `SE_INS = 1.0`,
matching `:237`. If `0.3` was the intended physics, that is a separate and much
larger question than a comment fix.

## UP-10 — insoluble-mode `num_eps` indexed by the soluble mode

`ukca_conden.F90:372-387`, inside `DO imode = mode_nuc_sol, mode_cor_sol`:

```fortran
372:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_ait_insol) > num_eps(imode) )   ! imode = 2
377:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_acc_insol) > num_eps(imode) )   ! imode = 3
382:  mask3i(:) = mask2(:) .AND. ( nd(:,mode_cor_insol) > num_eps(imode) )   ! imode = 4
387:  mask4i(:) = mask2(:) .AND. ( nd(:,mode_sup_insol) > num_eps(imode) )   ! imode = 4
```

`imode` is the **soluble** mode of the enclosing loop; the threshold should
belong to the insoluble mode being tested. Contrast `:366`, which correctly uses
`num_eps(imode)` for `nd(:,imode)`.

Whether that matters depends entirely on which pair of `num_eps` entries the
substitution lands on. For `i_mode_setup = 8` they are

```
num_eps = [1e-8, 1e-8, 1e-8, 1e-14, 1e-8, 1e-14, 1e-14, 1e-20]
mode      nuc   ait   acc   cor    ait   acc    cor    sup
          <------ soluble ------>  <------ insoluble ------>
```

so only **one of the four lines is both wrong and reachable**:

| line | tests mode | uses `num_eps` | should use | effect |
|---|---|---|---|---|
| `:372` | 5 ait_insol | `(2)` = 1e-8 | `(5)` = 1e-8 | **none** — equal |
| `:377` | 6 acc_insol | `(3)` = 1e-8 | `(6)` = 1e-14 | **10⁶ too strict** |
| `:382` | 7 cor_insol | `(4)` = 1e-14 | `(7)` = 1e-14 | **none** — equal |
| `:387` | 8 sup_insol | `(4)` = 1e-14 | `(8)` = 1e-20 | 10⁶ too strict, but **unreachable** |

**`:377` is the live defect.** The threshold is six orders of magnitude too
high, so condensation onto the accumulation-insoluble mode is **suppressed**
whenever `1e-14 < nd(acc_insol) <= 1e-8` — a range the mode occupies while it
is being depleted by ageing. Suppressed condensation means no contribution to
`ageterm1`, so it also feeds back into the ageing rate.

**`:387` cannot be reached by any supported configuration.** `mode_sup_insol`
is active only where `mode_choice(8) = 1`, which is true of exactly two setups —
12 (`sussbcocduntnh_8mode_8cpt`) and 13 (`sussbcocdump_8mode`). Setup 8 is
`i_sussbcocdu_7mode`, `mode_choice = [1,1,1,1,1,1,1,0]`: seven modes, and mode 8
is **off**. Neither 12 nor 13 is implemented by `glomap_box_config_mod`'s
`init_indices`, so there is no reference driver that can exercise `:387` at all.
Confirmed empirically: `mask4i` is false in all 14,400 records across the four
committed branch-dump goldens.

**Impact.** **Changes results on `i_mode_setup = 8`**, via `:377` — the only
supported setup with `mode_acc_insol` active alongside `mode_acc_sol`. The other
three lines are a latent trap rather than a live error: two are exact no-ops
today only because two `num_eps` entries happen to be equal, and would become
live the moment a setup gave them different values.

**Suggested patch.** Fix all four, not only the one that currently bites — the
no-ops are no-ops by coincidence:

```diff
--- a/src/ukca/ukca_conden.F90
+++ b/src/ukca/ukca_conden.F90
-          mask3i(:) = mask2(:) .AND. ( nd(:,mode_acc_insol) > num_eps(imode) )
+          mask3i(:) = mask2(:) .AND.                                           &
+                      ( nd(:,mode_acc_insol) > num_eps(mode_acc_insol) )
```

and the analogous substitution at `:372` (`mode_ait_insol`), `:382`
(`mode_cor_insol`) and `:387` (`mode_sup_insol`).

**History.** Found during the phase A review. The original write-up named `:387`
and `i_mode_setup = 8` as the mechanism, which the phase B review showed to be
wrong on both counts — `mode_sup_insol` is not active in setup 8, and `:387` is
unreachable. The defect is real; the analysis was not. Corrected here and in
`docs/fidelity.md`.

## UP-11 — the negative-size diagnostic overflows its own buffer

`ukca_volume_mode.F90:856-877` builds the message for the
`wetdp`/`drydp`/`wvol`/`dvol`/`rhopar` `<= 0` abort:

```fortran
WRITE(cmessage,'(5(A,E15.6,A,I0),A)')                                      &
     'Minimum value of "wetdp" or "drydp" or "wvol"' //newline//           &
     ...five MINVAL/MINLOC blocks...                                       &
     'Further debugging output in stdout.'
```

`cmessage` is `CHARACTER(LEN=errormessagelength)` and
`errormessagelength_mod.F90:32` sets that to **256**. The assembled text is
roughly **412** characters: two header lines of ~85, five blocks of ~58
(`'MINVAL(x(:,imode)) : '` + `E15.6` + `' at location : '` + `I0` + newline),
and a 35-character trailer.

**Trust the code.** The intent is clear from the message; the buffer is what is
wrong.

**Impact.** Writing past the record length of an internal file raises
`Fortran runtime error: End of record`, which aborts **before** `ereport` is
reached. So the branch never produces the diagnostic it was written to produce:
a run that hits a non-positive diameter or density dies with a message about
record lengths instead of the five `MINVAL`/`MINLOC` values someone went to
real trouble to assemble. The whole block is effectively dead code, and it is
dead exactly when it is most needed.

Found while testing the gate-A `ereport` shim: with the shim making the abort
non-fatal, driving a mode to zero mass reaches this `WRITE` and kills the
process anyway — with the shim's own diagnostic never printed either.

**Why the port is unaffected.** ADR-006: diagnostic and consistency routines
are recorded as deliberately not ported, and the port has no fixed-length
message buffers. Nothing to reproduce.

**Suggested patch.** Widen the buffer at the point of use rather than raising
`errormessagelength` globally, which many other routines depend on:

```diff
--- a/src/ukca/ukca_volume_mode.F90
+++ b/src/ukca/ukca_volume_mode.F90
-CHARACTER(LEN=errormessagelength) :: cmessage
+! The negative-size diagnostic assembles ~412 characters; errormessagelength
+! is 256, and overflowing an internal-file record aborts before ereport runs.
+CHARACTER(LEN=1024) :: cmessage
```

Alternatively split it into several `umPrint` calls and pass `ereport` a short
summary, which is what the trailing "Further debugging output in stdout"
suggests was intended.
