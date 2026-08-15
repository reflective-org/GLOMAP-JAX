# Fidelity flags

Every flag in `FidelityConfig` (`src/glomap_jax/config.py`) has a default that
**reproduces the Fortran**, including where the Fortran is wrong. That is not
deference; it is the only way a trajectory comparison against the reference
means anything.

`tests/test_fidelity_registry.py` fails if a flag exists without a section here,
or a section exists without a flag, so this file cannot drift out of date
silently.

**Both-settings tests do not exist yet.** They land with the phase that ports
the routine each flag governs. Two of them will additionally need a `src/box/`
overlay to produce a non-default reference at all: `l_fix_ukca_water_content`
is hard-set in code rather than namelist-exposed
(`glomap_box_config_mod.F90:322`), and `s_cond_s_zero_when_cond_off` has no
Fortran counterpart by construction (see UP-6).

Flipping a flag to the non-Fortran setting gives a model that is arguably more
correct and definitely not GLOMAP. Do it deliberately, never to make a test
pass.

---

## `coag_intra_factor3`

**Default `True` — reproduce the Fortran.** Upstream defect UP-1.

`ukca_solvecoagnucl_v.F90:259` integrates `dN/dt = A·N²` as
`1/(1/N − 3·A·Δt)`. The exact solution is `1/(1/N − A·Δt)`; there is no factor 3.
The header at line 78 repeats the same error, so code and comment agree with
each other and both disagree with the mathematics.

**Reachable every substep.** In `ukca_coagwithnucl.F90:462` the top insoluble
modes skip inter-modal coagulation entirely, so `B ≡ 0` and `C ≡ 0`, making the
discriminant exactly zero in any precision. This branch therefore fires on every
substep for coarse and super-coarse insoluble modes, where intra-modal number
loss runs three times too fast. It never blows up (`A < 0`, so the denominator
only grows), which is why it has survived.

Setting `False` uses the analytic solution and will disagree with every Fortran
golden.

## `ageing_totage_rescale_noop`

**Default `True` — reproduce the Fortran.** Upstream defect UP-3.

`ukca_ageing.F90:296-298` assigns `naged = nd(jl,imode)` and then, on the next
line, computes `totage = totage * nd(jl,imode) / naged` — a factor of exactly
1.0. The intended rescale is not implemented.

**Scope: diagnostics only.** `totage` is read only by the `SUM(totage) > 0` gate
and by `bud_aer_mas`; the prognostic `md` update uses `totage1`. So this flag
changes Gate B budget comparisons and never `nd`/`md`/`mdt`.

**Do not "fix" this into the prognostic path.** Scaling the transfer down would
lose mass: `ukca_conden` records condensate onto insoluble modes only in
`ageterm1` and never adds it to insoluble `md`, so ageing must move all of it.

## `s_cond_s_zero_when_cond_off`

**Default `True` — the port is necessarily better-defined here.** Upstream
defect UP-6.

`s_cond_s` is a local in `ukca_aero_step` (line 504), assigned only under
`IF (cond_on == 1)` and read under `IF (nucl_on == 1)`. With `cond_on=0,
nucl_on=1` it is read having never been assigned. The Vehkamäki path guards
`s_cond_s > 0` so garbage silently zeroes nucleation, but the BLN path at
`ukca_calcnucrate.F90:413` feeds it straight into `EXP` unguarded.

JAX has no undefined memory, so the port must choose a value. `0.0` makes the
Vehkamäki guard fail and the BLN factor collapse to `exp(0) = 1`.

**No Fortran golden can exist for that combination** until UP-6 is fixed, which
is also why `glomap-box` ships `cond_only`, `coag_only` and `all_off` namelists
but no `nucl_only`.

## `conden_insol_num_eps_by_sol_mode`

**Default `True` — reproduce the Fortran.** Upstream defect UP-10, found during
the phase A adversarial review and absent from the original plan.

`ukca_conden.F90:372-387` gates condensation onto each insoluble mode with
`num_eps` indexed by the enclosing **soluble** mode rather than the insoluble
mode being tested. Line 366 in the same routine uses `num_eps(imode)` correctly
for `nd(:,imode)`, which is what makes this look like a copy-paste slip rather
than intent.

Whether it matters depends on which pair of `num_eps` entries each line lands
on. With `num_eps = [1e-8, 1e-8, 1e-8, 1e-14, 1e-8, 1e-14, 1e-14, 1e-20]`, only
`:377` is both wrong and reachable:

```fortran
mask3i(:) = mask2(:) .AND. ( nd(:,mode_acc_insol) > num_eps(imode) )  ! imode = 3
```

`num_eps(3) = 1e-8` where `num_eps(mode_acc_insol) = 1e-14` belongs — six orders
of magnitude too **strict**, so condensation onto the accumulation-insoluble
mode is suppressed while `1e-14 < nd <= 1e-8`. `:372` and `:382` are exact
no-ops because the two entries happen to be equal; `:387` is unreachable,
because `mode_sup_insol` is active only in setups 12 and 13 and neither is
implemented by the box model.

**Changes results on `i_mode_setup = 8`** — via `:377`, the accumulation pair.
Setup 8 is `sussbcocdu_7mode`, `mode_choice = [1,1,1,1,1,1,1,0]`: it has
`mode_acc_insol` but **not** `mode_sup_insol`.

**Testability.** Both settings are distinguishable only on a setup with an
insoluble accumulation mode, i.e. setup 8 among those supported. Setting `False`
lowers the threshold to `1e-14` and admits condensation the Fortran suppresses,
so it will disagree with any setup-8 golden.

## `drydiam_undersize_reset`

**Default `True` — reproduce the Fortran.** Not a defect.

`ukca_calc_drydiam.F90:245-262` silently rewrites `md`, `mdt`, `dvol` and
`drydp` for any mode whose dry diameter falls below `ddplim0 × 0.1`, resetting
composition to `mlo × mfrac_0`.

Two things make it easy to get wrong. It is **ungated** — no `checkmd_nd`, no
`iextra_checks` — and it runs four times per `nmts` step, making it the most
frequently applied state mutation in the model. And it covers **modes 1–3 only**
(`DO imode = mode_nuc_sol, mode_acc_sol`), not all eight.

## `l_fix_ukca_water_content`

**Default `True` — matches the box model**, which pins it in code
(`glomap_box_config_mod.F90:322`).

The upstream switch does two things: it corrects a factor-ten typo in the
H⁺/NO₃⁻ `j=6` coefficient (`-1.220611402e2` → `e3`), and it restructures the
per-electrolyte RH floor. Without the fix, `aw` is set once before the ion-pair
loop and clamped upward *inside* it, so it ratchets to the maximum `rh_min` seen
so far and pairs later in loop order see a wrongly elevated water activity.

That cumulative clamp means the unfixed branch is **loop-carried in `aw` as well
as `cli`** — a broadcast implementation silently reproduces the *fixed*
behaviour while the flag claims otherwise.

## `l_fix_neg_pvol_wat`

**Default `True` — matches the box model.**

Registered against `ukca_vapour`, not `ukca_volume_mode`. In `volume_mode`
(lines 882-898) it only adds a fatal `ereport` guard on negative `pvol_wat` or
`mdwat` and has no numerical effect. Its actual numerical effect is in
`ukca_vapour`, where it changes `wts = MIN(99.0, MAX(41.0, ws*100))` to
`MAX(41.0, ws*100)` — which matters for the stratospheric density branch.

Registering it against the wrong routine would make its default meaningless.

## `l_fix_ukca_hygroscopicities`

**Default `True` — matches the box model.** Uses kappa-Köhler hygroscopicities
(Petters and Kreidenweis 2007) rather than the legacy dissociating-ion counts.

## `checkmd_nd`

**Default `False` — omitting it is exact.**

`ukca_check_md_nd` declares `nd`, `mdt` and `md` as `INTENT(IN)` and only prints
and warns. It clamps nothing and redistributes nothing, so not porting it has
zero effect on results. Verified, and asserted by a test that `checkmd_nd=1`
gives bit-identical Fortran output to `checkmd_nd=0`.

## `iextra_checks`

**Default `0`. Values above 1 raise `NotImplementedError`.**

`iextra_checks > 1` activates `ukca_mode_check_mdt`, which does mutate state: it
zeroes number concentration for modes whose total mass falls outside
`[mlo×0.001, mhi×1000]` and resets composition to the mid-point, so mass is
silently removed rather than redistributed. It also mutates the caller's mask
in place, which needs the sequential `icp` scan.

Not ported. `ukca_mode_check_artefacts` has no caller in the box model at all.
