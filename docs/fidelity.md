# Fidelity flags

Every flag in `FidelityConfig` (`src/glomap_jax/config/fidelity.py`) has a default that
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
The header at line 77 repeats the same error, so code and comment agree with
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

**Latent, not live** — a third revision of this claim, and the reason is in
`docs/UPSTREAM_DEFECTS.md`. `:377` is gated by `topmode > mode_ait_insol`, and
`topmode` is 5 unless `l_dust_mp_ageing` is set, so the line does not run in the
default configuration at all. Force the switch on with setup 8 and it runs but
the mask is still false: `init_state` puts `nd(mode_acc_insol)` at exactly
`1e-14`, and the test is strictly greater, so both the wrong threshold and the
right one give false.

**Testability: currently none.** No configuration this repository can build
distinguishes the two settings, so the both-settings test `docs/fidelity.md`
requires of every flag cannot be written yet. It needs `l_dust_mp_ageing`,
setup 8, and a constructed initial `nd(mode_acc_insol)` strictly inside
`(1e-14, 1e-8]`.

**So why keep the flag at all**, when UP-4 lost its flag for being unreachable?
Because the two are different kinds of unreachable. UP-4 is unreachable by
construction — `delgc_cond` is bounded in `[0, gc]`, so no input reaches it.
UP-10 is unreachable by *configuration*: a legal setting of `l_dust_mp_ageing`
plus a legal `nd` reaches it, and a UM run with dust microphysical ageing may
well. The flag records a choice the port will have to make; the missing test is
tracked, not pretended away.

## `drydiam_undersize_reset`

**Default `True` — reproduce the Fortran.** Not a defect.

`ukca_calc_drydiam.F90:245-262` silently rewrites `md`, `mdt`, `dvol` and
`drydp` for any mode whose dry diameter falls below `ddplim0 × 0.1`, resetting
composition to `mlo × mfrac_0`.

Two things make it easy to get wrong. It is **ungated** — no `checkmd_nd`, no
`iextra_checks` — and it runs twice per `nmts` step (four times per
`ukca_aero_step` call, five per chemistry step counting the box driver's own),
making it the most
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
`ukca_vapour`. With the flag **true** — the box model's setting — `:184` gives
`wts = MIN(99.0, MAX(41.0, ws*100))`; the unfixed branch at `:188` is
`MAX(41.0, ws*100)`, with no upper clamp. So `True` is the clamped form, which
matters for the stratospheric density branch. (This entry previously named the
two the wrong way round, which would have led a porter to implement the unfixed
branch as the default.)

Registering it against the wrong routine would make its default meaningless.

## `l_fix_ukca_hygroscopicities`

**Default `True` — matches the box model.** Uses kappa-Köhler hygroscopicities
(Petters and Kreidenweis 2007) rather than the legacy dissociating-ion counts.

## `checkmd_nd`

**Default `False` — omitting it is exact.**

`ukca_check_md_nd` declares `nd`, `mdt` and `md` as `INTENT(IN)` and only prints
and warns. It clamps nothing and redistributes nothing, so not porting it has
zero effect on results — an argument from the `INTENT(IN)` declarations, which
is sound but is not a measurement.

**No test asserts this yet.** An earlier version of this entry claimed one
existed; it never did. Task 79 is where it lands: run the Fortran with
`checkmd_nd=1` and `=0` and assert bit-identical output.

## `iextra_checks`

**Default `0`. Values above 1 raise `NotImplementedError`.**

`iextra_checks > 1` activates `ukca_mode_check_mdt`, which does mutate state: it
zeroes number concentration for modes whose total mass falls outside
`[mlo×0.001, mhi×1000]` and resets composition to the mid-point, so mass is
silently removed rather than redistributed. It also mutates the caller's mask
in place, which needs the sequential `icp` scan.

Not ported. `ukca_mode_check_artefacts` has no caller in the box model at all.

## `cbrt_exact`

**Default `False` — reproduce the Fortran.** Not an upstream defect: a place
where JAX offers something better and taking it would break the port.

`ukca_um_legacy_mod.F90:450` defines `cubrt_v` as literally

```fortran
y(i) = x(i) ** (1.0 / 3.0)
```

That is a power, not a cube root, and the two are not the same computation.
Measured over 1,865 swept points: `x ** (1.0/3.0)` in JAX is **bit-identical**
to the Fortran; `jnp.cbrt` differs on **1,756** of them by up to **1.3e-14**,
which is **0.13 times** `RTOL_ALGEBRAIC`, not a hundred times it as this said
for three reviews. The magnitude was never the argument; the branch flip is.

The reason that matters is not the size. `cubrt_v` produces `drydp`, and
`drydp` is compared directly against `dp_thresh1` (`ukca_remode.F90:234` — merge
or not) and against `ddplim0*0.1` (`ukca_calc_drydiam.F90:250` — rewrite `md`
and `mdt` or not). Both are step changes, so a parcel sitting within 1.3e-14 of
either threshold goes one way in the reference and the other in the port, and
the trajectories separate by O(1).

They also disagree about negatives: `x ** (1.0/3.0)` is a non-integer power of a
negative and is `NaN`, while `jnp.cbrt` returns the real root. That is *more*
correct, and is exactly why it cannot be the faithful path. Unreachable today —
`dvol >= 0` everywhere `cubrt_v` is called — but it is the failure a `cbrt` port
would produce the first time it was not.

Setting `True` selects `jnp.cbrt`. It will disagree with every committed golden,
which is the correct behaviour for an accuracy option: it belongs to order 2,
alongside diffrax, not to the faithful path.

## `l_fix_nacl_density`

**Default `True` — matches the box model**, which is worth stating carefully
because for once the default is *not* "reproduce the literal".

`ukca_mode_setup.F90` lays down `rhocomp(cp_cl) = 1600.0` kg m⁻³ for sea salt,
then substitutes `rho_nacl = 2165.0` at `:433-435` when the flag is on. 2165 is
the correct density of NaCl; 1600 is not. `glomap_box_config_mod` sets the flag
`.TRUE.`, so the reference this port is validated against uses 2165 — and
reproducing the reference means defaulting to the corrected value.

The switch is applied to `rhocomp` **before** the mode masses are derived from
it, so it moves `mmid`, `mlo` and `mhi` too — by the full 2165/1600 ratio, 35%,
on any mode carrying sea salt. Applying it after the masses would leave them
built from the uncorrected density, silently. Captured at both settings as the
`nacl_off` golden.

It also reaches `no_ions`, but not independently: `:678-679` tests
`l_fix_ukca_hygroscopicities .AND. l_fix_nacl_density`, so with
hygroscopicities off this flag has no effect on that table at all.

Setting `False` reverts to the literal 1600 and will disagree with every
golden.
