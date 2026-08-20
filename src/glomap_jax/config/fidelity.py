"""Static configuration and the fidelity-flag registry.

Frozen, hashable dataclasses so they can be passed to ``jax.jit`` as static
arguments: every flag becomes a compile-time Python branch with no runtime cost.

The fidelity registry is the important part. This is a *port*, so where the
Fortran does something arguably wrong, the default reproduces the Fortran. That
is not deference — it is the only way a trajectory comparison means anything.
Each flag carries a written rationale in ``docs/fidelity.md``, and
``tests/test_fidelity_registry.py`` fails on any flag that lacks one, whose
default disagrees with its hand-written table, or that no module under ``src/``
reads.

It does **not** check that a flag is exercised at both settings, and
**both-settings tests do not exist yet** -- ``docs/fidelity.md`` says so in
bold. This docstring claimed the opposite, which is worse than saying nothing:
a reader trusts the registry to have caught what it never looked at.

"Obviously a bug, so I fixed it" is how a port stops being a port. The clearest
case below is ``ageing_totage_rescale_noop``, where the naive fix would lose
mass outright.

Not every upstream defect earns a flag. A flag is for a defect the port must
*choose* to reproduce; where there is nothing to choose -- an unreachable
branch, a documentation error, an unsupported switch -- the disposition is
recorded in ``docs/UPSTREAM_DEFECTS.md`` and enforced by
``tests/test_upstream_defects.py`` instead.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FidelityConfig:
    """Upstream quirks. Every default reproduces the Fortran.

    Flipping one to False produces a model that is arguably more correct and
    definitely not GLOMAP. Do that deliberately, never to make a test pass.
    """

    # UP-1. ukca_solvecoagnucl_v.F90:259 solves dN/dt = A*N^2 as
    # 1/(1/N - 3*A*dt); the exact integral has no factor 3. Measured in the
    # branch dump: fires on EVERY substep of every shipped case, for the top
    # *soluble* mode -- mode_cor_sol has no larger soluble mode to coagulate
    # with and no nucleation source, so B = C = 0 and the discriminant is
    # exactly zero. The insoluble-mode argument this comment used to give is
    # the pre-measurement one and is the weaker case: mode 7 is active only in
    # setups 6 and 8, and mode 8 in none the box model supports.
    # Number decays three times too fast. Correcting it by default would break
    # every trajectory comparison.
    coag_intra_factor3: bool = True

    # UP-3. ukca_ageing.F90:296-298 overwrites `naged` before using it as the
    # divisor, so the totage rescale is exactly 1.0. Affects the ageing entries
    # of bud_aer_mas only -- md uses totage1 -- so this changes Gate B budget
    # comparisons, never prognostic state. Do NOT "fix" it into the prognostic
    # path: ukca_conden records insoluble condensate only in ageterm1, so ageing
    # must transfer all of it or mass is lost.
    ageing_totage_rescale_noop: bool = True

    # UP-4 deliberately has NO flag. ukca_conden.F90:353-354 clamps with
    # `delgc_cond = delgc_cond/gc` where `= gc` was intended, but the guard is
    # unreachable: three lines above, delgc_cond = gc*(1-exp(-x)) with x >= 0
    # bounds it in [0, gc]. Both settings would be bit-identical, so no
    # both-settings test could ever distinguish them and the flag would sit
    # here forever as an untestable decision. The unreachability is asserted
    # instead -- see tests/test_upstream_defects.py, which checks the guard is
    # false in every record of every committed branch-dump golden.

    # UP-6. s_cond_s is read by ukca_calcnucrate when cond_on=0 and nucl_on=1,
    # having never been assigned. JAX has no undefined memory, so the port must
    # choose: 0.0 makes the Vehkamaki guard (s_cond_s > 0) fail and the BLN
    # factor collapse to exp(0)=1. This is a place where the port is
    # necessarily better-defined than the reference, and no Fortran golden can
    # exist for that combination until UP-6 is fixed upstream.
    s_cond_s_zero_when_cond_off: bool = True

    # UP-10, found in the phase A review. ukca_conden.F90:372-387 gates
    # insoluble-mode condensation with num_eps indexed by the enclosing SOLUBLE
    # mode rather than the insoluble mode being tested. Only :377 is both wrong
    # and reachable: num_eps(mode_acc_sol) = 1e-8 gates mode_acc_insol, whose
    # own threshold is 1e-14, so condensation is suppressed by a factor of 1e6
    # too strict. :372 and :382 are no-ops because the entries happen to be
    # equal, and :387 is unreachable -- mode_sup_insol is active only in setups
    # 12 and 13, neither of which the box model implements.
    # LATENT, not live -- this was called results-changing on setup 8 twice
    # before it was measured, and that claim is retracted, not qualified:
    # :377 is gated by topmode > mode_ait_insol, and topmode
    # is 5 unless l_dust_mp_ageing is set. Force it on with setup 8 and the mask
    # is still false -- init_state puts nd(mode_acc_insol) at exactly 1e-14 and
    # the test is strictly greater. No both-settings test is possible yet.
    conden_insol_num_eps_by_sol_mode: bool = True

    # Not a defect: ukca_calc_drydiam.F90:245-262 silently rewrites md/mdt for
    # modes 1-3 (nuc/ait/acc soluble only, NOT all eight) whose diameter falls
    # below ddplim0*0.1. Ungated and applied twice per nmts step -- four times
    # per ukca_aero_step call, five per chemistry step at nmts=1 counting the
    # driver's update_size, and 2 + 2*nmts in general -- making it the most
    # frequent state mutation in the model.
    drydiam_undersize_reset: bool = True

    # Upstream switch. Fixes a factor-ten typo in the H+/NO3- coefficient AND
    # restructures the per-pair RH floor; without it, `aw` ratchets upward
    # cumulatively across the ion-pair loop. The box model pins this True.
    l_fix_ukca_water_content: bool = True

    # Upstream switch. In ukca_volume_mode it is abort-only with no numerical
    # effect; its actual numerical effect is in ukca_vapour, where it changes
    # wts = MIN(99.0, MAX(41.0, ws*100)) to MAX(41.0, ...). Registered against
    # the routine where it matters.
    l_fix_neg_pvol_wat: bool = True

    # Upstream switch, uses kappa-Kohler hygroscopicities. Box model default.
    l_fix_ukca_hygroscopicities: bool = True

    # Consistency routines. ukca_check_md_nd is INTENT(IN) throughout and purely
    # diagnostic, so omitting it is exact. ukca_mode_check_artefacts has no
    # caller in the box model and ukca_mode_check_mdt is gated on
    # iextra_checks > 1, which the box model leaves at 0.
    checkmd_nd: bool = False
    iextra_checks: int = 0

    # NOT an upstream defect -- a place where JAX offers something better than
    # the Fortran and taking it would break the port. cubrt_v is literally
    # `x ** (1.0/3.0)`; jnp.cbrt is a genuinely better cube root, returns a real
    # root for negative x where the power form gives NaN, and disagrees on 1763
    # of 1865 swept points by up to 1.3e-14. That output is drydp, which is
    # compared against dp_thresh1 and ddplim0*0.1 -- both step changes -- so the
    # difference flips a merge or an undersize reset rather than shifting a
    # digit. Default False = reproduce the Fortran. See core/numerics.cbrt.
    cbrt_exact: bool = False

    # The corrected NaCl density. UKCA's literal rhocomp(cp_cl) is 1600 kg/m3;
    # the real value is 2165, and l_fix_nacl_density substitutes it. A genuine
    # fidelity flag -- it selects between a wrong number and its correction --
    # and the box model defaults it ON, so True is what reproduces the
    # reference here even though False is what the literal says.
    #
    # It also reaches no_ions, but only when l_fix_ukca_hygroscopicities is
    # also on: ukca_mode_setup.F90:678-679 tests both, so this is not an
    # independent knob for that table.
    l_fix_nacl_density: bool = True

    def __post_init__(self):
        if self.iextra_checks > 1:
            raise NotImplementedError(
                "iextra_checks > 1 activates ukca_mode_check_mdt, which zeroes "
                "number concentration for out-of-range modes and so changes mass "
                "budgets. Not ported; see docs/KEY_DECISIONS.md."
            )
