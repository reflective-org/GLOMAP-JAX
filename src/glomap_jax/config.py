"""Static configuration and the fidelity-flag registry.

Frozen, hashable dataclasses so they can be passed to ``jax.jit`` as static
arguments: every flag becomes a compile-time Python branch with no runtime cost.

The fidelity registry is the important part. This is a *port*, so where the
Fortran does something arguably wrong, the default reproduces the Fortran. That
is not deference — it is the only way a trajectory comparison means anything.
Each flag carries a written rationale in ``docs/fidelity.md`` and is tested at
both settings, and ``tests/test_fidelity_registry.py`` fails on any flag that
lacks either.

"Obviously a bug, so I fixed it" is how a port stops being a port. Two of the
flags below are cases where the naive fix is actively wrong: ``ageing_totage_``
``rescale_noop`` would lose mass if "corrected", and ``conden_delgc_over_gc``
guards a branch that cannot execute.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Mode configuration
# ---------------------------------------------------------------------------

# i_mode_setup values the box driver accepts. 7 is not a case upstream, and
# 10/12/13 (ncp = 9 or 10) are rejected by glomap_box_config_mod.F90:391.
SUPPORTED_MODE_SETUPS = (1, 2, 3, 4, 5, 6, 8)

MODE_SETUP_NAMES = {
    1: "suss_4mode",
    2: "sussbcoc_5mode",
    3: "sussbcoc_4mode",
    4: "sussbcocso_5mode",
    5: "sussbcocso_4mode",
    6: "duonly_2mode",
    8: "sussbcocdu_7mode",
}


@dataclass(frozen=True)
class ModelConfig:
    """Which GLOMAP configuration to run, and which processes are active."""

    i_mode_setup: int = 1

    # Process switches, matching the box namelist.
    cond_on: bool = True
    nucl_on: bool = True
    coag_on: bool = True
    bln_on: bool = False

    # Scheme selectors. Defaults are the box model's.
    i_nuc_method: int = 2  # 2 = BHN, +BLN in the BL if bln_on
    ibln: int = 1  # 1 activation, 2 kinetic, 3 PNAS
    icoag: int = 1  # 1 GLOMAP kernel, 2 M7, 3 UM. 4 is broken upstream.
    imerge: int = 1  # 1 mid-points, 2 edges, 3 dynamic
    ifuchs: int = 1  # 1 Fuchs(1964), 2 Fuchs-Sutugin(1971)
    idcmfp: int = 1  # diffusion / mean-free-path variant
    icondiam: int = 1  # 1 geometric mean, 2 condensation diameter
    intraoff: bool = False
    interoff: bool = False

    # Cloud processing. 0 = off; ukca_aero_step gates it on iactmethod > 0, so
    # this is off by construction rather than incidentally via zeroed fields.
    iactmethod: int = 0

    # Substepping.
    nmts: int = 1  # microphysics substeps per chemistry step
    nzts: int = 15  # condensation/nucleation competition substeps

    def __post_init__(self):
        # Static validation at construction, which is trace time. The Fortran
        # ereports on these; under jit we cannot raise, so catch them here.
        if self.i_mode_setup not in SUPPORTED_MODE_SETUPS:
            raise ValueError(
                f"i_mode_setup={self.i_mode_setup} is not supported; "
                f"expected one of {SUPPORTED_MODE_SETUPS}"
            )
        if self.icoag == 4:
            raise NotImplementedError(
                "icoag=4 is broken upstream: ukca_coag_coff_v.F90:339-340 reads "
                "mfppi/mfppj, which are only assigned inside the mutually "
                "exclusive icoag==1 block, so it always reads undefined memory. "
                "There is no correct reference to validate against. See "
                "docs/UPSTREAM_DEFECTS.md (UP-5)."
            )
        if self.icoag not in (1, 2, 3):
            raise ValueError(f"icoag={self.icoag} out of range (1-3 supported)")
        if self.i_nuc_method == 1:
            raise ValueError(
                "i_nuc_method=1 does not exist: ukca_calcnucrate.F90:280-285 "
                "ereports for anything outside 2-3, and the header marks it "
                "'Do not use!!'"
            )
        if self.i_nuc_method not in (2, 3):
            raise ValueError(f"i_nuc_method={self.i_nuc_method} must be 2 or 3")
        if self.nmts < 1 or self.nzts < 1:
            raise ValueError("nmts and nzts must be >= 1")


# ---------------------------------------------------------------------------
# Fidelity flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityConfig:
    """Upstream quirks. Every default reproduces the Fortran.

    Flipping one to False produces a model that is arguably more correct and
    definitely not GLOMAP. Do that deliberately, never to make a test pass.
    """

    # UP-1. ukca_solvecoagnucl_v.F90:259 solves dN/dt = A*N^2 as
    # 1/(1/N - 3*A*dt); the exact integral has no factor 3. Reachable EVERY
    # substep for coarse and super-coarse insoluble modes, where inter-modal
    # coagulation is skipped so B = C = 0 and the discriminant is exactly zero.
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

    # UP-4. ukca_conden.F90:353-354 clamps with `delgc_cond = delgc_cond/gc`
    # where `= gc` was intended. The guard is unreachable -- delgc_cond =
    # gc*(1-exp(-x)) with x >= 0 is bounded in [0, gc] -- so both settings are
    # bit-identical today. Kept as a flag so the invariant is asserted rather
    # than assumed.
    conden_delgc_over_gc: bool = True

    # UP-6. s_cond_s is read by ukca_calcnucrate when cond_on=0 and nucl_on=1,
    # having never been assigned. JAX has no undefined memory, so the port must
    # choose: 0.0 makes the Vehkamaki guard (s_cond_s > 0) fail and the BLN
    # factor collapse to exp(0)=1. This is a place where the port is
    # necessarily better-defined than the reference, and no Fortran golden can
    # exist for that combination until UP-6 is fixed upstream.
    s_cond_s_zero_when_cond_off: bool = True

    # Not a defect: ukca_calc_drydiam.F90:245-262 silently rewrites md/mdt for
    # modes 1-3 (nuc/ait/acc soluble only, NOT all eight) whose diameter falls
    # below ddplim0*0.1. Ungated and applied four times per nmts step, making it
    # the most frequent state mutation in the model.
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

    def __post_init__(self):
        if self.iextra_checks > 1:
            raise NotImplementedError(
                "iextra_checks > 1 activates ukca_mode_check_mdt, which zeroes "
                "number concentration for out-of-range modes and so changes mass "
                "budgets. Not ported; see docs/KEY_DECISIONS.md."
            )


@dataclass(frozen=True)
class SolverConfig:
    """Order-2 diffrax settings. Unused by the faithful path."""

    solver: str = "Kvaerno5"
    rtol: float = 1e-6
    atol: float = 1e-9
    max_steps: int = 4096
    # Tuned in carma-jax: Chord was measured 1.6x faster than diffrax's default
    # VeryChord by accepting larger outer steps at the same final accuracy.
    root_finder: str = "Chord"
    pid_gains: tuple[float, float, float] = field(default=(0.3, 0.3, 0.0))
