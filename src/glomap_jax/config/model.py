"""Static configuration and the fidelity-flag registry.

Frozen, hashable dataclasses so they can be passed to ``jax.jit`` as static
arguments: every flag becomes a compile-time Python branch with no runtime cost.

The fidelity registry is the important part. This is a *port*, so where the
Fortran does something arguably wrong, the default reproduces the Fortran. That
is not deference — it is the only way a trajectory comparison means anything.
Each flag carries a written rationale in ``docs/fidelity.md`` and is tested at
both settings, and ``tests/test_fidelity_registry.py`` fails on any flag that
lacks either.

"Obviously a bug, so I fixed it" is how a port stops being a port. The clearest
case below is ``ageing_totage_rescale_noop``, where the naive fix would lose
mass outright.

Not every upstream defect earns a flag. A flag is for a defect the port must
*choose* to reproduce; where there is nothing to choose -- an unreachable
branch, a documentation error, an unsupported switch -- the disposition is
recorded in ``docs/UPSTREAM_DEFECTS.md`` and enforced by
``tests/test_upstream_defects.py`` instead.
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
        # The same ereport cited above for i_nuc_method also covers ibln
        # (ukca_calcnucrate.F90:280-281), which an earlier version left
        # unvalidated -- ModelConfig(ibln=99) constructed happily.
        if self.ibln not in (1, 2, 3):
            raise ValueError(f"ibln={self.ibln} must be 1, 2 or 3")
        if self.icondiam not in (1, 2):
            raise ValueError(
                f"icondiam={self.icondiam} must be 1 or 2; "
                "ukca_conden.F90:279-283 ereports on CASE DEFAULT"
            )
        if self.imerge not in (1, 2, 3):
            raise ValueError(f"imerge={self.imerge} must be 1, 2 or 3")
        if self.ifuchs not in (1, 2):
            raise ValueError(f"ifuchs={self.ifuchs} must be 1 or 2")
        if self.idcmfp not in (1, 2):
            raise ValueError(f"idcmfp={self.idcmfp} must be 1 or 2")
        if self.nmts < 1 or self.nzts < 1:
            raise ValueError("nmts and nzts must be >= 1")


# ---------------------------------------------------------------------------
# Fidelity flags
# ---------------------------------------------------------------------------


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
