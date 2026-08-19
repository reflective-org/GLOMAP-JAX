"""Physical constants and numerical thresholds, as the reference defines them.

**These are the reference's values, not CODATA's, and the difference is
deliberate.** UKCA carries `avogadro = 6.022e23` and
`boltzmann = 1.3804e-23`; the 2019 SI definitions are `6.02214076e23` and
`1.380649e-23`. Substituting the better values changes every concentration
conversion in the model by ~4e-5 relative — four orders of magnitude above
`RTOL_ALGEBRAIC` — and no golden would match.

That is the same trap as `jnp.cbrt` (see `core/numerics`): the more accurate
choice is the wrong one for a faithful port. If a future order-2 mode wants
CODATA, it belongs behind a flag with its own goldens, not as a quiet
substitution here.

Every value below cites the file and line it came from, and
`tests/test_constants.py` re-parses those files and compares — so a constant
cannot be retyped wrongly, and cannot silently drift if the vendored tree is
ever updated.

Nothing derived lives here. `mm_da = avogadro * boltzmann / rgas`
(`ukca_cond_coff_v.F90:152`) is computed where it is used, because a derived
quantity cached in a constants table is a second source of truth.
"""

from __future__ import annotations

from typing import Final

# --- ukca_config_constants_mod.F90 -----------------------------------------
# Assigned at runtime by init_config_constants, not PARAMETERs, so the line
# numbers below are the assignments rather than the declarations.

AVOGADRO: Final[float] = 6.022e23
"""Molecules per mole. `ukca_config_constants_mod.F90:129`. Not CODATA."""

BOLTZMANN: Final[float] = 1.3804e-23
"""J K-1. `ukca_config_constants_mod.F90:130`. Not CODATA."""

RMOL: Final[float] = 8.314
"""Universal gas constant, J K-1 mol-1. `ukca_config_constants_mod.F90:122`."""

RHO_SO4: Final[float] = 1769.0
"""Density of a sulfate particle, kg m-3. `ukca_config_constants_mod.F90:131`."""

# --- ukca_constants.F90 -----------------------------------------------------

PI: Final[float] = 3.14159265358979323846
"""`ukca_constants.F90:37`. Written to more digits than float64 holds, as the
Fortran does; both round to the same double."""

ZERODEGC: Final[float] = 273.15
"""0 C in K. `ukca_constants.F90:41`."""

MMSUL: Final[float] = 0.09808
"""Molar mass of H2SO4, kg mol-1. `ukca_constants.F90:59`."""

NMOL: Final[float] = 1.0e2
"""Molecules per new particle at nucleation. `ukca_constants.F90:44`."""

# --- numerical thresholds ---------------------------------------------------
# Not physics. Each one is a branch: a value either side of it selects a
# different code path, which is why they are gathered here rather than left
# inline. See docs/harness.md on gate 0.

CONC_EPS: Final[float] = 1.0e-8
"""Gas concentration below which condensation is skipped.
`ukca_constants.F90:47`. Gates `mask1` in `ukca_conden`."""

DN_EPS: Final[float] = 1.0e-8
"""Number change below which `nd` is left alone. `ukca_constants.F90:48`.
Gates `WHERE (ABS(deln) > dn_eps)` in `ukca_coagwithnucl` — see issue #18, this
predicate is not yet instrumented by the branch dump."""

EPS_AB: Final[float] = 1.0e-20
"""`|A|` and `|B|` below this count as zero in `ukca_solvecoagnucl_v:177`,
selecting between its five closed forms. A local, not a module constant, in the
Fortran."""

EPS_D: Final[float] = EPS_AB * EPS_AB
"""Discriminant tolerance, `ukca_solvecoagnucl_v:178`. Written as the product
rather than `1.0e-40` because that is how the Fortran forms it, and the two are
not required to give the same double."""

SQD_CLAMP: Final[float] = 50.0
"""`SQD*DTZ` is clamped here to keep `EXP` finite
(`ukca_solvecoagnucl_v:218-221`). A discontinuity in the solution, not a
rounding guard."""

XXX_EPS: Final[float] = 1.0e-3
"""Taylor switch in `ukca_coagwithnucl:247`: above it the exponential is
evaluated, below it the linear expansion is used."""

J_EPS: Final[float] = 1.0e-3
"""Nucleation rate below which nucleation is skipped entirely
(`ukca_calcnucrate:343,370,415`). Fully on or fully off, never scaled."""

__all__ = [
    "AVOGADRO",
    "BOLTZMANN",
    "CONC_EPS",
    "DN_EPS",
    "EPS_AB",
    "EPS_D",
    "J_EPS",
    "MMSUL",
    "NMOL",
    "PI",
    "RHO_SO4",
    "RMOL",
    "SQD_CLAMP",
    "XXX_EPS",
    "ZERODEGC",
]
