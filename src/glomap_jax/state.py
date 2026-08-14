"""Traced aerosol state, as NamedTuples.

Convention, following the mature ports in this family: traced state is a
``NamedTuple`` (a pytree out of the box, no registration needed), static config
is a frozen dataclass. Every field is annotated with its shape and its units,
because a GLOMAP field's units are frequently not what its name suggests --
``md`` is molecules per particle, not a mass, and ``s0g`` is a volume mixing
ratio times the gridbox air mass, not a concentration.

Shapes. ``nmodes = 8`` and ``ncp_max = 10`` are Fortran PARAMETERs, so those
extents are static. All seven supported mode setups resolve to ``ncp = 6``; the
``ncp = 9/10`` setups are rejected by the box driver. What genuinely varies per
setup is ``nchemg`` (0, 9, 11), ``nadvg`` (2, 11, 13) and ``nbudaer`` (**seven**
distinct values: 8, 46, 89, 104, 107, 123, 138), so those are padded to their
maxima and masked, which lets a single compiled kernel serve every configuration.

Note ``s0g`` and ``s0g_dot`` live in **different index spaces**, which is easy to
miss because they are declared adjacently upstream. ``s0g`` is sized ``nadvg``
(advected tracers) and ``s0g_dot`` is sized ``nchemg`` (chemistry tracers), with
``nadvg = 2 + nchemg``.

Mode index order is fixed regardless of setup:
``0 nuc-sol, 1 ait-sol, 2 acc-sol, 3 cor-sol, 4 ait-ins, 5 acc-ins,
6 cor-ins, 7 sup-ins``. Inactive modes are masked, never removed.

Component index order:
``0 SU, 1 BC, 2 OC, 3 NaCl, 4 DU, 5 SO, 6 NO3, 7 NaNO3, 8 NH4, 9 MP``.
"""

from typing import NamedTuple

import jax.numpy as jnp

# Static extents, matching ukca_mode_setup.F90:68-73.
NMODES = 8
NMODES_SOL = 4
NMODES_INS = 4
NCP_MAX = 10
# nchemgmax, a PARAMETER in ukca_setup_indices.F90. condensable/mm_gas/dimen are
# already dimensioned to it upstream, so padding here is safe.
NCHEMG_MAX = 50
# s0g is indexed in the ADVECTED-tracer space, not the chemistry space:
# ukca_aero_step.F90:453 declares s0g(nbox, nadvg), and nadvg = 2 + nchemg
# (ukca_setup_indices.F90:647). Sizing it by NCHEMG_MAX happens to be large
# enough today (max nadvg = 13) but is the wrong axis and would break silently.
NADVG_MAX = 2 + NCHEMG_MAX
# Largest nbudaer across the seven supported setups (sussbcocdu_7mode). The
# seven values are 8, 46, 89, 104, 107, 123, 138.
NBUDAER_MAX = 138

# Mode indices (0-based; Fortran is 1-based).
MODE_NUC_SOL, MODE_AIT_SOL, MODE_ACC_SOL, MODE_COR_SOL = 0, 1, 2, 3
MODE_AIT_INS, MODE_ACC_INS, MODE_COR_INS, MODE_SUP_INS = 4, 5, 6, 7

# Component indices (0-based).
CP_SU, CP_BC, CP_OC, CP_CL, CP_DU = 0, 1, 2, 3, 4
CP_SO, CP_NO3, CP_NN, CP_NH4, CP_MP = 5, 6, 7, 8, 9

MODE_NAMES = (
    "nucsol",
    "aitsol",
    "accsol",
    "corsol",
    "aitins",
    "accins",
    "corins",
    "supins",
)
CP_NAMES = ("su", "bc", "oc", "cl", "du", "so", "no3", "nn", "nh4", "mp")


class AerosolState(NamedTuple):
    """Prognostic aerosol state — the arrays ukca_aero_step declares INTENT(IN OUT)."""

    nd: jnp.ndarray  # (nbox, NMODES)          number concentration [cm-3]
    md: jnp.ndarray  # (nbox, NMODES, NCP_MAX) per-component mass [molecules ptcl-1]
    mdt: jnp.ndarray  # (nbox, NMODES)          total mass [molecules ptcl-1]
    mdwat: jnp.ndarray  # (nbox, NMODES)          water [molecules ptcl-1]


class DerivedSize(NamedTuple):
    """Diagnostic size and density, recomputed from AerosolState each call.

    Held separately from the prognostic state because these are outputs of
    ukca_calc_drydiam / ukca_volume_mode, and because they are FROZEN across the
    nzts competition loop — a fact the diffrax branch depends on.
    """

    drydp: jnp.ndarray  # (nbox, NMODES)          geometric mean dry diameter [m]
    wetdp: jnp.ndarray  # (nbox, NMODES)          geometric mean wet diameter [m]
    dvol: jnp.ndarray  # (nbox, NMODES)          dry volume [m3]
    wvol: jnp.ndarray  # (nbox, NMODES)          wet volume [m3]
    rhopar: jnp.ndarray  # (nbox, NMODES)          particle density incl. water [kg m-3]
    pvol: jnp.ndarray  # (nbox, NMODES, NCP_MAX) per-component partial volume [-]
    pvol_wat: jnp.ndarray  # (nbox, NMODES)          water partial volume [-]


class GasState(NamedTuple):
    """Gas phase.

    Units are the Fortran's, deliberately. ukca_aero_step recovers a volume
    mixing ratio as ``s0g / sm`` and a number concentration as
    ``(s0g / sm) * aird``, so:

        s0g[jv]     = (conc [molecules cm-3] / aird) * sm
        s0g_dot[jv] =  prod [molecules cm-3 s-1] / aird     (vmr per second)

    The molar-mass ratio is applied internally by ukca_aero_step, twice and
    consistently. Do not apply it here.
    """

    s0g: jnp.ndarray  # (nbox, NADVG_MAX)  vmr * gridbox air mass [kg]
    s0g_dot: jnp.ndarray  # (nbox, NCHEMG_MAX) chemical tendency [vmr s-1]


class Environment(NamedTuple):
    """Air properties, constant over a chemistry step."""

    t: jnp.ndarray  # (nbox,) temperature [K]
    tsqrt: jnp.ndarray  # (nbox,) sqrt(T) [K^0.5]
    pmid: jnp.ndarray  # (nbox,) centre-level pressure [Pa]
    rh: jnp.ndarray  # (nbox,) relative humidity [0-1]
    s: jnp.ndarray  # (nbox,) specific humidity [kg kg-1]
    aird: jnp.ndarray  # (nbox,) air number density [molecules cm-3]
    airdm3: jnp.ndarray  # (nbox,) air number density [molecules m-3]
    rhoa: jnp.ndarray  # (nbox,) air density [kg m-3]
    mfpa: jnp.ndarray  # (nbox,) mean free path of air [m]
    dvisc: jnp.ndarray  # (nbox,) dynamic viscosity [kg m-1 s-1]
    sm: jnp.ndarray  # (nbox,) gridbox air mass [kg]
    height: jnp.ndarray  # (nbox,) height above surface [m]
    htpbl: jnp.ndarray  # (nbox,) boundary layer depth [m]


class BoxState(NamedTuple):
    """Everything the time loop carries."""

    aerosol: AerosolState
    size: DerivedSize
    gas: GasState
    # (nbox, NBUDAER_MAX + 1) per-process mass fluxes [molecules cm-3 per dtc].
    # Index 0 is NOT a null sink: upstream guards every write with
    # `IF (nmasxxx > 0)`, so slot 0 is never written. Kept only so 1-based
    # Fortran budget indices can be used unshifted.
    bud_aer_mas: jnp.ndarray
    # (nbox, NMODES) count of mode-merge events; reset per chemistry step.
    n_merge: jnp.ndarray
