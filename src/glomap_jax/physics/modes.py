"""Mode and component tables — the port of `ukca_mode_setup` (phase C).

`common_mode_setup_interface` dispatches on `i_mode_setup` to one routine per
configuration, each of which lays out a table of literals and then derives the
rest. This module does the same, and the split matters:

* **Literals** live in `_SETUPS` below, transcribed from the Fortran with the
  source line beside each one. They are data, not decisions.
* **Derived quantities are recomputed, never copied from the golden.** That is
  the acceptance criterion for this phase, and it is the whole point — copying
  `mmid` out of the reference would produce a table that matches and a port
  that has not implemented anything. Recomputing it means `mmid` is wrong here
  if the formula is wrong, which is what we want to find out.

Byte equality, not `allclose`. These tables feed every process routine, so a
diameter one ulp out is not a small error downstream — it is a different model,
and the branch predicates that read `drydp` would start disagreeing. See
`tests/test_modes.py`.

Two traps this module is shaped around, both found while capturing the golden:

* **`topmode` is not the highest active mode.** It is `nmodes` when
  `l_dust_mp_ageing` and `mode_ait_insol` (5) otherwise, regardless of
  `mode_choice` (`ukca_mode_setup.F90:418-422`).
* **`component_mode` is a permission table, not a presence table.**
  `component` is the intersection of "allowed in this mode", "this component is
  chosen" and "this mode is on".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glomap_jax.core.constants import AVOGADRO, PI

NMODES = 8
"""`ukca_mode_setup.F90:68`. A PARAMETER, not a runtime value: the tables are
always full width and inactive slots still carry values, so a port that stored
only the active modes would change every index."""

# Mode slots. 1-4 soluble, 5-8 insoluble, in every setup -- `modesol` is
# [1,1,1,1,0,0,0,0] everywhere and only `mode_choice` varies.
MODE_NUC_SOL, MODE_AIT_SOL, MODE_ACC_SOL, MODE_COR_SOL = 0, 1, 2, 3
MODE_AIT_INSOL, MODE_ACC_INSOL, MODE_COR_INSOL, MODE_SUP_INSOL = 4, 5, 6, 7

MODESOL = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)

# Component slots referenced by the density switches. `ukca_mode_setup.F90:76,78`.
CP_BC, CP_CL = 1, 3

# Corrected densities the switches substitute. `:187,195,199`.
RHO_NACL = 2165.0
RHO_BC_MG_MIX = 1800.0
RHO_BC_TUNED = 1900.0

# i_tune_bc values. Only meaningful when l_radaer is on.
I_UKCA_BC_TUNED = 1
I_UKCA_BC_MG_MIX = 2


@dataclass(frozen=True)
class ModeTables:
    """One `glomap_variables_type`, as plain arrays.

    Frozen so it can be a static argument to `jax.jit`: the tables are
    configuration, fixed at trace time, not traced state.
    """

    setup: int
    ncp: int
    topmode: int
    component_names: tuple[str, ...]

    mode_choice: np.ndarray
    modesol: np.ndarray
    mode: np.ndarray

    ddplim0: np.ndarray
    ddplim1: np.ndarray
    ddpmid: np.ndarray
    sigmag: np.ndarray
    x: np.ndarray
    num_eps: np.ndarray
    mmid: np.ndarray
    mlo: np.ndarray
    mhi: np.ndarray
    fracbcem: np.ndarray
    fracocem: np.ndarray

    component_choice: np.ndarray
    soluble_choice: np.ndarray
    soluble: np.ndarray
    mm: np.ndarray
    rhocomp: np.ndarray
    no_ions: np.ndarray

    component_mode: np.ndarray
    component: np.ndarray
    mfrac_0: np.ndarray


# ---------------------------------------------------------------------------
# Literals, transcribed from ukca_mode_setup.F90 with their line numbers.
# ---------------------------------------------------------------------------

_SUSS_4MODE = {
    "ncp": 6,  # :46
    "component_names": ("h2so4", "bcarbon", "ocarbon", "nacl", "dust", "sec_org"),  # :49-50
    "mode_choice": [1, 1, 1, 1, 0, 0, 0, 0],  # :53
    "component_choice": [1, 0, 0, 1, 0, 0],  # :56
    "soluble_choice": [1, 0, 0, 1, 0, 0],  # :57
    "component_mode": [  # :60-67, "allowed in <mode>"
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
    ],
    "ddplim0": [1.0e-9, 1.0e-8, 1.0e-7, 0.5e-6, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-6],  # :70-71
    "ddplim1": [1.0e-8, 1.0e-7, 0.5e-6, 1.0e-5, 1.0e-7, 1.0e-6, 1.0e-5, 5.0e-5],  # :73-74
    "sigmag": [1.59, 1.59, 1.40, 2.0, 1.59, 1.59, 2.0, 1.8],  # :77
    "num_eps": [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-14, 1.0e-8, 1.0e-14, 1.0e-14, 1.0e-20],  # :90-91
    "mfrac_0": [  # :94-101
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ],
    "mm": [0.098, 0.012, 0.0168, 0.05844, 0.100, 0.0168],  # :104
    "rhocomp": [1769.0, 1500.0, 1500.0, 1600.0, 2650.0, 1500.0],  # :109-110
    # no_ions is switch-dependent; see _no_ions below. :157-165
    "no_ions": {
        (True, True): [1.88, 0.0, 0.06, 2.23, 0.0, 0.06],
        (True, False): [1.88, 0.0, 0.06, 3.04, 0.0, 0.06],
        "default": [3.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    },
    "fracbcem": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # :167
    "fracocem": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # :168
}

_SETUPS: dict[int, dict] = {1: _SUSS_4MODE}


def supported_setups() -> tuple[int, ...]:
    return tuple(sorted(_SETUPS))


# ---------------------------------------------------------------------------
# Derived quantities. Recomputed, never copied.
# ---------------------------------------------------------------------------


def _x(sigmag: np.ndarray) -> np.ndarray:
    """`ukca_mode_setup.F90:80-85`: `EXP(4.5 * LOG(sg) * LOG(sg))`.

    Written as the Fortran writes it — two separate `LOG` calls multiplied,
    not `LOG(sg)**2`. They need not give the same double.
    """
    log_sg = np.log(sigmag)
    return np.exp(4.5 * log_sg * log_sg)


def _ddpmid(ddplim0: np.ndarray, ddplim1: np.ndarray) -> np.ndarray:
    """`:129-134`: the geometric mean, written as `EXP(0.5*(LOG a + LOG b))`.

    NOT `sqrt(a*b)`. Algebraically identical, numerically not, and `ddpmid`
    feeds `mmid` which feeds the mode-merging thresholds.
    """
    return np.exp(0.5 * (np.log(ddplim0) + np.log(ddplim1)))


def _mode_masses(
    ddplim0: np.ndarray,
    ddpmid: np.ndarray,
    ddplim1: np.ndarray,
    x: np.ndarray,
    mfrac_0: np.ndarray,
    rhocomp: np.ndarray,
    mm: np.ndarray,
    ncp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`:136-155`.

    Two associativity traps here, and I walked into the second one.

    `rhommav` is summed over components in index order, so the accumulation is
    a loop rather than a vector `sum` — a pairwise or vectorised reduction
    associates differently and need not give the same double.

    And the three products are written in the Fortran's exact factor order:

        (pi/6) * (d**3) * (rhommav*avogadro) * x

    left-associated, evaluated as `(((pi/6) * d**3) * (rhommav*avogadro)) * x`.
    Factoring out the `(pi/6) * (rhommav*avogadro) * x` that all three share is
    the obvious optimisation and it is wrong: it reassociates, and all three
    masses stopped being byte-equal. Multiplication is commutative in the reals
    and not in float64. Leave the redundancy.

    And `d**3` is written `d * d * d`. gfortran expands an integer literal
    exponent into repeated multiplication; numpy's `**` calls `pow()`. The two
    disagree by one ulp on two of the eight modes here — enough to fail byte
    equality, and this is the last place you would look for it.
    """
    mmid = np.empty(NMODES, dtype=np.float64)
    mlo = np.empty(NMODES, dtype=np.float64)
    mhi = np.empty(NMODES, dtype=np.float64)

    for imode in range(NMODES):
        rhommav = 0.0
        for icp in range(ncp):
            rhommav = rhommav + mfrac_0[imode, icp] * (rhocomp[icp] / mm[icp])
        dm, d0, d1 = ddpmid[imode], ddplim0[imode], ddplim1[imode]
        mmid[imode] = (PI / 6.0) * (dm * dm * dm) * (rhommav * AVOGADRO) * x[imode]
        mlo[imode] = (PI / 6.0) * (d0 * d0 * d0) * (rhommav * AVOGADRO) * x[imode]
        mhi[imode] = (PI / 6.0) * (d1 * d1 * d1) * (rhommav * AVOGADRO) * x[imode]

    return mmid, mlo, mhi


def _component(
    component_mode: np.ndarray, component_choice: np.ndarray, mode_choice: np.ndarray
) -> np.ndarray:
    """`:185-199`. Allowed in this mode AND chosen AND the mode is on.

    `component_mode` is a *permission* table ("allowed in nuc_sol" in the
    source), not a presence table — assuming they were the same is a mistake
    that survives five of the seven setups.
    """
    allowed = component_mode == 1
    chosen = (component_choice == 1)[None, :]
    active = (mode_choice == 1)[:, None]
    return allowed & chosen & active


def build(
    setup: int,
    *,
    l_dust_mp_ageing: bool = False,
    l_fix_nacl_density: bool = True,
    l_fix_ukca_hygroscopicities: bool = True,
    l_radaer: bool = False,
    i_tune_bc: int = I_UKCA_BC_TUNED,
) -> ModeTables:
    """Build the tables for one `i_mode_setup`.

    The switches change the tables in a specific ORDER, and the order matters:
    `rhocomp` is laid down as a literal, then patched by `l_radaer`/`i_tune_bc`
    and `l_fix_nacl_density`, and only then do `mmid`/`mlo`/`mhi` get computed
    from it (`ukca_mode_setup.F90:109-155`). Applying a density switch after
    the masses would leave the masses built from the unpatched density —
    silently, and by 35% on the NaCl mode.

    Defaults match `glomap_box_config_mod`'s, so `build(1)` reproduces what the
    box model builds.
    """
    if setup not in _SETUPS:
        raise NotImplementedError(
            f"i_mode_setup = {setup} is not ported yet; have {supported_setups()}"
        )
    lit = _SETUPS[setup]
    ncp = lit["ncp"]

    mode_choice = np.array(lit["mode_choice"], dtype=np.int32)
    component_choice = np.array(lit["component_choice"], dtype=np.int32)
    soluble_choice = np.array(lit["soluble_choice"], dtype=np.int32)
    component_mode = np.array(lit["component_mode"], dtype=np.int32)
    ddplim0 = np.array(lit["ddplim0"], dtype=np.float64)
    ddplim1 = np.array(lit["ddplim1"], dtype=np.float64)
    sigmag = np.array(lit["sigmag"], dtype=np.float64)
    mfrac_0 = np.array(lit["mfrac_0"], dtype=np.float64)
    mm = np.array(lit["mm"], dtype=np.float64)
    rhocomp = np.array(lit["rhocomp"], dtype=np.float64)

    # Density switches, applied BEFORE the masses are derived from rhocomp.
    # :424-431
    if l_radaer:
        if i_tune_bc == I_UKCA_BC_TUNED:
            rhocomp[CP_BC] = RHO_BC_TUNED
        elif i_tune_bc == I_UKCA_BC_MG_MIX:
            rhocomp[CP_BC] = RHO_BC_MG_MIX
    # :433-435
    if l_fix_nacl_density:
        rhocomp[CP_CL] = RHO_NACL

    x = _x(sigmag)
    ddpmid = _ddpmid(ddplim0, ddplim1)
    mmid, mlo, mhi = _mode_masses(ddplim0, ddpmid, ddplim1, x, mfrac_0, rhocomp, mm, ncp)

    key = (l_fix_ukca_hygroscopicities, l_fix_nacl_density)
    no_ions = np.array(
        lit["no_ions"].get(
            key if l_fix_ukca_hygroscopicities else "default", lit["no_ions"]["default"]
        ),
        dtype=np.float64,
    )

    return ModeTables(
        setup=setup,
        ncp=ncp,
        # :117-121 -- NOT the highest active mode.
        topmode=NMODES if l_dust_mp_ageing else MODE_AIT_INSOL + 1,
        component_names=tuple(lit["component_names"]),
        mode_choice=mode_choice,
        modesol=MODESOL.copy(),
        mode=mode_choice > 0,  # :183
        ddplim0=ddplim0,
        ddplim1=ddplim1,
        ddpmid=ddpmid,
        sigmag=sigmag,
        x=x,
        num_eps=np.array(lit["num_eps"], dtype=np.float64),
        mmid=mmid,
        mlo=mlo,
        mhi=mhi,
        fracbcem=np.array(lit["fracbcem"], dtype=np.float64),
        fracocem=np.array(lit["fracocem"], dtype=np.float64),
        component_choice=component_choice,
        soluble_choice=soluble_choice,
        soluble=soluble_choice == 1,  # :195-197
        mm=mm,
        rhocomp=rhocomp,
        no_ions=no_ions,
        component_mode=component_mode,
        component=_component(component_mode, component_choice, mode_choice),
        mfrac_0=mfrac_0,
    )


__all__ = ["NMODES", "ModeTables", "build", "supported_setups"]
