"""Gas-phase index tables — the port of `ukca_setup_indices` (task 31).

`glomap_box_config_mod`'s `init_indices` calls **two** routines per setup: a
gas-phase one and a mode one. This module is the gas-phase half. It turns
`i_mode_setup` into the 174 integer scalars and four length-50 arrays that say
where each gas species lives in `s0g`, which of them condense, into which
aerosol component, and with what molar mass and molecular diameter.

Byte equality, not a tolerance. These are indices: `mm_gas` and `dimen` feed
`ukca_cond_coff_v` directly, and `condensable_choice` selects which *component*
a vapour condenses into. One wrong entry is not a small error downstream.

Four routines, seven setups
---------------------------

The gas side collapses, and the collapse is the thing most likely to mislead a
reader coming from the mode tables, where all seven setups differ::

    setup 1        -> ukca_indices_sv1          nchemg = 9
    setups 2, 3, 8 -> ukca_indices_orgv1_soto3  nchemg = 11, Sec_Org -> CP_OC
    setups 4, 5    -> ukca_indices_orgv1_soto6  nchemg = 11, Sec_Org -> CP_SO
    setup 6        -> ukca_indices_nochem       nchemg = 0

So `build(2)`, `build(3)` and `build(8)` are equal field for field, and any
check phrased as "the tables differ across setups" is asserting something
false. Setups 2/3/8 and 4/5 differ in exactly one number — the aerosol
component `Sec_Org` condenses into, 3 vs 6.

1-based to 0-based, and the sentinel
------------------------------------

Fortran indexes from 1 and uses **0 for "this species is not in this setup"**,
guarded at every use site — `IF (mh2so4 > 0)` in `glomap_box_state_mod.F90:185`
is the canonical one. Naively subtracting 1 from everything turns "absent" into
-1, which is a *valid* Python index (the last element), and turns index 1 into
0, which then fails a `> 0` guard that was never converted.

So the port stores **0-based indices with `ABSENT = -1`**, and:

* `ABSENT` is never used to index anything. Test presence with
  `idx != ABSENT`, never with `idx > 0` — the 0-based index of the first
  species is 0, and `0 > 0` is false.
* The counts (`nchemg`, `nadvg`, `noffox`, `ntrag`, `nbudchem`, `gasbudget`,
  `ngasbudget`, `ichem`) are **not** indices and are **not** shifted.
* `condensable` is derived from the *unshifted* `condensable_choice`. Deriving
  it after the shift would drop H2SO4, whose component index is 1 in every
  setup that has it — the single most consequential way to get this wrong.

Never initialised on any box-model path
---------------------------------------

`budget`, `nbudget`, `traqu` and `ntraqu` are module variables of
`ukca_setup_indices` that are assigned **only** in `ukca_indices_traqu38` and
`ukca_indices_traqu9`, neither of which `init_indices` calls; and `idustdep`,
`ndustdep` and `nbudaertot` are declared and assigned nowhere at all. Reading
any of them is undefined, so they are deliberately not captured and not ported.
`tests/test_gas_indices.py` re-parses the vendored source and fails if that
ever stops being true.

`ntraer` and `nbudaer` live in the same module but are set by the *mode*-side
routine, so they belong to the aerosol budget map (task 32), not here. They are
captured into the golden anyway — `validation/capture_gas_indices.py` reads
them, and `nbudaer` is cross-checked against `wrap_sizes` — so that task starts
from a reference rather than from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glomap_jax.physics._gas_literals import GAS_LITERALS

ABSENT = -1
"""0-based stand-in for the Fortran's `0` = "not present in this setup".

Deliberately not `None` (which would need a branch at every use) and
deliberately not `0` (which is a valid 0-based index). Anything that indexes
with it is a bug the array bounds will not catch, because -1 is legal in
numpy — so presence is tested, never assumed."""

NCHEMGMAX = GAS_LITERALS["nchemgmax"]
"""`ukca_setup_indices.F90:608`, a PARAMETER. `mm_gas`, `dimen`,
`condensable_choice` and `condensable` are all `dimension(nchemgmax)` and stay
full width: entries past `nadvg` are dummies the Fortran still allocates, and a
port that trimmed them would change every index."""

_SETUP_ROUTINE = GAS_LITERALS["setup_routine"]
_GROUPS = GAS_LITERALS["groups"]
_ROUTINES = GAS_LITERALS["routines"]

# The gas scalars any part of the vendored tree actually reads, from the
# `USE ukca_setup_indices, ONLY:` lists in ukca_aero_step, ukca_conden,
# ukca_calcnucrate, ukca_wetox, ukca_fine_no3_mod, ukca_coarse_no3_mod and
# glomap_box_*. Everything else in the table is carried for fidelity and is
# reachable through `s0` / `st` / `budget` / `reaction`.
_LIVE_S0 = ("mh2so4", "msec_org", "msec_orgi", "msotwo", "mh2o2", "mh2o2f", "mhno3", "mnh3")


def supported_setups() -> tuple[int, ...]:
    return tuple(sorted(_SETUP_ROUTINE))


def _to_zero_based(value: int) -> int:
    """One Fortran index to one Python index. `0` (absent) becomes `ABSENT`.

    The branch is redundant arithmetic — `0 - 1` is already -1 — and it is
    written out anyway, because the two cases mean different things and only
    one of them is an index. `ABSENT` being what plain subtraction produces is
    the reason -1 was chosen over any other sentinel, not a coincidence to
    lean on silently.
    """
    return ABSENT if value == 0 else value - 1


@dataclass(frozen=True)
class GasIndices:
    """One gas-phase configuration, as plain integers and numpy arrays.

    Every `int` in `s0`, `st`, `budget` and `reaction` is **0-based**, with
    `ABSENT` for species the setup does not carry. Every `int` named as a count
    is unshifted.
    """

    setup: int
    routine: str

    # Counts and switches. Not indices; never shifted.
    nchemg: int
    ichem: int
    noffox: int
    nbudchem: int
    gasbudget: int
    ngasbudget: int
    nadvg: int
    ntrag: int

    # 0-based index maps, keyed by the Fortran variable name in lower case.
    s0: dict[str, int]
    st: dict[str, int]
    budget: dict[str, int]
    reaction: dict[str, int]

    # dimension(nchemgmax) tables, indexed by the 0-based `s0` index.
    mm_gas: np.ndarray
    dimen: np.ndarray
    condensable_choice: np.ndarray
    condensable: np.ndarray

    # Kept so the byte-equality gate can compare against the Fortran capture
    # without the test having to reimplement the shift it is checking.
    raw: dict[str, int] = field(repr=False, default_factory=dict)

    @property
    def mh2so4(self) -> int:
        return self.s0["mh2so4"]

    @property
    def msec_org(self) -> int:
        return self.s0["msec_org"]

    @property
    def msec_orgi(self) -> int:
        return self.s0["msec_orgi"]

    @property
    def msotwo(self) -> int:
        return self.s0["msotwo"]

    @property
    def mh2o2(self) -> int:
        return self.s0["mh2o2"]

    @property
    def mh2o2f(self) -> int:
        return self.s0["mh2o2f"]

    @property
    def mhno3(self) -> int:
        return self.s0["mhno3"]

    @property
    def mnh3(self) -> int:
        return self.s0["mnh3"]

    def condensable_species(self) -> tuple[int, ...]:
        """0-based `s0` slots of the condensable vapours, in index order.

        `ukca_aero_step.F90:831-833` is `IF (ichem == 1)` then
        `DO jv = 1, nchemg` then `IF (condensable(jv))`. The bound is `nchemg`,
        not `nchemgmax` — the dummy tail is allocated and never reached — and
        the whole loop is skipped when `ichem` is 0, which is setup 6.
        """
        return tuple(int(i) for i in np.nonzero(self.condensable[: self.nchemg])[0])


def build(setup: int) -> GasIndices:
    """Build the gas-phase index set for one `i_mode_setup`.

    No switches: unlike the mode tables, nothing in `l_radaer`, `i_tune_bc`,
    `l_fix_nacl_density`, `l_fix_ukca_hygroscopicities` or `l_dust_mp_ageing`
    reaches `ukca_setup_indices`. The setup number is the whole input.
    """
    if setup not in _SETUP_ROUTINE:
        raise NotImplementedError(
            f"i_mode_setup = {setup} has no gas-phase routine in "
            f"glomap_box_config_mod's init_indices; have {supported_setups()}"
        )
    routine = _SETUP_ROUTINE[setup]
    lit = _ROUTINES[routine]
    raw: dict[str, int] = dict(lit["scalars"])

    # Derived, recomputed rather than copied -- `ukca_setup_indices.F90:707-708`
    # and the same two lines in each of the other three routines.
    nchemg = raw["nchemg"]
    noffox = raw["noffox"]
    nadvg = 2 + nchemg
    ntrag = nadvg + noffox

    choice_1based = np.array(lit["arrays"]["condensable_choice"], dtype=np.int32)
    # Derived from the UNSHIFTED array. `condensable=(condensable_choice > 0)`
    # is a Fortran 1-based test: H2SO4's component index is 1, so applying the
    # 0-based shift first and then testing `> 0` silently drops sulphate from
    # every setup that has it.
    condensable = choice_1based > 0

    return GasIndices(
        setup=setup,
        routine=routine,
        nchemg=nchemg,
        ichem=raw["ichem"],
        noffox=noffox,
        nbudchem=raw["nbudchem"],
        gasbudget=raw["gasbudget"],
        ngasbudget=raw["ngasbudget"],
        nadvg=nadvg,
        ntrag=ntrag,
        s0={n: _to_zero_based(raw[n]) for n in _GROUPS["s0"]},
        st={n: _to_zero_based(raw[n]) for n in _GROUPS["st"]},
        budget={n: _to_zero_based(raw[n]) for n in _GROUPS["budget"]},
        reaction={n: _to_zero_based(raw[n]) for n in _GROUPS["reaction"]},
        mm_gas=np.array(lit["arrays"]["mm_gas"], dtype=np.float64),
        dimen=np.array(lit["arrays"]["dimen"], dtype=np.float64),
        # Component index, so it gets the same treatment: 1-based into the
        # component table, 0 meaning "does not condense".
        condensable_choice=np.array(
            [_to_zero_based(int(v)) for v in choice_1based], dtype=np.int32
        ),
        condensable=condensable,
        raw=raw,
    )


def live_scalar_names() -> tuple[str, ...]:
    """The `s0` entries the vendored tree actually reads. See `_LIVE_S0`."""
    return _LIVE_S0
