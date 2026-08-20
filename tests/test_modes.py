"""Tasks 25-29: `physics/modes.py` for every supported setup, byte-equal.

**`array_equal`, not `allclose`.** These tables feed every process routine, so
a diameter one ulp out is not a small error downstream — `drydp` is compared
against `dp_thresh1` and `ddplim0*0.1`, both step changes, and a one-ulp shift
flips them for any parcel on the boundary. A tolerance here would defer the
failure to somewhere it cannot be diagnosed.

**Derived quantities are recomputed, not copied.** That is the phase's
acceptance criterion and the reason these tests mean anything: reading `mmid`
out of the golden would produce a module that passes and implements nothing.

Getting to byte equality took three corrections, each a different way for
algebraically-identical code to give a different double. They are pinned below
as individual tests, because each one is a mistake the next person will make.

No `fortran` marker — the golden is committed, so this runs in CI.
"""

from pathlib import Path

import numpy as np
import pytest

from glomap_jax.physics import modes

GOLDEN = Path(__file__).parent / "goldens" / "modes.f64.tables.npz"

ARRAY_FIELDS = (
    "mode_choice",
    "modesol",
    "mode",
    "ddplim0",
    "ddplim1",
    "ddpmid",
    "sigmag",
    "x",
    "num_eps",
    "mmid",
    "mlo",
    "mhi",
    "fracbcem",
    "fracocem",
    "component_choice",
    "soluble_choice",
    "soluble",
    "mm",
    "rhocomp",
    "no_ions",
    "component_mode",
    "component",
    "mfrac_0",
)


@pytest.fixture(scope="module")
def golden():
    assert GOLDEN.is_file(), "run `python validation/capture_modes.py`"
    return np.load(GOLDEN, allow_pickle=False)


SETUPS = modes.supported_setups()


@pytest.fixture(scope="module")
def built():
    return modes.build(1)


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("field", ARRAY_FIELDS)
def test_field_is_byte_equal_to_the_fortran(golden, setup, field):
    got = np.asarray(getattr(modes.build(setup), field))
    if got.dtype == bool:
        got = got.astype(np.int32)
    np.testing.assert_array_equal(
        got, golden[f"s{setup}_{field}"], err_msg=f"setup {setup}, {field}"
    )


@pytest.mark.parametrize("setup", SETUPS)
def test_scalars_match(golden, setup):
    t = modes.build(setup)
    assert t.ncp == int(golden[f"s{setup}_ncp"])
    assert t.topmode == int(golden[f"s{setup}_topmode"])
    assert list(t.component_names) == list(golden[f"s{setup}_component_names"])


def test_every_supported_setup_is_ported(golden):
    assert set(SETUPS) == set(int(s) for s in golden["_setups"])


@pytest.mark.parametrize("setup", SETUPS)
def test_every_captured_field_is_checked(golden, setup):
    """A field the golden carries and this file never compares is an untested
    part of the port that looks tested."""
    captured = {k.split("_", 1)[1] for k in golden if k.startswith(f"s{setup}_")}
    checked = set(ARRAY_FIELDS) | {"ncp", "topmode", "component_names", "nmodes"}
    assert captured - checked == set(), f"uncompared: {sorted(captured - checked)}"


def test_unsupported_setups_raise_rather_than_return_wrong_tables():
    """10-13 exist in UKCA but the box model rejects them, so there is no
    reference. Returning plausible tables for one would be worse than failing."""
    for setup in (7, 9, 10, 11, 12, 13):
        with pytest.raises(NotImplementedError, match="not ported"):
            modes.build(setup)


def test_the_setups_are_not_all_the_same_tables():
    """Every byte-equality test above would still pass if `build` ignored its
    argument and the golden had been captured from one setup seven times."""
    signatures = {s: modes.build(s).mode.tobytes() for s in SETUPS}
    assert len(set(signatures.values())) >= 4


def test_dust_only_setup_has_no_soluble_modes():
    """Setup 6: condensation, nucleation and ageing are all structurally
    no-ops, which is what makes it worth gating on."""
    t = modes.build(6)
    assert not t.mode[:4].any()
    assert t.mode[modes.MODE_ACC_INSOL] and t.mode[modes.MODE_COR_INSOL]


def test_mode_sup_insol_is_never_active():
    """Slot 8 needs setup 12 or 13, neither of which the box model implements.
    Any code path guarded on it has no reference in either setting of a
    fidelity flag -- see UP-10."""
    assert not any(modes.build(s).mode[modes.MODE_SUP_INSOL] for s in SETUPS)


# --------------------------------------------------------------------------
# The three ways algebraically-identical code gave a different double.
# Each cost a debugging round; each is pinned so a "simplification" fails
# with a reason rather than as an anonymous mismatch.
# --------------------------------------------------------------------------


def test_no_ions_needs_both_switches_not_either(golden):
    """`:168-175` tests `l_fix_ukca_hygroscopicities .AND. l_fix_nacl_density`
    first, so NaCl density only reaches `no_ions` when hygroscopicities is also
    on — it is not an independent knob. Reading it as one selects the default
    branch and gets every setup wrong in the same way, which is what happened."""
    both = modes.build(1, l_fix_ukca_hygroscopicities=True, l_fix_nacl_density=True)
    hygro = modes.build(1, l_fix_ukca_hygroscopicities=True, l_fix_nacl_density=False)
    neither = modes.build(1, l_fix_ukca_hygroscopicities=False, l_fix_nacl_density=True)

    np.testing.assert_array_equal(both.no_ions, golden["s1_no_ions"])
    assert not np.array_equal(both.no_ions, hygro.no_ions)
    # nacl_density alone must NOT move it -- that is the trap.
    np.testing.assert_array_equal(
        neither.no_ions,
        modes.build(1, l_fix_ukca_hygroscopicities=False, l_fix_nacl_density=False).no_ions,
    )


def test_the_cube_must_be_repeated_multiplication_not_pow(golden):
    """gfortran expands `d**3` with an integer literal exponent into `d*d*d`;
    numpy's `**` calls `pow()`. They disagree by one ulp on two of the eight
    modes, which is enough to fail byte equality — and it is the last place
    anyone would look."""
    d = golden["s1_ddpmid"]
    differ = (d**3) != (d * d * d)
    assert differ.any(), "pow and repeated multiplication now agree; re-derive this"
    assert differ.sum() == 2


def test_the_mass_products_must_keep_the_fortran_factor_order(golden, built):
    """`(pi/6) * d**3 * (rhommav*avogadro) * x`, left-associated. Factoring out
    the `(pi/6) * (rhommav*avogadro) * x` shared by all three masses is the
    obvious optimisation and reassociates the product, which broke all three."""
    mfrac, rho, mm = golden["s1_mfrac_0"], golden["s1_rhocomp"], golden["s1_mm"]
    dm, x = golden["s1_ddpmid"], golden["s1_x"]
    ncp = int(golden["s1_ncp"])

    rhommav = np.array([sum(mfrac[i, c] * (rho[c] / mm[c]) for c in range(ncp)) for i in range(8)])
    faithful = np.array(
        [
            (modes.PI / 6.0) * (dm[i] * dm[i] * dm[i]) * (rhommav[i] * modes.AVOGADRO) * x[i]
            for i in range(8)
        ]
    )
    refactored = np.array(
        [
            ((modes.PI / 6.0) * (rhommav[i] * modes.AVOGADRO) * x[i]) * (dm[i] * dm[i] * dm[i])
            for i in range(8)
        ]
    )
    np.testing.assert_array_equal(faithful, golden["s1_mmid"])
    assert not np.array_equal(refactored, golden["s1_mmid"]), (
        "reassociating no longer changes the result; the guard above is now vacuous"
    )


def test_density_switches_apply_before_the_masses_are_derived(golden):
    """`rhocomp` is a literal, then patched by `l_fix_nacl_density`, and only
    then do the masses get built from it. Applying the switch afterwards leaves
    the masses built from the unpatched density — silently, and by 35% on the
    NaCl mode."""
    patched = modes.build(1, l_fix_nacl_density=True)
    unpatched = modes.build(1, l_fix_nacl_density=False)

    assert patched.rhocomp[modes.CP_CL] == modes.RHO_NACL
    assert unpatched.rhocomp[modes.CP_CL] == 1600.0
    np.testing.assert_array_equal(patched.rhocomp, golden["s1_rhocomp"])

    # The mass of the NaCl-bearing coarse soluble mode must move with it.
    ratio = patched.mmid[modes.MODE_COR_SOL] / unpatched.mmid[modes.MODE_COR_SOL]
    assert abs(ratio - modes.RHO_NACL / 1600.0) < 1e-12
    assert not np.array_equal(patched.mmid, unpatched.mmid)


def test_ddpmid_is_the_exp_log_form_not_sqrt(golden):
    """`EXP(0.5*(LOG a + LOG b))`, not `sqrt(a*b)`. Algebraically identical,
    numerically not, and `ddpmid` feeds `mmid` and the merge thresholds."""
    lo, hi = golden["s1_ddplim0"], golden["s1_ddplim1"]
    np.testing.assert_array_equal(np.exp(0.5 * (np.log(lo) + np.log(hi))), golden["s1_ddpmid"])


def test_x_uses_two_log_calls_not_a_square(golden):
    """`EXP(4.5 * LOG(sg) * LOG(sg))` as the Fortran writes it."""
    sg = golden["s1_sigmag"]
    log_sg = np.log(sg)
    np.testing.assert_array_equal(np.exp(4.5 * log_sg * log_sg), golden["s1_x"])


def test_component_is_the_three_way_intersection(golden, built):
    """Allowed in this mode AND chosen AND the mode is on. `component_mode`
    alone is a permission table — treating it as presence is wrong on most
    setups."""
    permitted = golden["s1_component_mode"].astype(bool)
    present = built.component
    assert not (present & ~permitted).any()
    assert (permitted & ~present).any(), "the two tables are identical here"


# --------------------------------------------------------------------------
# Task 30: every switch, at both settings, byte-equal.
#
# The acceptance criterion is "both settings", not "the default" — a flag whose
# non-default branch is never compared is a decision that looks tested. Each
# combination below is a separately captured golden; see validation/capture_modes.py.
# --------------------------------------------------------------------------

COMBOS = {
    "nacl_off": {"l_fix_nacl_density": False},
    "hygro_off": {"l_fix_ukca_hygroscopicities": False},
    "both_off": {"l_fix_nacl_density": False, "l_fix_ukca_hygroscopicities": False},
    "bc_tuned": {"l_radaer": True, "i_tune_bc": 1},
    "bc_mg_mix": {"l_radaer": True, "i_tune_bc": 2},
    "bc_oob": {"l_radaer": True, "i_tune_bc": 3},
    "dust_ageing": {"l_dust_mp_ageing": True},
}


@pytest.mark.parametrize("combo", sorted(COMBOS))
@pytest.mark.parametrize("setup", SETUPS)
def test_switch_combination_is_byte_equal(golden, setup, combo):
    built = modes.build(setup, **COMBOS[combo])
    for field in ARRAY_FIELDS:
        got = np.asarray(getattr(built, field))
        if got.dtype == bool:
            got = got.astype(np.int32)
        np.testing.assert_array_equal(
            got, golden[f"v_{combo}_s{setup}_{field}"], err_msg=f"{combo}/s{setup}/{field}"
        )
    assert built.topmode == int(golden[f"v_{combo}_s{setup}_topmode"])


@pytest.mark.parametrize("combo", sorted(COMBOS))
def test_each_combination_actually_changes_something(golden, combo):
    """Except `bc_oob`, which is captured precisely because it changes nothing.

    Without this, a combination whose switches never reached the Fortran would
    pass every byte-equality check above by being identical to the default —
    which is exactly what happened while building this: the namelist injection
    silently failed and all seven combinations matched the default."""
    watched = ("rhocomp", "no_ions", "mmid", "mlo", "mhi", "topmode")
    moved = [
        f for f in watched if not np.array_equal(golden[f"v_{combo}_s1_{f}"], golden[f"s1_{f}"])
    ]
    if combo == "bc_oob":
        assert not moved, "an out-of-range i_tune_bc should fall through silently"
    else:
        assert moved, f"{combo} is identical to the default; did its switches apply?"


def test_an_out_of_range_i_tune_bc_falls_through_silently(golden):
    """`ukca_mode_setup.F90:425-430` has no `CASE DEFAULT`, so a value that is
    neither 1 nor 2 leaves `rhocomp(cp_bc)` at its literal instead of failing.

    Reproduced rather than corrected: the port matches the reference including
    its silences, and this is the kind of silence a user hits by typo. It is
    the same shape as UP-5's unchecked `icoag`."""
    literal = golden["s1_rhocomp"][modes.CP_BC]
    assert golden["v_bc_tuned_s1_rhocomp"][modes.CP_BC] == modes.RHO_BC_TUNED
    assert golden["v_bc_mg_mix_s1_rhocomp"][modes.CP_BC] == modes.RHO_BC_MG_MIX
    assert golden["v_bc_oob_s1_rhocomp"][modes.CP_BC] == literal
    np.testing.assert_array_equal(
        modes.build(1, l_radaer=True, i_tune_bc=3).rhocomp, golden["s1_rhocomp"]
    )


def test_i_tune_bc_is_inert_without_l_radaer(golden):
    """It is read only inside `IF (l_radaer_in)`, and the box model defaults
    that off — so the BC density tuning is unreachable by default."""
    for value in (1, 2, 3):
        np.testing.assert_array_equal(
            modes.build(1, l_radaer=False, i_tune_bc=value).rhocomp, golden["s1_rhocomp"]
        )


def test_dust_ageing_moves_topmode_and_nothing_else(golden):
    """`topmode` is the only thing it touches — but it is read by `conden`,
    `ageing` and `coagwithnucl` loop bounds, so the effect downstream is large."""
    assert int(golden["v_dust_ageing_s1_topmode"]) == modes.NMODES
    assert int(golden["s1_topmode"]) == modes.MODE_AIT_INSOL + 1
    for field in ARRAY_FIELDS:
        np.testing.assert_array_equal(
            golden[f"v_dust_ageing_s1_{field}"], golden[f"s1_{field}"], err_msg=field
        )


def test_the_nacl_density_switch_moves_the_masses_by_the_density_ratio(golden):
    """Confirms the switch is applied to `rhocomp` before the masses derive
    from it, not after."""
    on = golden["s1_mmid"][modes.MODE_COR_SOL]
    off = golden["v_nacl_off_s1_mmid"][modes.MODE_COR_SOL]
    assert abs(on / off - modes.RHO_NACL / 1600.0) < 1e-12
