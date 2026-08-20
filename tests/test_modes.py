"""Task 25: `physics/modes.py` for `i_mode_setup = 1`, byte-equal.

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


@pytest.fixture(scope="module")
def built():
    return modes.build(1)


@pytest.mark.parametrize("field", ARRAY_FIELDS)
def test_field_is_byte_equal_to_the_fortran(golden, built, field):
    got = np.asarray(getattr(built, field))
    if got.dtype == bool:
        got = got.astype(np.int32)
    np.testing.assert_array_equal(got, golden[f"s1_{field}"], err_msg=field)


def test_scalars_match(golden, built):
    assert built.ncp == int(golden["s1_ncp"])
    assert built.topmode == int(golden["s1_topmode"])
    assert list(built.component_names) == list(golden["s1_component_names"])


def test_every_captured_field_is_checked(golden):
    """A field the golden carries and this file never compares is an untested
    part of the port that looks tested."""
    captured = {k[3:] for k in golden if k.startswith("s1_")}
    checked = set(ARRAY_FIELDS) | {"ncp", "topmode", "component_names", "nmodes"}
    assert captured - checked == set(), f"uncompared: {sorted(captured - checked)}"


def test_unported_setups_raise_rather_than_return_wrong_tables(built):
    with pytest.raises(NotImplementedError, match="not ported yet"):
        modes.build(2)


# --------------------------------------------------------------------------
# The three ways algebraically-identical code gave a different double.
# Each cost a debugging round; each is pinned so a "simplification" fails
# with a reason rather than as an anonymous mismatch.
# --------------------------------------------------------------------------


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
