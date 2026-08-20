"""Tasks 25-29: `physics/modes.py` for every supported setup, byte-equal.

**`array_equal`, not `allclose`.** These tables feed every process routine, so
a diameter one ulp out is not a small error downstream — `drydp` is compared
against `dp_thresh1` and `ddplim0*0.1`, both step changes, and a one-ulp shift
flips them for any parcel on the boundary. A tolerance here would defer the
failure to somewhere it cannot be diagnosed.

**Derived quantities are recomputed, not copied.** That is the phase's
acceptance criterion and the reason these tests mean anything: reading `mmid`
out of the golden would produce a module that passes and implements nothing.

Getting to byte equality took four corrections, each a different way for
algebraically-identical code to give a different double. They are pinned below
as individual tests, because each one is a mistake the next person will make,
alongside three further guards for simplifications nobody has actually made.
`BYTE_EQUALITY_CORRECTIONS` and `FIDELITY_GUARDS` name the seven, and
`test_the_header_counts_match_the_pinned_tests` keeps this paragraph and that
list the same fact rather than two things to remember.

**What the test count here means.** Two parametrisations dominate: 23 fields x
7 setups, and 7 switch combinations x 7 setups. That reads as seven independent
reference points per field and for most fields it is one — 17 of the 23 are
byte-identical across all seven setups, and `bc_oob` is a no-op in all seven.
This is redundancy in the *reference*, not vacuity: a per-setup mistake in the
port is still caught, and precisely, which the tests at the bottom of this file
measure rather than assert on faith.

No `fortran` marker — the golden is committed, so this runs in CI.
"""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from glomap_jax.physics import modes

REPO = Path(__file__).resolve().parents[1]
FORTRAN_MODE_SETUP = REPO / "fortran" / "src" / "ukca" / "ukca_mode_setup.F90"
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
# The four ways algebraically-identical code gave a different double, plus
# three guards for ways it could have and did not.
# Each cost a debugging round; each is pinned so a "simplification" fails
# with a reason rather than as an anonymous mismatch.
#
# Each test here therefore does TWO things:
#
#   1. runs the module's own code and compares it against the golden, so the
#      simplification it is named for actually reddens it; and
#   2. asserts the naive form really does give a different double, so the pin
#      cannot quietly go vacuous if a future numpy or platform closes the gap.
#
# (1) was missing until a mutation audit: four of these tests recomputed the
# faithful form out of numpy and the golden arrays and never called `modes` at
# all. Making the named simplification in `physics/modes.py` left every one of
# them GREEN while reddening 56-77 other tests in this file -- precisely the
# anonymous mismatch the section exists to prevent. If you add a test here,
# apply its mutation and watch this test fail before believing it.
# --------------------------------------------------------------------------


def test_no_ions_needs_both_switches_not_either(golden):
    """`:678-685` tests `l_fix_ukca_hygroscopicities .AND. l_fix_nacl_density`
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


def _golden_rhommav(golden, setup=1):
    """`rhommav` per mode, accumulated in index order out of golden inputs."""
    mfrac, rho, mm = (golden[f"s{setup}_{f}"] for f in ("mfrac_0", "rhocomp", "mm"))
    ncp = int(golden[f"s{setup}_ncp"])
    out = np.empty(modes.NMODES, dtype=np.float64)
    for imode in range(modes.NMODES):
        acc = 0.0
        for icp in range(ncp):
            acc = acc + mfrac[imode, icp] * (rho[icp] / mm[icp])
        out[imode] = acc
    return out


def _mass(golden, d, cube, order):
    """One mass column, built from golden inputs with a chosen cube and order.

    `cube` is how `d**3` is spelled and `order` how the four factors associate;
    everything else is held fixed, so a difference is attributable.
    """
    rhommav, x = _golden_rhommav(golden), golden["s1_x"]
    out = np.empty(modes.NMODES, dtype=np.float64)
    for i in range(modes.NMODES):
        cubed = cube(d[i])
        out[i] = order((modes.PI / 6.0), cubed, rhommav[i] * modes.AVOGADRO, x[i])
    return out


def _repeated(v):
    """gfortran's expansion of `v**3` with an integer literal exponent."""
    return v * v * v


def _pow(v):
    """numpy's `**`, which calls `pow()`."""
    return v**3


def _fortran_order(a, b, c, d):
    """`(pi/6) * d3 * (rhommav*avogadro) * x`, left-associated as written."""
    return a * b * c * d


def _factored_order(a, b, c, d):
    """The obvious optimisation: the cube pulled out of the shared factors."""
    return (a * c * d) * b


def test_the_cube_must_be_repeated_multiplication_not_pow(golden, built):
    """gfortran expands `d**3` with an integer literal exponent into `d*d*d`;
    numpy's `**` calls `pow()`. They disagree by one ulp on two of the eight
    `ddpmid` entries, which is enough to fail byte equality — and it is the
    last place anyone would look.

    The module's own masses are checked against both spellings, so writing
    `dm**3` in `_mode_masses` fails HERE, by name. Before, it failed only as 56
    anonymous byte-equality mismatches: `mmid` over 7 setups plus all 49 switch
    combinations, none of them naming the cube."""
    for name, d in (
        ("mmid", golden["s1_ddpmid"]),
        ("mlo", golden["s1_ddplim0"]),
        ("mhi", golden["s1_ddplim1"]),
    ):
        got = getattr(built, name)
        np.testing.assert_array_equal(
            got, _mass(golden, d, _repeated, _fortran_order), err_msg=f"{name}: not d*d*d"
        )
        pow_form = _mass(golden, d, _pow, _fortran_order)
        assert not np.array_equal(got, pow_form), f"{name} matches the pow form"

    # The ulp evidence itself, so the guard above cannot become vacuous
    # unnoticed if a future numpy makes pow() exact for these inputs.
    for name, d, expected in (
        ("ddpmid", golden["s1_ddpmid"], 2),
        ("ddplim0", golden["s1_ddplim0"], 2),
        ("ddplim1", golden["s1_ddplim1"], 3),
    ):
        differ = (d**3) != (d * d * d)
        assert differ.sum() == expected, (
            f"{name}: pow and repeated multiplication now differ on "
            f"{differ.sum()} of 8 entries, not {expected}; re-derive this"
        )


def test_the_mass_products_must_keep_the_fortran_factor_order(golden, built):
    """`(pi/6) * d**3 * (rhommav*avogadro) * x`, left-associated. Factoring out
    the `(pi/6) * (rhommav*avogadro) * x` shared by all three masses is the
    obvious optimisation and reassociates the product, which broke all three.

    Again both sides are checked against the module: `built` must equal the
    left-associated form and must NOT equal the factored one, so making the
    optimisation in `_mode_masses` reddens this test rather than the same 56
    others the cube mutation hits."""
    for name, d, moved in (
        ("mmid", golden["s1_ddpmid"], 4),
        ("mlo", golden["s1_ddplim0"], 4),
        ("mhi", golden["s1_ddplim1"], 5),
    ):
        got = getattr(built, name)
        np.testing.assert_array_equal(
            got,
            _mass(golden, d, _repeated, _fortran_order),
            err_msg=f"{name}: not the Fortran factor order",
        )
        factored = _mass(golden, d, _repeated, _factored_order)
        assert not np.array_equal(got, factored), (
            f"{name}: reassociating no longer changes the result; this guard is now vacuous"
        )
        assert (got != factored).sum() == moved, (
            f"{name}: reassociating moves {(got != factored).sum()} of 8 modes, not {moved}"
        )


def test_the_rhommav_loop_order_is_faithful_but_unexercised(golden):
    """The one fidelity constraint in `_mode_masses` that nothing here can
    falsify — pinned as latent rather than left reading like a measurement.

    `_mode_masses` accumulates `rhommav` over components in index order because
    that is what `ukca_mode_setup.F90:649-655` does. Its docstring used to say a
    vectorised reduction "need not give the same double", in the same voice as
    the cube and factor-order claims next to it — but those two are *asserted*
    to matter and this one is not: replacing the loop with
    `float(np.sum(mfrac_0[imode, :ncp] * (rhocomp[:ncp] / mm[:ncp])))` left all
    247 tests this file then had green, and still moves no number.

    The reason is structural, not luck. `mfrac_0` holds only 0.0, 0.5 and 1.0
    and never more than two non-zero entries per mode, so `rhommav` is a sum of
    at most two terms and every association gives the same double. This test
    measures exactly that, over all 7 setups x 8 modes plus all 7 switch
    variants, so the claim in the module stays true or this fails.

    The loop stays regardless: it is the faithful form, and a future setup with
    three or more non-zero components would make the difference live. That is
    the day this test is supposed to fail. Compare
    `test_numerics_reference.py::test_cbrt_and_the_power_form_disagree_about_negative_inputs`,
    the same pattern for the same reason."""
    variants = [f"s{s}" for s in SETUPS]
    variants += [f"v_{c}_s{s}" for c in sorted(COMBOS) for s in SETUPS]
    rows = 0
    for key in variants:
        setup_key = key if key.startswith("s") else f"s{key.rsplit('_s', 1)[1]}"
        ncp = int(golden[f"{setup_key}_ncp"])
        mfrac, rho, mm = (golden[f"{key}_{f}"] for f in ("mfrac_0", "rhocomp", "mm"))
        assert set(np.unique(mfrac).tolist()) <= {0.0, 0.5, 1.0}
        for imode in range(modes.NMODES):
            rows += 1
            terms = [mfrac[imode, icp] * (rho[icp] / mm[icp]) for icp in range(ncp)]
            assert sum(t != 0.0 for t in terms) <= 2, (
                f"{key} mode {imode} now has more than two non-zero components; "
                f"the loop order in _mode_masses may have become live -- re-measure"
            )
            in_order = 0.0
            for t in terms:
                in_order = in_order + t
            reversed_order = 0.0
            for t in reversed(terms):
                reversed_order = reversed_order + t
            vectorised = float(np.sum(mfrac[imode, :ncp] * (rho[:ncp] / mm[:ncp])))
            assert in_order == reversed_order == vectorised, (
                f"{key} mode {imode}: reduction order now changes rhommav. The "
                f"constraint in _mode_masses just became live -- say so there."
            )
    assert rows == 8 * len(SETUPS) * (1 + len(COMBOS)) == 448


def test_mode_masses_accumulates_rhommav_without_a_vectorised_reduction(monkeypatch):
    """The structural half of the pin above.

    Because the numerical difference is unreachable with the committed tables,
    no comparison of values can notice `rhommav` being rewritten as
    `float(np.sum(...))`: that edit moved nothing, and left the whole file
    green. So the shape is asserted directly instead — `np.sum` is made to
    raise, and `build` must still produce tables for every setup. This is the
    only test that reddens on that edit, and it is deliberately about the form
    rather than the answer."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "_mode_masses must accumulate rhommav in component index order, as "
            "ukca_mode_setup.F90:649-655 does, not with a vectorised reduction"
        )

    monkeypatch.setattr(modes.np, "sum", forbidden)
    for setup in SETUPS:
        modes.build(setup)


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


def test_ddpmid_is_the_exp_log_form_not_sqrt(golden, built):
    """`EXP(0.5*(LOG a + LOG b))`, not `sqrt(a*b)`. Algebraically identical,
    numerically not, and `ddpmid` feeds `mmid` and the merge thresholds.

    `modes._ddpmid` is called directly, so writing the `sqrt` form there fails
    here — and not only as 63 anonymous byte-equality failures, which is what
    it produced before: `ddpmid` and `mmid` over 7 setups, plus all 49 switch
    combinations."""
    lo, hi = golden["s1_ddplim0"], golden["s1_ddplim1"]
    np.testing.assert_array_equal(modes._ddpmid(lo, hi), golden["s1_ddpmid"])
    np.testing.assert_array_equal(built.ddpmid, golden["s1_ddpmid"])

    sqrt_form = np.sqrt(lo * hi)
    assert (sqrt_form != golden["s1_ddpmid"]).sum() == 6, (
        f"sqrt(a*b) now differs on {(sqrt_form != golden['s1_ddpmid']).sum()} of 8 modes, "
        f"not 6; re-derive this before relaxing the rule"
    )


def test_x_uses_two_log_calls_not_a_square(golden, built):
    """`EXP(4.5 * LOG(sg) * LOG(sg))` as the Fortran writes it.

    `LOG(sg)**2` moves exactly one of the eight entries — slot 8,
    `mode_sup_insol`, `sigmag = 1.8` — and that slot is never active in any
    supported setup. It is still byte-compared, because the tables are full
    width and slot 8's `x` feeds slot 8's `mmid`, `mlo` and `mhi`, so the
    one-ulp shift reddens 77 tests here: those 4 fields over 7 setups plus all
    49 switch combinations. None of them names the cause. `modes._x` is called
    directly, so this test does."""
    sg = golden["s1_sigmag"]
    np.testing.assert_array_equal(modes._x(sg), golden["s1_x"])
    np.testing.assert_array_equal(built.x, golden["s1_x"])

    log_sg = np.log(sg)
    moved = np.exp(4.5 * log_sg**2) != golden["s1_x"]
    assert moved.sum() == 1, (
        f"LOG(sg)**2 now differs on {moved.sum()} of 8 modes, not 1; re-derive this"
    )
    assert np.nonzero(moved)[0].tolist() == [modes.MODE_SUP_INSOL]


def test_component_is_the_three_way_intersection(golden, built):
    """Allowed in this mode AND chosen AND the mode is on. `component_mode`
    alone is a permission table — treating it as presence is wrong on every
    setup; see `test_component_needs_all_three_factors` for the counts."""
    permitted = golden["s1_component_mode"].astype(bool)
    present = built.component
    assert not (present & ~permitted).any()
    assert (permitted & ~present).any(), "the two tables are identical here"


def test_component_needs_all_three_factors(golden):
    """How much of `component` each partial reading gets right, measured.

    `modes._component` is `allowed & chosen & active` (`:700-704`). Three
    weaker readings are available and the docstring used to claim the
    permission-table one "survives five of the seven setups". It survives none.
    Re-derived here so the number in `physics/modes.py` is answerable to
    committed data rather than to memory:

    * `component == component_mode`               0 of 7
    * `component_mode & component_choice`         3 of 7 (setups 1, 2, 4)
    * `component_mode & mode_choice`              1 of 7 (setup 6)
    * the full three-way intersection             7 of 7

    The two partial readings are the dangerous ones: each is right on a
    non-empty subset, so a port checked against one setup can adopt either.
    """
    survives = {
        "permission_only": [],
        "no_mode_choice": [],
        "no_component_choice": [],
        "faithful": [],
    }
    for setup in SETUPS:
        permitted = golden[f"s{setup}_component_mode"].astype(bool)
        chosen = (golden[f"s{setup}_component_choice"] == 1)[None, :]
        active = (golden[f"s{setup}_mode_choice"] == 1)[:, None]
        reference = golden[f"s{setup}_component"].astype(bool)

        np.testing.assert_array_equal(modes.build(setup).component, reference)
        for name, candidate in (
            ("permission_only", permitted),
            ("no_mode_choice", permitted & chosen),
            ("no_component_choice", permitted & active),
            ("faithful", permitted & chosen & active),
        ):
            if np.array_equal(candidate, reference):
                survives[name].append(setup)

    assert survives["permission_only"] == []
    assert survives["no_mode_choice"] == [1, 2, 4]
    assert survives["no_component_choice"] == [6]
    assert survives["faithful"] == list(SETUPS)


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


# What each combination moves, relative to the default, in EVERY setup. The
# sets are setup-independent -- measured, not assumed: `rhocomp`, `no_ions` and
# the three mass columns are byte-identical across the seven setups to begin
# with (see `test_most_fields_are_identical_across_all_seven_setups`), so a
# switch that moves one moves it everywhere and by the same bytes.
EXPECTED_MOVES = {
    "nacl_off": {"rhocomp", "no_ions", "mmid", "mlo", "mhi"},
    "hygro_off": {"no_ions"},
    "both_off": {"rhocomp", "no_ions", "mmid", "mlo", "mhi"},
    "bc_tuned": {"rhocomp", "mmid", "mlo", "mhi"},
    "bc_mg_mix": {"rhocomp", "mmid", "mlo", "mhi"},
    "bc_oob": set(),
    "dust_ageing": {"topmode"},
}


@pytest.mark.parametrize("combo", sorted(COMBOS))
@pytest.mark.parametrize("setup", SETUPS)
def test_each_combination_actually_changes_something(golden, setup, combo):
    """Except `bc_oob`, which is captured precisely because it changes nothing.

    Without this, a combination whose switches never reached the Fortran would
    pass every byte-equality check above by being identical to the default —
    which is exactly what happened while building this: the namelist injection
    silently failed and all seven combinations matched the default.

    Two things this now says that the setup-1-only version did not.

    `bc_oob` is a no-op in **all seven** setups, not just setup 1. So all seven
    of its `test_switch_combination_is_byte_equal` parametrisations re-assert
    what `test_field_is_byte_equal_to_the_fortran` already covers — they are
    worth keeping as the pin on "an out-of-range `i_tune_bc` falls through
    silently", but they are not seven extra reference points.

    And the moved-field set is *exactly* the same for every setup, which is the
    same redundancy seen elsewhere in this reference. Asserting the exact set
    rather than "something moved" is what makes a switch reaching one field too
    many or too few a failure."""
    watched = ("rhocomp", "no_ions", "mmid", "mlo", "mhi", "topmode")
    moved = {
        f
        for f in watched
        if not np.array_equal(golden[f"v_{combo}_s{setup}_{f}"], golden[f"s{setup}_{f}"])
    }
    assert moved == EXPECTED_MOVES[combo], f"{combo}/setup {setup}"
    if combo == "bc_oob":
        assert not moved, "an out-of-range i_tune_bc should fall through silently"
    else:
        assert moved, f"{combo} is identical to the default; did its switches apply?"


def test_an_out_of_range_i_tune_bc_falls_through_silently(golden):
    """`ukca_mode_setup.F90:630-635` has no `CASE DEFAULT`, so a value that is
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


# --------------------------------------------------------------------------
# What the test count in this file means -- measured, not inferred.
#
# 250-odd tests here, and two parametrisations are most of them. A reader who
# takes "seven setups, byte-equal" as seven independent reference points is
# right for six fields and wrong for seventeen. Nothing below adds coverage;
# they make the existing coverage's shape a checked fact, so it cannot drift
# into a claim nobody re-derives.
# --------------------------------------------------------------------------

# Measured over the committed golden. `mode_choice` and its derived `mode`,
# the two emission-fraction tables, and `component_choice` with its derived
# `component`, are the only fields that differ between setups at all.
VARYING_FIELDS = (
    "mode_choice",
    "mode",
    "fracbcem",
    "fracocem",
    "component_choice",
    "component",
)


def test_most_fields_are_identical_across_all_seven_setups(golden):
    """17 of the 23 fields are byte-identical in all seven setups.

    So `test_field_is_byte_equal_to_the_fortran`'s 161 parametrisations are not
    161 independent comparisons: for those 17 fields they are one reference
    value checked seven times. That is a property of `ukca_mode_setup` -- the
    per-setup routines are near-copies and only the mode and component
    selections differ -- not a defect in the capture, and it is why
    `test_the_setups_are_not_all_the_same_tables` gates on `mode` rather than
    on the tables as a whole.

    It is also why the four `l_fix_*` / `l_radaer` switch combinations move the
    same fields in every setup (`EXPECTED_MOVES`).
    """
    identical = [
        f for f in ARRAY_FIELDS if len({golden[f"s{s}_{f}"].tobytes() for s in SETUPS}) == 1
    ]
    varying = [f for f in ARRAY_FIELDS if f not in identical]
    assert tuple(varying) == VARYING_FIELDS
    assert len(identical) == 17
    assert len(identical) + len(varying) == len(ARRAY_FIELDS) == 23


def test_a_wrong_literal_in_one_setup_fails_only_that_setup(golden, monkeypatch):
    """The redundancy above is in the reference, not in the coverage.

    Byte-identical golden values across setups could mean the port has one
    shared reference point pretending to be seven, OR that seven separate
    tables genuinely agree. It is the second: `build(setup)` reads
    `_SETUPS[setup]`, so perturbing one setup's literals reddens that setup's
    parametrisation and leaves the other six green. Demonstrated rather than
    argued, because the whole point of the section is not arguing.
    """
    for victim in SETUPS:
        literals = copy.deepcopy(modes._SETUPS)
        literals[victim]["ddplim0"] = list(literals[victim]["ddplim0"])
        literals[victim]["ddplim0"][2] *= 1.0 + 2.0**-40
        monkeypatch.setattr(modes, "_SETUPS", literals)

        for setup in SETUPS:
            agrees = np.array_equal(modes.build(setup).ddplim0, golden[f"s{setup}_ddplim0"])
            assert agrees == (setup != victim), (
                f"perturbing setup {victim}'s ddplim0 changed whether setup "
                f"{setup} matches its golden"
            )
        monkeypatch.undo()


# --------------------------------------------------------------------------
# The Fortran line citations, checked against the Fortran.
#
# Every computation citation in `physics/modes.py` and in this file was wrong,
# in one of two ways, and each was wrong plausibly.
#
# Most landed in `ukca_mode_setup`'s declaration section: `:80-85` for `x` is
# the `cp_*` PARAMETERs, `:183` for `mode` is the closing bracket of the
# `coag_mode` RESHAPE, `:185-199` for `component` is the `rho_nacl`/`rho_bc_*`
# block. The rest -- `:418-422` for `topmode` and the two switch blocks --
# named the right code inside `ukca_mode_allcp_4mode` (`:305-509`), which reads
# almost identically to the routine that actually runs and which
# `common_mode_setup_interface` never dispatches. Exactly right for a routine
# nothing calls is the harder of the two to notice.
#
# Nothing checked them, which is how they survived. This does.
# --------------------------------------------------------------------------

# citation -> (line number, text that line must contain), 1-based.
CITATIONS = {
    ":68": ((68, "PARAMETER :: nmodes=8"),),
    ":76,78": ((76, "cp_bc=2"), (78, "cp_cl=4")),
    ":187,195,199": (
        (187, "rho_nacl = 2165.0"),
        (195, "rho_bc_mg_mix = 1800.0"),
        (199, "rho_bc_tuned = 1900.0"),
    ),
    ":305-509": (
        (305, "SUBROUTINE ukca_mode_allcp_4mode"),
        (509, "END SUBROUTINE ukca_mode_allcp_4mode"),
    ),
    ":511-714": (
        (511, "SUBROUTINE ukca_mode_suss_4mode"),
        (714, "END SUBROUTINE ukca_mode_suss_4mode"),
    ),
    ":574-581": (
        (574, "component_mode(1,1:ncp)"),
        (574, "allowed in nuc_sol"),
        (581, "component_mode(8,1:ncp)"),
    ),
    ":593-597": ((593, "DO imode=1,nmodes"), (594, "%x(imode)=EXP(4.5 *"), (597, "END DO")),
    ":619-672": ((619, "%rhocomp(1:ncp)"), (672, "END DO")),
    ":623-627": (
        (623, "IF (l_dust_mp_ageing) THEN"),
        (624, "%topmode = nmodes"),
        (626, "%topmode = mode_ait_insol"),
        (627, "END IF"),
    ),
    ":629-636": (
        (629, "IF ( l_radaer_in ) THEN"),
        (630, "SELECT CASE (i_tune_bc_in)"),
        (636, "END IF"),
    ),
    ":630-635": ((630, "SELECT CASE (i_tune_bc_in)"), (635, "END SELECT")),
    ":638-640": (
        (638, "IF ( l_fix_nacl_density_in ) THEN"),
        (639, "%rhocomp(cp_cl) = rho_nacl"),
        (640, "END IF"),
    ),
    ":642-646": (
        (642, "DO imode=1,nmodes"),
        (643, "%ddpmid(imode) = EXP( 0.5 *"),
        (646, "END DO"),
    ),
    ":648-672": (
        (648, "DO imode=1,nmodes"),
        (657, "%mmid(imode) = ( pi / 6.0 )"),
        (662, "%mlo(imode)  = ( pi / 6.0 )"),
        (667, "%mhi(imode)  = ( pi / 6.0 )"),
        (672, "END DO"),
    ),
    ":649-655": (
        (649, "rhommav=0.0"),
        (650, "DO icp=1,ncp"),
        (651, "rhommav = rhommav +"),
        (655, "END DO"),
    ),
    ":678-685": (
        (678, "IF (l_fix_ukca_hygroscopicities_in .AND."),
        (681, "ELSE IF (l_fix_ukca_hygroscopicities_in) THEN"),
        (683, "ELSE"),
        (685, "END IF"),
    ),
    ":694": ((694, "%mode      = ( glomap_variables_local%mode_choice > 0 )"),),
    ":700-704": (
        (700, "%component_mode( imode, icp ) == 1"),
        (701, "%component_choice( icp ) == 1"),
        (702, "%mode_choice(imode) == 1"),
        (703, "%component( imode, icp ) = .TRUE."),
    ),
    ":706-708": ((706, "%soluble_choice(icp) == 1"), (707, "%soluble(icp) = .TRUE.")),
}

# A citation is written `ukca_mode_setup.F90:NNN`, `` `:NNN` `` or `# :NNN`.
# Anything else with a colon and a digit in it -- a slice, a dict key -- is not
# a citation and must not be picked up.
_CITED = re.compile(r"(?:F90|`|#[ 	]*):(\d+(?:[-,]\d+)*)")
_CITING_FILES = (
    REPO / "src" / "glomap_jax" / "physics" / "modes.py",
    Path(__file__),
    Path(__file__).parent / "test_mode_tables.py",
)


# The wrong citations, kept because the section comment above names them and a
# named number should be checkable too. Each one resolved to real code -- just
# not the code it was cited for, which is why nobody spotted it by reading.
MISCITATIONS = {
    ":80-85": ((80, "cp_so=6"), (84, "cp_mp=10")),  # cited for `x`
    ":183": ((183, "[ nmodes, nmodes ] )"),),  # cited for `mode`
    ":185-199": ((187, "rho_nacl = 2165.0"), (199, "rho_bc_tuned = 1900.0")),  # for `component`
    ":418-422": (  # cited for `topmode`; right code, dead routine
        (418, "IF (l_dust_mp_ageing) THEN"),
        (421, "%topmode = mode_ait_insol"),
        (422, "END IF"),
    ),
}


def _assert_citations_resolve(citations, lines):
    for cite, expected in citations.items():
        bounds = [int(n) for n in re.split(r"[-,]", cite[1:])]
        for lineno, text in expected:
            assert min(bounds) <= lineno <= max(bounds), f"{cite} does not contain {lineno}"
            assert text in lines[lineno - 1], (
                f"{cite}: line {lineno} is {lines[lineno - 1]!r}, expected to contain {text!r}"
            )


def test_the_fortran_line_citations_resolve():
    """Every cited line holds what the citing text says it holds.

    Fails if the vendored tree is updated and a citation is not -- the failure
    `tests/test_coag_mode.py::test_the_declaration_is_at_the_line_the_docstring_cites`
    exists for, generalised to every citation this module carries.
    """
    _assert_citations_resolve(
        CITATIONS, FORTRAN_MODE_SETUP.read_text(encoding="utf-8").splitlines()
    )


def test_the_old_citations_pointed_where_the_section_comment_says():
    """The post-mortem above, checked like everything else.

    `:80-85`, `:183` and `:185-199` are declarations unrelated to what cited
    them; `:418-422` is the correct `topmode` branch inside the dead
    `ukca_mode_allcp_4mode`, which is the failure mode worth remembering.
    """
    lines = FORTRAN_MODE_SETUP.read_text(encoding="utf-8").splitlines()
    _assert_citations_resolve(MISCITATIONS, lines)

    start = next(
        n for n, line in enumerate(lines, 1) if line.startswith("SUBROUTINE ukca_mode_allcp_4mode")
    )
    end = next(
        n
        for n, line in enumerate(lines, 1)
        if line.startswith("END SUBROUTINE ukca_mode_allcp_4mode")
    )
    assert (start, end) == (305, 509)
    assert start < 418 and 422 < end, "`:418-422` is no longer inside the dead routine"
    for cite in MISCITATIONS:
        assert cite not in CITATIONS, f"{cite} is cited live and as a miscitation"


def test_no_citation_points_outside_the_routine_that_is_dispatched():
    """Setup 1 is built by `ukca_mode_suss_4mode`, `:511-714`.

    Every citation to a computation must land inside it. The three that do not
    are the module-level PARAMETERs shared by all setups, and the one deliberate
    reference to the dead routine.
    """
    module_scope = {":68", ":76,78", ":187,195,199", ":305-509"}
    for cite in CITATIONS:
        if cite in module_scope or cite == ":511-714":
            continue
        bounds = [int(n) for n in re.split(r"[-,]", cite[1:])]
        assert 511 <= min(bounds) and max(bounds) <= 714, (
            f"{cite} is outside ukca_mode_suss_4mode -- which routine is it in?"
        )


def test_every_citation_in_the_port_is_one_this_file_checks():
    """An unchecked citation is how the whole set drifted into a dead routine.

    Covers `physics/modes.py`, this file and `test_mode_tables.py`. A new
    `:NNN` in any of them must be added to `CITATIONS` (or, if it is being
    quoted as a past mistake, to `MISCITATIONS`) before it can be committed.
    """
    known = set(CITATIONS) | set(MISCITATIONS)
    found = set()
    for path in _CITING_FILES:
        found |= {f":{m}" for m in _CITED.findall(path.read_text(encoding="utf-8"))}
    assert found - known == set(), f"uncheckable citations: {sorted(found - known)}"


def test_the_dead_routine_really_is_dead():
    """`:305-509` is cited only to warn people off it, so the warning has to be
    true: nothing in the vendored tree calls `ukca_mode_allcp_4mode`, and
    `common_mode_setup_interface` has no `CASE` for it."""
    calls = []
    for path in sorted((REPO / "fortran" / "src").rglob("*.F90")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("!") or "ukca_mode_allcp_4mode" not in stripped:
                continue
            if stripped.startswith(("SUBROUTINE", "END SUBROUTINE")):
                continue
            calls.append((path.name, lineno, stripped))
    assert calls == [], f"ukca_mode_allcp_4mode is reachable after all: {calls}"


# --------------------------------------------------------------------------
# The header's count of corrections, made the same fact as the tests.
# --------------------------------------------------------------------------

# The four corrections it took to get the tables byte-equal. `docs/
# architecture.md` and `docs/porting-notes.md` carry the same four.
BYTE_EQUALITY_CORRECTIONS = (
    "test_the_cube_must_be_repeated_multiplication_not_pow",
    "test_the_mass_products_must_keep_the_fortran_factor_order",
    "test_density_switches_apply_before_the_masses_are_derived",
    "test_no_ions_needs_both_switches_not_either",
)

# Hazards of the same kind that the port got right first time. Pinned anyway,
# because "we happened not to make this mistake" is not a guard.
FIDELITY_GUARDS = (
    "test_ddpmid_is_the_exp_log_form_not_sqrt",
    "test_x_uses_two_log_calls_not_a_square",
    "test_component_is_the_three_way_intersection",
)

_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def test_the_header_counts_match_the_pinned_tests():
    """The header said "three corrections" while four were pinned below it and
    three other documents said four. Counting from the lists instead."""
    here = globals()
    for name in BYTE_EQUALITY_CORRECTIONS + FIDELITY_GUARDS:
        assert callable(here.get(name)), f"{name} is named in the header lists but absent"

    corrections = _NUMBER_WORDS[len(BYTE_EQUALITY_CORRECTIONS)]
    guards = _NUMBER_WORDS[len(FIDELITY_GUARDS)]
    assert f"took {corrections} corrections" in __doc__
    assert f"{guards} further guards" in __doc__
