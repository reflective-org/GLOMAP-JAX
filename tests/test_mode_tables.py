"""Task 24: the mode and component tables, captured for all seven setups.

`common_mode_setup_interface` turns `i_mode_setup` plus five density and
hygroscopicity switches into `glomap_variables_type`. Every process routine
reads it, so phase C ports it under **byte equality** rather than a tolerance:
a mode diameter one ulp out is not a small error downstream, it is a different
model.

This file checks the captured golden is a usable reference. The port itself
(tasks 25-29) then compares against it with `array_equal`.

No `fortran` marker — the golden is committed, so this runs in CI.
"""

from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "goldens" / "modes.f64.tables.npz"
SETUPS = (1, 2, 3, 4, 5, 6, 8)

MODE_REAL = (
    "fracbcem",
    "fracocem",
    "ddplim0",
    "ddpmid",
    "ddplim1",
    "mmid",
    "mlo",
    "mhi",
    "num_eps",
    "sigmag",
    "x",
)
MODE_INT = ("mode_choice", "modesol", "mode")
CP_REAL = ("mm", "rhocomp", "no_ions")
CP_INT = ("component_choice", "soluble_choice", "soluble")


@pytest.fixture(scope="module")
def tables():
    assert GOLDEN.is_file(), "run `python validation/capture_modes.py`"
    return np.load(GOLDEN, allow_pickle=False)


def test_every_supported_setup_was_captured(tables):
    assert list(tables["_setups"]) == list(SETUPS)


def test_unsupported_setups_are_absent(tables):
    """10, 11, 12 and 13 exist in UKCA but `glomap_box_config_mod`'s
    `init_indices` has no CASE for them, so there is no reference to capture."""
    captured = {int(k.split("_")[0][1:]) for k in tables if k.startswith("s")}
    assert captured & {10, 11, 12, 13} == set()


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("field", MODE_REAL + MODE_INT)
def test_per_mode_tables_are_full_width(tables, setup, field):
    """Eight entries always, even where only two modes are active. The array is
    `dimension(nmodes)` in the Fortran and inactive slots still carry values —
    a port that stored only the active ones would silently change indexing."""
    assert tables[f"s{setup}_{field}"].shape == (8,)


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("field", CP_REAL + CP_INT)
def test_per_component_tables_match_ncp(tables, setup, field):
    assert tables[f"s{setup}_{field}"].shape == (int(tables[f"s{setup}_ncp"]),)


@pytest.mark.parametrize("setup", SETUPS)
def test_every_setup_has_six_components(tables, setup):
    """`ncp_max` is 10 and the `ncp = 9`/`10` routines exist, but they belong to
    setups 10/12/13, which the box model rejects. So the padding discussion the
    plan worried about is moot for order 1."""
    assert int(tables[f"s{setup}_ncp"]) == 6


@pytest.mark.parametrize("setup", SETUPS)
def test_solubility_is_structural_not_per_setup(tables, setup):
    """`modesol` is `[1,1,1,1,0,0,0,0]` in every setup — slots 1-4 are always
    soluble and 5-8 always insoluble. Only `mode_choice` varies. Conflating the
    two is the easiest way to mis-port the mode layout."""
    np.testing.assert_array_equal(tables[f"s{setup}_modesol"], [1, 1, 1, 1, 0, 0, 0, 0])


@pytest.mark.parametrize("setup", SETUPS)
def test_mode_matches_mode_choice(tables, setup):
    """`mode` is the LOGICAL the routines branch on; `mode_choice` is the
    INTEGER that built it. They must agree, or one of them is stale."""
    np.testing.assert_array_equal(tables[f"s{setup}_mode"], tables[f"s{setup}_mode_choice"])


def test_mode_sup_insol_is_never_active(tables):
    """Slot 8 needs setup 12 or 13, neither of which the box model implements.
    So it is off in every configuration this port can validate — which means any
    code path guarded on it has no reference, in either setting of a fidelity
    flag. See UP-10."""
    for setup in SETUPS:
        assert int(tables[f"s{setup}_mode"][7]) == 0, f"setup {setup}"


def test_active_mode_counts_are_the_documented_ones(tables):
    """A count that moves means the mode setup changed, which invalidates every
    golden keyed to it."""
    expected = {1: 4, 2: 5, 3: 4, 4: 5, 5: 4, 6: 2, 8: 7}
    actual = {s: int(tables[f"s{s}_mode"].sum()) for s in SETUPS}
    assert actual == expected


def test_the_dust_only_setup_has_no_soluble_modes(tables):
    """Setup 6 is `mode_choice = [0,0,0,0,0,1,1,0]` — condensation, nucleation
    and ageing are all structurally no-ops there, which makes it the edge case
    worth gating on."""
    assert list(tables["s6_mode"]) == [0, 0, 0, 0, 0, 1, 1, 0]


@pytest.mark.parametrize("setup", SETUPS)
def test_topmode_is_a_switch_not_the_highest_active_mode(tables, setup):
    """`topmode` reads like "highest active mode" and is not.
    `ukca_mode_setup.F90:418-422` sets it to `nmodes` when `l_dust_mp_ageing`
    and to `mode_ait_insol` (5) otherwise — regardless of `mode_choice`.

    The box model defaults the switch off, so it is 5 everywhere, *including*
    setup 8 where modes 6 and 7 are active. Loops written `DO imode = 1,
    topmode` therefore stop at 5 and never reach them. Verified against the
    binding: flipping `l_dust_mp_ageing` to `.TRUE.` gives 8."""
    assert int(tables[f"s{setup}_topmode"]) == 5


@pytest.mark.parametrize("setup", SETUPS)
def test_diameter_bounds_are_ordered_and_positive(tables, setup):
    lo, mid, hi = (tables[f"s{setup}_{f}"] for f in ("ddplim0", "ddpmid", "ddplim1"))
    assert (lo > 0).all() and (mid > 0).all() and (hi > 0).all()
    assert (lo < mid).all() and (mid < hi).all()


@pytest.mark.parametrize("setup", SETUPS)
def test_mass_bounds_are_ordered(tables, setup):
    lo, mid, hi = (tables[f"s{setup}_{f}"] for f in ("mlo", "mmid", "mhi"))
    assert (lo < mid).all() and (mid < hi).all()


@pytest.mark.parametrize("setup", SETUPS)
def test_num_eps_spans_the_range_that_makes_up10_matter(tables, setup):
    """`num_eps` differing between a soluble mode and the insoluble mode it
    gates is the whole substance of UP-10. Pinned so the analysis there stays
    anchored to real numbers."""
    num_eps = tables[f"s{setup}_num_eps"]
    assert num_eps.min() > 0
    assert num_eps.max() / num_eps.min() >= 1e6


@pytest.mark.parametrize("setup", SETUPS)
def test_component_names_are_present_and_stripped(tables, setup):
    names = tables[f"s{setup}_component_names"]
    assert len(names) == int(tables[f"s{setup}_ncp"])
    assert all(n and n == n.strip() for n in names)


@pytest.mark.parametrize("setup", SETUPS)
def test_mfrac_0_rows_sum_to_one_for_active_modes(tables, setup):
    """Initial mass fractions across components. An active mode whose fractions
    do not sum to 1 would start the model with the wrong total mass."""
    mfrac = tables[f"s{setup}_mfrac_0"]
    active = tables[f"s{setup}_mode"].astype(bool)
    sums = mfrac[active].sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-12)


@pytest.mark.parametrize("setup", SETUPS)
def test_present_components_are_a_subset_of_permitted_ones(tables, setup):
    """The two tables answer different questions, which is easy to miss:
    `component_mode` is which components are *allowed* in each mode
    (`ukca_mode_setup.F90:369-372`, "allowed in nuc_sol"), while `component` is
    which are actually *present* for this setup.

    So the invariant is containment, not equality. Asserting equality was my
    first guess and it fails on five of the seven setups — a port that treated
    them as interchangeable would grant components to modes that should not
    carry them."""
    present = tables[f"s{setup}_component"].astype(bool)
    permitted = tables[f"s{setup}_component_mode"].astype(bool)
    violations = present & ~permitted
    assert not violations.any(), (
        f"setup {setup}: components present in modes that do not permit them at "
        f"{list(zip(*np.nonzero(violations)))}"
    )


@pytest.mark.parametrize("setup", SETUPS)
def test_permitted_components_are_not_all_present(tables, setup):
    """The containment above would be trivially true if the two tables were
    equal, so check they genuinely differ — otherwise the test proves nothing."""
    present = tables[f"s{setup}_component"].astype(bool)
    permitted = tables[f"s{setup}_component_mode"].astype(bool)
    assert (permitted & ~present).any(), (
        f"setup {setup}: every permitted component is present, so the "
        f"containment check is vacuous here"
    )


@pytest.mark.parametrize("setup", SETUPS)
def test_components_are_only_present_in_active_modes(tables, setup):
    inactive = ~tables[f"s{setup}_mode"].astype(bool)
    assert not tables[f"s{setup}_component"].astype(bool)[inactive].any()


def test_setups_are_not_all_identical(tables):
    """If the capture silently ran the same setup seven times, every test above
    would still pass."""
    signatures = {s: tables[f"s{s}_mode"].tobytes() for s in SETUPS}
    assert len(set(signatures.values())) >= 4, "the setups look suspiciously alike"
