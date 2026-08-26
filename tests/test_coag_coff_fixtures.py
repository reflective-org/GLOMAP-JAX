"""Task 48: the `ukca_coag_coff_v` leaf sweep, and what it settles.

No `fortran` marker: the archive is committed, so all of this runs in CI.

Three supported methods -- Jacobson's transition-regime kernel, HAM/M7's
mean-radius approximation, and the original UM sulphate scheme -- and every
shipped namelist runs the first. `icoag = 4` is refused rather than captured:
it reads `mfppi`/`mfppj`, assigned only inside the `icoag == 1` block, so there
is no correct answer to record (UP-5).

Three things this archive establishes that a trajectory cannot.

**1. The methods read different arguments.** At `icoag = 3`, five of the nine
array arguments are dead: `vi`, `vj`, `rhoi`, `rhoj` and `mfpa`. At
`icoag = 2`, `vi` and `vj` are dead -- `vmid` is rebuilt from `rmid` at `:294`
rather than taken from the caller. Each "dead" is asserted as a byte equality
between rows differing in that argument alone, each "live" as an inequality.

The `mfpa` case is the sharp one. The routine's header describes `icoag = 3` as
using `MFP = MFPA = 6.6e-8*p0*T/(p*T0)`, but `:317` uses the bare
`ukca_mfp_ref = 6.6e-8` PARAMETER and never touches the `mfpa` argument, so
neither the pressure-temperature scaling nor the caller's own mean free path
enters. The header describes a calculation the code does not perform.

**2. The kernel is byte-symmetric.** Swapping `(ri, rj)`, `(vi, vj)` and
`(rhoi, rhoj)` together reproduces `kij` bit for bit at all three methods. That
is not free: it holds because every combination of the i and j quantities is an
addition, and floating-point addition is commutative. A formulation that summed
three terms, or divided before adding, need not have been symmetric.

It matters because phase C recorded that `coag_mode` is symmetric on all 64
entries, so a transposed transcription of that table would be byte-equal to the
correct one and undetectable. The kernel cannot distinguish the two orders
either. The `(imode, jmode)` convention therefore rests on
`ukca_calc_coag_kernel`'s subscripts alone, with nothing downstream to catch a
transposition -- which the next task has to know.

**3. `coag_on = 0` zeroes unmasked rows.** `:239-242` returns before the mask
is consulted, so it is the only path on which a row that the mask selected
comes back zero. Distinguishing "masked off" from "coagulation switched off"
matters to the driver, which passes an all-true mask.

The last test feeds mutated copies of the archive back through the capture's
own guards and requires each to be rejected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coag_coff.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_coag_coff_leaf as cap  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coag_coff_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def calls(sweep):
    return [
        {"icoag": int(sweep["call_icoag"][i]), "coag_on": int(sweep["call_coag_on"][i])}
        for i in range(sweep["call_icoag"].size)
    ]


@pytest.fixture(scope="module")
def grid(sweep):
    keys = ("ri", "rj", "vi", "vj", "rhoi", "rhoj", "mfpa", "dvisc", "t", "mask", "block")
    return {k: sweep[k] for k in keys}


def test_the_archive_matches_the_grid_the_capture_script_builds(sweep, calls):
    built = cap.build_grid()
    for key in ("ri", "rj", "vi", "vj", "rhoi", "rhoj", "mfpa", "dvisc", "t", "mask", "block"):
        np.testing.assert_array_equal(sweep[key], built[key], err_msg=key)
    assert calls == cap.build_calls()


def test_all_three_supported_methods_are_present_and_the_fourth_is_not(calls):
    """`icoag = 4` is UP-5: it reads `mfppi`/`mfppj`, assigned only inside the
    `icoag == 1` block, so the golden must not contain it."""
    assert {c["icoag"] for c in calls} == {1, 2, 3}
    assert {c["coag_on"] for c in calls} == {0, 1}


def test_the_volumes_are_the_ones_the_driver_would_pass(sweep, grid):
    """`ukca_calc_coag_kernel.F90:245-246` passes `wvol` beside `wetdp/2`, and
    `wvol = (pi/6)*wetdp**3`, i.e. `(4/3)*pi*r**3`. The main and axis blocks
    must therefore carry consistent pairs -- otherwise the sweep is validating
    the routine on inputs the model never produces."""
    rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("main"))
    expected = np.array([cap.volume(float(r)) for r in grid["ri"][rows]])
    np.testing.assert_array_equal(grid["vi"][rows], expected)


def test_the_mask_is_a_select_not_a_multiply(sweep, grid):
    off = grid["block"] == cap.BLOCKS.index("masked_off")
    assert off.sum() == 7
    poisoned = sweep["ri"][off]
    assert np.isnan(poisoned).any() and np.isinf(poisoned).any() and (poisoned == 0.0).any()
    assert np.all(sweep["kij"][:, :, off] == 0.0)


def test_coag_on_zero_zeroes_rows_the_mask_selected(sweep, grid, calls):
    """`:239-242` returns before `WHERE (mask(:))`, so this is not the same
    statement as the test above."""
    on = grid["mask"] == 1
    for i, c in enumerate(calls):
        vals = sweep["kij"][:, i, on]
        if c["coag_on"] == 0:
            assert np.all(vals == 0.0), cap.call_label(c)
        else:
            assert np.all(vals > 0.0) and np.all(np.isfinite(vals)), cap.call_label(c)


def test_the_routine_is_setup_independent(sweep):
    assert sweep["setups"].tolist() == [1, 4]
    np.testing.assert_array_equal(sweep["kij"][0], sweep["kij"][1])


@pytest.mark.parametrize(("block", "const_icoags"), cap.CONSTANT_BLOCK_AT)
def test_a_block_is_constant_exactly_where_its_argument_is_unread(
    sweep, grid, calls, block, const_icoags
):
    """`rhoi`, `rhoj` and `mfpa` reach `kij` at methods 1 and 2 and not at 3,
    where `:316-318` uses only `t`, `dvisc`, `ri`, `rj` and a PARAMETER."""
    rows = grid["block"] == cap.BLOCKS.index(block)
    assert rows.sum() > 1
    for i, c in enumerate(calls):
        if c["coag_on"] == 0:
            continue
        vals = sweep["kij"][0, i, rows]
        constant = bool(np.all(vals == vals[0]))
        assert constant == (c["icoag"] in const_icoags), cap.call_label(c)


def test_the_particle_volumes_are_dead_at_methods_two_and_three(sweep, grid, calls):
    """The decoupled block is the `ri` block with `vi` and `vj` multiplied by
    four and nothing else changed, so equality is exactly "the argument is not
    read". Method 1 uses them in `veli = SQRT(term1*t/(rhoi*vi))` (`:257`);
    method 2 rebuilds `vmid` from `rmid` and method 3 has no velocity at all."""
    ri_rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("ri"))
    dc_rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("vi_decoupled"))
    assert ri_rows.size == dc_rows.size
    np.testing.assert_array_equal(grid["ri"][ri_rows], grid["ri"][dc_rows])
    assert np.all(grid["vi"][dc_rows] == grid["vi"][ri_rows] * cap.VI_DECOUPLE)
    for i, c in enumerate(calls):
        if c["coag_on"] == 0:
            continue
        same = np.array_equal(sweep["kij"][0, i, ri_rows], sweep["kij"][0, i, dc_rows])
        assert same == (c["icoag"] in (2, 3)), cap.call_label(c)


@pytest.mark.parametrize("block", cap.ALWAYS_VARYING)
def test_the_arguments_every_method_reads_do_move_the_answer(sweep, grid, calls, block):
    rows = grid["block"] == cap.BLOCKS.index(block)
    for i, c in enumerate(calls):
        if c["coag_on"] == 0:
            continue
        vals = sweep["kij"][0, i, rows]
        assert not np.all(vals == vals[0]), f"{block} under {cap.call_label(c)}"


def test_the_kernel_is_byte_symmetric_under_an_i_j_swap(sweep, grid, calls):
    """Bit for bit, at all three methods. It survives rounding because every
    combination of the i and j quantities is an addition -- `dcoefi + dcoefj`
    at `:279`, `deli*deli + delj*delj` at `:281`, `veli*veli + velj*velj` at
    `:283` -- and addition is commutative.

    The consequence is not comfort. `coag_mode` is symmetric on all 64 entries
    (phase C), so nothing in the table catches a transposition; this says
    nothing in the kernel does either, and `ukca_calc_coag_kernel`'s subscripts
    are the only thing that fixes the convention."""
    a = np.flatnonzero(grid["block"] == cap.BLOCKS.index("swap_a"))
    b = np.flatnonzero(grid["block"] == cap.BLOCKS.index("swap_b"))
    assert a.size == b.size >= 6
    # The pairs really are swapped, and really are asymmetric to begin with.
    np.testing.assert_array_equal(grid["ri"][a], grid["rj"][b])
    np.testing.assert_array_equal(grid["rhoi"][a], grid["rhoj"][b])
    assert np.all(grid["ri"][a] != grid["rj"][a])
    assert np.all(grid["rhoi"][a] != grid["rhoj"][a])
    for i, c in enumerate(calls):
        if c["coag_on"] == 0:
            continue
        np.testing.assert_array_equal(
            sweep["kij"][0, i, a], sweep["kij"][0, i, b], err_msg=cap.call_label(c)
        )


def test_the_three_methods_disagree_with_each_other(sweep, calls):
    """Otherwise `icoag` never reached the Fortran and every byte equality
    above is comparing one method against itself."""
    live = [i for i, c in enumerate(calls) if c["coag_on"] == 1]
    for a in range(len(live)):
        for b in range(a + 1, len(live)):
            assert not np.array_equal(sweep["kij"][0, live[a]], sweep["kij"][0, live[b]])


def test_the_grid_reaches_both_limits_of_the_cunningham_correction(sweep, grid):
    """`cci = 1 + kn*(1.257 + 0.4*EXP(-1.1/kn))` tends to 1 as `kn -> 0` and to
    `1 + 1.657*kn` as `kn -> inf`. A sweep confined to the transition regime
    validates nothing about either asymptote, and the modes this kernel is
    applied to span five decades of radius."""
    lo, hi = cap.verify_knudsen_coverage(grid)
    assert lo == float(sweep["_kn_min"]) and hi == float(sweep["_kn_max"])
    assert lo < 0.01 and hi > 100.0


MUTATIONS = (
    "zero_a_masked_row",
    "break_setup_independence",
    "wake_up_a_dead_volume",
    "flatten_a_live_axis",
    "break_the_symmetry",
    "leak_through_coag_on_zero",
    "make_two_methods_agree",
)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_captures_own_guards_reject_a_corrupted_archive(sweep, grid, calls, mutation):
    """Name the mutation that would fail the guard, then apply it."""
    kij = np.array(sweep["kij"])

    def rows(name):
        return np.flatnonzero(grid["block"] == cap.BLOCKS.index(name))

    if mutation == "zero_a_masked_row":
        kij[0, 0, rows("masked_off")[0]] = 1.0e-30
    elif mutation == "break_setup_independence":
        kij[1, 0, 5] = np.nextafter(kij[1, 0, 5], np.inf)
    elif mutation == "wake_up_a_dead_volume":
        i = cap._find(calls, 2, 1)
        kij[:, i, rows("vi_decoupled")[3]] *= 1.001
    elif mutation == "flatten_a_live_axis":
        i = cap._find(calls, 1, 1)
        r = rows("t")
        kij[:, i, r] = kij[0, i, r[0]]
    elif mutation == "break_the_symmetry":
        i = cap._find(calls, 3, 1)
        kij[:, i, rows("swap_b")[2]] = np.nextafter(kij[0, i, rows("swap_b")[2]], np.inf)
    elif mutation == "leak_through_coag_on_zero":
        i = cap._find(calls, 1, 0)
        kij[:, i, rows("main")[0]] = 1.0e-12
    elif mutation == "make_two_methods_agree":
        a, b = cap._find(calls, 1, 1), cap._find(calls, 2, 1)
        kij[:, b] = kij[:, a]

    with pytest.raises(SystemExit):
        cap.verify_outputs(kij, grid, calls)


def test_the_guards_accept_the_archive_as_committed(sweep, grid, calls):
    report = cap.verify_outputs(np.array(sweep["kij"]), grid, calls)
    assert report["live_rows"] == 273
    assert report["masked_rows"] == 7
    assert report["swap_pairs"] == 6
