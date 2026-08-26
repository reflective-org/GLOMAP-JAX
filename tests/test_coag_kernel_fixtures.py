"""Task 50: the `ukca_calc_coag_kernel` sweep, and the slot map.

No `fortran` marker: the archive is committed, so all of this runs in CI.

`ukca_calc_coag_kernel` is a driver. Every number it returns came from
`ukca_coag_coff_v`, which task 48 pinned. This archive exists for the one thing
no value comparison can check.

**The kernel is byte-symmetric under an `(i, j)` swap** (task 48) and
`coag_mode` is symmetric on all 64 entries (phase C). So a transposed
convention writes the *right number into the wrong slot*, and every value test
in this project still passes. The only discriminator is which entries are
non-zero. `expected_slots` re-derives that set from the Fortran's loop bounds
and the capture compares it against what the compiled routine filled -- so a
misreading of the loops fails the capture rather than becoming the port's
specification.

Three families, and they are not one triangle: soluble x larger-soluble
(upper), soluble x larger-insoluble (upper, across the split), and
**insoluble x smaller-soluble (lower)**. With every mode active that is 19
slots and no `(i, j)` appears in both orders.

Three soluble/insoluble pairs are filled in neither order -- `(2,5)`, `(3,6)`,
`(4,7)` in the Fortran's 1-based indices, the same-size-class pairs. That is
not a gap the port has to work around: `ukca_coagwithnucl.F90:307`, `:345` and
`:467` read `kij_arr` with the same three loop bounds, so it never asks for
them. It reads *less*, in fact: its insoluble inner loop stops at `topmode`,
which is `mode_ait_insol` unless `l_dust_mp_ageing` is on.

**The size convention is measured, not read.** `:281-286` selects the
partner's size by `modesol` -- soluble partners enter wet, insoluble ones dry
-- while `:297-299` takes `wetdp` unconditionally for an insoluble `imode`. So
the same insoluble mode enters wet as `i` and dry as `j`. In the model those
coincide (an insoluble mode carries no water), which is exactly the coincidence
that would hide the routing, so the grid feeds `drydp = 0.7*wetdp` for every
mode and the capture calls `ukca_coag_coff_v` directly under both conventions
to see which one each slot matches: 8 dry, 43 wet, over the seven setups.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coag_kernel.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_coag_kernel_leaf as cap  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coag_kernel_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def data(sweep):
    return {k: sweep[k] for k in ("kii", "kij", "direct_wetj", "direct_dryj", "mode", "modesol")}


@pytest.fixture(scope="module")
def grid(sweep):
    return {k: sweep[k] for k in ("wetdp", "drydp", "wvol", "dvol", "rhopar", "mfpa", "dvisc", "t")}


def test_the_archive_matches_the_grid_the_capture_script_builds(grid):
    built = cap.build_grid()
    for key in grid:
        np.testing.assert_array_equal(grid[key], built[key], err_msg=key)


def test_all_seven_supported_setups_are_present(sweep):
    assert sweep["setups"].tolist() == [1, 2, 3, 4, 5, 6, 8]


def test_the_dry_and_wet_sizes_differ_for_every_mode(grid):
    """The fixture's whole ability to say which size the driver passes rests on
    this. In the model an insoluble mode has `drydp == wetdp`, and that
    coincidence would make the two conventions indistinguishable."""
    assert np.all(grid["drydp"] < grid["wetdp"])
    assert np.all(grid["dvol"] < grid["wvol"])


def test_every_mode_carries_a_distinct_size_and_density(grid):
    """With two modes sharing a value, a slot mix-up *within* a family would be
    invisible as well as one between families."""
    for row in grid["wetdp"]:
        assert len(set(row.tolist())) == 8
    assert len(set(grid["rhopar"][0].tolist())) == 8


@pytest.mark.parametrize("s_i", range(7))
def test_the_filled_slots_are_the_ones_the_loop_bounds_predict(sweep, data, s_i):
    """The transposition check. Compares the set of non-zero `(i, j)` entries
    against `expected_slots`, which is a reading of `:243`, `:262`, `:276` and
    `:308-309` -- not of the golden."""
    want_kii, want_kij = cap.expected_slots(data["mode"][s_i], data["modesol"][s_i])
    c_i = list(zip(sweep["call_icoag"].tolist(), sweep["call_coag_on"].tolist())).index((1, 1))
    kii, kij = data["kii"][s_i, c_i], data["kij"][s_i, c_i]
    got_kii = {m for m in range(8) if np.any(kii[:, m] != 0.0)}
    got_kij = {(i, j) for i in range(8) for j in range(8) if np.any(kij[:, i, j] != 0.0)}
    assert got_kii == want_kii
    assert got_kij == want_kij


@pytest.mark.parametrize("s_i", range(7))
def test_no_slot_is_filled_in_both_orders(data, s_i):
    """What makes `kij_arr` directional at all. If any `(i,j)` and `(j,i)` were
    both filled, the transposition question would be undecidable from the
    zero pattern too."""
    _, want = cap.expected_slots(data["mode"][s_i], data["modesol"][s_i])
    assert not {(i, j) for i, j in want if (j, i) in want}


def test_the_lower_triangular_family_is_actually_populated(data):
    """`:308-322` writes insoluble `imode` against *smaller* soluble `jmode`.
    If no setup reached it, the "not one triangle" claim would be untested and
    a port that wrote only the upper triangle would pass everything else."""
    lower = set()
    for s_i in range(7):
        _, want = cap.expected_slots(data["mode"][s_i], data["modesol"][s_i])
        lower |= {(i, j) for i, j in want if i > j}
    assert lower, "no setup reaches the insoluble-imode loop"
    assert lower <= {(4, 2), (4, 3), (5, 3)}


def test_the_same_size_class_pairs_are_filled_by_neither_family(data):
    """`(2,5)`, `(3,6)` and `(4,7)` 1-based -- `(1,4)`, `(2,5)`, `(3,6)`
    zero-based. Recorded because it looks like an omission and is not:
    `ukca_coagwithnucl` reads with the same bounds and never asks for them."""
    all_on = np.ones(8, dtype=np.int32)
    _, want = cap.expected_slots(all_on, np.array([1, 1, 1, 1, 0, 0, 0, 0]))
    for pair in ((1, 4), (2, 5), (3, 6)):
        assert pair not in want and pair[::-1] not in want, pair


@pytest.mark.parametrize("s_i", range(7))
def test_each_filled_slot_matches_exactly_one_size_convention(sweep, data, s_i):
    """The measurement, repeated from the committed archive. A slot whose `j`
    is insoluble must match the dry-size call and not the wet one, and vice
    versa -- and a slot matching neither would mean the driver passes something
    this file has not understood."""
    modesol = data["modesol"][s_i]
    _, want = cap.expected_slots(data["mode"][s_i], modesol)
    c_i = list(zip(sweep["call_icoag"].tolist(), sweep["call_coag_on"].tolist())).index((1, 1))
    seen_dry = 0
    for i, j in sorted(want):
        wet_ok = np.array_equal(data["kij"][s_i, c_i][:, i, j], data["direct_wetj"][s_i][:, i, j])
        dry_ok = np.array_equal(data["kij"][s_i, c_i][:, i, j], data["direct_dryj"][s_i][:, i, j])
        assert wet_ok != dry_ok, f"slot ({i},{j}) matches neither convention or both"
        assert dry_ok == (modesol[j] == 0), f"slot ({i},{j})"
        seen_dry += dry_ok
    if any(modesol[j] == 0 for _, j in want):
        assert seen_dry > 0


def test_the_dry_branch_is_reached_somewhere(data, sweep):
    """`:283-285` is the `ELSE` arm. If no setup reached it the convention
    measurement above would be vacuous everywhere it is asserted."""
    total = 0
    for s_i in range(7):
        modesol = data["modesol"][s_i]
        _, want = cap.expected_slots(data["mode"][s_i], modesol)
        total += sum(1 for _, j in want if modesol[j] == 0)
    assert total == 8, f"{total} slots take the dry branch, measured 8"


MUTATIONS = (
    "transpose_a_slot",
    "drop_the_lower_family",
    "fill_a_forbidden_pair",
    "leak_coag_on_zero",
)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_captures_own_guards_reject_a_corrupted_archive(sweep, data, grid, mutation):
    d = {k: np.array(v) for k, v in data.items()}
    s_i = 6  # setup 8, seven active modes
    c_i = list(zip(sweep["call_icoag"].tolist(), sweep["call_coag_on"].tolist())).index((1, 1))

    if mutation == "transpose_a_slot":
        d["kij"][s_i, c_i][:, 3, 0] = d["kij"][s_i, c_i][:, 0, 3]
        d["kij"][s_i, c_i][:, 0, 3] = 0.0
    elif mutation == "drop_the_lower_family":
        d["kij"][s_i, c_i][:, 4, 2] = 0.0
    elif mutation == "fill_a_forbidden_pair":
        d["kij"][s_i, c_i][:, 1, 4] = 1.0e-12
    elif mutation == "leak_coag_on_zero":
        off = list(zip(sweep["call_icoag"].tolist(), sweep["call_coag_on"].tolist())).index((1, 0))
        d["kii"][s_i, off][:, 0] = 1.0e-12

    with pytest.raises(SystemExit):
        cap.verify_outputs(d, grid)


def test_the_guards_accept_the_archive_as_committed(data, grid):
    report = cap.verify_outputs({k: np.array(v) for k, v in data.items()}, grid)
    assert report["setups"] == 7
    assert (report["dryj_slots"], report["wetj_slots"]) == (8, 43)
