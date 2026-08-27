"""Task 56: the `ukca_coagwithnucl` sweep, and the branch issue #13 names.

No `fortran` marker: the archive is committed, so all of this runs in CI.
7 setups x 2 switch combos x 4 `(intraoff, interoff)` pairs x 20 rows.

**The `mdcpnew < 0` reset is reached 438 times.** Issue #13 records it as one
of the branches no trajectory fixture reaches, and it is the reason the `icp`
loop at `:544-575` cannot be `vmap`ped:

```fortran
DO icp=1,ncp
  ...
  WHERE (mask1(:) .AND. (mdcpnew(:) < 0.0))
    nd(:,imode)=0.0
    mask1(:)=.FALSE. ! set false so not used for other icp values
  END WHERE
```

`mask1` is mutated inside the loop and the source comment says why: once any
component's new mass goes negative, the mode's number is zeroed and every later
component is skipped. A broadcast over `icp` evaluates all components against
the original mask and gets a different answer. This archive is what lets the
port's test *demonstrate* that rather than assert it.

Getting there needs construction. `mdcpnew < 0` means the net transfer out of a
component exceeds the mass that was there, so the grid hands the routine large
inter-modal kernels with a tiny per-component `md` -- kernels being an input
here, already pinned by task 50.

`l_dust_mp_ageing` is inert in five of seven setups
---------------------------------------------------

It reaches this routine only through `topmode`, which bounds the soluble loop's
insoluble inner loop at `:342` and the insoluble outer loop at `:427`. Setups 1,
3 and 5 have four active modes; setups 2 and 4 have five, the fifth being
`mode_ait_insol`, which `topmode = 5` already covers. Setups **6 and 8** are
required to differ and do -- and note the contrast with `ukca_conden`, where
setup 6 collided because it has no condensable gas. Coagulation needs none.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coagwithnucl.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_coagwithnucl_leaf as cap  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coagwithnucl_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def data(sweep):
    keys = (
        "nd",
        "md",
        "mdt",
        "bud",
        "ageterm2",
        "mode",
        "topmode",
        "coag_mode",
        "ncp",
        "nchemg",
        "nbudaer",
        "in_nd",
        "in_md",
        "in_mdt",
        "in_delgc",
    )
    return {k: sweep[k] for k in keys}


def test_the_archive_matches_the_grid_the_capture_builds(sweep):
    built = cap.build_grid(cap.NCP_MAX, cap.NCHEMG_MAX)
    np.testing.assert_array_equal(sweep["block"], built["block"])
    np.testing.assert_array_equal(sweep["in_kii"], built["kii"])
    np.testing.assert_array_equal(sweep["in_kij"], built["kij"])


def test_every_setup_combo_and_switch_pair_is_present(sweep):
    assert sweep["setups"].tolist() == [1, 2, 3, 4, 5, 6, 8]
    assert [str(c) for c in sweep["combos"]] == ["default", "dust_ageing"]
    assert sweep["switches"].tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]


def test_the_source_still_has_the_loop_carried_icp_loop():
    """If upstream ever rewrote `:544-575` to stop mutating `mask1`, the port's
    `lax.scan` would be reproducing a routine that no longer exists."""
    info = cap.verify_source_structure()
    assert info["checked"] == 10
    assert info["budget_sites"] == 53


def test_the_negative_mass_reset_is_reached(data):
    """Issue #13's branch. Counted, because "it is covered" is what a fixture
    says right up until someone measures it."""
    total = 0
    for s_i in range(data["nd"].shape[0]):
        for c_i in range(data["nd"].shape[1]):
            for k_i in range(data["nd"].shape[2]):
                zeroed = (data["nd"][s_i, c_i, k_i] == 0.0) & (data["in_nd"][s_i, c_i] > 0.0)
                total += int(zeroed.sum())
    assert total == 438, f"{total} (row, mode) pairs reach the reset, measured 438"


def test_dust_ageing_is_inert_in_exactly_five_setups(sweep, data):
    """`topmode` bounds both insoluble loops, so the switch matters only where
    a mode above `mode_ait_insol` is active."""
    np.testing.assert_array_equal(data["topmode"][:, 0], np.full(7, 5))
    np.testing.assert_array_equal(data["topmode"][:, 1], np.full(7, 8))
    setups = sweep["setups"].tolist()
    changed = [
        s
        for i, s in enumerate(setups)
        if not np.array_equal(data["nd"][i, 0], data["nd"][i, 1])
        or not np.array_equal(data["bud"][i, 0], data["bud"][i, 1])
        or not np.array_equal(data["ageterm2"][i, 0], data["ageterm2"][i, 1])
    ]
    assert changed == [6, 8], f"dust ageing changed setups {changed}"


def test_the_switches_each_remove_a_term(sweep, data):
    """`intraoff` gates `A` at `:299` and `interoff` gates `B` at `:313`. Both
    default 0 everywhere, so all four combinations exist only here -- and each
    must change the answer, or the switch never reached the Fortran."""
    switches = [tuple(s) for s in sweep["switches"].tolist()]
    base = switches.index((0, 0))
    for k_i, pair in enumerate(switches):
        if pair == (0, 0):
            continue
        differs = any(
            not np.array_equal(data["nd"][s_i, c_i, k_i], data["nd"][s_i, c_i, base])
            for s_i in range(data["nd"].shape[0])
            for c_i in range(data["nd"].shape[1])
        )
        assert differs, f"switch pair {pair} changed nothing"


def test_coag_mode_is_recorded_and_symmetric(data):
    """Phase C measured `coag_mode` symmetric on all 64 entries and warned that
    a transposed transcription would be undetectable in the table. This is the
    routine where the subscript order has consequences -- `:534` scatters into
    `coag_mode(imode, jmode)` -- so the table travels with the golden."""
    cm = data["coag_mode"]
    assert cm.shape[-2:] == (8, 8)
    for s_i in range(cm.shape[0]):
        for c_i in range(cm.shape[1]):
            np.testing.assert_array_equal(cm[s_i, c_i], cm[s_i, c_i].T)
    # Setup-independent, as phase C measured.
    for s_i in range(1, cm.shape[0]):
        np.testing.assert_array_equal(cm[0, 0], cm[s_i, 0])


def test_the_hole_at_budget_slot_zero_is_never_written(data):
    assert np.all(data["bud"][..., 0] == 0.0)


def test_number_never_goes_negative_and_ageterm2_ran(data):
    assert np.all(data["nd"] >= 0.0)
    assert np.all(np.isfinite(data["md"]))
    assert np.any(data["ageterm2"] != 0.0)


MUTATIONS = ("write_the_hole", "negative_nd", "erase_the_reset")


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_captures_own_guards_reject_a_corrupted_archive(data, mutation):
    d = {k: np.array(v) for k, v in data.items()}
    if mutation == "write_the_hole":
        d["bud"][0, 0, 0, 0, 0] = 1.0e-30
    elif mutation == "negative_nd":
        d["nd"][0, 0, 0, 0, 0] = -1.0
    else:
        # No row reaches the reset any more: the guard must refuse, because a
        # golden without it cannot distinguish a scan from a broadcast.
        d["nd"] = np.where(d["nd"] == 0.0, 1.0, d["nd"])

    with pytest.raises(SystemExit):
        cap.verify_outputs(d, lambda s_i, c_i: {"nd": d["in_nd"][s_i, c_i]})


def test_the_guards_accept_the_archive_as_committed(data):
    report = cap.verify_outputs(
        {k: np.array(v) for k, v in data.items()},
        lambda s_i, c_i: {"nd": data["in_nd"][s_i, c_i]},
    )
    assert report == {"negative_mass_rows": 438, "budget_written": 24, "ageterm2_written": 12}
