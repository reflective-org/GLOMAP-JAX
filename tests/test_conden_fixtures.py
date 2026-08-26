"""Task 53: the `ukca_conden` sweep -- 7 setups x 2 switch combos x 4 configs.

No `fortran` marker: the archive is committed, so all of this runs in CI.

`ukca_conden` is the widest routine in the port: four `INTENT(IN OUT)` arrays,
three `INTENT(OUT)`, and 30 budget write sites. Four things this archive
establishes, none of which a value comparison would.

**1. A budget index gates the physics.** `deltams` and `deltami` are zeroed at
`:363-364` and assigned *only inside* `IF (nmascond... > 0)`, and `md`/`mdt`
are then updated from `deltams`. An uncarried slot therefore loses the **mass**,
not just the diagnostic. Measured across all seven setups: every (active mode,
condensable component) pair has its slot carried, 54 of 54. The gating is inert
today and nothing enforces that it stays so. Issue #30.

**2. `:576-602` is the same block twice** -- the only one of the 30 write sites
that repeats. 30 sites, 29 distinct `(name, delta)` pairs, and the duplicate is
`(nmascondocaccins, deltami)`. The budget field is an accumulation so it
doubles; `ageterm1` is an assignment so it does not; `md`/`mdt` take `deltams`
so they do not either. Issue #29.

**3. `l_dust_mp_ageing` reaches this routine only through `topmode`,** and
setup 8 is the only supported setup where that changes anything. In setups 1-5
no mode above `mode_ait_insol` is active; in setup 6 no gas is condensable. The
capture declares those six collisions with that reason and requires setup 8's
two combos to **differ** -- which is the half that can fail.

**4. Setups 2 and 8 are the same program at `topmode = 5`.** Their
`condensable`, `condensable_choice`, `mm_gas`, `dimen`, `sigmag` and `num_eps`
tables are identical, all ten of setup 2's condensation budget names occupy the
same slot in setup 8, and their `mode` vectors differ only in modes 6 and 7 --
which `topmode = 5` excludes. Declared, and re-derived here from the ported
tables rather than taken on trust.

And two invariants asserted rather than reached: UP-4's `delgc_cond > gc` guard
is unreachable because `delgc_cond = gc*(1 - EXP(-sumnc*dtz))`, and `mask4i`
needs `mode_sup_insol`, which no supported setup activates.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "conden.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_conden_leaf as cap  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_conden_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def data(sweep):
    keys = (
        "md",
        "mdt",
        "gc",
        "bud",
        "delgc",
        "ageterm1",
        "s_cond_s",
        "mode",
        "topmode",
        "ncp",
        "nchemg",
        "nbudaer",
        "in_md",
        "in_mdt",
        "in_gc",
    )
    return {k: sweep[k] for k in keys}


@pytest.fixture(scope="module")
def grid(sweep):
    return {
        k[3:]: sweep[k]
        for k in sweep.files
        if k.startswith("in_") and k[3:] not in ("md", "mdt", "gc")
    }


def test_the_archive_matches_the_grid_the_capture_builds(grid):
    built = cap.build_grid(cap.NCP_MAX, cap.NCHEMG_MAX)
    for key in grid:
        np.testing.assert_array_equal(grid[key], built[key], err_msg=key)


def test_every_supported_setup_and_both_switch_combinations_are_present(sweep):
    assert sweep["setups"].tolist() == [1, 2, 3, 4, 5, 6, 8]
    assert [str(c) for c in sweep["combos"]] == ["default", "dust_ageing"]
    assert sweep["config"].shape == (4, 3)


def test_the_thirty_write_sites_are_twenty_nine_distinct_pairs():
    """Issue #29, re-derived from the vendored source on every run. If the
    duplicate is ever fixed upstream this fails, which is what should happen --
    the port reproduces it behind a flag."""
    counts = cap.verify_source_structure()
    assert counts == {"write_sites": 30, "distinct": 29}


def test_every_condensable_pair_has_a_budget_slot():
    """Issue #30. `deltams` stays zero without one, and `md` is updated from
    `deltams`, so an uncarried slot loses mass rather than a diagnostic."""
    assert cap.verify_budget_gating() == {"pairs": 54}


def test_the_hole_at_budget_slot_zero_is_never_written(data):
    """`bud_aer_mas` is declared `(nbox,0:nbudaer)` and every uncarried index
    points at slot 0. Phase C chose `NOT_CARRIED = 0` over `-1` because `-1`
    wraps under every scatter mode; this is the other half of that decision."""
    assert np.all(data["bud"][..., 0] == 0.0)


def test_dust_ageing_moves_topmode_and_only_setup_eight_notices(sweep, data):
    """`l_dust_mp_ageing` reaches `ukca_conden` only through `topmode`. Both
    halves: the switch must take (topmode 5 -> 8 everywhere) and must change
    the answer in exactly one setup."""
    np.testing.assert_array_equal(data["topmode"][:, 0], np.full(7, 5))
    np.testing.assert_array_equal(data["topmode"][:, 1], np.full(7, 8))
    setups = sweep["setups"].tolist()
    changed = [
        s
        for i, s in enumerate(setups)
        if not np.array_equal(data["gc"][i, 0], data["gc"][i, 1])
        or not np.array_equal(data["bud"][i, 0], data["bud"][i, 1])
    ]
    assert changed == [8], f"dust ageing changed setups {changed}, expected only 8"


def test_setups_two_and_eight_agree_only_because_topmode_excludes_the_extra_modes(sweep, data):
    """The collision the capture declares, re-derived from the ported tables
    rather than from the golden."""
    from glomap_jax.physics import budget_indices as bi
    from glomap_jax.physics import gas_indices as gi
    from glomap_jax.physics import modes as md

    g2, g8 = gi.build(2), gi.build(8)
    for field in ("condensable", "condensable_choice", "mm_gas", "dimen"):
        np.testing.assert_array_equal(getattr(g2, field), getattr(g8, field), err_msg=field)
    t2, t8 = md.build(2), md.build(8)
    np.testing.assert_array_equal(t2.sigmag, t8.sigmag)
    np.testing.assert_array_equal(t2.num_eps, t8.num_eps)
    assert t2.mode.astype(int).tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert t8.mode.astype(int).tolist() == [1, 1, 1, 1, 1, 1, 1, 0]

    m2, m8 = bi.build(2), bi.build(8)
    shared = [n for n in bi.BUDGET_NAMES if n.startswith("nmascond") and m2.is_carried(n)]
    assert len(shared) == 10
    assert all(m2.slot(n) == m8.slot(n) for n in shared)

    setups = sweep["setups"].tolist()
    i2, i8 = setups.index(2), setups.index(8)
    np.testing.assert_array_equal(data["gc"][i2, 0], data["gc"][i8, 0])
    # And they must part company once topmode lets modes 6 and 7 in.
    assert not np.array_equal(data["bud"][i2, 1], data["bud"][i8, 1])


def test_up4_is_unreachable_on_every_row(data):
    """`:353-355` computes `delgc_cond/gc` where `= gc` was intended, guarded
    by `delgc_cond > gc`. But `delgc_cond = gc*(1 - EXP(-sumnc*dtz))` and the
    exponential is non-negative. Asserted over the whole archive, which is what
    `docs/UPSTREAM_DEFECTS.md` gives as UP-4's disposition."""
    for s_i in range(data["gc"].shape[0]):
        for c_i in range(data["gc"].shape[1]):
            gc_in = data["in_gc"][s_i, c_i]
            for k_i in range(data["gc"].shape[2]):
                assert np.all(data["delgc"][s_i, c_i, k_i] <= gc_in)


def test_condensation_only_moves_mass_one_way(data):
    """Gas down, aerosol up, nothing negative. Cheap, and it is the check that
    would catch a sign error in the transfer that every other test here would
    let through."""
    for s_i in range(data["gc"].shape[0]):
        for c_i in range(data["gc"].shape[1]):
            for k_i in range(data["gc"].shape[2]):
                assert np.all(data["gc"][s_i, c_i, k_i] <= data["in_gc"][s_i, c_i])
                assert np.all(data["gc"][s_i, c_i, k_i] >= 0.0)
                assert np.all(data["mdt"][s_i, c_i, k_i] >= data["in_mdt"][s_i, c_i])
                assert np.all(data["s_cond_s"][s_i, c_i, k_i] >= 0.0)


def test_the_insoluble_half_of_the_routine_actually_ran(data):
    """`ageterm1` is written only inside `mask3i`, i.e. only when an insoluble
    mode is active and above threshold. If it were zero everywhere, every
    insoluble block would be unvalidated and a port that omitted them would
    pass."""
    nonzero = int(
        np.sum(
            [
                np.any(data["ageterm1"][i, j, k] != 0.0)
                for i in range(7)
                for j in range(2)
                for k in range(4)
            ]
        )
    )
    assert nonzero == 24, f"{nonzero} of 56 configurations reached ageterm1, measured 24"


def test_mask4i_is_never_reached(data):
    """It needs `mode_sup_insol`, and slot 8 needs setup 12 or 13, which the box
    model does not implement. Asserted rather than assumed: the whole
    super-coarse insoluble branch has no reference and is not ported."""
    assert not np.any(data["mode"][..., 7]), "mode_sup_insol is active somewhere after all"


MUTATIONS = ("write_the_hole", "increase_a_gas", "lose_the_duplicate")


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_captures_own_guards_reject_a_corrupted_archive(
    sweep, data, grid, monkeypatch, mutation
):
    """Name the mutation, then apply it."""
    d = {k: np.array(v) for k, v in data.items()}
    if mutation == "write_the_hole":
        d["bud"][0, 0, 0, 0, 0] = 1.0e-30
        with pytest.raises(SystemExit):
            cap.verify_outputs(d, grid)
        return
    if mutation == "increase_a_gas":
        d["gc"][0, 0, 0] = d["in_gc"][0, 0] + 1.0
        with pytest.raises(SystemExit):
            cap.verify_outputs(d, grid)
        return
    # The source guard, not the archive: stub the write-site count and the
    # staleness check must refuse it.
    monkeypatch.setattr(cap, "SOURCE", REPO / "fortran" / "src" / "ukca" / "ukca_calcnucrate.F90")
    with pytest.raises(SystemExit):
        cap.verify_source_structure()


def test_the_guards_accept_the_archive_as_committed(data, grid):
    report = cap.verify_outputs({k: np.array(v) for k, v in data.items()}, grid)
    assert report["mask4i_hits"] == 0
    assert report["insoluble_ageterm"] == 24
    assert report["budget_written"] == 48
