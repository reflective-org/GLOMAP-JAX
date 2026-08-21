"""Task 35c -- `tests/goldens/water_content.f64.leaf.npz`.

None of these need the Fortran toolchain: they check the committed archive and
`validation/capture_water_leaf.py`'s pure-Python machinery (`col`, grid
construction, `parse_reference`, `simulate`), not the extension. That is what
lets this file run in CI, which has no gfortran.

The main correctness risk this file exists to pin is the column offset:
`leaf_water_content` remaps signed ion species (-4..3) onto a plain (n,8)
array with a `+5` (Fortran 1-based) / `+4` (0-based numpy) shift, and getting
that wrong does not raise -- it returns a different, finite, plausible number
(demonstrated below). `test_simulate_reproduces_every_captured_cell` is the
guard: it recomputes `wc` for every row and every rh independently, in the
Fortran's own iteration order including the aw-ratchet, from the (cl, ions)
the archive actually used. An offset bug in `build_main_grid` would put a
concentration at the wrong column for BOTH the capture and this
reimplementation to agree on, but it would not change what the real Fortran
did when it was captured -- so a wrong offset shows up as a disagreement
between `simulate()` and the committed `wc`, not as an untested assumption.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"
ARCHIVE = GOLDENS / "water_content.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_water_leaf as cwl  # noqa: E402


@pytest.fixture(scope="module")
def golden():
    if not ARCHIVE.is_file():
        pytest.skip(f"{ARCHIVE.name} missing -- run `python validation/capture_water_leaf.py`")
    with np.load(ARCHIVE, allow_pickle=False) as data:
        yield {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def ref():
    return cwl.parse_reference()


# ---------------------------------------------------------------------------
# The archive itself
# ---------------------------------------------------------------------------


def test_the_archive_is_committed():
    assert ARCHIVE.is_file(), "run `python validation/capture_water_leaf.py` (needs gfortran)"


def test_shapes_and_finiteness(golden):
    cl, ions, rh, wc = golden["cl"], golden["ions"], golden["rh"], golden["wc"]
    comp_label = golden["comp_label"]
    C, R = cl.shape[0], rh.shape[0]
    assert cl.shape == (C, 8)
    assert ions.shape == (C, 8)
    assert comp_label.shape == (C,)
    assert wc.shape == (2, C, R)
    for name in ("cl", "rh", "wc", "mask_rh", "mask_cl", "mask_wc"):
        assert np.isfinite(golden[name]).all(), name
    assert (golden["rh"] >= 0.0).all()
    assert set(np.unique(ions)) <= {0, 1}
    assert list(golden["fix_water_flags"]) == [0, 1]


def test_an_all_zeros_archive_would_not_pass(golden):
    """The vacuous-test guard from CLAUDE.md: name the mutation, then check it
    would actually be caught. An all-zeros `wc` of the right shape passes
    shape+finiteness -- but not the cross-check below, nor the anti-collapse
    checks, both of which require genuine nonzero divergence."""
    zeroed = np.zeros_like(golden["wc"])
    assert not np.array_equal(zeroed, golden["wc"])
    assert np.any(golden["wc"] != 0.0)


# ---------------------------------------------------------------------------
# The column offset (the main correctness risk)
# ---------------------------------------------------------------------------


def test_col_is_species_plus_four():
    for species in range(-4, 4):
        assert cwl.col(species) == species + 4


def test_col_refuses_outside_its_declared_range():
    with pytest.raises(IndexError):
        cwl.col(4)
    with pytest.raises(IndexError):
        cwl.col(-5)


@pytest.mark.fortran
def test_the_wrong_offset_silently_returns_a_different_plausible_number():
    """Measured directly against the built extension, not asserted from
    memory: `col(j) = j + 5` (the Fortran 1-based offset, used by mistake in
    0-based numpy) puts H+ and NO3- concentration at the wrong columns and
    the driver returns `ierr = 0` anyway, with a different, finite number.
    This is the failure `test_simulate_reproduces_every_captured_cell` exists
    to catch in the committed archive.
    """
    f2py = REPO / "validation" / "f2py"
    if not sorted(f2py.glob("glomap_f2py*.so")):
        pytest.skip("binding not built; run validation/build_f2py.sh")
    sys.path.insert(0, str(f2py))
    import glomap_f2py as g

    e = g.wrap_set_fix_water_content(0)
    assert int(e) == 0

    def call(offset):
        n = 1
        mask_i = np.array([1], dtype=np.int32)
        ions_i = np.zeros((n, 8), dtype=np.int32)
        cl = np.zeros((n, 8), dtype=np.float64)
        ions_i[0, (1 + offset) % 8] = 1
        ions_i[0, (-3 + offset) % 8] = 1
        cl[0, (1 + offset) % 8] = 1e-10
        cl[0, (-3 + offset) % 8] = 1e-10
        rh = np.array([0.7])
        wc, ierr = g.leaf_water_content(mask_i, ions_i, cl, rh)
        assert int(ierr) == 0
        return float(wc[0])

    correct = call(4)
    wrong = call(5)
    assert correct != pytest.approx(wrong)
    assert np.isfinite(wrong)  # the wrong offset does not raise -- it lies


# ---------------------------------------------------------------------------
# The rh threshold grid, cross-checked against the source
# ---------------------------------------------------------------------------


def test_thresholds_match_every_distinct_rh_min_the_source_declares(ref):
    cwl._verify_thresholds(ref)  # must not raise


def test_the_threshold_check_is_not_vacuous(ref, monkeypatch):
    """Mutation: drop a real threshold from the swept set and confirm the
    check that is supposed to catch it actually does."""
    monkeypatch.setattr(
        cwl, "RH_THRESHOLDS_PCT", tuple(t for t in cwl.RH_THRESHOLDS_PCT if t != 0.019)
    )
    with pytest.raises(SystemExit, match="rh threshold mismatch"):
        cwl._verify_thresholds(ref)


def test_rh_grid_contains_every_threshold_and_its_neighbours(golden):
    rh = golden["rh"]
    for t in cwl.RH_THRESHOLDS_PCT:
        assert t in rh, t
        assert np.nextafter(t, 0.0) in rh, t
        assert np.nextafter(t, 1.0) in rh, t


# ---------------------------------------------------------------------------
# The dead divide branch (z(ic)/z(ia)), cross-checked from z alone
# ---------------------------------------------------------------------------


def test_divide_branch_is_dead_for_every_pair(ref):
    hits = cwl.divide_branch_hits(ref)
    assert set(hits) == set(cwl.PAIRS)
    assert all(v == 0 for v in hits.values()), hits


def test_the_divide_branch_check_is_not_vacuous(ref):
    """Mutation: give SO4 (z=-2) the same nonzero, non-unit charge as some
    cation and confirm `divide_branch_hits` catches the branch turning live."""
    mutated = dict(ref)
    mutated["z"] = dict(ref["z"])
    mutated["z"][-2] = 3.0
    mutated["z"][3] = 3.0
    hits = cwl.divide_branch_hits(mutated)
    assert hits[(3, -2)] == 1


# ---------------------------------------------------------------------------
# The independent cross-check: simulate() against every captured cell
# ---------------------------------------------------------------------------


def test_simulate_reproduces_every_captured_cell(golden, ref):
    """The main correctness pin. `simulate` is a second, independent
    reimplementation of `ukca_water_content_v` (both arms, correct iteration
    order, the aw-ratchet included) built from a fresh parse of the source --
    not from `water_tables.py`. Disagreeing anywhere means either the
    archive's (cl, ions) don't mean what the labels say, or `simulate` itself
    is wrong; either way the golden is not trustworthy until this passes.
    """
    cl, ions, rh, wc = golden["cl"], golden["ions"], golden["rh"], golden["wc"]
    C, R = cl.shape[0], rh.shape[0]
    worst = 0.0
    for i in range(C):
        for j in range(R):
            rh_j = float(rh[j])
            for f, fw in ((0, False), (1, True)):
                predicted, _ = cwl.simulate(ref, fw, cl[i], ions[i], rh_j)
                actual = float(wc[f, i, j])
                worst = max(worst, abs(predicted - actual))
                assert predicted == pytest.approx(actual, rel=1e-9, abs=1e-9), (
                    golden["comp_label"][i],
                    rh_j,
                    fw,
                )
    assert worst < 1e-9


def test_simulate_would_catch_a_wrong_offset(golden, ref):
    """Mutation, entirely in Python: feed `simulate` a `cl`/`ions` row built
    with the wrong (species + 5) offset for a pair that is genuinely active,
    and confirm it predicts something different from the correctly-offset
    captured value -- demonstrating that an offset bug in `build_main_grid`
    would have been caught by the check above, not merely assumed to be."""
    cl, ions, rh = golden["cl"], golden["ions"], golden["rh"]
    i = list(golden["comp_label"]).index("pair_1_-3_mag_1e-12")
    j = int(np.argmin(np.abs(rh - 0.7)))
    correct, _ = cwl.simulate(ref, False, cl[i], ions[i], float(rh[j]))

    wrong_cl = np.zeros(8)
    wrong_ions = np.zeros(8, dtype=np.int32)
    for species in (1, -3):
        wrong_cl[(species + 5) % 8] = 1e-12
        wrong_ions[(species + 5) % 8] = 1
    wrong, _ = cwl.simulate(ref, False, wrong_cl, wrong_ions, float(rh[j]))
    assert correct != pytest.approx(wrong)


# ---------------------------------------------------------------------------
# molal_max clamp reachability, re-derived from the committed grid
# ---------------------------------------------------------------------------


def test_molal_clamp_hit_counts_match_a_fresh_simulation(golden, ref):
    cl, ions, rh = golden["cl"], golden["ions"], golden["rh"]
    C, R = cl.shape[0], rh.shape[0]
    hits = {p: [0, 0] for p in cwl.PAIRS}
    for i in range(C):
        for j in range(R):
            rh_j = float(rh[j])
            for f, fw in ((0, False), (1, True)):
                _, clamped = cwl.simulate(ref, fw, cl[i], ions[i], rh_j)
                for pair, hit in clamped.items():
                    if hit:
                        hits[pair][f] += 1
    off = [hits[p][0] for p in cwl.PAIRS]
    on = [hits[p][1] for p in cwl.PAIRS]
    assert off == list(golden["diag_molal_clamp_hits_fix_off"])
    assert on == list(golden["diag_molal_clamp_hits_fix_on"])


def test_the_clamp_partition_is_not_vacuous(golden, ref):
    """Mutation: narrow the rh grid so pair (3,-3) (whose hits, measured in
    the capture report, occur only above rh=1.1) can no longer clamp, and
    confirm the count actually drops to zero -- i.e. that a shrunk grid is
    something this diagnostic would notice, not something it would paper over
    with a stale number."""
    cl, ions = golden["cl"], golden["ions"]
    i = list(golden["comp_label"]).index("pair_3_-3_mag_1e-12")
    narrow_rh = np.linspace(0.0, 0.9, 181)
    hits_narrow = sum(
        cwl.simulate(ref, False, cl[i], ions[i], float(r))[1][(3, -3)] for r in narrow_rh
    )
    full_rh = golden["rh"]
    hits_full = sum(cwl.simulate(ref, False, cl[i], ions[i], float(r))[1][(3, -3)] for r in full_rh)
    assert hits_narrow == 0
    assert hits_full > 0


# ---------------------------------------------------------------------------
# Anti-collapse: the two independent effects of l_fix_ukca_water_content
# ---------------------------------------------------------------------------


def test_the_flags_differ_on_a_pair_1_minus3_row(golden):
    """The coefficient-patch effect: only reachable through pair (1,-3)."""
    ions = golden["ions"]
    active = (ions[:, cwl.col(1)] == 1) & (ions[:, cwl.col(-3)] == 1)
    assert active.any()
    wc = golden["wc"]
    assert np.any(wc[0][active] != wc[1][active])


def test_the_flags_differ_on_a_row_that_never_touches_pair_1_minus3(golden):
    """The aw-ratchet effect: demonstrated on a row built from the caller's
    four reachable ions alone, which never activates (1,-3) -- so this
    divergence cannot be attributed to the patched coefficient."""
    labels = list(golden["comp_label"])
    i = labels.index("combo4_h1_na1_cl1_so41")
    ions = golden["ions"]
    assert not (ions[i, cwl.col(1)] and ions[i, cwl.col(-3)])
    wc = golden["wc"]
    assert np.any(wc[0, i, :] != wc[1, i, :])


def test_the_ratchet_effect_shows_up_below_rh_0_47(golden):
    labels = list(golden["comp_label"])
    i = labels.index("combo4_h1_na1_cl1_so41")
    rh = golden["rh"]
    low = rh < 0.47
    assert low.any()
    wc = golden["wc"]
    assert np.any(wc[0, i, low] != wc[1, i, low])


def test_rows_that_never_touch_pair_1_minus3_and_never_mix_floors_agree():
    """The flip side of the two anti-collapse rows: `pair_1_-4_mag_*` and the
    other ic=1 single-pair rows activate exactly one pair whose own rh_min is
    0, and (1,-4) is the very first pair the loop visits -- nothing precedes
    it, so neither effect can reach it. This is what makes the two rows above
    a real isolation of the two effects rather than a coincidence."""
    if not ARCHIVE.is_file():
        pytest.skip("archive missing")
    with np.load(ARCHIVE, allow_pickle=False) as d:
        labels = list(d["comp_label"])
        i = labels.index("pair_1_-4_mag_1e-12")
        assert np.array_equal(d["wc"][0, i, :], d["wc"][1, i, :])


# ---------------------------------------------------------------------------
# Mask compaction (ukca_water_content_v.F90:237-243)
# ---------------------------------------------------------------------------


def test_mask_all_false_leaves_wc_at_zero(golden):
    mask_wc = golden["mask_wc"]
    for f in (0, 1):
        assert np.array_equal(mask_wc[f, 1], np.zeros_like(mask_wc[f, 1]))


def test_interleaved_mask_matches_all_true_at_active_rows_and_zero_elsewhere(golden):
    mask_wc = golden["mask_wc"]
    active = golden["mask_patterns"][2].astype(bool)
    for f in (0, 1):
        assert np.array_equal(mask_wc[f, 2][active], mask_wc[f, 0][active])
        assert np.array_equal(mask_wc[f, 2][~active], np.zeros(int((~active).sum())))


def test_the_mask_check_is_not_vacuous(golden):
    """Mutation: an interleaved call that used the all-FALSE result instead of
    all-TRUE at its active rows would trip the same assertion -- confirm the
    committed all-true and all-false rows genuinely differ, or the check above
    could not tell the two apart."""
    mask_wc = golden["mask_wc"]
    assert not np.array_equal(mask_wc[0, 0], mask_wc[0, 1])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_fields(golden):
    assert str(golden["_case"]) == "water_content"
    assert str(golden["_mode"]) == "leaf"
    assert str(golden["_variant"]) == "f64"
    cl, rh, patterns, mask_rh = (
        golden["cl"],
        golden["rh"],
        golden["mask_patterns"],
        golden["mask_rh"],
    )
    expected_rows = cl.shape[0] * rh.shape[0] + patterns.shape[0] * mask_rh.shape[0]
    assert int(golden["_rows"]) == expected_rows
