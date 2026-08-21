"""Task 35e: what `volume_mode.f64.leaf.npz` must contain to be worth having.

**No `fortran` marker anywhere in this file.** The archive is committed, so
every assertion here runs in CI without a toolchain; regenerating it needs
gfortran, reading it does not.

The acceptance criterion this replaces was "shape+finiteness per fixture", which
an all-zeros archive of the right shape satisfies. What is asserted instead is
the set of facts the fixture exists to pin, each one re-derived from the
committed numbers rather than from the capture script's say-so:

* the stratospheric override at `ukca_volume_mode.F90:434-438` is applied **per
  point**, and `pmid == putls` is on the troposphere side of the strict `<`;
* the relative-humidity clamps at `:306-307` are strict, so `nextafter(0.9, 1)`
  is clamped and `nextafter(0.9, 0)` is not;
* `mask_nosol` (`:330`) really was reached, and the rows that reached it take
  `:601-606` and the `ELSE WHERE` at `:631`;
* `nd == num_eps` exactly is on the FALSE side of `:312`;
* `l_fix_ukca_water_content` moves `mdwat` in six setups and cannot in the
  seventh;
* `l_fix_neg_pvol_wat` moves an output only where `pmid < putls`;
* `nbox > 1` is byte-equal to the concatenation of `nbox = 1` calls;
* the deliberate-abort block reached `:481` exactly twice and no other error
  path.

Each test names the mutation that would fail it. Several are checks on the
*data*, so the mutation is a mutation of the golden: those are applied inline to
a copy of the array and the assertion is confirmed to flip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "volume_mode.f64.leaf.npz"
sys.path.insert(0, str(REPO / "validation"))
import capture_volume_mode_leaf as capture  # noqa: E402

CP_SU, CP_BC, CP_OC, CP_CL, CP_DU, CP_SO = range(6)
MMW = 0.0180154  # ukca_constants.F90:60


@pytest.fixture(scope="module")
def d():
    assert GOLDEN.is_file(), f"{GOLDEN.name} missing -- run `make goldens`"
    with np.load(GOLDEN, allow_pickle=False) as data:
        yield {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_the_archive_has_the_shape_the_capture_script_declares():
    """Mutation: change SETUPS or FIX_WATER in the capture script without
    re-capturing -- the leading axes stop matching and this fails."""
    with np.load(GOLDEN, allow_pickle=False) as data:
        nrows = int(data["_rows"])
        ns, nf = len(capture.SETUPS), len(capture.FIX_WATER)
        assert list(data["_setups"]) == list(capture.SETUPS)
        assert list(data["_fix_water"]) == list(capture.FIX_WATER)
        assert data["main_pvol"].shape == (ns, nf, nrows, 8, 6)
        assert data["main_mdwat"].shape == (ns, nf, nrows, 8)
        assert data["main_md_in"].shape == (ns, nrows, 8, 6)
        assert data["tab_component"].shape == (ns, 8, 6)
        assert list(data["_predicates"]) == list(capture.PREDICATES)
        assert float(data["_putls"]) == capture.PUTLS


def test_every_captured_array_is_finite(d):
    """Mutation: let one row divide by a zero denom and keep its output -- an
    inf or a NaN lands here. `rhosol` is `rhotmp/denom` with no guard, so this
    is not hypothetical."""
    for name, arr in d.items():
        if arr.dtype.kind == "f":
            assert np.isfinite(arr).all(), f"{name} is not finite"


def test_the_branch_hit_counts_are_all_nonzero_or_explained(d):
    """Criterion 3. Mutation: delete a dense cluster from one axis; the count
    for its predicate drops to zero. Applied for real against the capture --
    removing T = 180.0 takes `wts_floor_41` to 0 and the capture refuses to
    write."""
    hits = dict(zip(d["_predicates"], d["hits"]))
    expected_zero = set(d["_expected_zero"])
    assert expected_zero == set(capture.EXPECTED_ZERO)
    for name, n in hits.items():
        if name in expected_zero:
            assert n == 0, f"{name} was expected to be unreachable, got {n}"
        else:
            assert n > 0, f"{name} was never reached"


def test_the_deliberately_unreachable_predicates_have_a_reason(d):
    """A predicate listed as expected-zero with no reason is an excuse. Every
    entry of EXPECTED_ZERO must carry a non-empty explanation, and every name
    in it must be a real predicate."""
    for name, reason in capture.EXPECTED_ZERO.items():
        assert name in set(d["_predicates"]), f"{name} is not a predicate"
        assert len(reason) > 20, f"{name} has no real reason"


# ---------------------------------------------------------------------------
# The stratospheric override -- the branch no committed golden had reached
# ---------------------------------------------------------------------------


def test_the_strat_override_is_applied_per_point_not_per_call(d):
    """`:434` is `WHERE (mask .AND. pmid < putls)`, inside ONE call whose pmid
    column is mixed. So within a single record, the rows below putls must carry
    the `wts` water and the rows above it must not.

    Mutation: replace the below-putls mdwat with the above-putls value and the
    first assertion fails; replace the above-putls mdwat with the formula and
    the second does.
    """
    pmid = d["main_pmid"]
    putls = float(d["_putls"])
    strat, trop = pmid < putls, pmid >= putls
    assert strat.any() and trop.any(), "the column is not mixed"

    for si, setup in enumerate(d["_setups"]):
        nd = d["main_nd"][si]
        md = d["main_md_out"][si]
        num_eps = d["tab_num_eps"][si]
        mode = d["tab_mode"][si].astype(bool)
        modesol = d["tab_modesol"][si]
        comp = d["tab_component"][si].astype(bool)
        mm = d["tab_mm"][si]
        for fi in range(len(d["_fix_water"])):
            mdwat = d["main_mdwat"][si, fi]
            wts = d["main_wts"][si, fi]
            for m in range(8):
                if not (mode[m] and modesol[m] == 1 and comp[m, CP_SU]):
                    continue
                mask = nd[:, m] > num_eps[m]
                sel = mask & strat
                if not sel.any():
                    continue
                want = (100.0 / wts[sel] - 1.0) * md[sel, m, CP_SU] * mm[CP_SU] / MMW
                assert np.allclose(mdwat[sel, m], want, rtol=1e-12, atol=0.0), (
                    f"setup {setup} mode {m + 1}: the strat rows do not carry the wts water content"
                )
                other = mask & trop
                if other.any():
                    zsr = (100.0 / wts[other] - 1.0) * md[other, m, CP_SU] * mm[CP_SU] / MMW
                    # The ZSR water and the wts water must not coincide, or the
                    # test above would pass on rows the override never touched.
                    assert not np.allclose(mdwat[other, m], zsr, rtol=1e-8), (
                        f"setup {setup} mode {m + 1}: the tropospheric rows also "
                        "look like the strat formula, so this test proves nothing"
                    )


def test_putls_itself_is_on_the_troposphere_side(d):
    """`pmid < putls` is strict, so 1.5e4 must NOT be overridden while the
    double below it must be. Both are in the grid, at otherwise identical
    (t, s, rh, composition).

    Mutation: change the Fortran to `<=` and re-capture -- the row at putls
    starts matching its lower neighbour and the inequality below flips.
    """
    putls = float(d["_putls"])
    below = np.nextafter(putls, 0.0)
    above = np.nextafter(putls, np.inf)
    pmid, rh, variant = d["main_pmid"], d["main_rh"], d["main_variant"]
    # Block (a): the pmid axis at fixed everything else.
    base = (variant == "typical") & (rh == 0.9) & (d["main_nd_kind"] == "bulk")
    i_at = np.flatnonzero(base & (pmid == putls))
    i_below = np.flatnonzero(base & (pmid == below))
    i_above = np.flatnonzero(base & (pmid == above))
    assert len(i_at) == len(i_below) == len(i_above) == 1

    mdwat = d["main_mdwat"][:, 0]  # (setup, row, mode)
    at, lo, hi = mdwat[:, i_at[0]], mdwat[:, i_below[0]], mdwat[:, i_above[0]]
    assert np.array_equal(at, hi), (
        "putls and the double above it must both be tropospheric -- they are "
        "not byte-equal, so the predicate is not `<`"
    )
    assert not np.array_equal(at, lo), (
        "putls and the double below it came out identical, so the override did "
        "not fire one double below the threshold"
    )
    # And the mutation of the data itself: if `at` had been captured as `lo`,
    # the first assertion above would fail.
    assert not np.array_equal(lo, hi)


def test_t_and_s_reach_an_output_only_below_putls(d):
    """`t`, `pmid` and `s` enter `ukca_volume_mode` only through the
    `ukca_vapour` call at `:287`, and `wts`/`rhosol_strat` are read only inside
    the two `pmid < putls` blocks. So two tropospheric rows differing only in
    `t` must be byte-identical, and two stratospheric ones must not.

    Mutation: make the Fortran use `t` anywhere else and the first half fails.
    """
    pmid, t = d["main_pmid"], d["main_t"]
    strat_rows = np.flatnonzero((pmid == 1.0e4) & (d["main_variant"] == "typical"))
    ts = t[strat_rows]
    assert len(set(ts.tolist())) > 3, "the t axis is not swept below putls"
    mdwat = d["main_mdwat"][0, 0][strat_rows]
    assert len({tuple(r) for r in mdwat}) > 3, "t does not move mdwat below putls"

    # Same rows, tropospheric: only rh varies there, so pick the strat block's
    # control rows at 1.0e5 out of the dedicated l_fix_neg_pvol_wat block, which
    # sweeps t at both pressures.
    sp, st = d["strat_pmid"], d["strat_t"]
    trop = np.flatnonzero(sp == 1.0e5)
    assert len(set(st[trop].tolist())) > 1, "the strat block does not vary t at 1e5"
    out = d["strat_fn0_mdwat"][trop]
    assert len({tuple(r) for r in out}) == 1, (
        "two tropospheric rows differing only in t gave different mdwat, so t "
        "reaches an output outside the pmid < putls blocks"
    )


# ---------------------------------------------------------------------------
# The relative-humidity clamps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bound", "outside", "inside"),
    [
        (0.9, np.nextafter(0.9, 1.0), np.nextafter(0.9, 0.0)),
        (0.1, np.nextafter(0.1, 0.0), np.nextafter(0.1, 1.0)),
    ],
)
def test_the_humidity_clamps_are_strict(d, bound, outside, inside):
    """`:306` is `WHERE (corrh > 0.9) corrh = 0.9` and `:307` the mirror, both
    strict. So the double just outside the bound is clamped onto it and must
    give byte-identical output, and the double just inside must not.

    Mutation: relax either test to `>=` in the Fortran and re-capture -- the
    bound row itself would then be rewritten, which changes nothing, but
    `inside` would still differ; relax to `>= inside` and the second assertion
    flips. The falsifiable half is the second one, and it is the half that
    fails if the grid ever loses its neighbouring doubles.
    """
    pmid = d["main_pmid"]
    base = (d["main_variant"] == "typical") & (pmid == 1.0e5) & (d["main_t"] == 213.0)
    rh = d["main_rh"]

    def one(value):
        idx = np.flatnonzero(base & (rh == value))
        assert len(idx) >= 1, f"no rh = {value!r} row"
        block = d["main_mdwat"][:, :, idx]
        # Several blocks of the grid land on the same (t, pmid, s, rh); they
        # must agree, or "the row at rh" would not be well defined.
        for k in range(1, len(idx)):
            assert np.array_equal(block[:, :, 0], block[:, :, k])
        return block[:, :, 0]

    assert np.array_equal(one(bound), one(outside)), (
        f"rh = {outside!r} was not clamped onto {bound!r}"
    )
    assert not np.array_equal(one(bound), one(inside)), (
        f"rh = {inside!r} gave the same answer as {bound!r}, so the clamp is not "
        "strict or mdwat does not depend on rh at this resolution"
    )


def test_the_humidity_clamps_reach_beyond_the_namelist_range(d):
    """The highest `rel_humid` any shipped namelist sets is 0.90 and
    `glomap_box_config_mod:287` admits [0, 1], so 0.05 and 0.95 are
    configurations the box model would accept and no golden had ever run.

    Mutation: drop them from RH_AXIS -- `check_axes` refuses the capture."""
    rh = d["main_rh"]
    assert (rh > 0.9).any() and (rh < 0.1).any()
    assert 0.95 in rh and 0.05 in rh
    for value, bound in ((0.95, 0.9), (0.05, 0.1)):
        base = (
            (d["main_variant"] == "typical")
            & (d["main_pmid"] == 1.0e5)
            & (d["main_t"] == 213.0)
            & (d["main_nd_kind"] == "bulk")
        )
        a = np.flatnonzero(base & (rh == value))
        b = np.flatnonzero(base & (rh == bound))
        assert len(a) == 1 and len(b) >= 1
        assert np.array_equal(d["main_mdwat"][:, :, a[0]], d["main_mdwat"][:, :, b[0]])


# ---------------------------------------------------------------------------
# mask_nosol, mdsol < 0, and the num_eps tie
# ---------------------------------------------------------------------------


def _masks(d, si, fi=0):
    """(mask, mask_sol, mask_nosol, mask_neg) per mode, from committed data."""
    nd = d["main_nd"][si]
    md = d["main_md_out"][si]
    num_eps = d["tab_num_eps"][si]
    mode = d["tab_mode"][si].astype(bool)
    modesol = d["tab_modesol"][si]
    comp = d["tab_component"][si].astype(bool)
    soluble = d["tab_soluble"][si].astype(bool)
    out = {}
    for m in range(8):
        if not (mode[m] and modesol[m] == 1):
            continue
        mask = nd[:, m] > num_eps[m]
        mdsol = np.zeros(len(mask))
        for c in range(md.shape[2]):
            if comp[m, c] and soluble[c]:
                mdsol = np.where(mask, mdsol + md[:, m, c], mdsol)
        out[m] = (mask, mask & (mdsol > 0), mask & (mdsol == 0), mask & (mdsol < 0))
    return out


def test_mask_nosol_was_reached_and_took_its_branch(d):
    """`:601-606` zeroes the soluble partial volumes and the `ELSE WHERE` at
    `:631` sets `wvol = dvol`, `pvol_wat = 0` and `rhopar = rho_so4`. Reaching
    the mask is not the same as taking the branch; both are asserted.

    Mutation: overwrite one nosol row's `pvol[..., CP_SU]` with a non-zero and
    the first assertion fails -- applied inline below.
    """
    soluble = d["tab_soluble"]
    total = 0
    for si, setup in enumerate(d["_setups"]):
        for m, (_mask, _sol, nosol, _neg) in _masks(d, si).items():
            if not nosol.any():
                continue
            total += int(nosol.sum())
            pvol = d["main_pvol"][si, 0]
            wvol = d["main_wvol"][si, 0]
            dvol = d["main_dvol"][si]
            pvol_wat = d["main_pvol_wat"][si, 0]
            comp = d["tab_component"][si].astype(bool)
            for c in range(pvol.shape[2]):
                if comp[m, c] and soluble[si][c]:
                    assert (pvol[nosol, m, c] == 0.0).all(), (
                        f"setup {setup} mode {m + 1}: :601 did not zero the soluble "
                        f"partial volume of component {c + 1}"
                    )
            assert np.array_equal(wvol[nosol, m], dvol[nosol, m])
            assert (pvol_wat[nosol, m] == 0.0).all()
    assert total > 0, "mask_nosol was never reached, so this test proves nothing"

    # The mutation, applied: a non-zero soluble pvol on a nosol row must fail.
    si = list(d["_setups"]).index(2)
    m, (_a, _b, nosol, _c) = next(iter(_masks(d, si).items()))
    poisoned = d["main_pvol"][si, 0].copy()
    poisoned[np.flatnonzero(nosol)[0], m, CP_SU] = 1.0
    assert not (poisoned[nosol, m, CP_SU] == 0.0).all()


def test_mask_nosol_is_unreachable_exactly_where_it_is_claimed_to_be(d):
    """Setups 1 and 6 cannot reach it however the composition is chosen -- 1
    because every active soluble mode carries only `su` and `cl`, 6 because it
    has no active soluble mode. Both are recorded rather than omitted, so a
    mode-table change that makes them reachable is a failure here.

    Mutation: add an insoluble component to a setup-1 soluble mode and
    re-capture -- `check_hits` refuses the archive.
    """
    unreachable = set(int(v) for v in d["_nosol_unreachable"])
    assert unreachable == set(capture.NOSOL_UNREACHABLE)
    for si, setup in enumerate(d["_setups"]):
        got = sum(int(n.sum()) for _m, (_a, _b, n, _c) in _masks(d, si).items())
        if int(setup) in unreachable:
            assert got == 0, f"setup {setup} reached mask_nosol after all"
        else:
            assert got > 0, f"setup {setup} should reach mask_nosol and did not"


def test_the_unnamed_negative_mdsol_state_takes_neither_branch(d):
    """`mask_sol` is `mdsol > 0` and `mask_nosol` is `mdsol == 0`; `mdsol < 0`
    is neither and the source never names it. Such a row takes the `ELSE WHERE`
    at `:597` (soluble pvol = dvol*mfrac_0) and at `:631`, and must NOT have
    been zeroed by `:601`.

    Mutation: treat `mdsol <= 0` as nosol in a port -- `pvol` for the soluble
    component would go to zero and the last assertion fails.
    """
    seen = nonzero = 0
    for si, setup in enumerate(d["_setups"]):
        for m, (_mask, sol, nosol, neg) in _masks(d, si).items():
            if not neg.any():
                continue
            seen += int(neg.sum())
            assert not (neg & sol).any() and not (neg & nosol).any()
            wvol, dvol = d["main_wvol"][si, 0], d["main_dvol"][si]
            assert np.array_equal(wvol[neg, m], dvol[neg, m]), (
                f"setup {setup} mode {m + 1}: an mdsol < 0 row did not take the :631 ELSE WHERE"
            )
            # :597-599 exactly: the ELSE WHERE writes dvol*mfrac_0, which is
            # what separates this state from mask_nosol, where :601 writes 0.
            pvol = d["main_pvol"][si, 0]
            mfrac = d["tab_mfrac_0"][si]
            comp = d["tab_component"][si].astype(bool)
            soluble = d["tab_soluble"][si].astype(bool)
            for c in range(pvol.shape[2]):
                if comp[m, c] and soluble[c]:
                    assert np.array_equal(pvol[neg, m, c], dvol[neg, m] * mfrac[m, c]), (
                        f"setup {setup} mode {m + 1} cpt {c + 1}: an mdsol < 0 row "
                        "did not get the :598 default partial volume"
                    )
                    nonzero += int((pvol[neg, m, c] != 0.0).sum())
    assert seen > 0
    assert nonzero > 0, (
        "every mdsol < 0 row came out with pvol = 0 for every soluble component, "
        "which is also what :601 does to a nosol row -- the two states would be "
        "indistinguishable in this fixture"
    )


def test_num_eps_itself_is_below_the_mask_threshold(d):
    """`:312` is `nd > num_eps`, strict. The grid carries `num_eps` exactly and
    both neighbouring doubles at otherwise identical inputs, so the tie must
    behave like the double below it and not like the one above.

    Mutation: change the Fortran to `>=` and re-capture -- the tie row starts
    matching `eps_above` instead.
    """
    kinds = d["main_nd_kind"]
    base = (d["main_variant"] == "typical") & (d["main_pmid"] == 1.0e5)

    def one(kind):
        idx = np.flatnonzero(base & (kinds == kind))
        assert len(idx) == 1, kind
        return idx[0]

    exact, below, above = one("eps_exact"), one("eps_below"), one("eps_above")
    mdwat = d["main_mdwat"][:, 0]
    assert np.array_equal(mdwat[:, exact], mdwat[:, below]), (
        "nd = num_eps behaved like nd > num_eps, so :312 is not strict"
    )
    assert not np.array_equal(mdwat[:, exact], mdwat[:, above])
    assert (mdwat[:, exact] == 0.0).all(), ":447 sets mdwat = 0 where the mask is off"


# ---------------------------------------------------------------------------
# The two fidelity flags
# ---------------------------------------------------------------------------


def test_l_fix_ukca_water_content_moves_six_setups_and_cannot_move_the_seventh(d):
    """The flag's live arm is `:271-322`, not the `y(1,-3,6)` patch at `:235`:
    the unfixed path clamps `aw` to `rh_min(ic,ia)` cumulatively down the ion
    pair loop, so Na+/Cl- inherits NH4+/NO3-'s 0.62 floor. Setup 6 never calls
    `ukca_water_content_v` at all, because no active mode of it is soluble.

    Mutation: sweep the flag inside one process (#22, the SAVEd table is a
    one-way latch) and all seven setups collide.
    """
    identical = set(int(v) for v in d["_fix_water_identical"])
    assert identical == set(capture.FIX_WATER_IDENTICAL)
    for si, setup in enumerate(d["_setups"]):
        same = np.array_equal(d["main_mdwat"][si, 0], d["main_mdwat"][si, 1])
        if int(setup) in identical:
            assert same, f"setup {setup} was expected to be flag-invariant"
        else:
            assert not same, f"setup {setup} did not respond to the flag"


def test_l_fix_neg_pvol_wat_moves_an_output_only_below_putls(d):
    """Its whole numerical effect is `MIN(99.0, ...)` on `wts`
    (`ukca_vapour.F90:184` against `:188`), and `wts` is read only at `:436`.
    So the strat block's rows below putls must differ between the two settings
    and its control rows at 1.0e5 must be byte-equal.

    Mutation: capture both settings in one process -- both rows agree and the
    first assertion fails.
    """
    pmid, putls = d["strat_pmid"], float(d["_putls"])
    strat = pmid < putls
    lo, hi = d["strat_fn0_mdwat"], d["strat_fn1_mdwat"]
    assert not np.array_equal(lo[strat], hi[strat]), (
        "the two l_fix_neg_pvol_wat settings gave identical stratospheric water"
    )
    assert np.array_equal(lo[~strat], hi[~strat]), (
        "the flag moved a tropospheric row, so it reaches an output outside the pmid < putls block"
    )
    unclamped = d["strat_fn0_wts"]
    clamped = d["strat_fn1_wts"]
    assert (unclamped > 99.0).any(), "no row exercises the 99% ceiling"
    assert (clamped <= 99.0).all()
    assert np.array_equal(clamped, np.minimum(unclamped, 99.0)), (
        "the clamped wts is not MIN(99, unclamped), so the two arms differ by more than the ceiling"
    )


def test_the_switch_combinations_move_the_densities_they_name(d):
    """`nacl_off` moves rhocomp(cp_cl), `bc_mg_mix` moves rhocomp(cp_bc). Both
    reach `volume_mode` only through `mm_rhocp` and `mm_ovravcrhocp`, i.e.
    through `rhopar` and the insoluble partial volumes.

    Mutation: let `render_namelist`'s switch injection silently no-op -- both
    combos become the default record and the inequality below fails.
    """
    si = list(d["_setups"]).index(capture.COMBO_SETUP)
    fi = list(d["_fix_water"]).index(capture.COMBO_FIX_WATER)
    base_rho = d["tab_rhocomp"][si]
    base_rhopar = d["main_rhopar"][si, fi]
    for combo, cp in (("nacl_off", CP_CL), ("bc_mg_mix", CP_BC)):
        rho = d[f"combo_{combo}_rhocomp"]
        assert rho[cp] != base_rho[cp], f"{combo} did not move rhocomp[{cp}]"
        assert not np.array_equal(d[f"combo_{combo}_rhopar"], base_rhopar), (
            f"{combo} changed a density that rhopar divides by and rhopar did not move"
        )


# ---------------------------------------------------------------------------
# nbox, and the deliberate abort
# ---------------------------------------------------------------------------


def test_nbox_gt_1_equals_the_concatenation_of_nbox_1_calls(d):
    """Byte equality, not a tolerance. `ukca_volume_mode` has no cross-row
    coupling that reaches an output -- `SUM(ierr)` and the `MINVAL` guards gate
    ereports only -- and every vectorised comparison downstream depends on that.

    Mutation: compare each row against its neighbour instead; applied inline.
    """
    for fn in d["_fix_neg_pvol"]:
        fields = ("mdwat", "wvol", "wetdp", "rhopar", "pvol", "pvol_wat", "drydp", "dvol")
        for f in fields:
            many = d[f"strat_fn{fn}_{f}"]
            one = d[f"strat_fn{fn}_nbox1_{f}"]
            assert many.shape == one.shape
            assert np.array_equal(many, one), f"fn={fn} {f} differs at nbox = 1"
    shifted = np.roll(d["strat_fn0_mdwat"], 1, axis=0)
    assert not np.array_equal(d["strat_fn0_mdwat"], shifted), (
        "every strat row is identical, so the equality above would hold however "
        "the rows were paired"
    )


def test_the_deliberate_abort_reached_481_exactly_twice(d):
    """`:481` is `IF (SUM(ierr) > 0)` and `ierr` is built from `denom` alone
    (`:476`), so the `denom2 <= 0` row at `:533` is reported ONLY because the
    `denom <= 0` row opened the gate. Two rows, two reports, and the last
    message is the second row's.

    Mutation: make row 2's `denom` positive -- the counter moves by 0 and the
    capture refuses to write. Applied for real against the capture script.
    """
    assert list(d["abort_counts"]) == [2, 0, 0]
    assert "Demoninator <= 0 for i=2" in str(d["abort_message"])
    assert str(d["abort_routine"]).strip() == "UKCA_VOLUME_MODE"
    denom, denom2 = d["abort_denom"], d["abort_denom2"]
    assert denom[0] > 0.0 >= denom2[0], "row 1 must be the denom2-only failure"
    assert denom2[1] > 0.0 >= denom[1], "row 2 must be the denom-only failure"
    assert list(d["hits_abort"]) == [1, 1]
    # drydiam accepted both rows, which is what makes the denom2 row reachable.
    assert (d["abort_dvol"] > 0.0).all() and (d["abort_drydp"] > 0.0).all()


def test_the_abort_block_is_below_putls_and_at_the_unfixed_flag(d):
    """Both are forced, not incidental: `denom <= 0` with `mask_sol` is
    unreachable in the troposphere (the NaCl water coefficient makes
    `mm_cl + kappa_cl > mm_su`), and the negative `md(cp_su)` it needs makes the
    strat `mdwat` negative, which `l_fix_neg_pvol_wat` would catch at `:884`
    before `:481` was ever reached.

    Mutation: move the block to 1.0e5 -- `denom` comes out positive on both rows
    and the capture's sign assertion fails.
    """
    assert (d["abort_pmid"] < float(d["_putls"])).all()
    assert int(d["abort_fix_neg_pvol"]) == 0
    assert int(d["abort_setup"]) == capture.ABORT_SETUP


# ---------------------------------------------------------------------------
# The claims the module docstring makes about coverage
# ---------------------------------------------------------------------------


def test_the_grid_reaches_every_percent_bin_of_the_rhosol_strat_table(d):
    """`ukca_vapour:233` matches `(NINT(wts/5))*5` against `percent`, twelve
    entries from 40 to 95, and falls through to 1300.0 when nothing matches.
    Both the matches and the fall-through are exercised.

    Mutation: shrink T_AXIS to a single value -- the bin count collapses.
    """
    wts = np.concatenate([d["main_wts"].ravel(), d["strat_fn0_wts"].ravel()])
    rho = np.concatenate([d["main_rhosol_strat"].ravel(), d["strat_fn0_rhosol_strat"].ravel()])
    rounds = set((np.rint(wts / 5) * 5).astype(int).tolist())
    percent = set(range(40, 100, 5))
    assert percent <= rounds, f"never reached {sorted(percent - rounds)}"
    assert (rho == 1300.0).any(), "the fall-through was never taken"
    assert (rho != 1300.0).any(), "the lookup table was never hit"


def test_the_water_vapour_pressure_clamps_are_both_exercised(d):
    """`bh2o = 1.609*s*pmid/p0` is clamped to [2e-8, 2e-6]
    (`ukca_vapour.F90:141-142`), and both ends are reachable from the s axis.

    Mutation: drop 1.0e-8 from S_AXIS -- `bh2o_clamp_low` goes to zero and
    `check_hits` refuses the archive.
    """
    bh2o = 1.609 * d["main_s"] * d["main_pmid"] / 101325.0
    assert (bh2o < 2.0e-8).any() and (bh2o > 2.0e-6).any()
    hits = dict(zip(d["_predicates"], d["hits"]))
    assert hits["bh2o_clamp_low"] > 0 and hits["bh2o_clamp_high"] > 0
