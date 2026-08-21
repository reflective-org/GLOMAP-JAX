"""Task 35b: the `ukca_vapour` leaf sweep, and what it settles.

No `fortran` marker: the archive is committed, so all of this runs in CI. What
it checks is not "the numbers are finite" -- an all-zeros archive of the right
shape passes that, which is why the task's stated acceptance criterion was
replaced. It checks the four things the capture exists to establish, each of
which fails loudly if a later grid edit quietly drops it:

1. **The flag is real, and it reaches exactly one output.** `wts` differs
   between the two `l_fix_neg_pvol_wat` settings on 1,819 of 5,797 rows, and
   `rhosol_strat` is byte-identical on all of them. The second half is the
   provable one -- `:184` gives 99 where `:188` gives more, and
   `(NINT(wts/5))*5` sends both to 100 or above while `percent` stops at 95 --
   so it is asserted as an expected collision rather than left unstated. A
   both-settings test written against `rhosol_strat` could not fail;
   `docs/fidelity.md` invited exactly that one.

2. **Every branch is reached.** The archive carries a hit count per predicate,
   and every one is asserted non-zero except the two that are structurally
   unreachable at their flag setting.

3. **`rp` is dead.** Four values, one of them 0.0 -- which makes `:198` a
   division by zero -- give byte-identical outputs. If that ever fails, the
   analysis behind task 38's scope is wrong.

4. **The routine is row-independent.** One `nbox = 8` call equals eight
   `nbox = 1` calls, byte for byte, across the two whole-array sections at
   `:194-199`.

And one finding that is not in the plan: at `t = 15732.0/51.81`, `b` is
**exactly** zero, `d` is exactly `a*a`, and `xsb` is `0/0`. The plan recorded
`xsb = 0.849354` there, which is the limit, not the value. What the compiled
routine returns is `wts = 41.0` -- gfortran's `MAX(41.0, NaN)` keeping the
floor -- and hence a *table* density of 1293.28, not the 1300.0 fall-through.
`test_the_b_zero_pole_returns_the_floor_not_a_nan` pins it, because a port
using `jnp.maximum`, which propagates NaN, gets a different answer.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "vapour.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_vapour_leaf as cv  # noqa: E402

from conftest import assert_matches_reference, on_capture_platform  # noqa: E402

#: Enough to absorb a one-ulp `log` on a host whose libm is not the capture
#: platform's, and far too little to hide a moved grid: the nearest two roots
#: in the archive are 0.48 K apart, which is 1.7e13 ulps.
ROOT_ULP = 4


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_vapour_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def records(sweep):
    """The archive back in the shape `capture_vapour_leaf`'s guards take.

    Testing the guards against the real capture rather than against invented
    data: a guard that would reject the archive this repo has is the guard
    that is wrong.
    """
    out = {}
    for k, flag in enumerate(sweep["flag_values"].tolist()):
        out[int(flag)] = {
            "wts": sweep["wts"][k].tolist(),
            "rhosol_strat": sweep["rhosol_strat"][k].tolist(),
            "probe_wts": sweep["rp_probe_wts"][k].tolist(),
            "probe_rhosol": sweep["rp_probe_rhosol"][k].tolist(),
            "nbox8_wts": sweep["nbox8_wts"][k].tolist(),
            "nbox1_wts": sweep["nbox1_wts"][k].tolist(),
            "nbox8_rhosol": sweep["nbox8_rhosol"][k].tolist(),
            "nbox1_rhosol": sweep["nbox1_rhosol"][k].tolist(),
            "tie_y": sweep["tie_y"].tolist(),
            "flags": [1, int(flag), cv.SETUP],
        }
    return out


@pytest.fixture(scope="module")
def inputs(sweep):
    return {k: sweep[k] for k in ("t", "pmid", "s")}


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_the_archive_holds_one_input_row_per_output_column(sweep):
    n = int(sweep["_rows"])
    assert n == len(sweep["t"]) == len(sweep["pmid"]) == len(sweep["s"]) == len(sweep["block"])
    for name in ("wts", "rhosol_strat"):
        assert sweep[name].shape == (2, n), name
    assert sweep["flag_values"].tolist() == [0, 1]
    assert str(sweep["_case"]) == "vapour"
    assert str(sweep["_mode"]) == "leaf"
    assert str(sweep["_variant"]) == "f64"


def test_the_swept_inputs_span_the_physical_range(sweep):
    """The coarse axis is 150 to 340 K plus the K&L reference at 360; pressure
    covers four decades and reaches `putls`; humidity reaches zero and below."""
    t, pmid, s = sweep["t"], sweep["pmid"], sweep["s"]
    assert t.min() == 150.0 and t.max() == 360.0
    assert (t > 0).all(), "1/t at :143 and LOG(t) at :148 need a positive temperature"
    assert pmid.min() == 100.0 and pmid.max() == 1.05e5
    assert (pmid == 1.5e4).any(), "no row sits on putls (ukca_volume_mode.F90:258)"
    assert s.min() < 0.0 and (s == 0.0).any()
    assert np.isfinite(sweep["wts"]).all() and np.isfinite(sweep["rhosol_strat"]).all()


def test_the_pressure_and_humidity_axes_are_exactly_reproducible(sweep):
    """These abscissae contain no libm call -- decimal literals, divisions and
    `nextafter` only -- so unlike the temperature roots they must come back
    bit-identical on any machine. `np.logspace` here would not."""
    built = set()
    for pmid in cv.pmid_grid():
        for s in cv.s_grid(pmid):
            built.add((float(pmid), float(s)))
    swept = {
        (float(p), float(x))
        for p, x, b in zip(sweep["pmid"], sweep["s"], sweep["block"])
        if str(b).startswith("ps_")
    }
    assert swept == built


# --------------------------------------------------------------------------
# 1. The flag
# --------------------------------------------------------------------------


def test_the_two_flag_settings_are_not_the_same_record(sweep):
    """The failure this repo has already shipped once is a golden holding one
    configuration twice over, byte-equal and green."""
    w0, w1 = sweep["wts"]
    assert w0.tobytes() != w1.tobytes()
    assert (w0 != w1).sum() > 0


def test_wts_differs_exactly_where_the_cap_binds(sweep):
    """`:184` is `MIN(99, MAX(41, ws*100))` and `:188` is `MAX(41, ws*100)`, so
    the two arms agree everywhere except where `ws*100` exceeds 99 -- and there
    the capped arm is exactly 99 and the other is strictly above it."""
    w0, w1 = sweep["wts"]
    differs = w0 != w1
    assert differs.any()
    assert (w1[differs] == 99.0).all()
    assert (w0[differs] > 99.0).all()
    assert (w0[~differs] == w1[~differs]).all()
    assert differs[sweep["t"] > 310.54].any(), "the bmaxatm arm of the ws=99 cluster is gone"


def test_rhosol_strat_is_byte_identical_across_the_flag(sweep):
    """The expected collision, and the reason `docs/fidelity.md` was wrong.

    Both arms send `wts >= 99` to `round >= 100`, `percent` stops at 95
    (`ukca_vapour.F90:90`), so both fall through to the 1300.0 of `:223`. If
    this ever fails it is a finding about the Fortran, not a tolerance to
    loosen -- and it is the assertion that makes the both-settings test of
    `wts` above worth writing.
    """
    r0, r1 = sweep["rhosol_strat"]
    assert r0.tobytes() == r1.tobytes()
    w0, w1 = sweep["wts"]
    capped = w0 != w1
    assert (r0[capped] == 1300.0).all(), "a capped row reached the lookup table"


def test_the_density_is_the_documented_table_lookup(sweep):
    """`rhosol_strat` reconstructed from `wts` and the literals at `:90-100`.

    Exact, on every platform: the lookup is an integer comparison and one
    multiply-add, with nothing for a libm to disagree about. It is what proves
    the archive's densities are the Martin et al. table and not, say, the
    previous row's.
    """
    for k in (0, 1):
        expected = cv.rhosol_of(sweep["t"], sweep["wts"][k])
        assert_matches_reference(
            sweep["rhosol_strat"][k], expected, f"rhosol_strat, flag {k}", ulp=0
        )
    assert (sweep["rhosol_strat"] == 1300.0).any(), "no row falls through to :223"
    assert (sweep["rhosol_strat"] != 1300.0).any(), "no row reaches the lookup table"


def test_the_transcription_reproduces_the_fortran(sweep, inputs):
    """The capture's numpy transcription of `:136-189`, against the driver.

    This is what licenses the branch-hit counts: `bh2o`, `d` and `xsb` never
    leave the routine, so they can only be counted off a re-implementation, and
    an unchecked re-implementation is a guess.
    """
    bh2o = cv.bh2o_of(inputs["pmid"], inputs["s"])
    for k, flag in enumerate((0, 1)):
        assert_matches_reference(
            cv.wts_of(inputs["t"], bh2o, flag), sweep["wts"][k], f"wts, flag {flag}"
        )


# --------------------------------------------------------------------------
# 2. The branches
# --------------------------------------------------------------------------


def test_the_branch_names_are_the_ones_the_capture_declares(sweep):
    """A renamed or dropped predicate would otherwise leave the counts below
    asserting something other than what they say."""
    assert list(sweep["branch_names"]) == list(cv.BRANCH_NAMES)
    assert sweep["branch_hits"].shape == (2, len(cv.BRANCH_NAMES))


def test_every_branch_is_reached_or_declared_unreachable(sweep):
    for k, flag in enumerate((0, 1)):
        for j, name in enumerate(sweep["branch_names"]):
            count = int(sweep["branch_hits"][k, j])
            expected_zero = bool(sweep["branch_expected_zero"][k, j])
            assert expected_zero == ((flag, str(name)) in cv.EXPECTED_ZERO), name
            if expected_zero:
                assert count == 0, f"flag {flag}: {name} fired {count} times but cannot"
            else:
                assert count > 0, f"flag {flag}: nothing in the grid reaches {name}"


def test_the_recorded_hit_counts_are_what_the_grid_actually_produces(sweep, inputs):
    """The counts are committed data, so they can rot. Recomputing them from
    the committed inputs is what stops a grid edit from moving the grid and
    leaving the old counts in place.

    Exact on the capture platform. Elsewhere a one-ulp `LOG` can move the
    innermost point of a cluster across its own edge, which is a count of one,
    so off-platform the tolerance is two per predicate -- still four orders
    below the smallest count in the archive.
    """
    slack = 0 if on_capture_platform() else 2
    for k, flag in enumerate((0, 1)):
        got = cv.branch_hits(inputs["t"], inputs["pmid"], inputs["s"], sweep["wts"][k])
        gap = np.abs(got - sweep["branch_hits"][k])
        worst = int(gap.max())
        assert worst <= slack, (
            f"flag {flag}: {int((gap > slack).sum())} predicates moved, worst by {worst} "
            f"({[str(n) for n in sweep['branch_names'][gap > slack]]})"
        )


def test_both_edges_of_the_d_clamp_are_covered(sweep):
    """`IF (d < 0.0)` has two roots, not one. The plan's fixture list had only
    the cold one; above the hot root `2b > 0`, `xsb = -a/(2b) > 1` and `msb`
    changes sign, with `:176` never firing -- a distinct branch state."""
    names = list(sweep["branch_names"])
    for name in ("d_negative_cold", "d_negative_hot", "xsb_above_one"):
        assert int(sweep["branch_hits"][0, names.index(name)]) > 0, name


# --------------------------------------------------------------------------
# 3 and 4. rp, and vectorisation
# --------------------------------------------------------------------------


def test_rp_does_not_reach_either_output(sweep):
    """`rp` feeds `kelvin` at `:198` and `muh2so4`/`ph2so4` at `:203-216`, none
    of which is an INTENT(OUT). Byte equality across four values, one of them
    0.0 -- a division by zero at `:198`, which could not come back unchanged if
    the chain were live."""
    assert 0.0 in set(sweep["rp_probe_values"].tolist())
    for k in (0, 1):
        for field in ("rp_probe_wts", "rp_probe_rhosol"):
            block = sweep[field][k]
            for j in range(1, block.shape[0]):
                assert block[j].tobytes() == block[0].tobytes(), (
                    f"{field}: rp = {sweep['rp_probe_values'][j]!r} moved the output"
                )
        assert np.isfinite(sweep["rp_probe_wts"][k]).all()


def test_the_nbox_block_matches_eight_single_row_calls(sweep):
    """Three loops with two whole-array sections between them (`:194-199`). A
    single-row fixture cannot detect a mis-vectorised port; this can."""
    assert len(sweep["nbox_rows"]) == 8
    for k in (0, 1):
        for field in ("wts", "rhosol"):
            many, one = sweep[f"nbox8_{field}"][k], sweep[f"nbox1_{field}"][k]
            assert many.tobytes() == one.tobytes(), field


def test_the_nbox_rows_span_different_branch_states(sweep):
    """Eight rows that all sat in the same branch would compare a vectorised
    call against itself."""
    rows = sweep["nbox_rows"]
    assert len(set(sweep["wts"][0][rows].tolist())) > 1
    assert len(set(rows.tolist())) == 8


# --------------------------------------------------------------------------
# The NINT tie idiom, and the two poles
# --------------------------------------------------------------------------


def test_the_tie_block_rounds_half_away_from_zero(sweep):
    """`(NINT(wts/5))*5` at `:226` indexes a table, so a tie going the other way
    picks a different row rather than a nearby number. numpy and `jnp.round`
    round half to EVEN and would give 40 at 42.5; Fortran gives 45."""
    x, y = sweep["tie_x"], sweep["tie_y"]
    exact = np.array([v in set(cv.TIE_WTS) for v in x.tolist()])
    assert exact.sum() == len(cv.TIE_WTS)
    assert_matches_reference(y[exact], np.array(cv.TIE_WTS) + 2.5, "NINT ties", ulp=0)
    # numpy's half-to-even disagrees on exactly the odd multiples of 5.
    half_to_even = np.round(x[exact] / 5.0) * 5.0
    assert (half_to_even != y[exact]).sum() == 6
    # And the neighbours either side must go the way their side says.
    assert_matches_reference(y, cv.nint(x / 5.0) * 5.0, "vapour_round", ulp=0)


def test_the_b_zero_pole_returns_the_floor_not_a_nan(sweep):
    """`b = ks3 + ks4*(1/t)` is exactly 0 at `t = 15732.0/51.81`, so `:175` is
    `0/0` and `ws` is NaN -- and the compiled routine returns `wts = 41.0`,
    because gfortran's `MAX(41.0, NaN)` keeps the floor. The density is then a
    *table* value, not the 1300.0 fall-through.

    A port whose maximum propagates NaN (`jnp.maximum` does) gets NaN here.
    That is the whole reason this row is in the grid, and the plan's
    `xsb = 0.849354` for this point is the limit rather than the value.
    """
    t_pole = float(sweep["t_b_zero"])
    assert t_pole == 15732.0 / 51.81
    lo, at, hi = sweep["b_residual"]
    assert at == 0.0
    assert lo < 0.0 < hi, "one ulp either side of the pole b must change sign"

    rows = np.nonzero(sweep["t"] == t_pole)[0]
    assert len(rows) == 3, "one pole row per atmosphere"
    assert (sweep["wts"][:, rows] == 41.0).all()
    assert (sweep["rhosol_strat"][:, rows] != 1300.0).all()
    t_diff = 253.0 - t_pole
    assert_matches_reference(
        sweep["rhosol_strat"][0, rows],
        np.full(3, cv.DATA253[0] + cv.K_DIFF[0] * t_diff),
        "rhosol_strat at the b = 0 pole",
        ulp=0,
    )
    names = list(sweep["branch_names"])
    assert int(sweep["branch_hits"][0, names.index("xsb_is_nan")]) == 3


def test_the_xsb_one_pole_is_hit_and_also_returns_the_floor(sweep, inputs):
    """The plan said to straddle `xsb = 1` and "do not expect to hit it".

    It is hit, on six rows, and not by luck: near that root `xsb - 1` is a
    cancellation of order 1e-15, so `xsb` moves in plateaus about twelve ulps
    wide and one of them is the exact value 1.0, two consecutive doubles
    across. Bisecting `xsb > 1.0` lands on the plateau. There `msb` at `:178`
    is `+inf`, `ws` at `:179` is `inf/inf`, and `MAX(41.0, NaN)` again keeps
    the floor -- so these rows are the second route to the same NaN semantics
    a port has to reproduce.
    """
    names = list(sweep["branch_names"])
    assert int(sweep["branch_hits"][0, names.index("xsb_exactly_one")]) == 6
    assert int(sweep["branch_hits"][0, names.index("ws_is_nan")]) == 9

    it = cv.intermediates(inputs["t"], cv.bh2o_of(inputs["pmid"], inputs["s"]))
    rows = np.nonzero(it["xsb"] == 1.0)[0]
    if not on_capture_platform() and len(rows) != 6:
        pytest.skip(f"a different libm puts {len(rows)} rows on the pole, not six")
    assert np.isinf(it["msb"][rows]).all()
    assert (sweep["wts"][:, rows] == 41.0).all(), "MAX(41.0, NaN) no longer keeps the floor"
    expected = cv.rhosol_of(inputs["t"][rows], sweep["wts"][0][rows])
    assert_matches_reference(sweep["rhosol_strat"][0][rows], expected, "rhosol at xsb=1", ulp=0)


def test_every_root_bracket_is_two_adjacent_doubles(sweep):
    """A cluster centred on a transcribed decimal drifts off its edge by a ulp
    and stops testing anything. These are re-derived, and the archive records
    the bracket so the claim is checkable without re-running the capture."""
    lo, hi = sweep["root_lo"], sweep["root_hi"]
    assert len(lo) == 54, "three atmospheres times eighteen branch edges"
    assert (np.nextafter(lo, hi) == hi).all()
    assert (sweep["root_residual"][:, 0] != sweep["root_residual"][:, 1]).all()
    for value in np.concatenate([lo, hi]):
        assert (sweep["t"] == value).any(), f"{value!r} is a recorded root and is not swept"


def test_the_roots_are_reproduced_by_re_deriving_them(sweep):
    """The bisection, run again now, must land on the recorded brackets.

    Bit-identical on the capture platform and within a few ulps elsewhere,
    because the predicate contains `LOG`. Anything larger is a moved grid: the
    closest two roots in the archive are 0.48 K apart.
    """
    recorded = dict(zip([str(v) for v in sweep["root_labels"]], sweep["root_lo"]))
    for atmos, (pmid, s) in cv.atmospheres().items():
        bh2o = float(cv.bh2o_of(np.array(pmid), np.array(s)))
        for name, (lo, _hi) in cv.roots(bh2o).items():
            key = f"{atmos}/{name}"
            assert key in recorded, key
            assert_matches_reference([lo], [recorded[key]], key, ulp=ROOT_ULP)


# --------------------------------------------------------------------------
# The guards, and the mutations that must red them
# --------------------------------------------------------------------------


def test_the_guards_accept_the_committed_capture(records, inputs, sweep):
    cv.check_records(inputs, records)
    cv.check_branches(sweep["branch_hits"])
    cv.check_transcription(inputs, records)


def test_a_capture_that_ran_one_flag_setting_twice_is_refused(records, inputs):
    """One subprocess per flag exists because a flag swept in one process
    compares a configuration against itself and passes."""
    collapsed = {0: records[0], 1: dict(records[0])}
    with pytest.raises(SystemExit, match="collided"):
        cv.check_records(inputs, collapsed)


def test_a_density_that_moved_with_the_flag_is_refused(records, inputs):
    """The expected collision is required, not merely tolerated: losing it
    means the fall-through this archive documents has stopped happening."""
    moved = dict(records)
    bent = list(records[1]["rhosol_strat"])
    bent[0] = bent[0] + 1.0
    moved[1] = dict(records[1], rhosol_strat=bent)
    with pytest.raises(SystemExit, match=r"collided as \[none\]"):
        cv.check_records(inputs, moved)


def test_a_flag_that_moved_a_row_the_cap_does_not_reach_is_refused(records, inputs):
    """`:184` and `:188` differ where `ws*100 > 99` and nowhere else. A capture
    in which they differ somewhere else is measuring something other than the
    flag."""
    bent = list(records[1]["wts"])
    bent[0] = bent[0] + 1.0
    with pytest.raises(SystemExit, match="ws\\*100 > 99"):
        cv.check_records(inputs, {0: records[0], 1: dict(records[1], wts=bent)})


def test_an_rp_that_reached_the_output_is_refused(records, inputs):
    probe = [list(row) for row in records[0]["probe_wts"]]
    probe[2][0] = probe[2][0] + 1.0
    with pytest.raises(SystemExit, match="rp = "):
        cv.check_records(inputs, {0: dict(records[0], probe_wts=probe), 1: records[1]})


def test_a_mis_vectorised_nbox_block_is_refused(records, inputs):
    bent = list(records[0]["nbox1_wts"])
    bent[3] = bent[3] + 1.0
    with pytest.raises(SystemExit, match="nbox=8"):
        cv.check_records(inputs, {0: dict(records[0], nbox1_wts=bent), 1: records[1]})


def test_a_transcription_that_drifted_from_the_fortran_is_refused(records, inputs):
    bent = list(records[0]["wts"])
    bent[17] = bent[17] + 1.0
    with pytest.raises(SystemExit, match="transcription disagrees"):
        cv.check_transcription(inputs, {0: dict(records[0], wts=bent)})


def test_a_branch_that_no_row_reaches_is_refused(sweep):
    """The mutation named in the task: delete a dense cluster from the grid and
    the count for its predicate goes to zero. The capture must refuse to write
    rather than produce an archive that covers one decision fewer."""
    hits = sweep["branch_hits"].copy()
    hits[0, list(sweep["branch_names"]).index("d_negative_hot")] = 0
    with pytest.raises(SystemExit, match="d_negative_hot"):
        cv.check_branches(hits)


def test_a_branch_that_fired_where_it_cannot_is_refused(sweep):
    hits = sweep["branch_hits"].copy()
    hits[0, list(sweep["branch_names"]).index("wts_cap_99")] = 1
    with pytest.raises(SystemExit, match="no ceiling"):
        cv.check_branches(hits)


def test_a_root_that_is_not_on_an_edge_is_refused(monkeypatch):
    """`roots` requires exactly one sign change per predicate. Widen the pole
    guard until it swallows the `ws*100 = 97.5` edge at 305.82 K and the search
    finds none -- which must raise, not return a bracket that is not one.

    The same guard catches the other direction: without the upper limit at the
    hot `d = 0` root, `xsb` falls back through 1 and `ws*100` back through 99,
    97.5 and 92.5 somewhere around 350 to 380 K, and each of those predicates
    would have two roots rather than one.
    """
    monkeypatch.setattr(cv, "POLE_GUARD", 40.0)
    with pytest.raises(SystemExit, match="changes sign"):
        cv.roots(cv.BMAXATM)
