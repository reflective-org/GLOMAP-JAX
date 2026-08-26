"""Task 46: the `ukca_cond_coff_v` leaf sweep, and what it settles.

No `fortran` marker: the archive is committed, so all of this runs in CI.

What it checks is not that the numbers are finite. An archive of the right
shape full of plausible garbage passes that, and this repository has written
nine tests that could not fail across two reviews. What it checks is the one
thing the capture exists to establish and a trajectory fixture cannot: **which
argument reaches which output, at each of the four switch settings**, three of
which no shipped namelist has ever run.

The map, from `ukca_cond_coff_v.F90:171-181`:

* `rhoa` and `airdm3` are read at `idcmfp = 1` and not at 2
* `t` and `pmid` are read at `idcmfp = 2` and not at 1
* `dmol` is read at `idcmfp = 1` and not at 2
* `difvol` is read at `idcmfp = 2` and not at 1
* `tsqrt` and `rp` are read at both

Every "not" above is asserted as a byte equality between two calls or two rows
that differ in that argument alone, and every "is" as a byte *inequality* --
because only the second half can fail when a call never reached the Fortran.

Two findings are pinned here rather than left in prose:

1. **`se` is inert at the value the model runs.** `ukca_conden.F90:235-237`
   sets `se_sol` and `se_ins` both to `1.0`, so `1.0/se - 1.0` at `:210` is
   exactly zero and the Fuchs-Sutugin interfacial correction `akn` is exactly
   `1.0` on every row of every validated run. The routine's own header
   (`:60`) says `se` is `0.3`. Both are swept and the archive shows they
   differ, so "the header is wrong" is re-derivable from committed data.

2. **`cc/sinkarr` is `term6*dcoff_cp*1e-6` on both branches**, carrying no
   `rp`, no `denom`, no `fkn` and no `akn`. That is an algebraic identity of
   the source, so it holds along the whole `rp` axis, and it is the cheapest
   check that the two outputs were not computed independently.

The last test is the one that makes the rest mean anything: it feeds mutated
copies of the committed archive back through the capture's own guards and
requires each mutation to be rejected. A guard that passes a corrupted golden
guards nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "cond_coff.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_cond_coff_leaf as cap  # noqa: E402


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_cond_coff_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def calls(sweep):
    """The call table as the capture built it, rebuilt from the archive.

    Read back from the golden rather than recomputed from `build_calls()` so a
    grid edit that changes the call order without regenerating the archive
    fails here instead of silently comparing the wrong pairs.
    """
    return [
        {
            "name": str(sweep["call_name"][i]),
            "ifuchs": int(sweep["call_ifuchs"][i]),
            "idcmfp": int(sweep["call_idcmfp"][i]),
            "mmcg": float(sweep["call_mmcg"][i]),
            "se": float(sweep["call_se"][i]),
            "dmol": float(sweep["call_dmol"][i]),
            "difvol": float(sweep["call_difvol"][i]),
            "live": int(sweep["call_live"][i]),
        }
        for i in range(sweep["call_name"].size)
    ]


@pytest.fixture(scope="module")
def grid(sweep):
    return {k: sweep[k] for k in ("rp", "t", "tsqrt", "airdm3", "rhoa", "pmid", "mask", "block")}


def test_the_archive_matches_the_grid_the_capture_script_builds(sweep, calls):
    """The committed golden and the live capture script describe the same sweep.

    If someone widens an axis and does not re-capture, every byte-equality test
    below would compare rows that no longer mean what their block says.
    """
    built = cap.build_grid()
    for key in ("rp", "t", "tsqrt", "airdm3", "rhoa", "pmid"):
        np.testing.assert_array_equal(sweep[key], built[key], err_msg=key)
    np.testing.assert_array_equal(sweep["mask"], built["mask"])
    np.testing.assert_array_equal(sweep["block"], built["block"])
    assert [c["name"] for c in calls] == [c["name"] for c in cap.build_calls()]
    assert [(c["ifuchs"], c["idcmfp"]) for c in calls] == [
        (c["ifuchs"], c["idcmfp"]) for c in cap.build_calls()
    ]


def test_all_four_switch_settings_are_present(calls):
    """Three of these have never run in any namelist, which is the whole point
    of the sweep: `ifuchs` and `idcmfp` are 1 in all five shipped and added
    namelists."""
    assert {(c["ifuchs"], c["idcmfp"]) for c in calls} == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_both_live_condensables_are_swept(calls):
    """`mmcg`/`difvol` are not free parameters: `ukca_conden.F90:292` and
    `:324-328` fix them per condensable. Exactly two combinations are
    reachable, and both must be in the archive."""
    live = {(c["mmcg"], c["difvol"]) for c in calls if c["live"]}
    assert live == {(0.098, 51.96), (0.15, 204.14)}


def test_the_mask_is_a_select_not_a_multiply(sweep, grid):
    """Masked-off rows carry inf, NaN and 1e308. `WHERE` never evaluates them,
    so both outputs must be exactly the 0.0 the routine opens with. A port that
    writes `mask * term` gets `0.0 * inf = NaN` here."""
    off = grid["block"] == cap.BLOCKS.index("masked_off")
    assert off.sum() == 7
    # The poison has to actually be poisonous, or this test is decoration.
    poisoned = sweep["rp"][off]
    assert np.isnan(poisoned).any() and np.isinf(poisoned).any() and (poisoned == 0.0).any()
    for key in ("cc", "sinkarr"):
        assert np.all(sweep[key][:, :, off] == 0.0), f"{key} leaked through the mask"


def test_every_live_row_is_positive_and_finite(sweep, grid):
    on = grid["mask"] == 1
    for key in ("cc", "sinkarr"):
        vals = sweep[key][:, :, on]
        assert np.all(np.isfinite(vals)), key
        assert np.all(vals > 0.0), key


def test_the_routine_is_setup_independent(sweep):
    """Measured, not argued. `ukca_cond_coff_v` takes no `glomap_variables`
    and reads no per-setup table -- but so did `coag_mode`, which was still
    captured under every setup before that was called a fact."""
    assert sweep["setups"].tolist() == [1, 4]
    np.testing.assert_array_equal(sweep["cc"][0], sweep["cc"][1])
    np.testing.assert_array_equal(sweep["sinkarr"][0], sweep["sinkarr"][1])


@pytest.mark.parametrize(("arg", "other", "inert_idcmfp"), cap.INERT_AT)
@pytest.mark.parametrize("ifuchs", (1, 2))
@pytest.mark.parametrize("idcmfp", (1, 2))
def test_an_argument_is_read_at_one_idcmfp_and_not_the_other(
    sweep, calls, arg, other, inert_idcmfp, ifuchs, idcmfp
):
    """`difvol` reaches `term8` and `term8` is read only at `idcmfp = 2`;
    `dmol` reaches `term2`/`term3` and those are read only at `idcmfp = 1`.
    Both halves are asserted: inert must change nothing, live must change
    something."""
    a, b = cap._pair(calls, "h2so4", other, ifuchs, idcmfp)
    same = np.array_equal(sweep["cc"][0, a], sweep["cc"][0, b])
    assert same == (idcmfp == inert_idcmfp), (
        f"{arg} at ifuchs={ifuchs}, idcmfp={idcmfp} "
        f"{'did not move' if same else 'moved'} the answer"
    )


@pytest.mark.parametrize(("block", "const_idcmfp"), cap.CONSTANT_BLOCK_AT)
def test_a_block_is_constant_exactly_where_its_argument_is_unread(
    sweep, grid, calls, block, const_idcmfp
):
    """`airdm3` and `rhoa` reach `mfp_cp`/`dcoff_cp` only at `idcmfp = 1`;
    `pmid` reaches `dcoff_cp` only at `idcmfp = 2`. So each axis is flat at
    exactly one setting and not at the other."""
    rows = grid["block"] == cap.BLOCKS.index(block)
    assert rows.sum() > 1
    for i, c in enumerate(calls):
        vals = sweep["cc"][0, i, rows]
        constant = bool(np.all(vals == vals[0]))
        assert constant == (c["idcmfp"] == const_idcmfp), cap.call_label(c)


def test_t_is_unread_at_idcmfp_1_even_when_it_is_absurd(sweep, grid, calls):
    """`tsqrt` and `t` are separate arguments and the callee never checks that
    they agree. The decoupled block carries `tsqrt = SQRT(t_b)` with
    `t = 999.0`; at `idcmfp = 1` those rows must be byte-equal to the
    consistent `t = t_b` rows, which proves `t` is not read rather than merely
    not influential on this grid."""
    t_rows = grid["block"] == cap.BLOCKS.index("t")
    d_rows = grid["block"] == cap.BLOCKS.index("tsqrt_decoupled")
    assert np.array_equal(sweep["tsqrt"][t_rows], sweep["tsqrt"][d_rows])
    assert np.all(sweep["t"][d_rows] == cap.DECOUPLED_T)
    for i, c in enumerate(calls):
        same = np.array_equal(sweep["cc"][0, i, t_rows], sweep["cc"][0, i, d_rows])
        assert same == (c["idcmfp"] == 1), cap.call_label(c)


@pytest.mark.parametrize("ifuchs", (1, 2))
@pytest.mark.parametrize("idcmfp", (1, 2))
def test_the_headers_sticking_efficiency_is_not_the_one_the_model_runs(
    sweep, calls, ifuchs, idcmfp
):
    """`:60` documents `se = 0.3`; `ukca_conden.F90:235-237` runs `1.0`. If
    those gave the same answer the discrepancy would be cosmetic. They do
    not."""
    a, b = cap._pair(calls, "h2so4", "h2so4_se_header", ifuchs, idcmfp)
    assert not np.array_equal(sweep["cc"][0, a], sweep["cc"][0, b])


def test_the_interfacial_correction_is_rp_dependent_only_away_from_se_one(sweep, calls):
    """The sharper half of the same finding.

    At `se = 1.0`, `1.0/se - 1.0` is exactly `0.0`, so `akn` is exactly `1.0`
    and `cc = term6*dcoff*rp*fkn` at `ifuchs = 2`. At `se = 0.3` it is not, and
    `akn` depends on `kn = mfp_cp/rp` while every other factor in the product
    is either linear in `rp` or independent of it. So the ratio between the two
    answers must *vary along the rp axis* -- a constant ratio would mean `se`
    had entered as a scale factor, which is what a port that folds `akn` into
    the prefactor would produce.
    """
    a, b = cap._pair(calls, "h2so4", "h2so4_se_header", 2, 1)
    rows = np.flatnonzero(sweep["block"] == cap.BLOCKS.index("rp"))
    ratio = sweep["cc"][0, b, rows] / sweep["cc"][0, a, rows]
    span = ratio.max() / ratio.min()
    assert np.all(ratio < 1.0), "a lower sticking efficiency must lower the coefficient"
    assert 3.0 < span < 3.7, (
        f"the se=0.3 / se=1.0 ratio spans {span:.4f} across the rp axis, measured "
        "3.3157; a moved grid or a changed formula"
    )
    # The asymptote is a check on the reading, not a coincidence: as kn -> inf,
    # fkn -> 1/(1.33*kn) (`:208`), so 1.33*kn*fkn -> 1 and
    # akn -> 1/(1 + (1/se - 1)) = se. The smallest particles lose exactly the
    # factor se and the largest lose nothing, so the ratio is bounded below by
    # se and above by 1 -- strictly, at both ends.
    assert 0.3 < ratio.min() < 0.31, f"ratio floor {ratio.min():.6f} is not approaching se = 0.3"
    assert 0.99 < ratio.max() < 1.0, f"ratio ceiling {ratio.max():.6f} is not approaching 1.0"


def test_cc_over_sinkarr_carries_no_rp(sweep, grid, calls):
    """Both branches make `cc` and `sinkarr` from the same factors, so their
    ratio is `term6*dcoff_cp*1e-6` and independent of `rp`. Not byte-exact:
    the two expressions divide by `denom` in a different order."""
    rows = grid["block"] == cap.BLOCKS.index("rp")
    for i in range(len(calls)):
        ratio = sweep["cc"][0, i, rows] / sweep["sinkarr"][0, i, rows]
        spread = np.max(np.abs(ratio - ratio[0])) / abs(ratio[0])
        assert spread <= 1e-14, f"{cap.call_label(calls[i])}: spread {spread:.3e}"


def test_the_pressure_axis_can_see_the_divide_by_a_constant_rewrite(sweep):
    """`:179` divides by the literal `101325.0`, the site XLA rewrites into a
    multiply by the reciprocal (issue #23). Of twelve plausible pressures only
    two give a different double under the rewrite, so an axis of round numbers
    would pass against a port that had it. The recorded count is re-derived
    here rather than trusted."""
    c = 101325.0
    inv = 1.0 / c
    n = sum(1 for p in cap.PMID_AXIS if (p / c) != (p * inv))
    assert n == int(sweep["_pmid_discriminating"])
    assert n >= 5
    assert 101325.0 in cap.PMID_AXIS, "the exactly-1.0 anchor is gone"


MUTATIONS = (
    "zero_a_masked_row",
    "break_setup_independence",
    "make_an_inert_argument_live",
    "flatten_the_pressure_axis",
    "couple_t_at_idcmfp_1",
    "scale_sinkarr_alone",
)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_the_captures_own_guards_reject_a_corrupted_archive(sweep, grid, calls, mutation):
    """Name the mutation that would fail the guard, then apply it.

    Six ways this golden could be wrong and still look right. Each one must be
    caught by `verify_outputs`, which is what runs before `savez_compressed` at
    capture time -- so this is a test of the capture, not only of the archive.
    """
    cc = np.array(sweep["cc"])
    sinkarr = np.array(sweep["sinkarr"])

    if mutation == "zero_a_masked_row":
        off = np.flatnonzero(grid["block"] == cap.BLOCKS.index("masked_off"))
        cc[0, 0, off[0]] = 1.0e-20
    elif mutation == "break_setup_independence":
        cc[1, 3, 5] = np.nextafter(cc[1, 3, 5], np.inf)
    elif mutation == "make_an_inert_argument_live":
        _, b = cap._pair(calls, "h2so4", "h2so4_difvol_org", 1, 1)
        cc[:, b, 7] = np.nextafter(cc[0, b, 7], np.inf)
    elif mutation == "flatten_the_pressure_axis":
        rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("pmid"))
        for i, c in enumerate(calls):
            if c["idcmfp"] == 2:
                cc[:, i, rows] = cc[0, i, rows[0]]
    elif mutation == "couple_t_at_idcmfp_1":
        rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("tsqrt_decoupled"))
        for i, c in enumerate(calls):
            if c["idcmfp"] == 1:
                cc[:, i, rows[0]] *= 1.5
    elif mutation == "scale_sinkarr_alone":
        rows = np.flatnonzero(grid["block"] == cap.BLOCKS.index("rp"))
        sinkarr[:, :, rows[3]] *= 1.0001

    with pytest.raises(SystemExit):
        cap.verify_outputs(cc, sinkarr, grid, calls)


def test_the_guards_accept_the_archive_as_committed(sweep, grid, calls):
    """The other half of the mutation test: guards tight enough to reject all
    six must still pass the real thing, or they are simply broken."""
    report = cap.verify_outputs(np.array(sweep["cc"]), np.array(sweep["sinkarr"]), grid, calls)
    assert report["live_rows"] == 252
    assert report["masked_rows"] == 7
    assert report["inert_argument_pairs"] == 8
