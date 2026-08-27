"""Task 61: `ukca_remode`, byte-equal, and the second live loop-carried loop.

Against `tests/goldens/remode.f64.leaf.npz`: 7 setups x 3 `imerge` settings x
17 boxes, five outputs, **byte-equal on all twenty-one configurations**.

Issue #12 says no shipped namelist ever merges a mode, so the fixture is
entirely constructed. `drydp` is handed in rather than computed -- it is
`calc_drydiam`'s output, already byte-equal -- which is what makes constructing
the merge condition legitimate rather than inventing physics.

The mode loop is loop-carried and reachable
-------------------------------------------

`:206` writes `nd(jl,imode+1)` and `md(jl,imode+1,:)` on each pass and the next
pass reads both. **210 of the fixture's boxes merge twice**, so the second merge
sees the first's result. `broadcast=True` runs every mode against the entry
state, and `test_the_broadcast_form_differs` requires that to change the answer.

This is the second of CLAUDE.md's five loops shown live -- after
`ukca_coagwithnucl`'s `icp` loop -- and unlike `ukca_ageing`'s two, it needs
nothing exotic to reach.

Why byte equality is attainable
-------------------------------

`umErf` carries the whole merge fraction, and the phase B numerics sweep found
it bit-identical between gfortran and JAX on the capture platform. `EXP` appears
once, at `:252`, and on this grid it lands where XLA and libm agree -- so unlike
the six routines before `ukca_ageing`, there is no residual gap. That is a
property of this grid and not a guarantee, and
`test_exp_is_the_only_place_issue_28_could_bite` says where to look if it ever
stops holding.

A trap this task removed from the golden
----------------------------------------

`in_mdt` is `md.sum(axis=2)` over each setup's **own** `ncp`. Stored once at the
padded width, it is a value no configuration was ever given -- and comparing
against it made the port look wrong on 94 of 136 `mdt` entries while every
other field was byte-equal. It is now stored per setup, and
`test_the_stored_input_totals_match_their_own_widths` keeps it that way.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.remode as rm
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import modes as mm
from glomap_jax.physics._remode_literals import MERGE_BUDGET_SITES

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "remode.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_remode_leaf as cap  # noqa: E402
import extract_remode_literals as extractor  # noqa: E402

FIELDS = ("nd", "md", "mdt", "bud", "n_merge")


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_remode_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


def _cases(sweep):
    for s_i, setup in enumerate(sweep["setups"].tolist()):
        for k_i, imerge in enumerate(sweep["imerges"].tolist()):
            yield s_i, setup, k_i, imerge


def _run(sweep, s_i, setup, imerge, **kw):
    ncp = int(sweep["ncp"][s_i])
    nbud = int(sweep["nbudaer"][s_i])
    nbox = sweep["in_nd"].shape[0]
    return rm.remode(
        mm.build(setup),
        bi.build(setup),
        jnp.asarray(sweep["in_nd"]),
        jnp.asarray(sweep["in_md"][s_i][:, :, :ncp]),
        jnp.asarray(sweep["in_mdt"][s_i]),
        jnp.asarray(sweep["in_drydp"]),
        jnp.asarray(sweep["in_pmid"]),
        jnp.zeros((nbox, nbud + 1)),
        imerge=imerge,
        **kw,
    ), (ncp, nbud)


def _expected(sweep, s_i, k_i, ncp, nbud):
    return {
        "nd": sweep["nd"][s_i, k_i],
        "md": sweep["md"][s_i, k_i][:, :, :ncp],
        "mdt": sweep["mdt"][s_i, k_i],
        "bud": sweep["bud"][s_i, k_i][:, : nbud + 1],
        "n_merge": sweep["n_merge"][s_i, k_i],
    }


def test_the_port_is_byte_equal_to_the_compiled_routine(sweep):
    for s_i, setup, k_i, imerge in _cases(sweep):
        r, (ncp, nbud) = _run(sweep, s_i, setup, imerge)
        want = _expected(sweep, s_i, k_i, ncp, nbud)
        got = (r.nd, r.md, r.mdt, r.bud_aer_mas, r.n_merge)
        for name, g in zip(FIELDS, got):
            np.testing.assert_array_equal(
                np.asarray(g), want[name], err_msg=f"{name} setup {setup} imerge {imerge}"
            )


def test_the_stored_input_totals_match_their_own_widths(sweep):
    """`in_mdt` is per setup, because it sums over that setup's `ncp`. Stored
    once at the padded width it is a value no run was given, and the first
    draft of this fixture did exactly that."""
    for s_i in range(sweep["setups"].size):
        ncp = int(sweep["ncp"][s_i])
        np.testing.assert_array_equal(
            sweep["in_mdt"][s_i], sweep["in_md"][s_i][:, :, :ncp].sum(axis=2)
        )
    # And the padded width would NOT match, or this proves nothing.
    padded = sweep["in_md"][0].sum(axis=2)
    assert not np.array_equal(sweep["in_mdt"][0], padded)


def test_the_broadcast_form_differs(sweep):
    """The mode loop's dependency, demonstrated. Each pass writes the mode
    above; running every mode against the entry state is the plausible-looking
    mistake."""
    differing = 0
    for s_i, setup, k_i, imerge in _cases(sweep):
        seq, (ncp, nbud) = _run(sweep, s_i, setup, imerge)
        bro, _ = _run(sweep, s_i, setup, imerge, broadcast=True)
        pairs = list(
            zip(
                (seq.nd, seq.md, seq.mdt, seq.bud_aer_mas),
                (bro.nd, bro.md, bro.mdt, bro.bud_aer_mas),
            )
        )
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in pairs):
            differing += 1
            want = _expected(sweep, s_i, k_i, ncp, nbud)
            np.testing.assert_array_equal(np.asarray(seq.nd), want["nd"])
            assert not np.array_equal(np.asarray(bro.nd), want["nd"])
    assert differing > 0, (
        "the broadcast form never differed -- either no box merges twice or the "
        "sequential dependency is not implemented"
    )


def test_two_hundred_and_ten_boxes_merge_twice(sweep):
    """What makes the test above possible. Issue #12's whole point is that a
    trajectory reaches none of this."""
    total = 0
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        counts = (np.asarray(r.n_merge) > 0).sum(axis=1)
        total += int((counts >= 2).sum())
    assert total == 210


def test_imerge_three_merges_unconditionally(sweep):
    """`:234` is `(dp > dp_thresh1) .OR. (imerge == 3)`, so setting 3 is the
    only one that runs on a distribution no namelist would merge."""
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        # The block codes are stored as `in_block`, like every other
        # setup-independent input in this archive.
        blocks = [str(b) for b in sweep["blocks"]]
        unmerged = np.flatnonzero(sweep["in_block"] == blocks.index("unmerged"))
        fired = (np.asarray(r.n_merge)[unmerged] > 0).any()
        tables = mm.build(setup)
        if not any(tables.mode[:3]):
            continue
        assert fired == (imerge == 3), f"setup {setup} imerge {imerge}"


def test_the_accumulation_mode_does_not_merge_in_the_stratosphere(sweep):
    """`:203-204` drops `nmodemax_merge` to 2 below 1e4 Pa. The grid straddles
    the threshold with both neighbouring doubles, and the test is a strict
    `<` -- so `pmid == 1.0e4` exactly is tropospheric."""
    assert rm.P_STRAT == 1.0e4
    pmid = np.asarray(sweep["in_pmid"])
    assert (pmid < rm.P_STRAT).any() and (pmid == rm.P_STRAT).any() and (pmid > rm.P_STRAT).any()
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        strat = pmid < rm.P_STRAT
        assert not np.any(np.asarray(r.n_merge)[strat, 2] > 0), f"setup {setup} imerge {imerge}"


def test_both_silent_clamps_are_surfaced(sweep):
    """`frac_n < 0.5` and `frac_m < 0.001` are floored with no diagnostic.
    CLAUDE.md requires a cap to be reported, so both come back as masks -- and
    at least one must fire, or the surfacing is untested."""
    assert (rm.FRAC_N_FLOOR, rm.FRAC_M_FLOOR) == (0.5, 0.001)
    fired_n = fired_m = 0
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        fired_n += int(np.asarray(r.frac_n_clamped).sum())
        fired_m += int(np.asarray(r.frac_m_clamped).sum())
    assert fired_n > 0, "the frac_n floor never fired; the grid does not reach it"


def test_an_unsupported_imerge_raises(sweep):
    """`:215-231` assigns both thresholds only inside the three branches, so
    any other value reads them never having been assigned."""
    s_i, setup, _k_i, _imerge = next(iter(_cases(sweep)))
    for bad in (0, 4, -1):
        with pytest.raises(ValueError, match="imerge"):
            _run(sweep, s_i, setup, bad)


def test_number_is_conserved_across_the_merge(sweep):
    """A merge moves particles from one mode to the next; the total over the
    pair is unchanged up to rounding. Not exact: both sides are differences of
    nearly-equal numbers, as `ukca_ageing`'s conservation test found."""
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        nd_out = np.asarray(r.nd)
        nd_in = np.asarray(sweep["in_nd"])
        total_in, total_out = nd_in.sum(axis=1), nd_out.sum(axis=1)
        nz = total_in > 0.0
        assert np.all(np.abs(total_out[nz] - total_in[nz]) / total_in[nz] < 1e-12)
        assert np.all(nd_out >= 0.0)


def test_exp_is_the_only_place_issue_28_could_bite():
    """`EXP` appears once, at `:252` (`dp2`). If the byte equality above ever
    breaks on a new grid, that is where to look -- and `umErf` is not, because
    the phase B sweep found it bit-identical."""
    text = (REPO / "fortran" / "src" / "ukca" / "ukca_remode.F90").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("!"))
    assert code.count("EXP(") == 1
    assert "umErf(" in code


def test_the_generated_write_sites_are_not_stale(capsys):
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch):
    doctored = tmp_path / "_remode_literals.py"
    doctored.write_text("MERGE_BUDGET_SITES = ()\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "TARGET", doctored)
    assert extractor.main(["--check"]) == 1


def test_every_site_merges_into_the_mode_above_it():
    tags = {
        1: "su",
        2: "bc",
        3: "oc",
        4: "ss",
        5: "du",
        6: "so",
        7: "nh",
        8: "nt",
        9: "nn",
        10: "mp",
    }
    assert len(MERGE_BUDGET_SITES) == 19
    for imode, jmode, icp, name in MERGE_BUDGET_SITES:
        assert jmode == imode + 1
        assert name == f"nmasmerg{tags[icp]}intr{imode}{jmode}", name
        assert 1 <= imode <= 3, "only the three soluble modes merge"


def test_the_source_structure_is_unchanged():
    info = cap.verify_source_structure()
    assert info == {"checked": 9, "budget_names": 19}


def test_the_hole_at_budget_slot_zero_is_never_written(sweep):
    for s_i, setup, _k_i, imerge in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, imerge)
        assert np.all(np.asarray(r.bud_aer_mas)[:, 0] == 0.0)
