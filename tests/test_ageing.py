"""Task 59: `ukca_ageing`, byte-equal, and two CLAUDE.md claims corrected.

Against `tests/goldens/ageing.f64.leaf.npz`: 7 setups x 2 `l_dust_mp_ageing`
settings, four outputs, **byte-equal on all fourteen**.

That is the first port since phase D to reach exact agreement, and the reason
is simple: `ukca_ageing` has no transcendental in its live path at all, so
issue #28 has nothing to bite on. Every gap in the previous six routines was
`exp`; this one has none, and the equality is exact.

Two claims corrected
--------------------

CLAUDE.md lists `ukca_ageing` twice among five loops needing sequential
treatment: "over modes (7->4 and 8->4 collide) and over `jv`".

**The mode collision is unreachable.** `tmode` is `imode - 3` below
`mode_sup_insol` and `imode - 4` at it, so 7 and 8 both target mode 4 -- true of
the source. But `mode_sup_insol` needs setup 12 or 13, which the box model does
not implement, and measured across all seven setups at both switch settings the
modes entering the loop are a subset of `{5, 6, 7}` with targets always
distinct. So the mode loop is parallelisable in every configuration this project
can validate, and the "broadcast differs" test CLAUDE.md asks for **cannot be
written** for it. `test_the_mode_targets_never_collide` asserts that, and fails
the moment a setup activates mode 8.

**The `jv` collision is also unreachable.** `cp_coag_added(icp)` stops a
component's coagulation term being counted once per condensable gas that maps
to it -- but that needs **two** condensable gases on one component, and no
supported setup has two, because `msec_orgi` is absent in all of them. Dropping
the guard entirely changes nothing on any of the fourteen configurations.

So **neither** of the two collisions CLAUDE.md names for `ukca_ageing` can have
the "broadcast differs" test the rule asks for. Both are real in the source and
unreachable here, and the tests below assert the *reasons* rather than the
conclusions, so the day a setup activates mode 8 or adds a second organic
condensable, they fail.

Three association and rewrite errors this port shipped
------------------------------------------------------

All three were 1-4 ulp, all in setup 8's dust-ageing arm -- the only
configuration running more than one insoluble mode -- and none looked wrong:

1. `totage(:)*nd/naged` factored as `totage * (nd/naged)`, which is exactly
   `totage * 1.0` once `naged` has been clamped to `nd`, where the Fortran
   computes `(totage*nd)/nd`.
2. the `ageterm2` sum accumulated into a temporary and added once, where the
   Fortran accumulates term by term into both `totage_jv` and `totage(icp)`.
3. `/f_mm` and `/dimen` written as plain divisions. Both are scalar constants,
   so XLA rewrote them into multiplies by the reciprocal. **This was the one
   that actually mattered** -- (1) and (2) turned out to be equivalent on this
   data, and only `true_divide` closed the gap.

The third is the fourth time `numerics.true_divide` has been load-bearing, and
the first where two other plausible explanations were fixed first without
moving a single element.

UP-3, at both settings
----------------------

`:302-306` overwrites `naged` before using it as a divisor, so the rescale is
`totage * nd/nd` -- a no-op, where the comment says it "reduces ageing if
limited by insoluble particles". Both settings are exercised against the golden
and required to differ, and the default is the arm the reference agrees with.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.ageing as ag
from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import gas_indices as gi
from glomap_jax.physics import modes as mm
from glomap_jax.physics._ageing_literals import AGEING_BUDGET_SITES

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "ageing.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_ageing_leaf as cap  # noqa: E402
import extract_ageing_literals as extractor  # noqa: E402

FIELDS = ("nd", "md", "mdt", "bud")

#: Ageing does nothing at all in these setups: 1, 3 and 5 activate no insoluble
#: mode, and 6 carries no condensable gas.
NOOP_SETUPS = (1, 3, 5, 6)


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_ageing_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


def _cases(sweep):
    for s_i, setup in enumerate(sweep["setups"].tolist()):
        for c_i, combo in enumerate(str(c) for c in sweep["combos"]):
            yield s_i, setup, c_i, combo


def _run(sweep, s_i, setup, c_i, combo, **kw):
    ncp = int(sweep["ncp"][s_i, c_i])
    nchemg = int(sweep["nchemg"][s_i, c_i])
    nbud = int(sweep["nbudaer"][s_i, c_i])
    nbox = sweep["in_nd"].shape[2]
    tables = mm.build(setup, l_dust_mp_ageing=(combo == "dust_ageing"))
    out = ag.ageing(
        tables,
        gi.build(setup),
        bi.build(setup),
        jnp.asarray(sweep["in_nd"][s_i, c_i]),
        jnp.asarray(sweep["in_md"][s_i, c_i][:, :, :ncp]),
        jnp.asarray(sweep["in_mdt"][s_i, c_i]),
        jnp.asarray(sweep["in_ageterm1"][s_i, c_i][:, :, :nchemg]),
        jnp.asarray(sweep["in_ageterm2"][s_i, c_i][:, :, :, :ncp]),
        jnp.asarray(sweep["in_wetdp"]),
        jnp.zeros((nbox, nbud + 1)),
        **kw,
    )
    return out, (ncp, nbud)


def _expected(sweep, s_i, c_i, ncp, nbud):
    return {
        "nd": sweep["nd"][s_i, c_i],
        "md": sweep["md"][s_i, c_i][:, :, :ncp],
        "mdt": sweep["mdt"][s_i, c_i],
        "bud": sweep["bud"][s_i, c_i][:, : nbud + 1],
    }


def test_the_port_is_byte_equal_to_the_compiled_routine(sweep):
    """No transcendental in the live path, so no issue-#28 allowance: exact on
    every element of every configuration."""
    for s_i, setup, c_i, combo in _cases(sweep):
        out, (ncp, nbud) = _run(sweep, s_i, setup, c_i, combo)
        want = _expected(sweep, s_i, c_i, ncp, nbud)
        for name, got in zip(FIELDS, out):
            np.testing.assert_array_equal(
                np.asarray(got), want[name], err_msg=f"{name} setup {setup}/{combo}"
            )


def test_the_routine_has_no_transcendental_in_its_live_path():
    """Why the equality above is exact rather than bounded. If an `EXP` or
    `LOG` ever appears here, this port inherits issue #28 and the test above
    has to become a ulp window."""
    text = (REPO / "fortran" / "src" / "ukca" / "ukca_ageing.F90").read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("!"))
    for fn in ("EXP(", "LOG(", "SQRT(", "ATAN(", "TAN("):
        assert fn not in code.upper().replace(" ", ""), f"{fn} has appeared in ukca_ageing"


def test_true_divide_is_load_bearing(sweep, monkeypatch):
    """The error that actually mattered. `/f_mm` and `/dimen` are divisions by
    scalar constants, and XLA rewrites them. Two other plausible explanations
    -- a factored rescale and a reassociated sum -- were fixed first and moved
    nothing at all."""
    monkeypatch.setattr(ag.numerics, "true_divide", lambda x, c: jnp.asarray(x) / c)
    broke = 0
    for s_i, setup, c_i, combo in _cases(sweep):
        out, (ncp, nbud) = _run(sweep, s_i, setup, c_i, combo)
        want = _expected(sweep, s_i, c_i, ncp, nbud)
        if any(not np.array_equal(np.asarray(g), want[n]) for n, g in zip(FIELDS, out)):
            broke += 1
    assert broke > 0, "removing true_divide changed nothing; the guard is not load-bearing here"


def test_the_mode_targets_never_collide():
    """CLAUDE.md's "7->4 and 8->4 collide", checked. True of the source and
    unreachable: no supported setup activates `mode_sup_insol`. Fails the
    moment one does, which is when the port's mode loop must go sequential."""
    info = cap.verify_target_modes()
    assert info["mode_pairs"] == 10
    assert ag.target_mode(6) == 3 and ag.target_mode(7) == 3, "7 and 8 must both target mode 4"
    for setup in bi.supported_setups():
        for dust in (False, True):
            t = mm.build(setup, l_dust_mp_ageing=dust)
            assert not t.mode[7], f"setup {setup} activates mode_sup_insol"


def test_the_cp_coag_added_guard_is_inert_in_every_supported_setup(sweep):
    """CLAUDE.md's second ageing collision, and it is unreachable too.

    `cp_coag_added(icp)` at `:265` stops a component's coagulation term being
    counted once per condensable gas that maps to it. It can only bite where
    **two** condensable gases share a component -- `msec_org` and `msec_orgi`
    both mapping to `cp_oc`. Measured: no supported setup has two, because
    `msec_orgi` is index 0 (absent) in all of them, which phase C already
    recorded for a different reason.

    So `broadcast=True` -- dropping the guard entirely -- changes nothing, on
    any of the fourteen configurations. Both of the collisions CLAUDE.md names
    for `ukca_ageing` are structurally real in the source and unreachable in
    every configuration this project can validate, and neither can have the
    "broadcast differs" test the rule asks for.

    The guard stays in the port because the source has it and a setup with two
    organic condensables would need it. What is asserted here is the *reason*
    it is currently inert, so the day a setup adds one, this fails.
    """
    for setup in bi.supported_setups():
        g = gi.build(setup)
        choices = [
            int(g.condensable_choice[j]) for j in range(len(g.condensable)) if g.condensable[j]
        ]
        assert len(choices) == len(set(choices)), (
            f"setup {setup} has two condensable gases sharing a component, so "
            "cp_coag_added is now load-bearing and needs a differs-test"
        )
        assert g.msec_orgi < 0 or not bool(g.condensable[g.msec_orgi])

    for s_i, setup, c_i, combo in _cases(sweep):
        seq, _ = _run(sweep, s_i, setup, c_i, combo)
        bro, _ = _run(sweep, s_i, setup, c_i, combo, broadcast=True)
        for name, a, b in zip(FIELDS, seq, bro):
            np.testing.assert_array_equal(
                np.asarray(a), np.asarray(b), err_msg=f"{name} setup {setup}/{combo}"
            )


def test_up3_changes_the_answer_at_both_settings(sweep):
    """`:302-306` overwrites `naged` before using it as the divisor, so the
    rescale is `totage * nd/nd`. Both arms against the golden, and the default
    is the one the reference agrees with."""
    differing = 0
    for s_i, setup, c_i, combo in _cases(sweep):
        on, (ncp, nbud) = _run(
            sweep, s_i, setup, c_i, combo, fidelity=FidelityConfig(ageing_totage_rescale_noop=True)
        )
        off, _ = _run(
            sweep, s_i, setup, c_i, combo, fidelity=FidelityConfig(ageing_totage_rescale_noop=False)
        )
        if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(on, off)):
            differing += 1
            want = _expected(sweep, s_i, c_i, ncp, nbud)
            np.testing.assert_array_equal(np.asarray(on[0]), want["nd"])
    assert differing > 0, (
        "the flag changed nothing -- no row reached naged > nd, which is the only "
        "place the rescale runs"
    )


def test_ageing_does_nothing_in_four_of_the_seven_setups(sweep):
    """1, 3 and 5 activate no insoluble mode; 6 activates two but carries no
    condensable gas. Recorded because it bounds what any trajectory comparison
    of this routine could ever show."""
    for s_i, setup, c_i, combo in _cases(sweep):
        out, _ = _run(sweep, s_i, setup, c_i, combo)
        unchanged = np.array_equal(np.asarray(out[0]), sweep["in_nd"][s_i, c_i])
        assert unchanged == (setup in NOOP_SETUPS), f"setup {setup}/{combo}"


def test_number_moves_from_insoluble_to_soluble_and_is_conserved(sweep):
    """Ageing transfers particles; it does not create or destroy them.

    Not asserted as an exact equality, and the test's first draft was: both
    sides are differences of nearly-equal numbers -- `nd - (nd - naged)` and
    `(nd_t + naged) - nd_t` -- so each loses digits to cancellation and they
    disagree at 1e-13 even though the port is byte-equal to the reference. The
    conservation is real; the exact equality was an artefact of how the test
    measured it.
    """
    for s_i, setup, c_i, combo in _cases(sweep):
        out, _ = _run(sweep, s_i, setup, c_i, combo)
        nd_out = np.asarray(out[0])
        nd_in = sweep["in_nd"][s_i, c_i]
        assert np.all(nd_out >= 0.0)
        for imode in range(4, 7):
            tmode = ag.target_mode(imode)
            lost = nd_in[:, imode] - nd_out[:, imode]
            gained = nd_out[:, tmode] - nd_in[:, tmode]
            assert np.all(lost >= 0.0), f"insoluble mode {imode + 1} gained number"
            assert np.all(gained >= 0.0), f"soluble mode {tmode + 1} lost number"
            moving = lost > 0.0
            if moving.any():
                rel = np.abs(gained[moving] - lost[moving]) / lost[moving]
                assert rel.max() < 1e-12, (
                    f"setup {setup}/{combo} mode {imode + 1}: the target gained "
                    f"{rel.max():.2e} relative more than the source lost"
                )
        # And the total over the two modes is conserved to the same order.
        total_in = nd_in.sum(axis=1)
        total_out = nd_out.sum(axis=1)
        nz = total_in > 0.0
        assert np.all(np.abs(total_out[nz] - total_in[nz]) / total_in[nz] < 1e-12)


def test_the_generated_write_sites_are_not_stale(capsys):
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch):
    doctored = tmp_path / "_ageing_literals.py"
    doctored.write_text("AGEING_BUDGET_SITES = ()\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "TARGET", doctored)
    assert extractor.main(["--check"]) == 1


def test_the_one_name_written_twice_takes_two_different_quantities():
    """21 sites, 20 names. `nmasagedocintr52` receives both
    `totage(cp_oc)*f_mm` and `naged*md(...,cp_oc)`, because organic carbon is a
    condensable *and* a primary insoluble component -- so unlike issue #29 this
    is two quantities, not one counted twice."""
    assert len(AGEING_BUDGET_SITES) == 21
    names = [r[3] for r in AGEING_BUDGET_SITES]
    assert len(set(names)) == 20
    twice = {n for n in names if names.count(n) > 1}
    assert twice == {"nmasagedocintr52"}
    kinds = sorted(r[4] for r in AGEING_BUDGET_SITES if r[3] == "nmasagedocintr52")
    assert kinds == ["naged", "totage"]


def test_every_site_agrees_with_its_own_name():
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
    for imode, tmode, icp, name, kind in AGEING_BUDGET_SITES:
        assert name == f"nmasaged{tags[icp]}intr{imode}{tmode}", name
        assert tmode == (imode - 3 if imode < 8 else imode - 4)
        assert kind in ("totage", "naged")


def test_the_hole_at_budget_slot_zero_is_never_written(sweep):
    for s_i, setup, c_i, combo in _cases(sweep):
        out, _ = _run(sweep, s_i, setup, c_i, combo)
        assert np.all(np.asarray(out[3])[:, 0] == 0.0)
