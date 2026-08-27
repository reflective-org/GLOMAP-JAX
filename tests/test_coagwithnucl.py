"""Task 57: `ukca_coagwithnucl`, and the first proof a scan is needed.

Against `tests/goldens/coagwithnucl.f64.leaf.npz`: 7 setups x 2 switch combos x
4 `(intraoff, interoff)` pairs x 20 rows, five outputs.

**116 elements differ across the whole archive, worst 4 ulp, and substituting
libm's `exp` removes every one.** `bud_aer_mas` and `ageterm2` are byte-equal
everywhere. Issue #28's sixth routine.

The broadcast form differs, and now that is measured
----------------------------------------------------

CLAUDE.md lists five routines that need sequential treatment over a loop and
says each must get a test asserting the broadcast version **differs** -- written
as part of the port, "since a scan that was never wrong is a scan nobody can
show is needed". None of the five had one, because none was ported.

This is the first. `coagwithnucl(..., broadcast=True)` evaluates the `icp` loop
at `:544-575` against the original `mask1` instead of the one the loop mutates,
and it differs on **24 of the 56 configurations**, moving 82 `md` elements, 32
`mdt` and 220 `bud_aer_mas`.

It only differs because the fixture reaches `mdcpnew < 0` -- issue #13's branch,
438 times. On a trajectory grid the two forms would agree and the test would be
vacuous, which is exactly why the constructed fixture had to come first.

The bug this port shipped and the fixture caught
------------------------------------------------

`coag_mode` is stored 0-based in `physics/coag_mode.py` and 1-based in the
Fortran and in the golden. The first draft indexed the golden's copy directly,
so every destination shifted by one mode -- mass leaving the nucleation mode
landed in the Aitken mode. 62 of setup 1's `md` entries were wrong and not one
of them looked wrong: the numbers were the right order of magnitude, positive,
and mass-conserving in total. Only a byte comparison against the reference
found it.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.coagwithnucl as cw
import glomap_jax.physics.solvecoagnucl as sv
from glomap_jax.physics import budget_indices as bi
from glomap_jax.physics import coag_mode as cmm
from glomap_jax.physics import gas_indices as gi
from glomap_jax.physics import modes as mm
from glomap_jax.physics._coagwithnucl_literals import COAG_BUDGET_SITES

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "coagwithnucl.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import extract_coagwithnucl_literals as extractor  # noqa: E402

EXP_ULP = 4
EXP_AFFECTED = {"nd": 27, "md": 64, "mdt": 25}
EXACT_FIELDS = ("bud", "ageterm2")
FIELDS = ("nd", "md", "mdt", "bud", "ageterm2")

#: Measured. The broadcast form's disagreement with the sequential one.
BROADCAST_DIFFERS_ON = 24
BROADCAST_MOVES = {"md": 82, "mdt": 32, "bud": 220}


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_coagwithnucl_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


def _cases(sweep):
    for s_i, setup in enumerate(sweep["setups"].tolist()):
        for c_i, combo in enumerate(str(c) for c in sweep["combos"]):
            for k_i, pair in enumerate(sweep["switches"].tolist()):
                yield s_i, setup, c_i, combo, k_i, tuple(pair)


def _args(sweep, s_i, setup, c_i, combo):
    ncp = int(sweep["ncp"][s_i, c_i])
    nchemg = int(sweep["nchemg"][s_i, c_i])
    nbud = int(sweep["nbudaer"][s_i, c_i])
    nbox = sweep["in_nd"].shape[2]
    tables = mm.build(setup, l_dust_mp_ageing=(combo == "dust_ageing"))
    return (
        (
            tables,
            sweep["coag_mode"][s_i, c_i] - 1,
            bi.build(setup),
            gi.build(setup),
            jnp.asarray(sweep["in_nd"][s_i, c_i]),
            jnp.asarray(sweep["in_md"][s_i, c_i][:, :, :ncp]),
            jnp.asarray(sweep["in_mdt"][s_i, c_i]),
            jnp.asarray(sweep["in_delgc"][s_i, c_i][:, :nchemg]),
            jnp.asarray(sweep["in_kii"]),
            jnp.asarray(sweep["in_kij"]),
            jnp.zeros((nbox, nbud + 1)),
        ),
        (ncp, nbud),
    )


def _expected(sweep, s_i, c_i, k_i, ncp, nbud):
    return {
        "nd": sweep["nd"][s_i, c_i, k_i],
        "md": sweep["md"][s_i, c_i, k_i][:, :, :ncp],
        "mdt": sweep["mdt"][s_i, c_i, k_i],
        "bud": sweep["bud"][s_i, c_i, k_i][:, : nbud + 1],
        "ageterm2": sweep["ageterm2"][s_i, c_i, k_i][:, :, :, :ncp],
    }


def _run(sweep, s_i, setup, c_i, combo, pair, **kw):
    args, (ncp, nbud) = _args(sweep, s_i, setup, c_i, combo)
    intraoff, interoff = pair
    r = cw.coagwithnucl(*args, dtz=float(sweep["dtz"]), intraoff=intraoff, interoff=interoff, **kw)
    return r, (ncp, nbud)


def test_the_port_matches_the_compiled_routine(sweep):
    counts = dict.fromkeys(FIELDS, 0)
    worst = 0.0
    for s_i, setup, c_i, combo, k_i, pair in _cases(sweep):
        r, (ncp, nbud) = _run(sweep, s_i, setup, c_i, combo, pair)
        want = _expected(sweep, s_i, c_i, k_i, ncp, nbud)
        for name, got in zip(FIELDS, (r.nd, r.md, r.mdt, r.bud_aer_mas, r.ageterm2)):
            got = np.asarray(got)
            off = got != want[name]
            counts[name] += int(off.sum())
            nz = off & (want[name] != 0)
            if nz.any():
                worst = max(
                    worst,
                    float(
                        (
                            np.abs(got[nz] - want[name][nz]) / np.spacing(np.abs(want[name][nz]))
                        ).max()
                    ),
                )
    for name in EXACT_FIELDS:
        assert counts[name] == 0, f"{name} is not exact: {counts[name]} elements differ"
    assert {k: v for k, v in counts.items() if v} == EXP_AFFECTED
    assert worst <= EXP_ULP


def test_the_whole_gap_is_the_exponential(sweep, monkeypatch):
    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(cw, "jnp", _Libm())
    monkeypatch.setattr(sv, "jnp", _Libm())
    for s_i, setup, c_i, combo, k_i, pair in _cases(sweep):
        r, (ncp, nbud) = _run(sweep, s_i, setup, c_i, combo, pair)
        want = _expected(sweep, s_i, c_i, k_i, ncp, nbud)
        for name, got in zip(FIELDS, (r.nd, r.md, r.mdt, r.bud_aer_mas, r.ageterm2)):
            np.testing.assert_array_equal(np.asarray(got), want[name], err_msg=f"{name} {pair}")


def test_the_broadcast_form_differs(sweep):
    """What CLAUDE.md asks for, for the first of its five loop-carried loops.

    `:544-575` mutates `mask1` inside the `icp` loop -- once a component's new
    mass goes negative the mode's number is zeroed and every later component is
    skipped. Evaluating the components against the original mask is the
    plausible-looking mistake, and it must change the answer or the sequential
    form is unmotivated.
    """
    differing = 0
    moved = {}
    for s_i, setup, c_i, combo, _k_i, pair in _cases(sweep):
        seq, _ = _run(sweep, s_i, setup, c_i, combo, pair)
        bro, _ = _run(sweep, s_i, setup, c_i, combo, pair, broadcast=True)
        this = {
            name: int((np.asarray(a) != np.asarray(b)).sum())
            for name, a, b in zip(
                ("nd", "md", "mdt", "bud"),
                (seq.nd, seq.md, seq.mdt, seq.bud_aer_mas),
                (bro.nd, bro.md, bro.mdt, bro.bud_aer_mas),
            )
        }
        this = {k: v for k, v in this.items() if v}
        if this:
            differing += 1
            for k, v in this.items():
                moved[k] = moved.get(k, 0) + v
    assert differing == BROADCAST_DIFFERS_ON, (
        f"the broadcast form differs on {differing} of 56 configurations, measured "
        f"{BROADCAST_DIFFERS_ON}; if this ever reaches 0 the fixture has stopped "
        "reaching mdcpnew < 0 and the sequential form is no longer motivated"
    )
    assert moved == BROADCAST_MOVES


def test_the_broadcast_form_is_the_one_that_is_wrong(sweep):
    """Not just different -- wrong. The sequential form is the one the
    reference agrees with, on every configuration where they part."""
    for s_i, setup, c_i, combo, k_i, pair in _cases(sweep):
        seq, (ncp, nbud) = _run(sweep, s_i, setup, c_i, combo, pair)
        bro, _ = _run(sweep, s_i, setup, c_i, combo, pair, broadcast=True)
        if np.array_equal(np.asarray(seq.md), np.asarray(bro.md)):
            continue
        want = _expected(sweep, s_i, c_i, k_i, ncp, nbud)
        # The budget field is byte-equal for the sequential form and not for
        # the broadcast one, which is the sharpest statement available.
        np.testing.assert_array_equal(np.asarray(seq.bud_aer_mas), want["bud"])
        assert not np.array_equal(np.asarray(bro.bud_aer_mas), want["bud"])
        return
    pytest.fail("the two forms never parted; the comparison proves nothing")


def test_coag_mode_must_be_zero_based(sweep):
    """The bug this port shipped. The Fortran table and the golden are 1-based;
    `physics/coag_mode.COAG_MODE` is 0-based. Passing the wrong one shifts every
    destination by a mode -- mass leaving the nucleation mode lands in the
    Aitken mode -- and every number stays positive and plausible."""
    golden = sweep["coag_mode"][0, 0]
    np.testing.assert_array_equal(golden - 1, cmm.COAG_MODE)
    np.testing.assert_array_equal(golden, cmm.COAG_MODE_FORTRAN)

    s_i, setup, c_i, combo, k_i, pair = next(iter(_cases(sweep)))
    args, (ncp, nbud) = _args(sweep, s_i, setup, c_i, combo)
    good = cw.coagwithnucl(*args, dtz=float(sweep["dtz"]), intraoff=pair[0], interoff=pair[1])
    shifted = list(args)
    shifted[1] = sweep["coag_mode"][s_i, c_i]  # the 1-based table, unconverted
    with np.errstate(all="ignore"):
        bad = cw.coagwithnucl(*shifted, dtz=float(sweep["dtz"]), intraoff=pair[0], interoff=pair[1])
    want = _expected(sweep, s_i, c_i, k_i, ncp, nbud)
    np.testing.assert_array_equal(np.asarray(good.md), want["md"])
    assert not np.array_equal(np.asarray(bad.md), want["md"])


def test_the_generated_write_sites_are_not_stale(capsys):
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch):
    doctored = tmp_path / "_coagwithnucl_literals.py"
    doctored.write_text("COAG_BUDGET_SITES = ()\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "TARGET", doctored)
    assert extractor.main(["--check"]) == 1


def test_every_write_site_agrees_with_its_own_name():
    """`nmascoagsuintr12` is su, mode 1 -> mode 2. The extractor asserts the
    suffix against the guard it sits in; this asserts it again from the
    committed table, so a hand edit cannot slip past."""
    assert len(COAG_BUDGET_SITES) == 53
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
    for imode, jmode, icp, name in COAG_BUDGET_SITES:
        assert name == f"nmascoag{tags[icp]}intr{imode}{jmode}", name
        assert 1 <= imode <= 8 and 1 <= jmode <= 8


def test_iextra_checks_above_one_raises(sweep):
    s_i, setup, c_i, combo, _k_i, pair = next(iter(_cases(sweep)))
    args, _ = _args(sweep, s_i, setup, c_i, combo)
    with pytest.raises(ValueError, match="ukca_mode_check_mdt"):
        cw.coagwithnucl(
            *args, dtz=float(sweep["dtz"]), intraoff=pair[0], interoff=pair[1], iextra_checks=2
        )


def test_the_solver_state_is_surfaced(sweep):
    """`clamped` and `failed` come back per mode. The Fortran drops the first
    and reports the second by stopping the process."""
    for s_i, setup, c_i, combo, _k_i, pair in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, c_i, combo, pair)
        assert np.asarray(r.clamped).shape == np.asarray(r.nd).shape
        assert not np.any(np.asarray(r.failed)), "a row hit the solver's error branch"


def test_number_never_goes_negative(sweep):
    for s_i, setup, c_i, combo, _k_i, pair in _cases(sweep):
        r, _ = _run(sweep, s_i, setup, c_i, combo, pair)
        assert np.all(np.asarray(r.nd) >= 0.0)
        assert np.all(np.isfinite(np.asarray(r.md)))
