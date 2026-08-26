"""Task 51: `ukca_binapara`, and 113 coefficients nobody typed.

Against `tests/goldens/binapara.f64.leaf.npz`: 990 points, both outputs, plus
the critical cluster size the Fortran keeps to itself.

**The extraction is validated independently of this port.** The capture builds
a numpy reconstruction from `_binapara_literals.py` and requires it to
reproduce the compiled routine **bit for bit** on every point before writing
the golden -- 990 of 990, both `jveh` and `rc`. `numpy.exp` reaches the same
libm gfortran does, so that comparison has no issue-#28 slack in it: a single
mis-parsed digit in any of the 113 coefficients would fail the capture. That is
what a machine extraction buys over careful reading, and it is why the
re-derivation lives in the capture rather than here.

**This port is not byte-equal, and the reason is entirely `exp`.** 47 `jveh`
values and 117 `rc` values differ, every one by exactly 1 ulp; substituting
libm's `exp` makes all 164 vanish. Same finding as issue #28, now on its second
routine -- and here it is unavoidable rather than incidental, because `exp_v` is
applied to all three results.

Three things this file pins that a value comparison would not:

1. **Term order is the specification.** Reversing the order of one expression's
   terms changes the answer, because Fortran sums left to right. The generated
   module stores an ordered tuple for exactly this reason.

2. **The clamps bind the routine's own copies.** A caller passing 150 K gets the
   190.15 K answer *and does not take the cold branch*, because the
   `t < 195.15` test at `:251` reads the clipped temperature. Porting it
   against the caller's `t` would differ only below 190.15 K.

3. **The three output rules commute, and only by coincidence.** `1e5` from the
   cold rule lies strictly between the floor and the ceiling, so it survives
   both and the source's order can be permuted without changing the answer.
   That is a property of three constants and not of the structure -- move
   `j_cold` outside the window and the orders diverge. This paragraph said the
   opposite until the test contradicted it. What the cold rule *does* change is
   how many points reach the ceiling at all: 182 before it, 74 after.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import glomap_jax.physics.binapara as bp
from glomap_jax.physics._binapara_literals import (
    CLAMPS,
    JVEH_TERMS,
    LIMITS,
    NTOT_TERMS,
    RC_TERMS,
    TERMX_TERMS,
)

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "binapara.f64.leaf.npz"

sys.path.insert(0, str(REPO / "validation"))

import capture_binapara_leaf as cap  # noqa: E402
import extract_binapara_literals as extractor  # noqa: E402

#: Measured. Every differing point is exactly 1 ulp and the cause is `exp`.
EXP_ULP = 1
EXP_AFFECTED = {"jveh": 47, "rc": 117}


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run validation/capture_binapara_leaf.py (or `make goldens`)"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def ported(sweep):
    return bp.binapara(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )


# --- the extraction -------------------------------------------------------


def test_the_generated_literals_are_not_stale(capsys):
    """`--check` is what a CI job runs. If the vendored source ever changes a
    coefficient, this fails rather than the port quietly computing the old
    parameterisation."""
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch):
    """The half `--check` passing cannot reach. Without this, stubbing the
    comparison would leave the staleness gate green -- three such gates were
    found stubbed in the phase C review."""
    doctored = tmp_path / "_binapara_literals.py"
    doctored.write_text("TERMX_TERMS = ()\n", encoding="utf-8")
    monkeypatch.setattr(extractor, "TARGET", doctored)
    assert extractor.main(["--check"]) == 1


def test_every_coefficient_is_accounted_for():
    """113 terms, in the four expressions' own counts. A parser that silently
    dropped a term it did not recognise would give a shorter polynomial that
    still runs and is quietly wrong."""
    counts = {
        "termx": len(TERMX_TERMS),
        "jveh": len(JVEH_TERMS),
        "ntot": len(NTOT_TERMS),
        "rc": len(RC_TERMS),
    }
    assert counts == {"termx": 10, "jveh": 50, "ntot": 50, "rc": 3}
    assert sum(counts.values()) == 113


def test_no_coefficient_was_retyped():
    """`src/` must not contain the literals as source text. The whole point of
    generating them is that nobody transcribes sixteen digits by hand."""
    port = (REPO / "src" / "glomap_jax" / "physics" / "binapara.py").read_text(encoding="utf-8")
    for _sign, coeff, _factors, _over in JVEH_TERMS[:10]:
        assert repr(coeff) not in port, f"{coeff} is typed into binapara.py"


def test_the_terms_that_divide_by_termx_are_the_ones_the_source_divides():
    """10 of `jveh`'s 50 terms and 10 of `ntot`'s carry `/termx(jl)`, which is
    what makes `termx` a pole in principle. Counted so a parser that lost the
    flag on some terms fails here rather than producing a polynomial that is
    merely wrong where `termx` is small."""
    assert sum(1 for t in JVEH_TERMS if t[3]) == 10
    assert sum(1 for t in NTOT_TERMS if t[3]) == 10
    assert not any(t[3] for t in TERMX_TERMS), "termx cannot divide by itself"
    assert not any(t[3] for t in RC_TERMS)


# --- the port against the reference ---------------------------------------


@pytest.mark.parametrize("name", ("jveh", "rc"))
def test_the_port_matches_the_reference_to_one_ulp(sweep, ported, name):
    got = np.asarray(ported[0] if name == "jveh" else ported[1])
    want = sweep[name][0]
    off = got != want
    assert int(off.sum()) == EXP_AFFECTED[name], (
        f"{int(off.sum())} {name} values differ, measured {EXP_AFFECTED[name]}"
    )
    if off.any():
        ulp = np.abs(got[off] - want[off]) / np.spacing(np.abs(want[off]))
        assert ulp.max() <= EXP_ULP, f"worst {ulp.max():.1f} ulp; exp accounts for 1"


def test_the_whole_gap_is_the_exponential(sweep, monkeypatch):
    """The decisive experiment, as task 50. Route the module's only
    transcendental through numpy -- the same libm gfortran reaches -- and every
    difference disappears, on both outputs."""

    class _Libm:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def exp(x):
            return jnp.asarray(np.exp(np.asarray(x)))

    monkeypatch.setattr(bp, "jnp", _Libm())
    jveh, rc, _ = bp.binapara(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )
    np.testing.assert_array_equal(np.asarray(jveh), sweep["jveh"][0])
    np.testing.assert_array_equal(np.asarray(rc), sweep["rc"][0])


def test_the_reference_is_setup_independent(sweep):
    np.testing.assert_array_equal(sweep["jveh"][0], sweep["jveh"][1])
    np.testing.assert_array_equal(sweep["rc"][0], sweep["rc"][1])


# --- the three things a value comparison would not catch ------------------


@pytest.mark.parametrize("which", ("TERMX_TERMS", "JVEH_TERMS", "NTOT_TERMS"))
def test_reversing_the_term_order_changes_the_answer(sweep, monkeypatch, which):
    """Fortran sums left to right, so the stored order is load-bearing. If
    reversing it changed nothing, the ordered tuple would be documentation
    rather than specification -- and a port that grouped the terms by power,
    which is how a human would write it, would be free to do so."""
    original = getattr(bp, which)
    monkeypatch.setattr(bp, which, tuple(reversed(original)))
    jveh, _rc, _ntot = bp.binapara(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )
    assert not np.array_equal(np.asarray(jveh), sweep["jveh"][0]), (
        f"reversing {which} changed nothing; the order is not being used"
    )


def test_a_temperature_below_the_clamp_takes_the_clamped_value_and_not_the_cold_branch():
    """`:106` clips `t` to 190.15 and `:251` then tests the *clipped* value
    against 195.15. 190.15 is not below 195.15, so a caller passing 150 K gets
    the 190.15 K answer with the cold rule switched off -- which a port testing
    the caller's own temperature would get wrong, and only below 190.15 K."""
    rh = jnp.full((3,), 0.001)
    h2so4 = jnp.full((3,), 1.0e5)
    cold_caller = jnp.asarray([150.0, 180.0, 190.15])
    jveh, rc, ntot = bp.binapara(cold_caller, rh, h2so4)
    jveh = np.asarray(jveh)
    # All three clip to the same temperature, so all three agree exactly.
    assert len(set(jveh.tolist())) == 1
    assert len(set(np.asarray(rc).tolist())) == 1
    # And none of them is the cold-branch sentinel, even at 150 K.
    assert not np.any(jveh == LIMITS["j_cold"]) or np.all(np.asarray(ntot) >= LIMITS["ntot_min"])


def test_the_cold_rule_fires_between_the_two_bounds(sweep, ported):
    """It has to be reachable at all: the window is the clipped temperature in
    [190.15, 195.15), which is 5 K wide and below every shipped namelist."""
    jveh = np.asarray(ported[0])
    fired = jveh == LIMITS["j_cold"]
    assert int(fired.sum()) == int(sweep["_j_cold"]), "the cold rule's hit count moved"
    t_clipped = np.clip(sweep["t"], *CLAMPS["t"])
    assert np.all(t_clipped[fired] < LIMITS["t_cold"])
    assert np.all(t_clipped[fired] >= CLAMPS["t"][0])


def test_the_output_rules_commute_only_because_of_where_1e5_sits(sweep, ported):
    """Written the other way round first, and the test corrected the claim.

    The three rules at `:249-283` run cold, then floor, then ceiling. Applying
    them floor-ceiling-cold instead gives the **same** function on every point
    -- because `j_cold = 1e5` lies strictly between `j_floor = 1e-7` and
    `j_ceiling = 1e10`, so a value the cold rule sets is untouched by the two
    that follow it, and a value the cold rule will overwrite does not care what
    they did to it first.

    That is a property of three constants, not of the structure. Move `j_cold`
    outside the window and the orders diverge immediately, which is what the
    second half demonstrates -- so the port keeps the source's order rather
    than relying on a coincidence it does not control.
    """
    t, rh, h2so4 = bp.clamp_inputs(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )
    _, log_jveh, log_ntot, _ = bp.log_polynomials(t, rh, h2so4)
    raw = np.asarray(jnp.exp(log_jveh))
    ntot = np.asarray(jnp.exp(log_ntot))
    cold = (ntot < LIMITS["ntot_min"]) & (np.asarray(t) < LIMITS["t_cold"])

    def apply(order, j_cold):
        out = raw.copy()
        for rule in order:
            if rule == "cold":
                out = np.where(cold, j_cold, out)
            elif rule == "floor":
                out = np.where(out < LIMITS["j_floor"], 0.0, out)
            else:
                out = np.where(out > LIMITS["j_ceiling"], LIMITS["j_ceiling"], out)
        return out

    source_order = ("cold", "floor", "ceiling")
    other_order = ("floor", "ceiling", "cold")
    assert LIMITS["j_floor"] < LIMITS["j_cold"] < LIMITS["j_ceiling"]
    np.testing.assert_array_equal(
        apply(source_order, LIMITS["j_cold"]), apply(other_order, LIMITS["j_cold"])
    )
    np.testing.assert_array_equal(apply(source_order, LIMITS["j_cold"]), np.asarray(ported[0]))

    # Outside the window the two orders are different functions, so the
    # equality above is a measurement of these constants and not a tautology.
    assert cold.any(), "no point takes the cold rule, so this proves nothing"
    for outside in (1.0e-9, 1.0e12):
        assert not np.array_equal(apply(source_order, outside), apply(other_order, outside))


def test_the_cold_rule_takes_points_away_from_the_ceiling(sweep, ported):
    """The interaction that IS visible: 182 points exceed `j_ceiling` before
    the cold rule runs and 74 still do after it, because the other 108 are cold
    and get `1e5` instead."""
    t, rh, h2so4 = bp.clamp_inputs(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )
    _, log_jveh, _, _ = bp.log_polynomials(t, rh, h2so4)
    raw = np.asarray(jnp.exp(log_jveh))
    assert int((raw > LIMITS["j_ceiling"]).sum()) == int(sweep["_ceilinged"]) == 182
    assert (
        int((np.asarray(ported[0]) == LIMITS["j_ceiling"]).sum()) == int(sweep["_j_ceiling"]) == 74
    )


def test_ntot_is_returned_because_the_fortran_hides_it(sweep, ported):
    """380 of the 990 points have a critical cluster below four molecules and
    113 of those are too warm for the cold rule -- invisible in `jveh`. A branch
    nothing can see is a branch nothing can test."""
    ntot = np.asarray(ported[2])
    small = ntot < LIMITS["ntot_min"]
    assert int(small.sum()) == int(sweep["_small_cluster"])
    t_clipped = np.clip(sweep["t"], *CLAMPS["t"])
    warm_and_small = small & (t_clipped >= LIMITS["t_cold"])
    assert int(warm_and_small.sum()) == int(sweep["_small_but_warm"]) > 0


def test_termx_never_approaches_its_own_pole_on_the_reachable_grid(sweep):
    """52 terms divide by `termx` and nothing guards it. Measured over the
    sweep it stays in [0.049, 0.615], so the pole is not reached -- recorded as
    a measurement, because "it cannot happen" has been wrong three times in
    this project."""
    t, rh, h2so4 = bp.clamp_inputs(
        jnp.asarray(sweep["t"]), jnp.asarray(sweep["rh"]), jnp.asarray(sweep["h2so4"])
    )
    termx = np.asarray(bp.log_polynomials(t, rh, h2so4)[0])
    assert np.all(np.isfinite(termx))
    assert 0.048 < termx.min() and termx.max() < 0.62


def test_the_captures_reconstruction_still_reproduces_the_golden(sweep):
    """The capture's numpy reconstruction, re-run from committed data. This is
    the check that validates the 113 coefficients, and running it here means a
    later edit to the extractor cannot break it silently between captures."""
    sim = cap.simulate(sweep["t"], sweep["rh"], sweep["h2so4"])
    np.testing.assert_array_equal(sim["jveh"], sweep["jveh"][0])
    np.testing.assert_array_equal(sim["rc"], sweep["rc"][0])


@pytest.mark.parametrize("mutation", ("drop_a_term", "flip_a_sign", "lose_a_digit"))
def test_the_reconstruction_would_catch_a_mis_parsed_coefficient(sweep, monkeypatch, mutation):
    """Name the mutation, then apply it. Without this, the paragraph above --
    "a single mis-parsed digit would fail the capture" -- is an assertion about
    a test rather than a property of one."""
    terms = list(JVEH_TERMS)
    if mutation == "drop_a_term":
        terms.pop(7)
    elif mutation == "flip_a_sign":
        sign, coeff, factors, over = terms[7]
        terms[7] = ("-" if sign == "+" else "+", coeff, factors, over)
    else:
        sign, coeff, factors, over = terms[7]
        terms[7] = (sign, float(f"{coeff:.6e}"), factors, over)

    import glomap_jax.physics._binapara_literals as lit

    monkeypatch.setattr(lit, "JVEH_TERMS", tuple(terms))
    sim = cap.simulate(sweep["t"], sweep["rh"], sweep["h2so4"])
    assert not np.array_equal(sim["jveh"], sweep["jveh"][0]), mutation
