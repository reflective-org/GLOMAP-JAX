"""Task 21: the numerics leaf sweep, and what it settles about task 34.

The plan called the transcendental compat layer "the sleeper" — an `erf`
discrepancy becomes a merge/no-merge flip in `ukca_remode`, and discovering it
inside phase I costs a day. This file replaces that worry with measurements
taken against the Fortran itself, over dense grids that land *on* each hazard
rather than near it.

The measurements, over 15,382 points:

| primitive | JAX vs gfortran | verdict |
|---|---|---|
| `erf` | bit-identical, 4330/4330 | the sleeper does not materialise |
| `log`, `1/x` | bit-identical | safe |
| `exp` | 456/3199 differ, max 2.1e-16 (1 ulp) | within tolerance, but real |
| `x ** (1/3)` | bit-identical | **this** is what `cubrt_v` computes |
| `np.cbrt` | 1756/1865 differ, max 1.3e-14 | must not be used |
| `NINT` | 64/642 differ — every tie | must not use `jnp.round` |

No `fortran` marker: the golden is committed, so these run in CI. That matters
more here than elsewhere — these are the assertions that stop someone reaching
for `jnp.round` or `jnp.cbrt` in phase D, months from now.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import RTOL_TRANSCENDENTAL

GOLDEN = Path(__file__).parent / "goldens" / "numerics.f64.leaf.npz"

PRIMITIVES = ["erf", "cubrt", "exp", "log", "oneover", "nint", "vapour_round"]


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run `make goldens` (or validation/capture_leaf.py)"
    return np.load(GOLDEN, allow_pickle=False)


# --------------------------------------------------------------------------
# The sweep itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", PRIMITIVES)
def test_every_primitive_was_swept(sweep, name):
    x, y = sweep[f"{name}_x"], sweep[f"{name}_y"]
    assert x.shape == y.shape
    assert len(x) > 400, f"{name}: {len(x)} points is not a dense grid"
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert (np.diff(x) > 0).all(), f"{name}: grid is not strictly increasing"


def test_the_erf_grid_lands_on_zero_not_merely_near_it(sweep):
    """`FRAC_N` is cut at 0.5, i.e. at `erf(x) == 0`. A grid that straddles zero
    without containing it would miss the only point that decides a merge."""
    x = sweep["erf_x"]
    assert 0.0 in x
    assert (np.abs(x[x != 0]).min()) <= 1e-300, "no approach to zero at full scale"
    assert sweep["erf_y"][x == 0].item() == 0.0


def test_the_exp_grid_lands_on_the_solvecoagnucl_clamp(sweep):
    """`SQD*DTZ > 50` is a hard clamp — a branch, not a large value — so both
    representable neighbours of 50 are swept."""
    x = sweep["exp_x"]
    assert 50.0 in x
    assert np.nextafter(50.0, 0.0) in x
    assert np.nextafter(50.0, 100.0) in x


def test_the_cubrt_grid_includes_exact_cubes(sweep):
    """Where an honest cube root returns an integer and a power form need
    not."""
    x = sweep["cubrt_x"]
    for k in (2, 5, 17, 64):
        assert float(k) ** 3 in x


def test_the_nint_grid_lands_on_every_tie_and_both_neighbours(sweep):
    x = sweep["nint_x"]
    for tie in (-2.5, -0.5, 0.5, 2.5, 63.5):
        assert tie in x
        assert np.nextafter(tie, 0.0) in x
        assert np.nextafter(tie, 1e9) in x


def test_the_vapour_grid_covers_the_whole_clamped_domain(sweep):
    """`ukca_vapour` clamps `wts` to [41, 99] before dividing by 5."""
    x = sweep["vapour_round_x"]
    assert x.min() <= 41.0 and x.max() >= 99.0
    assert 42.5 in x and 47.5 in x, "the ties the /5 produces"


# --------------------------------------------------------------------------
# What the compat layer (task 34) must and must not do
# --------------------------------------------------------------------------


def test_jax_erf_is_bit_identical_to_gfortran(sweep):
    """The plan's sleeper risk, measured and dismissed.

    `ukca_remode`'s `FRAC_N = 0.5*(1 + erf(x))` is cut at exactly 0.5, so a
    single-ulp disagreement here would be a merge/no-merge flip and an O(1)
    trajectory difference. There is no disagreement: 4330/4330 points agree
    bit for bit. The compat layer needs no erf shim."""
    got = np.asarray(jax.scipy.special.erf(jnp.asarray(sweep["erf_x"])))
    np.testing.assert_array_equal(got, sweep["erf_y"])


def test_jax_log_and_reciprocal_are_bit_identical_to_gfortran(sweep):
    np.testing.assert_array_equal(np.asarray(jnp.log(jnp.asarray(sweep["log_x"]))), sweep["log_y"])
    np.testing.assert_array_equal(
        np.asarray(1.0 / jnp.asarray(sweep["oneover_x"])), sweep["oneover_y"]
    )


def test_jax_exp_differs_from_gfortran_by_at_most_one_ulp(sweep):
    """The one primitive that is not bit-identical. 456 of 3199 points differ,
    all by a single ulp — XLA's exp is not the platform libm's.

    Well inside `RTOL_TRANSCENDENTAL`, so no shim is needed. Recorded rather
    than waved away because `exp` feeds `DELGC_COND` and the coagulation
    solver, and a 1-ulp difference is still enough to flip a comparison sitting
    exactly on a threshold. If a gate-0 branch disagreement ever traces back to
    an `exp`, this is the reason."""
    got = np.asarray(jnp.exp(jnp.asarray(sweep["exp_x"])))
    expected = sweep["exp_y"]
    differing = int((got != expected).sum())
    assert 0 < differing < len(expected) // 2, f"{differing} of {len(expected)} differ"
    np.testing.assert_allclose(got, expected, rtol=RTOL_TRANSCENDENTAL)
    worst = np.max(np.abs(got - expected) / np.abs(expected))
    assert worst < 1e-15, f"exp drifted beyond one ulp: {worst:.3e}"


def test_cbrt_must_be_written_as_x_to_the_one_third(sweep):
    """`cubrt_v` is literally `y = x ** (1.0/3.0)`. It is not a cube root
    function, and the two are not the same computation.

    `x ** (1.0/3.0)` in JAX reproduces the Fortran bit for bit. `np.cbrt`
    disagrees on 1756 of 1865 points, by up to 1.3e-14 — a hundred times
    `RTOL_ALGEBRAIC`. Since `cubrt_v` is what produces `drydp`, and `drydp`
    feeds `ukca_remode`'s merge threshold and `ukca_calc_drydiam`'s undersize
    reset, that is a branch-flipping difference and not a cosmetic one."""
    x = sweep["cubrt_x"]
    np.testing.assert_array_equal(np.asarray(jnp.asarray(x) ** (1.0 / 3.0)), sweep["cubrt_y"])
    assert not np.array_equal(np.cbrt(x), sweep["cubrt_y"]), (
        "np.cbrt now agrees with x**(1/3); re-measure before relaxing the rule"
    )


def test_cbrt_and_the_power_form_disagree_about_negative_inputs(sweep):
    """`x ** (1.0/3.0)` is a non-integer power of a negative and is NaN;
    `np.cbrt` returns the real root.

    Unreachable today — `dvol` is non-negative everywhere `cubrt_v` is called —
    so this is a latent difference, not a live one. It is pinned because it is
    the failure a `np.cbrt` port would produce the first time a negative volume
    appeared, and NaN-versus-a-number is much easier to diagnose when someone
    wrote down that it was expected."""
    assert np.isnan(sweep["cubrt_negative_y"]).all()
    assert np.isfinite(np.cbrt(sweep["cubrt_negative_x"])).all()


def test_round_half_to_even_is_wrong_for_this_code(sweep):
    """Fortran `NINT` rounds half away from zero; numpy and `jnp.round` round
    half to even. Every one of the 64 ties in the grid disagrees.

    `jnp.round` is the obvious thing to reach for and it is wrong here."""
    x, expected = sweep["nint_x"], sweep["nint_y"]
    for label, got in (
        ("np.round", np.round(x)),
        ("jnp.round", np.asarray(jnp.round(jnp.asarray(x)))),
    ):
        mismatched = got != expected
        assert mismatched.sum() == 64, f"{label}: {mismatched.sum()} mismatches, expected 64"
        assert np.all(np.abs(x[mismatched] % 1.0) == 0.5), (
            f"{label} disagrees somewhere other than an exact tie"
        )


def test_nint_away_from_ties_is_unambiguous(sweep):
    """Only the ties are contentious, which is what makes a targeted shim
    sufficient rather than needing a reimplementation of rounding."""
    x, expected = sweep["nint_x"], sweep["nint_y"]
    off_tie = np.abs(x % 1.0) != 0.5
    np.testing.assert_array_equal(np.round(x[off_tie]), expected[off_tie])


def test_the_vapour_lookup_index_is_wrong_under_round_half_to_even(sweep):
    """The live consumer: `ukca_vapour.F90:226` computes `(NINT(wts/5))*5` and
    uses the result to index a table, so a tie that rounds the other way
    selects a different table entry — not a slightly different number."""
    x, expected = sweep["vapour_round_x"], sweep["vapour_round_y"]
    naive = np.round(x / 5) * 5
    mismatched = naive != expected
    assert mismatched.sum() == 6
    assert set(x[mismatched]) == {42.5, 52.5, 62.5, 72.5, 82.5, 92.5}


def test_powr_v_takes_a_scalar_exponent(sweep):
    """Not elementwise pairs. An elementwise port would compile, run, and be a
    different routine."""
    exponents, x, y = sweep["pow_exponents"], sweep["pow_x"], sweep["pow_y"]
    assert y.shape == (len(exponents), len(x))
    tiny = np.finfo(np.float64).tiny
    for i, p in enumerate(exponents):
        got = np.asarray(jnp.asarray(x) ** p)
        # Subnormal results are excluded and tested separately below: JAX
        # flushes them to zero and gfortran does not.
        normal = np.abs(y[i]) >= tiny
        np.testing.assert_array_equal(got[normal], y[i][normal])


def test_jax_flushes_subnormal_results_to_zero_and_gfortran_does_not(sweep):
    """A hazard the plan did not have, found by the `**` sweep.

    JAX on this backend flushes the *result of any arithmetic operation* to
    zero when it would be subnormal — eager and under `jit`, and even for
    `x + 0.0`. gfortran and numpy compute it. A subnormal *constant* survives
    conversion untouched, which makes the behaviour easy to miss: the value is
    representable and round-trips, it just cannot be produced.

    **Latent, not live.** Nothing in GLOMAP is known to reach the float64
    subnormal range: `num_eps` bottoms out at 1e-20 and
    `eps_d = eps_ab**2 = 1e-40`, both comfortably normal in double precision.
    (They are *not* in single precision, which is one more reason the port is
    float64-only — see ADR-001.)

    Pinned anyway, because the failure it would produce is a zero where the
    reference has a number, which then flows into a `> eps` comparison and
    flips a branch. If a gate-0 disagreement ever appears with no other
    explanation, look here first. Issue #15."""
    tiny = np.finfo(np.float64).tiny
    subnormal = tiny / 2

    # A subnormal constant survives -- it is representable, and this is the
    # part that makes the behaviour easy to miss.
    assert float(jnp.float64(subnormal)) == subnormal

    # But nothing can produce one.
    assert float(jnp.float64(tiny) * 0.5) == 0.0
    assert float(jnp.float64(subnormal) + 0.0) == 0.0
    assert float(jax.jit(lambda v: v * 0.5)(jnp.float64(tiny))) == 0.0
    assert float(jnp.exp(jnp.float64(-745.0))) == 0.0

    assert np.float64(tiny) * 0.5 == subnormal, "numpy must still compute subnormals"
    assert np.exp(-745.0) > 0.0

    # And the reference does too, which is the half that matters.
    y = sweep["pow_y"]
    inverse = sweep["pow_exponents"].tolist().index(-1.0)
    produced = y[inverse][np.abs(y[inverse]) < tiny]
    assert (produced > 0).any(), "the sweep no longer produces a subnormal to compare"
