"""The first ported code, and its acceptance: `glomap_jax.core.numerics`.

Every assertion here compares the JAX implementation against the **committed
Fortran golden**, not against a hand-written expectation. That is the whole
point of having spent phases A and B on a harness: the reference is on disk, so
"does the port match" is a measurement rather than an argument.

`tests/test_numerics_reference.py` is the neighbouring file and answers a
different question — it checks what *raw JAX* does against the golden, which is
how the rules below were discovered. This file checks that the port applies
them.

No `fortran` marker: the golden is committed, so this runs in CI.
"""

from pathlib import Path

import jax
import numpy as np
import pytest

from conftest import CROSS_PLATFORM_ULP, RTOL_TRANSCENDENTAL, assert_matches_reference
from glomap_jax.config import FidelityConfig
from glomap_jax.core import numerics

GOLDEN = Path(__file__).parent / "goldens" / "numerics.f64.leaf.npz"


@pytest.fixture(scope="module")
def sweep():
    assert GOLDEN.is_file(), "run `make goldens`"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.mark.parametrize(
    ("fn", "key"),
    [
        ("erf", "erf"),
        ("cbrt", "cubrt"),
        ("nint", "nint"),
        ("vapour_round", "vapour_round"),
    ],
)
def test_matches_the_fortran_bit_for_bit(sweep, fn, key):
    """Bit-identical, not `allclose`. These are the primitives every later phase
    is built on; a one-ulp drift here becomes a flipped branch downstream, and
    accepting it now would make every subsequent tolerance meaningless.

    `nint` and `vapour_round` are held exact on every platform: they are integer
    results off a comparison, with no rounding to disagree about. `erf` and
    `cbrt` are held exact only where the goldens were captured -- see
    `assert_matches_reference`, and the `linux-reference` CI job that
    re-establishes the strong claim on x86_64 by building gfortran there.
    """
    got = np.asarray(getattr(numerics, fn)(sweep[f"{key}_x"]))
    exact_everywhere = 0 if fn in ("nint", "vapour_round") else CROSS_PLATFORM_ULP
    assert_matches_reference(got, sweep[f"{key}_y"], f"{fn} vs {key}", ulp=exact_everywhere)


def test_nint_survives_the_double_just_below_a_half(sweep):
    """The bug the golden caught on the first attempt.

    `sign(x) * floor(|x| + 0.5)` is the obvious formulation and is wrong at
    exactly two of the 642 swept points: for `x = ±0.49999999999999994`,
    `|x| + 0.5` rounds *up* to exactly 1.0, giving ±1 where Fortran gives 0.

    Pinned separately from the sweep so that if someone simplifies `nint` back
    to the obvious form, the failure names the reason instead of reporting two
    anonymous mismatches."""
    edge = np.array([-0.49999999999999994, 0.49999999999999994])
    assert np.abs(edge[0]) + 0.5 == 1.0, "the hazard is gone; re-derive this test"
    np.testing.assert_array_equal(np.asarray(numerics.nint(edge)), [0.0, 0.0])


def test_nint_rounds_halves_away_from_zero_not_to_even():
    ties = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(
        np.asarray(numerics.nint(ties)), [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    )
    assert not np.array_equal(np.round(ties), np.asarray(numerics.nint(ties)))


def test_cbrt_exact_is_a_different_function_and_is_not_the_default(sweep):
    """`FidelityConfig.cbrt_exact` defaults to reproducing the Fortran.

    Asserted rather than assumed because the flag is the one place in the
    registry where the non-default setting is *more* numerically correct, which
    makes it the one most likely to be flipped by someone trying to help."""
    assert FidelityConfig().cbrt_exact is False

    faithful = np.asarray(numerics.cbrt(sweep["cubrt_x"], exact=False))
    assert_matches_reference(faithful, sweep["cubrt_y"], "cbrt(exact=False)")


def test_how_far_jnp_cbrt_is_from_the_faithful_form_on_this_jax(sweep):
    """The measurement behind the rule, kept separate because it is not a gate.

    On jax 0.11.0 the two disagree on 94% of the swept grid by up to 1.3e-14.
    On jax 0.9.2 they were observed to agree bit for bit on every point — so
    the size of the gap is a property of the installed XLA, not of this port,
    and asserting it would make the suite red on a version where `jnp.cbrt`
    happens to lower to the same thing.

    Nothing here is load-bearing: fidelity is pinned by the test above, which
    compares the default against the Fortran and does not mention `jnp.cbrt`.
    What this records is *why* the default is the power form, and it reports
    the number rather than hiding it behind a threshold.
    """
    x = sweep["cubrt_x"]
    faithful = np.asarray(numerics.cbrt(x, exact=False))
    exact = np.asarray(numerics.cbrt(x, exact=True))
    differ = faithful != exact
    if differ.any():
        rel = np.abs((faithful[differ] - exact[differ]) / exact[differ])
        tail = f", max relative {np.max(rel):.2e}"
    else:
        tail = " (none -- they agree bit for bit here)"
    print(
        f"\njax {jax.__version__}: jnp.cbrt differs from x**(1/3) on "
        f"{differ.sum()}/{len(x)} points{tail}"
    )


@pytest.mark.filterwarnings("ignore:invalid value encountered in power")
def test_cbrt_disagrees_about_negatives_in_the_direction_that_matters(sweep):
    """The power form is NaN, `jnp.cbrt` returns the real root. `cbrt` is more
    correct, which is exactly why it cannot be the faithful path.

    The `invalid value in power` warning is the point, not a nuisance: it is
    numpy reporting the same thing gfortran reports by producing NaN."""
    negatives = sweep["cubrt_negative_x"]
    assert np.isnan(np.asarray(numerics.cbrt(negatives, exact=False))).all()
    assert np.isfinite(np.asarray(numerics.cbrt(negatives, exact=True))).all()
    np.testing.assert_array_equal(
        np.asarray(numerics.cbrt(negatives, exact=False)), sweep["cubrt_negative_y"]
    )


def test_vapour_round_selects_the_same_table_entry_as_the_fortran(sweep):
    """`ukca_vapour.F90:226` uses this to index a lookup table, so a tie that
    rounds the other way picks a different entry — not a nearby number."""
    x = sweep["vapour_round_x"]
    naive = np.round(x / 5) * 5
    ours = np.asarray(numerics.vapour_round(x))
    np.testing.assert_array_equal(ours, sweep["vapour_round_y"])
    disagree = naive != sweep["vapour_round_y"]
    assert set(x[disagree]) == {42.5, 52.5, 62.5, 72.5, 82.5, 92.5}
    assert (ours[disagree] == sweep["vapour_round_y"][disagree]).all()


def test_safe_divide_masks_without_poisoning_the_gradient():
    """Single-`where` division gives a NaN cotangent: reverse mode differentiates
    the branch that was not taken. Order 2 needs this to be finite."""
    import jax
    import jax.numpy as jnp

    num = jnp.array([1.0, 2.0])
    den = jnp.array([2.0, 0.0])
    mask = jnp.array([True, False])

    value = numerics.safe_divide(num, den, mask)
    np.testing.assert_allclose(np.asarray(value), [0.5, 0.0], rtol=RTOL_TRANSCENDENTAL)

    grad = jax.grad(lambda d: numerics.safe_divide(num, d, mask).sum())(den)
    assert np.isfinite(np.asarray(grad)).all(), f"NaN cotangent: {grad}"

    naive = jax.grad(lambda d: jnp.where(mask, num / d, 0.0).sum())(den)
    assert not np.isfinite(np.asarray(naive)).all(), (
        "the single-where form no longer produces a NaN cotangent; if JAX "
        "changed this, safe_divide may no longer be necessary"
    )


def test_masked_sum_excludes_rather_than_multiplies():
    """`0.0 * inf` is `NaN`, and `ratio1 = mm / (avogadro * rhocomp)` is
    evaluated over the full component extent including padding.

    Note the hazard needs a *numeric* mask to bite: a boolean mask multiplies as
    `False * inf -> 0.0` under JAX's promotion rules, so the naive form only
    fails once someone writes `mask.astype(float) * term` — which is exactly
    what a reader reaching for a weighted sum tends to write."""
    import jax.numpy as jnp

    term = jnp.array([1.0, jnp.inf, 3.0])
    mask = jnp.array([True, False, True])
    assert float(numerics.masked_sum(term, mask)) == 4.0

    naive = jnp.sum(mask.astype(jnp.float64) * term)
    assert np.isnan(float(naive)), "the hazard this guards against is gone"
