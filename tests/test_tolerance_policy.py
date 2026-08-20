"""Task 7 acceptance: the tolerance helpers behave as the policy describes."""

import numpy as np
import pytest

import conftest
from conftest import CROSS_PLATFORM_ULP, assert_close, assert_matches_reference, max_rel_err


def test_atol_scale_permits_a_physically_meaningless_difference_at_zero():
    # An unset budget slot or an inactive mode's water content is exactly zero
    # in one implementation and denormal-small in the other. Pure rtol fails.
    expected = np.array([1.0e3, 0.0])
    actual = np.array([1.0e3, 1.0e-300])
    with pytest.raises(AssertionError):
        assert_close(actual, expected, rtol=1e-13)
    assert_close(actual, expected, rtol=1e-13, atol_scale=1e-12)


def test_atol_scale_does_not_mask_a_real_discrepancy():
    expected = np.array([1.0e3, 1.0e3])
    actual = np.array([1.0e3, 1.1e3])
    with pytest.raises(AssertionError):
        assert_close(actual, expected, rtol=1e-13, atol_scale=1e-12)


def test_max_rel_err_ignores_entries_below_the_floor():
    expected = np.array([1.0, 1.0e-30])
    actual = np.array([1.0 + 1e-12, 5.0e-30])  # 400% error on a ~zero entry
    assert max_rel_err(actual, expected, abs_floor=1e-20) == pytest.approx(1e-12, rel=1e-3)


def test_shape_mismatch_is_reported_before_values():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close(np.zeros(3), np.zeros(4), rtol=1e-13)


# --- assert_matches_reference: the gate that decides when byte equality is fair ---
#
# Added after CI on ubuntu x86_64 failed six leaf comparisons by <=2 ulp against
# goldens captured with gfortran on Darwin arm64. The helper is now what stands
# between "the port is wrong" and "this is not the machine that built the
# reference", so it needs to be wrong in neither direction.


def _one_ulp_up(x):
    return np.nextafter(np.float64(x), np.inf)


def test_on_the_capture_platform_one_ulp_is_a_failure(monkeypatch):
    """The whole design rests on byte equality where it is achievable. If the
    strict path quietly accepted a ulp, every downstream tolerance would be
    resting on nothing."""
    monkeypatch.setattr(conftest, "on_capture_platform", lambda: True)
    expected = np.array([1.0, 2.0, 3.0])
    actual = expected.copy()
    actual[1] = _one_ulp_up(actual[1])
    with pytest.raises(AssertionError):
        assert_matches_reference(actual, expected, "one ulp")
    assert_matches_reference(expected.copy(), expected, "identical")


def test_off_the_capture_platform_the_window_is_bounded_and_real(monkeypatch):
    """Two ulp passes, three does not. A structural porting error is orders of
    magnitude out, so the relaxed path still catches everything it is meant to
    -- which is the only reason relaxing it is defensible."""
    monkeypatch.setattr(conftest, "on_capture_platform", lambda: False)
    expected = np.array([1.0, 2.0, 3.0])

    within = expected.copy()
    for _ in range(CROSS_PLATFORM_ULP):
        within[1] = _one_ulp_up(within[1])
    assert_matches_reference(within, expected, "at the bound")

    beyond = _one_ulp_up(within[1])
    outside = expected.copy()
    outside[1] = beyond
    with pytest.raises(AssertionError, match="more than 2 ulp"):
        assert_matches_reference(outside, expected, "past the bound")


def test_the_error_names_both_platforms(monkeypatch):
    """A bare `assert_array_equal` failure on CI cost real time to diagnose:
    six mismatches, no indication that the goldens came from another machine."""
    monkeypatch.setattr(conftest, "on_capture_platform", lambda: False)
    with pytest.raises(AssertionError, match="captured on"):
        assert_matches_reference(np.array([1.0]), np.array([2.0]), "far off")


def test_ulp_zero_stays_exact_even_off_the_capture_platform(monkeypatch):
    """`nint` and `vapour_round` return integers off a comparison. There is no
    rounding for a platform to disagree about, so a drift there is a bug
    wherever it appears."""
    monkeypatch.setattr(conftest, "on_capture_platform", lambda: False)
    expected = np.array([1.0, 2.0])
    actual = expected.copy()
    actual[0] = _one_ulp_up(actual[0])
    with pytest.raises(AssertionError):
        assert_matches_reference(actual, expected, "integer-valued", ulp=0)


def test_nan_matches_nan_rather_than_failing_the_window(monkeypatch):
    """`x ** (1.0/3.0)` is NaN for negative x and the golden records that. A
    comparison against NaN is false, so without this the relaxed path would
    report the faithful answer as a violation."""
    monkeypatch.setattr(conftest, "on_capture_platform", lambda: False)
    both = np.array([np.nan, 1.0])
    assert_matches_reference(both.copy(), both, "nan against nan")
    with pytest.raises(AssertionError):
        assert_matches_reference(np.array([np.nan, 1.0]), np.array([0.5, 1.0]), "nan against real")
