"""Task 7 acceptance: the tolerance helpers behave as the policy describes."""

import numpy as np
import pytest

from conftest import assert_close, max_rel_err


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
    actual = np.array([1.0 + 1e-12, 5.0e-30])   # 400% error on a ~zero entry
    assert max_rel_err(actual, expected, abs_floor=1e-20) == pytest.approx(1e-12, rel=1e-3)


def test_shape_mismatch_is_reported_before_values():
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_close(np.zeros(3), np.zeros(4), rtol=1e-13)
