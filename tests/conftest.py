"""Shared fixtures and the tolerance policy.

Tolerance policy (see docs/porting-notes.md). Loosening any of these to make a
test pass is a finding to investigate, not a knob to turn. Per-test overrides
are review-blocking.

The numbers are calibrated for THIS code, not inherited wholesale. Two
adjustments relative to the aer3d policy they are modelled on:

* aer3d's RTOL_TRAJECTORY = 1e-9 gates a **10 second** run, and aer3d already
  needed a 1000x looser constant for 100 s. A GLOMAP box run is 48 x 1800 s =
  24 hours with 15 substeps each, so the 24-hour case is a SOAK at RTOL_SOAK and
  the primary trajectory gate is a bounded number of steps from a golden state.
* aer3d's `atol_scale` floor is kept, and matters more here: GLOMAP `num_eps`
  values reach 1e-20, `bud_aer_mas` is mostly exact zeros, and `pvol`/`mdwat`
  are zero for inactive modes. A pure-relative comparison fails on 0-vs-1e-300.

Note what these tolerances CANNOT catch. This code's dominant failure mode is a
flipped branch, not precision drift: ~10 sites compare a computed float against
a threshold and select a different closed form. A flip gives an O(1) difference
between two correct float64 implementations, so it blows past any rtol. That is
what the branch-agreement gate (Gate 0) is for, and why it is the highest-value
check in the suite rather than a nicety.
"""

import numpy as np
import pytest

import glomap_jax

# Pure algebra: kernels, diameters, unit conversions.
RTOL_ALGEBRAIC = 1e-13
# exp/log/pow-heavy leaves: binapara polynomials, water-content fits.
RTOL_TRANSCENDENTAL = 1e-12
# A single process call against its Fortran input/output pair.
RTOL_STEP = 1e-11
# Bounded-step trajectory from a golden state. NOT the 24-hour run.
RTOL_TRAJECTORY = 1e-9
# The 24-hour, 48-step soak. Marked `slow`.
RTOL_SOAK = 1e-6
# The jit/scan driver must agree with the eager one to near machine precision;
# they are the same arithmetic in a different execution order.
RTOL_JIT_VS_EAGER = 1e-14


def pytest_sessionstart(session):
    """Fail the whole session immediately if float64 is not on.

    Every golden below assumes it. Discovering otherwise as a mysterious
    mismatch fifty tests later wastes an afternoon.
    """
    glomap_jax._assert_x64()


def assert_close(actual, expected, rtol, atol_scale=0.0, err_msg=""):
    """Compare with a relative tolerance and an optional absolute floor.

    ``atol_scale`` is a fraction of ``max|expected|``, not an absolute number, so
    the floor scales with the field being compared. Without it, quantities that
    are legitimately zero -- unset budget slots, water content of inactive modes
    -- fail on differences that are physically meaningless.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (
        f"shape mismatch: {actual.shape} vs {expected.shape}. {err_msg}"
    )
    atol = atol_scale * float(np.max(np.abs(expected))) if atol_scale else 0.0
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)


def max_rel_err(actual, expected, abs_floor=0.0):
    """Largest relative error, ignoring entries below ``abs_floor``.

    Entries that are effectively zero are excluded rather than allowed to
    dominate the statistic, which is what makes the reported number useful for
    deciding whether a discrepancy is real.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    mask = np.abs(expected) > abs_floor
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(actual[mask] - expected[mask]) / np.abs(expected[mask])))


@pytest.fixture(scope="session")
def goldens_dir():
    from pathlib import Path

    return Path(__file__).parent / "goldens"
