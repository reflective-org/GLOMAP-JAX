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

import json
import platform
from pathlib import Path

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
    return Path(__file__).parent / "goldens"


# --- Reference bit-equality, and where it stops being a property of the port ---
#
# The leaf sweeps assert that a JAX primitive reproduces gfortran's result bit
# for bit. That is true, and it is what lets later phases gate on byte equality
# rather than tolerance -- but it is a property of a *platform pair*, not of the
# port, and the project only ever measured it on one.
#
# CI proved the point: on ubuntu x86_64 the committed goldens, captured with
# gfortran on Darwin arm64, disagree with JAX by up to 2 ulp -- `erf` on 35% of
# its grid, `x**(1.0/3.0)` on 4.6% of its. The same tests are green on macOS.
# The job that failed has no gfortran, so it could not have re-derived the
# reference; it was comparing this platform's JAX against another platform's
# Fortran, which the manifest's own docstring already says is not a valid
# comparison. Nothing acted on that until now.
#
# So: bit equality is required where the goldens were captured, and a bounded
# ulp gap is asserted everywhere else. The bound still catches a real porting
# error -- anything structural is orders of magnitude out, not two ulp. The
# strong claim is re-established per platform by the `linux-reference` CI job,
# which builds gfortran there and re-captures before comparing.

CROSS_PLATFORM_ULP = 2

# Per-primitive bounds, each one measured rather than chosen. Darwin arm64
# (Homebrew gfortran 16.1.0, the capture platform) against ubuntu x86_64:
#
#   erf              2 of 4330 points past 2 ulp, worst 4 ulp at erf(x) = 0.4928
#                    -- mid-range, not near zero, so it is glibc's erf against
#                    Apple's and not a cancellation artefact
#   x ** (1.0/3.0)   86 of 1865 differ, none past 1 ulp
#   x ** p           1 of 1865 differs, 1 ulp
#   log, 1/x         identical
#
# Exceeding one of these means look, not bump. The point of a measured bound is
# that it moves only when someone re-measures and writes down what they found.
CROSS_PLATFORM_ULP_BY_PRIMITIVE = {"erf": 4}

_MANIFEST = Path(__file__).parent / "goldens" / "MANIFEST.json"


def capture_platform() -> str | None:
    """`uname -srm` of the machine that compiled the reference, or None.

    Recorded by `validation/build_reference.sh` into `fortran/TOOLCHAIN.txt` and
    copied into the manifest at capture time.
    """
    if not _MANIFEST.is_file():
        return None
    return json.loads(_MANIFEST.read_text(encoding="utf-8")).get("toolchain", {}).get("uname")


def on_capture_platform() -> bool:
    """Whether bit equality with the committed goldens is a fair thing to ask.

    Compares OS and machine, deliberately not the kernel version: a point
    release does not change libm or XLA lowering, and requiring an exact match
    would downgrade every developer's gate the next time they update.
    """
    recorded = capture_platform()
    if not recorded:
        return False
    fields = recorded.split()
    return len(fields) == 3 and fields[0] == platform.system() and fields[2] == platform.machine()


def _ulp_window(expected, n):
    lo = hi = np.asarray(expected, dtype=np.float64)
    for _ in range(n):
        lo = np.nextafter(lo, -np.inf)
        hi = np.nextafter(hi, np.inf)
    return lo, hi


def assert_matches_reference(actual, expected, what, ulp=CROSS_PLATFORM_ULP):
    """Bit-identical on the capture platform; within `ulp` elsewhere.

    Pass `ulp=0` for a quantity that must be exact on every platform -- integer
    results, `NINT`, table lookups. Those have no rounding to disagree about,
    and letting them drift would hide a real bug.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if ulp == 0 or on_capture_platform():
        np.testing.assert_array_equal(actual, expected, err_msg=what)
        return

    both_nan = np.isnan(actual) & np.isnan(expected)
    lo, hi = _ulp_window(expected, ulp)
    within = ((actual >= lo) & (actual <= hi)) | both_nan
    if within.all():
        return

    off = ~within
    gap = np.abs(actual[off] - expected[off]) / np.spacing(np.abs(expected[off]))
    worst = np.argmax(gap)
    raise AssertionError(
        f"{what}: {off.sum()} of {off.size} points differ from the reference by "
        f"more than {ulp} ulp.\nThe goldens were captured on "
        f"{capture_platform()!r} and this is "
        f"{platform.system()} {platform.machine()}, so a small gap is expected "
        f"and a large one is a porting error.\n"
        f"worst: {gap[worst]:.1f} ulp, expected {expected[off][worst]!r}, "
        f"got {actual[off][worst]!r}\n"
        f"all violations, in ulp: {np.sort(gap)[::-1][:12]}"
    )
