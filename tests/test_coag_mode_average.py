"""`FidelityConfig.coag_mode_average`: the mode-averaged coagulation control.

A comparison control, not a port of any Fortran, so there is no golden. What is
tested is that the default is the Fortran, that the two settings differ, and
that the averaged coefficient is the integral it claims to be:

* the default call, the explicit `"native"` call and the Fortran path agree
  bit for bit, and `"integral"` differs on every filled slot -- mutation: make
  `"integral"` the default, or let it fall through to `coag_coff` unchanged;
* `"integral"` matches an independent trapezoid integral in ln r of the same
  `coag_coff` over both log-normal distributions -- mutations: drop the sqrt(2)
  in the node placement, use the mode's mean volume instead of each node's own
  sphere, or forget to normalise the weights by sqrt(pi);
* 48 nodes agree with 96 -- mutation: a node count too low for sigma = 2
  (32 nodes fail it: 4.9e-10 against 128);
* in the narrow-mode limit it collapses to the single-particle coefficient
  evaluated with the median sphere's volume -- mutation: un-normalised weights;
* at a 10 nm, sigma = 1.59 mode the native/integral ratio is the 0.42 the
  intercomparison paper reports (Fig. S3) -- mutation: integral == native.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from conftest import RTOL_QUADRATURE
from glomap_jax.physics.coag_coff import coag_coff
from glomap_jax.physics.coag_kernel import calc_coag_kernel
from glomap_jax.physics.modes import NMODES

# 220 K, 50 hPa sulfate-like inputs; any physical values would do.
T, MFPA, DVISC, RHO = 220.0, 1.0e-6, 1.44e-5, 1769.0
SIGMAG = np.array([1.59, 1.59, 1.4, 2.0, 1.59, 1.59, 2.0, 1.8])
MODE = np.array([1, 1, 1, 1, 0, 0, 0, 0])
MODESOL = np.array([1, 1, 1, 1, 0, 0, 0, 0])
DP0 = np.array([10e-9, 40e-9, 200e-9, 2e-6, 20e-9, 100e-9, 1e-6, 2e-6])


def _inputs(sigmag=SIGMAG, mean_volume=True):
    dp = np.stack([DP0, 1.5 * DP0])                         # two boxes
    factor = np.exp(4.5 * np.log(sigmag) ** 2) if mean_volume else 1.0
    vol = np.pi / 6.0 * dp**3 * factor
    rho = np.full_like(dp, RHO)
    env = [np.full(2, MFPA), np.full(2, DVISC), np.full(2, T)]
    return [MODE, MODESOL, dp, vol, dp, vol, rho, *env]


def _kernel(args, **kw):
    kii, kij = calc_coag_kernel(*args, coag_on=1, icoag=1, **kw)
    return np.asarray(kii), np.asarray(kij)


def _trapezoid(ri, rj, si, sj, n=801, span=8.0):
    """Independent <K>: trapezoid in ln r over +-span sigma of each mode."""
    zi = np.linspace(-span, span, n)
    pdf = np.exp(-0.5 * zi**2) / np.sqrt(2.0 * np.pi)
    a = ri * np.exp(zi * np.log(si))
    b = rj * np.exp(zi * np.log(sj))
    A, B = np.meshgrid(a, b, indexing="ij")
    k = np.asarray(coag_coff(np.ones(A.shape, bool), A, B, np.pi / 0.75 * A**3,
                             np.pi / 0.75 * B**3, np.full(A.shape, RHO), np.full(A.shape, RHO),
                             np.full(A.shape, MFPA), np.full(A.shape, DVISC), np.full(A.shape, T),
                             coag_on=1, icoag=1))
    w = np.outer(pdf, pdf) * (zi[1] - zi[0]) ** 2
    return float(np.sum(k * w))


def test_default_is_native_and_integral_differs():
    args = _inputs()
    kii0, kij0 = _kernel(args)
    kii1, kij1 = _kernel(args, mode_average="native", sigmag=SIGMAG)
    np.testing.assert_array_equal(kii0, kii1)
    np.testing.assert_array_equal(kij0, kij1)
    kii2, kij2 = _kernel(args, mode_average="integral", sigmag=SIGMAG)
    for native, averaged in ((kii0, kii2), (kij0, kij2)):
        filled = native != 0
        np.testing.assert_array_equal(filled, averaged != 0)  # same slots
        assert np.all(np.abs(averaged[filled] / native[filled] - 1) > 0.01)


@pytest.mark.parametrize("i,j", [(1, 1), (0, 2), (2, 3)])
def test_integral_matches_independent_trapezoid(i, j):
    kii, kij = _kernel(_inputs(), mode_average="integral", sigmag=SIGMAG)
    got = kii[0, i] if i == j else kij[0, i, j]
    want = _trapezoid(DP0[i] / 2, DP0[j] / 2, SIGMAG[i], SIGMAG[j])
    np.testing.assert_allclose(got, want, rtol=RTOL_QUADRATURE, atol=0)


def test_default_nodes_agree_with_twice_as_many():
    args = _inputs()
    kii_a, kij_a = _kernel(args, mode_average="integral", sigmag=SIGMAG)
    kii_b, kij_b = _kernel(args, mode_average="integral", sigmag=SIGMAG, nodes=96)
    np.testing.assert_allclose(kii_a, kii_b, rtol=RTOL_QUADRATURE, atol=0)
    np.testing.assert_allclose(kij_a, kij_b, rtol=RTOL_QUADRATURE, atol=0)


def test_narrow_modes_collapse_to_the_single_particle():
    narrow = np.full(NMODES, 1.0 + 1e-7)
    args = _inputs(narrow, mean_volume=False)
    kii0, kij0 = _kernel(args)
    kii1, kij1 = _kernel(args, mode_average="integral", sigmag=narrow)
    np.testing.assert_allclose(kii1, kii0, rtol=1e-9, atol=0)
    np.testing.assert_allclose(kij1, kij0, rtol=1e-9, atol=0)


def test_native_over_integral_matches_the_paper_at_10_nm():
    args = _inputs()
    kii0, _ = _kernel(args)
    kii1, _ = _kernel(args, mode_average="integral", sigmag=SIGMAG)
    assert 0.40 < kii0[0, 0] / kii1[0, 0] < 0.45


def test_bad_arguments_raise():
    args = _inputs()
    with pytest.raises(ValueError):
        _kernel(args, mode_average="mean")
    with pytest.raises(ValueError):
        _kernel(args, mode_average="integral")
