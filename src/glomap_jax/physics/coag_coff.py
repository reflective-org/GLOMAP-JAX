"""`ukca_coag_coff_v`: the Brownian coagulation coefficient of two modes.

Ported from `fortran/src/ukca/ukca_coag_coff_v.F90`, byte-equal to the compiled
routine on the capture platform over `tests/goldens/coag_coff.f64.leaf.npz`.

One output, `kij` in cm^3 s-1. `ukca_calc_coag_kernel` calls it once per mode
for the intra-modal `kii` and once per ordered mode pair for `kij`, and
`ukca_coagwithnucl` turns those into transfer rates.

Three methods, and they read different arguments
------------------------------------------------

`icoag` selects Jacobson's full transition-regime kernel (1, the only one any
shipped namelist runs), the HAM/M7 mean-radius approximation (2), or the
original UM sulphate scheme (3). They are not the same expression with
different coefficients -- each reads a different subset of the arguments:

| argument | `icoag = 1` | `icoag = 2` | `icoag = 3` |
|---|---|---|---|
| `ri`, `rj`, `dvisc`, `t` | read | read | read |
| `rhoi`, `rhoj` | read | read (via `rhomid`) | **not read** |
| `mfpa` | read | read | **not read** |
| `vi`, `vj` | read | **not read** | **not read** |

At `icoag = 3` five of the nine are dead, and that includes `mfpa`: `:316-318`
uses the local `UKCA_MFP_REF` PARAMETER, not the caller's mean free path. The
routine's header describes `MFP = MFPA = 6.6e-8*p0*T/(p*T0)` for this method,
which is not what it computes -- neither the scaling nor the argument is there.
The same header calls the Cunningham correction "1.59"; the code writes
`1.591`. Port from the code.

`icoag = 4` is not implemented
------------------------------

`:339-340` reads `mfppi` and `mfppj`, which are assigned only inside the
`IF (icoag == 1)` block at `:270` and `:281`. The four blocks are sequential
and not mutually exclusive, so `icoag = 4` means block 1 did not run and both
arrays are read never having been written. There is no correct reference, so
this raises rather than producing plausible garbage. UP-5; see
`docs/unsupported.md`.

`coag_on = 0` returns before the mask
-------------------------------------

`:239-242` sets `kij = 0` for **every** element and returns, without consulting
the mask. So "coagulation is switched off" and "this box is masked out" produce
the same zeros by different routes, and only the first zeroes a row the mask
selected. Reproduced exactly, because the driver passes an all-true mask and
the distinction is the only thing that makes `coag_on` visible.

Where the arithmetic had to be written a particular way
-------------------------------------------------------

* **The cubes are repeated multiplication.** `(dpi + mfppi)**3` and
  `rmid**3` are integer literal exponents, which gfortran expands to
  multiplications while `pow` need not agree. Phase C measured `d**3` against
  `d*d*d` at one ulp apart on two of eight modes.

* **`termv3` divides by `3.0` through `numerics.true_divide`.**
  `2.0e6*boltzmann*t/3.0/dvisc` (`:316`) divides an array by a scalar constant,
  the site XLA rewrites into a multiply by the reciprocal.

* **Every division is `numerics.safe_divide` under the mask**, for the reason
  `cond_coff` gives.

* **The Cunningham exponent keeps the Fortran's sign placement.**
  `EXP(-1.1/kni)` parses as `EXP(-(1.1/kni))` in Fortran and as
  `exp((-1.1)/kni)` here; negation is exact, so the two agree bit for bit, and
  the second spelling is the one `safe_divide` can guard.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.core import numerics
from glomap_jax.core.constants import BOLTZMANN, PI

#: `REAL, PARAMETER :: ukca_mfp_ref=6.6e-8` at `ukca_coag_coff_v.F90:227`, the
#: mean free path of air at the surface. Read only by `icoag = 3`, which uses
#: it *instead of* the `mfpa` argument.
UKCA_MFP_REF = 6.6e-8


def _vmid(rmid: Array) -> Array:
    """`(pi/0.75)*rmid(:)**3` at `:294`.

    The cube is a SEPARATE `powi` operand, so the prefactor multiplies the
    *finished* cube. Written `(PI/0.75) * rmid * rmid * rmid` it associates as
    `(((c*r)*r)*r)` instead, which differs by one ulp on 84 rows of the leaf
    grid and carries through to 14 rows of `kij`. That is what the byte-equality
    gate caught on the first comparison of task 49, and it is a function rather
    than an expression so the mutation stays applicable from a test.
    """
    return (PI / 0.75) * (rmid * rmid * rmid)


def _cunningham(kn: Array, mask: Array) -> Array:
    """`1.0 + kn*(1.257 + 0.4*EXP(-1.1/kn))`, the slip correction at `:266`.

    1.257, 0.4 and 1.1 are Seinfeld & Pandis pg. 465 following Allen & Raabe
    (1982); the source comments note the UM originally used 1.249, 0.42 and
    -0.87. Those are not a flag -- the older values appear nowhere in the live
    code, only in a comment at `:333-334`.
    """
    return 1.0 + kn * (1.257 + 0.4 * jnp.exp(numerics.safe_divide(-1.1, kn, mask)))


def coag_coff(
    mask: Array,
    ri: Array,
    rj: Array,
    vi: Array,
    vj: Array,
    rhoi: Array,
    rhoj: Array,
    mfpa: Array,
    dvisc: Array,
    t: Array,
    *,
    coag_on: int,
    icoag: int,
) -> Array:
    """`kij`, the coagulation coefficient for the I-J pair, in cm^3 s-1.

    Symmetric bit-for-bit under a simultaneous swap of `(ri, rj)`, `(vi, vj)`
    and `(rhoi, rhoj)`, at all three methods -- so the kernel cannot fix the
    `(imode, jmode)` convention and `ukca_calc_coag_kernel`'s subscripts are
    the only thing that does. See `tests/test_coag_coff_fixtures.py`.
    """
    if icoag == 4:
        raise ValueError(
            "icoag=4 is broken upstream (UP-5): ukca_coag_coff_v.F90:339-340 reads "
            "mfppi/mfppj, assigned only inside the mutually exclusive icoag==1 "
            "block, so it always reads undefined memory. There is no correct "
            "reference to validate against."
        )
    if icoag not in (1, 2, 3):
        raise ValueError(f"icoag={icoag} out of range (1-3 supported)")

    ri, rj = jnp.asarray(ri), jnp.asarray(rj)
    vi, vj = jnp.asarray(vi), jnp.asarray(vj)
    rhoi, rhoj = jnp.asarray(rhoi), jnp.asarray(rhoj)
    mfpa, dvisc, t = jnp.asarray(mfpa), jnp.asarray(dvisc), jnp.asarray(t)
    mask = jnp.asarray(mask)

    zero = jnp.zeros_like(ri)

    # `:239-242`. Before the mask, so every row is zero and not only the
    # masked ones.
    if coag_on == 0:
        return zero

    term1 = 8.0 * BOLTZMANN / PI
    term2 = BOLTZMANN / (6.0 * PI)
    term3 = 8.0 / PI
    term4 = 4.0e6 * PI
    term5 = 16.0e6 * PI

    if icoag == 1:
        # `:254-286`, Jacobson p.446 for the transition regime.
        dpi = 2.0 * ri
        veli = jnp.sqrt(numerics.safe_divide(term1 * t, rhoi * vi, mask))
        kni = numerics.safe_divide(mfpa, ri, mask)
        cci = _cunningham(kni, mask)
        dcoefi = numerics.safe_divide(term2 * cci * t, ri * dvisc, mask)
        mfppi = numerics.safe_divide(term3 * dcoefi, veli, mask)
        deli = _delta(dpi, mfppi, mask)

        dpj = 2.0 * rj
        velj = jnp.sqrt(numerics.safe_divide(term1 * t, rhoj * vj, mask))
        knj = numerics.safe_divide(mfpa, rj, mask)
        ccj = _cunningham(knj, mask)
        dcoefj = numerics.safe_divide(term2 * ccj * t, rj * dvisc, mask)
        mfppj = numerics.safe_divide(term3 * dcoefj, velj, mask)
        delj = _delta(dpj, mfppj, mask)

        rtot = ri + rj
        dtot = dcoefi + dcoefj
        termv1 = numerics.safe_divide(rtot, rtot + jnp.sqrt(deli * deli + delj * delj), mask)
        termv2 = numerics.safe_divide(
            numerics.safe_divide(4.0 * dtot, jnp.sqrt(veli * veli + velj * velj), mask),
            rtot,
            mask,
        )
        kij = numerics.safe_divide(term4 * rtot * dtot, termv1 + termv2, mask)

    elif icoag == 2:
        # `:292-304`, HAM/M7 as described in Stier et al (2005). `vmid` is
        # rebuilt from `rmid`, which is why `vi` and `vj` are dead here.
        rmid = 0.5 * (ri + rj)
        vmid = _vmid(rmid)
        rhomid = 0.5 * (rhoi + rhoj)
        vel = jnp.sqrt(numerics.safe_divide(term1 * t, rhomid * vmid, mask))
        kn = numerics.safe_divide(mfpa, rmid, mask)
        cc = _cunningham(kn, mask)
        dcoef = numerics.safe_divide(term2 * cc * t, rmid * dvisc, mask)
        termv1 = numerics.safe_divide(4.0 * dcoef, vel * rmid, mask)
        mfpp = numerics.safe_divide(2.0 * dcoef, PI * vel, mask)
        termv2 = numerics.safe_divide(rmid, rmid + mfpp, mask)
        kij = numerics.safe_divide(term5 * rmid * dcoef, termv1 + termv2, mask)

    else:
        # `:315-319`, the original UM sulphate scheme. `mfpa` is NOT read: the
        # local PARAMETER stands in for it, and the Cunningham correction is
        # the constant 1.591 rather than a function of size.
        termv3 = numerics.safe_divide(numerics.true_divide(2.0e6 * BOLTZMANN * t, 3.0), dvisc, mask)
        inv_ri = numerics.safe_divide(1.0, ri, mask)
        inv_rj = numerics.safe_divide(1.0, rj, mask)
        termv4 = (
            1.591
            * UKCA_MFP_REF
            * (numerics.safe_divide(inv_ri, ri, mask) + numerics.safe_divide(inv_rj, rj, mask))
        )
        kij = termv3 * (ri + rj) * (inv_ri + inv_rj + termv4)

    # `:238`: kij is zeroed before any branch, so a masked box carries 0.0.
    # Selected, never multiplied -- the masked fixture rows hold inf and NaN.
    return jnp.where(mask, kij, zero)


def _delta(dp: Array, mfpp: Array, mask: Array) -> Array:
    """`:271-273`, the mean distance from the sphere centre after a particle
    mean free path.

    Both `**3` are integer literal exponents, so both are repeated
    multiplication here: `pow` and `x*x*x` need not give the same double, and
    phase C measured them one ulp apart.
    """
    a = dp + mfpp
    b = dp * dp + mfpp * mfpp
    return numerics.safe_divide(a * a * a - jnp.sqrt(b * b * b), 3.0 * dp * mfpp, mask) - dp
