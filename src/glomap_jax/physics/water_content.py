"""`ukca_water_content_v` — ZSR water uptake (task 40).

Aerosol water content from ion concentrations and relative humidity, via
Jacobson's Table B.10 fits. Sixteen calls per chemistry step at `nmts = 1`,
which makes it the most frequently evaluated routine in phase D.

## It is loop-carried, twice, and the two carries are different

The pair loop runs `ic = 1..3` outer, `ia = -4..-1` inner, and iteration N
reads what N-1 wrote. Two distinct carried quantities:

**`cli`, in both arms.** `:262-265` takes the limiting ion pair concentration
and *subtracts it from both ion pools*:

    clp(ic,ia) = MIN(cli(ic)/n(ic), cli(ia)/n(ia))
    cli(ic)    = cli(ic) - n(ic)*clp(ic,ia)
    cli(ia)    = cli(ia) - n(ia)*clp(ic,ia)

So an ion exhausted by an early pair is unavailable to a later one. "Compute
all twelve pairs, then apply" is a different model, not a faster spelling of
this one.

**`aw`, in the unfixed arm only.** This is the whole of what
`l_fix_ukca_water_content` does beyond the one patched coefficient, and the two
arms are structurally different rather than differing in a constant:

* fixed (`:274`): `aw = rh` is re-read **inside** the loop, so each pair's
  `rh_min` floor applies to the original humidity.
* unfixed (`:300`): `aw = rh` is set **once before** the loop, so each pair's
  floor raises `aw` permanently for every pair after it. The floor ratchets,
  in pair order.

That is why the flag changes answers for compositions that never touch the
patched (1,-3) coefficient, and why a both-settings test needs a low-humidity
row as well as a nitrate row.

## The stoichiometry branch is dead

`:255-259` divides `n` through by the charges when `z(ic) == z(ia)` and
`z(ia) != 1`. With `z = [1,1,2,1,0,1,1,1]` over `-4..3` no cation has charge 2,
so the condition is never true. Reproduced anyway, and asserted dead in the
tests rather than dropped -- if a future `z` gains a divalent cation the branch
becomes live and the port should already be right.

## Faithful forms

* The polynomial is accumulated term by term in ascending order, `mb += y[k] *
  aw**k`, exactly as `:286-293` writes it. Not Horner, not `polyval`: both
  associate differently. `aw**0` is written and contributes `y[0]*1.0`.
* `MIN(mb, molal_max)` goes through `numerics.fortran_min` -- gfortran's `MIN`
  returns its first argument on a NaN second, `jnp.minimum` propagates.
* The ZSR sum `dum += clp/mb` is an ordered fold over pairs in the same
  `(ic, ia)` order.
* `wc = (1.0/18.0e-3)*dum`, written as `:332` writes it. Worth knowing that
  the distinction is inert here: XLA rewrites division by a compile-time
  constant into multiplication by its reciprocal, so `dum/18.0e-3` and
  `(1.0/18.0e-3)*dum` are the *same* computation in JAX, eager included --
  measured, 0 of 200,000 random values differ. In numpy they differ on 30%.
  The spelling is kept faithful anyway, because the next such pair may not be
  a compile-time constant.

## Full width, not compacted

The Fortran gathers the masked rows into `idx(:m)` and works on the prefix.
That is a memory optimisation with no numerical content: every operation is
elementwise per row. This port keeps full width and gates with the mask, which
is what `vmap` and `jit` need and what keeps the shapes static. Rows outside
the mask come back as zero, matching a caller that pre-zeroes `wc` -- which the
Fortran does *not* do, so `leaf_water_content` zeroes it in the driver instead.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from ..core import numerics
from . import water_tables as wt

__all__ = ["ION_CHARGE", "NCOEFF", "PAIRS", "stoichiometry", "water_content"]

# ukca_water_content_v.F90:132, DATA (z(i),i=-nanion,ncation).
# Index -4..3 maps to 0..7 through wt.ion_slot.
ION_CHARGE = (1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0)

NCOEFF = wt.NCOEFF

# The loop order the carries depend on: cation outer, anion inner.
PAIRS = tuple((c, a) for c in range(1, wt.NCATION + 1) for a in range(-wt.NANION, 0))

_EPS = float(jnp.finfo(jnp.float64).eps)


def stoichiometry(cation: int, anion: int) -> tuple[float, float]:
    """`(n_cation, n_anion)` for one pair (`:253-259`).

    Note the crossing: `n(ic) = z(ia)` and `n(ia) = z(ic)`. For (1,-2) that is
    `n(1) = z(-2) = 2` and `n(-2) = z(1) = 1`, so the limiting concentration is
    `MIN(cli(1)/2, cli(-2)/1)`. Reading it uncrossed swaps the divisors.
    """
    z_c = ION_CHARGE[wt.ion_slot(cation)]
    z_a = ION_CHARGE[wt.ion_slot(anion)]
    n_c, n_a = z_a, z_c
    if abs(z_c - z_a) < _EPS and abs(z_a - 1.0) > _EPS:
        # Dead for the shipped charges; see the module docstring.
        n_c, n_a = n_c / z_c, n_a / z_a
    return n_c, n_a


def _pair_concentrations(cl: Array, ions: Array, mask: Array):
    """The loop-carried pair scan (`:250-267`). Returns `(clp, present)`."""
    cli = jnp.asarray(cl, dtype=jnp.float64)
    clp: dict[tuple[int, int], Array] = {}
    present: dict[tuple[int, int], Array] = {}

    for cation, anion in PAIRS:
        ic, ia = wt.ion_slot(cation), wt.ion_slot(anion)
        n_c, n_a = stoichiometry(cation, anion)
        here = mask & ions[:, ia] & ions[:, ic]
        present[(cation, anion)] = here

        limiting = jnp.minimum(cli[:, ic] / n_c, cli[:, ia] / n_a)
        taken = jnp.where(here, limiting, 0.0)
        clp[(cation, anion)] = taken
        # Both pools are drawn down before the next pair sees them.
        cli = cli.at[:, ic].add(-n_c * taken)
        cli = cli.at[:, ia].add(-n_a * taken)

    return clp, present


def _molalities(rh: Array, present, *, fix_water_content: bool):
    """Binary electrolyte molalities (`:270-321`).

    The two arms differ in *where* `aw` is refreshed, which makes the unfixed
    arm's humidity floor cumulative. See the module docstring.
    """
    coefficients = wt.coefficients(fix_water_content)
    mb: dict[tuple[int, int], Array] = {}
    aw = jnp.asarray(rh, dtype=jnp.float64)  # the unfixed arm's carried value

    for cation, anion in PAIRS:
        row, col = wt.pair_index(cation, anion)
        floor = wt.LIMITS_TABLE[row, col, 0] / 1.0e2
        if fix_water_content:
            # Re-read the original humidity for this pair only.
            pair_aw = jnp.where(rh < floor, floor, rh)
        else:
            aw = jnp.where(aw < floor, floor, aw)
            pair_aw = aw

        value = jnp.zeros_like(pair_aw)
        for k in range(NCOEFF):
            value = value + coefficients[row, col, k] * pair_aw**k
        value = numerics.fortran_min(value, wt.LIMITS_TABLE[row, col, 1])
        mb[(cation, anion)] = jnp.where(present[(cation, anion)], value, 0.0)

    return mb


def water_content(cl: Array, rh: Array, ions: Array, mask: Array, *, fix_water_content: bool):
    """`wc`, in mol cm-3 of air. Rows outside `mask` come back zero."""
    clp, present = _pair_concentrations(cl, ions, mask)
    mb = _molalities(rh, present, fix_water_content=fix_water_content)

    dum = jnp.zeros(jnp.shape(rh), dtype=jnp.float64)
    for pair in PAIRS:
        here = present[pair]
        # Guarded so the masked-out lanes never divide by the zero `mb` holds
        # there; the Fortran's WHERE does the same by not evaluating them.
        safe = jnp.where(here, mb[pair], 1.0)
        dum = dum + jnp.where(here, clp[pair] / safe, 0.0)

    return jnp.where(mask, (1.0 / 18.0e-3) * dum, 0.0)
