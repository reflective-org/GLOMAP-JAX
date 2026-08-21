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

## The cation labels disagree between the caller and the table

`ukca_volume_mode.F90:397` fills `cl(:,3)` from `cp_cl` and calls it Na, and
`:405` fills `cl(:,2)` from `cp_nh4`. The table labels cation row **2** Na and
row **3** NH4 (`ukca_water_content_v.F90:176-225`). Since the pair loop indexes
`y(ic,...)` with the same `ic` it reads `cli(:,ic)` from, one of the two is
wrong and one salt's concentrations are meeting the other's fit.

It is not cosmetic here. `cp_nh4 = 9` while `ncp = 6`, so `:402`'s guard
(`UBOUND(component,DIM=2) >= cp_no3`, i.e. `6 >= 7`) is false and `cl(:,2)` is
never assigned in any box setup. The only populated cation slots are 1 and 3 —
so sea salt, the dominant soluble mass, selects the row labelled NH4.

**The port takes no position.** It reproduces the raw index arithmetic and is
byte-equal to the Fortran, so it inherits whichever way this falls. No fidelity
flag, because a flag would imply we know which setting is right. Issue #24.

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
* **The powers come from `powi`, not from `**`.** gfortran expands an integer
  literal exponent through GCC's `powi` chain, and `jnp`'s `x**5` and `x**6`
  disagree with that chain on 35% and 55% of the live range. This was missed
  when the routine was first ported: task 40's grid swept `rh` at six values
  and none of them happened to be one of the disagreements, so the port passed
  a byte-equality gate it did not meet. `aw = 0.9` and `aw = 0.47` -- the
  humidity clamp and the Na+/Cl- floor -- are both disagreements, and
  `test_volume_mode.py` reaches them.
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

__all__ = ["ION_CHARGE", "NCOEFF", "PAIRS", "powi", "stoichiometry", "water_content"]

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


def powi(x: Array) -> tuple[Array, ...]:
    """`x**0 .. x**7` as gfortran expands an integer literal exponent.

    GCC lowers a constant integer exponent through its `powi` table, which is a
    specific chain of multiplications and **not** what any `pow` computes:

        p2 = x*x   p3 = x*p2   p4 = p2*p2   p5 = p2*p3   p6 = p3*p3   p7 = p3*p4

    Measured over 200,000 samples in [0.1, 0.95] against that chain: `jnp`'s
    `x**k` agrees for k in {0,1,2,3,4,7} and disagrees on **69,691 (34.8%)** of
    points for `x**5` and **109,311 (54.7%)** for `x**6`. `numpy`'s `**`
    disagrees for every k from 3 to 7.

    That gap is live, not theoretical. `aw = 0.9` -- the top of the box model's
    admissible humidity range and the value `corrh` clamps to -- is one of the
    `x**5` disagreements, and `aw = 0.47` -- the Na+/Cl- `rh_min` floor, which
    every sub-47% humidity is raised to -- is one of the `x**6` disagreements.
    Both reach `wc` at a relative 1.2e-11, which is 1e11 times the byte
    equality this port is gated on.

    `x**0` is 1.0 for every `x` including 0.0 in Fortran, so the chain starts
    from a literal one rather than emitting a `pow` that would NaN.
    """
    p2 = x * x
    p3 = x * p2
    p4 = p2 * p2
    p5 = p2 * p3
    p6 = p3 * p3
    p7 = p3 * p4
    return (jnp.ones_like(x), x, p2, p3, p4, p5, p6, p7)


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

        powers = powi(pair_aw)
        value = jnp.zeros_like(pair_aw)
        for k in range(NCOEFF):
            value = value + coefficients[row, col, k] * powers[k]
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
