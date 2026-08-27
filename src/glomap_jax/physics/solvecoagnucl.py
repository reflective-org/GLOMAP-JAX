"""`ukca_solvecoagnucl_v`: the analytic solution of `dN/dt = A·N² + B·N + C`.

Ported from `fortran/src/ukca/ukca_solvecoagnucl_v.F90`. `A` is intra-modal
coagulation, `B` inter-modal, `C` new particle formation, and the routine picks
between five closed forms by the sign of `A` and of the discriminant
`D = 4AC - B²`.

Nine branches, not eight
------------------------

The header documents six cases and the branch dump names eight codes. There is
a ninth, and this port is where it is written down. `:198-199`::

    logic1 = mask .AND. ((a > eps_ab) .OR. (a < -eps_ab))
    logic2 = mask .AND. ((a < eps_ab) .AND. (a > -eps_ab))

Both tests are **strict** and both use `eps_ab`, so `|A| == eps_ab` **exactly**
satisfies neither. Those rows take no closed form: `ndnew` keeps the `nd` it
was initialised with at `:190` and `deln` comes back exactly zero, with no
`ierr` and no diagnostic. `logic3` is written `<=`/`>=`, so `B` has no such
gap. Issue #31, found by the leaf capture rather than by reading the source.

That is reproduced rather than flagged. A fidelity flag would imply there is an
alternative worth running; there is not — the gap is a shape of the input
space, and the port simply takes no branch there.

Issue #13 is why the fixture is constructed
-------------------------------------------

Four of the eight documented codes are unreachable from any trajectory, because
a trajectory does not choose `A`, `B` and `C`. `1a_term4` in particular needs
`EXP(sqd·dtz)/term3` to round to exactly `1.0`, which cannot be sampled — the
capture solves for it.

Two upstream quirks
-------------------

* **UP-1, `coag_intra_factor3`.** `:259` writes `1/(1/nd - 3·A·dtz)` where the
  analytic solution of `dN/dt = A·N²` is `1/(1/N₀ - A·t)`. Gate 0 established
  this is the branch the top soluble mode takes every substep of every shipped
  namelist, because it has no larger mode to coagulate into and no nucleation,
  so `B = C = 0` and `D = 0`.
* **UP-2**, documentation only: the header swaps arctan and log relative to
  what the code computes. `1a` (`D < 0`) uses the exponential form and `1b`
  (`D > 0`) the arctan/tan one; the header says the opposite.

The clamp is silent, and this port says so
------------------------------------------

`:225-228` caps `sqd·dtz` at 50 because `EXP` overflows, with no diagnostic.
CLAUDE.md requires caps to be surfaced, so `solvecoagnucl` returns the clamp
mask alongside `deln` — the Fortran cannot, and a caller that wants to know how
often the solver saturated has nowhere else to look.

The error case is fatal upstream
--------------------------------

`logic1ca` (`A ≠ 0`, `D == 0`, `B ≠ 0`) sets `ierr = 1` and `:291` calls
`ereport`, which does `STOP 1`. Reproduced as a returned mask rather than an
exception: this is array code, and one poisoned row must not take the whole
column with it. The caller decides.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core import numerics
from glomap_jax.core.constants import EPS_AB, EPS_D, SQD_CLAMP

__all__ = ["SolveResult", "solvecoagnucl"]


class SolveResult(NamedTuple):
    """`deln`, plus the two things the Fortran computes and cannot return.

    `clamped` and `failed` exist because CLAUDE.md requires a cap to be
    surfaced and a failure to be loud. The Fortran surfaces the first not at
    all and the second by stopping the process.
    """

    deln: Array
    clamped: Array
    failed: Array


def solvecoagnucl(
    mask: Array,
    a: Array,
    b: Array,
    c: Array,
    nd: Array,
    *,
    dtz: float,
    fidelity: FidelityConfig = FidelityConfig(),
) -> SolveResult:
    """`dN` over one `dtz`, for every row of a masked column."""
    mask = jnp.asarray(mask)
    a, b, c, nd = (jnp.asarray(x) for x in (a, b, c, nd))

    ndnew = nd  # `:190`, initialised to nd and NOT to zero.
    d = 4.0 * a * c - b * b

    # `:198-200`. Written exactly as the Fortran writes them, strict tests and
    # all, so that `|a| == eps_ab` falls through both -- issue #31.
    logic1 = mask & ((a > EPS_AB) | (a < -EPS_AB))
    logic2 = mask & ((a < EPS_AB) & (a > -EPS_AB))
    logic3 = (b <= EPS_AB) & (b >= -EPS_AB)

    logic1a = logic1 & (d < -EPS_D)
    logic1b = logic1 & (d > EPS_D)
    logic1c = logic1 & (d <= EPS_D) & (d >= -EPS_D)
    logic1ca = logic1c & ~logic3
    logic1cb = logic1c & logic3
    logic2a = logic2 & ~logic3
    logic2b = logic2 & logic3

    # --- A != 0, D < 0: `:216-247` ---
    sqd_neg = jnp.sqrt(jnp.where(logic1a, -d, 1.0))
    term1 = 2.0 * a * nd + b + sqd_neg
    term2 = 2.0 * a * nd + b - sqd_neg
    term3 = jnp.where(logic1a, numerics.safe_divide(term1, term2, logic1a), 0.0)
    logic1aok = logic1a & ((term3 > EPS_AB) | (term3 < -EPS_AB))

    # `:225-228`. Silent upstream; surfaced here.
    sqd_tms_dtz = sqd_neg * dtz
    clamped = logic1aok & (sqd_tms_dtz > SQD_CLAMP)
    sqd_tms_dtz = jnp.where(clamped, SQD_CLAMP, sqd_tms_dtz)
    term4 = 1.0 - numerics.safe_divide(jnp.exp(sqd_tms_dtz), term3, logic1aok)
    logic1aok2 = logic1aok & ((term4 > EPS_AB) | (term4 < -EPS_AB))

    # `(2*sqd/term4 - b - sqd)/2/a`, in the Fortran's own association: the two
    # divisions are separate, not folded into `/(2*a)`.
    solved = numerics.safe_divide(
        numerics.safe_divide(
            numerics.safe_divide(2.0 * sqd_neg, term4, logic1aok2) - b - sqd_neg,
            2.0,
            logic1aok2,
        ),
        a,
        logic1aok2,
    )
    ndnew = jnp.where(logic1aok2, solved, ndnew)

    # --- A != 0, D > 0: `:250-255` ---
    sqd_pos = jnp.sqrt(jnp.where(logic1b, d, 1.0))
    t1 = numerics.safe_divide(2.0 * a * nd + b, sqd_pos, logic1b)
    t2 = jnp.arctan(t1) + numerics.safe_divide(sqd_pos * dtz, 2.0, logic1b)
    branch_b = numerics.safe_divide(
        numerics.safe_divide(sqd_pos * jnp.tan(t2) - b, 2.0, logic1b), a, logic1b
    )
    ndnew = jnp.where(logic1b, branch_b, ndnew)

    # --- A != 0, D == 0, B == 0: `:259`. UP-1's spurious factor 3. ---
    factor = 3.0 if fidelity.coag_intra_factor3 else 1.0
    inv = numerics.safe_divide(jnp.ones_like(nd), nd, logic1cb)
    branch_cb = numerics.safe_divide(jnp.ones_like(nd), inv - factor * a * dtz, logic1cb)
    ndnew = jnp.where(logic1cb, branch_cb, ndnew)

    # --- A == 0, B != 0: `:262-266` ---
    ex = jnp.exp(b * dtz)
    branch_2a = numerics.safe_divide(ex * (b * nd + c) - c, b, logic2a)
    ndnew = jnp.where(logic2a, branch_2a, ndnew)

    # --- A == 0, B == 0: `:270`. Pure nucleation. ---
    ndnew = jnp.where(logic2b, nd + c * dtz, ndnew)

    # `:274-275`. `deln` is the DIFFERENCE, and the cancellation is part of the
    # closed form: at nd = 1e3 and c*dtz = 1e-4 the 2b arm gives
    # 9.999999997489795e-05 and not 1e-4. Computing `c*dtz` directly would
    # disagree with the Fortran on every row where `nd` dominates.
    deln = jnp.where(mask, ndnew - nd, 0.0)
    return SolveResult(deln=deln, clamped=clamped, failed=logic1ca)
