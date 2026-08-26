"""`ukca_binapara`: Vehkamaki (2002) binary homogeneous nucleation.

Ported from `fortran/src/ukca/ukca_binapara_mod.F90`. Three polynomials in
temperature, `log(rh)` and `log(h2so4)` -- 113 machine-extracted coefficients
in `_binapara_literals.py` -- an exponential on each result, and three
post-exponential rules whose order matters.

This is the whole of binary nucleation in this model. `ukca_calcnucrate.F90:257`
hard-codes `i_bhn_method = i_bhn_method_vekhamaki`, so the Kulmala (1998)
branch beside it is unreachable and is not ported (`docs/unsupported.md`).

The term order is the specification
-----------------------------------

Fortran sums left to right, so `a + b - c + d` is `((a + b) - c) + d`. The
literals module stores each expression as an **ordered** tuple of terms and
`_evaluate` sums them in that order and no other. Regrouping -- by power of
`logrh`, or by collecting the 26 terms that carry `/termx` -- is algebraically
identical and numerically different, and would look like the tidier code.

`termx` divides 10 of `jveh`'s 50 terms and 10 of `ntot`'s, so it is a pole in
principle. Measured over the leaf grid it stays in [0.049, 0.615], and nothing
in the routine guards it; the golden records what the compiled routine returns
across that range rather than what a guard would.

The clamps are applied to the routine's own copies
--------------------------------------------------

`:106-118` clips `t`, `rh` and `h2so4`, and **every later reference reads the
clipped value** -- including the `t < 195.15` test at `:251`. So a caller
passing 150 K gets the 190.15 K answer and does *not* take the cold branch,
because 190.15 is not below 195.15. Porting that test against the caller's
temperature would change the answer only below 190.15 K, which is exactly the
region a physically-minded grid would omit.

The three output rules, in order
--------------------------------

1. `ntot < 4` **and** clipped `t < 195.15` sets `jveh = 1e5`;
2. then `jveh < 1e-7` sets it to 0;
3. then `jveh > 1e10` sets it to 1e10.

`1e5` sits strictly between the floor and the ceiling, so a value set by rule 1
survives rules 2 and 3 -- and, measured, the three rules therefore **commute**:
permuting them changes nothing. That is a property of three constants and not
of the structure, so the source's order is kept rather than relied on. Move
`j_cold` outside `[1e-7, 1e10]` and the orders diverge at once, which
`test_the_output_rules_commute_only_because_of_where_1e5_sits` demonstrates.

What rule 1 does change is how many points reach the ceiling at all: 182 exceed
it before the cold rule runs and 74 still do after.

`ntot` is not an output of the Fortran routine, so "the critical cluster fell
below four molecules" is invisible in `jveh` unless the temperature rule also
fires. It is returned here as a third result -- 380 of the grid's 990 points
reach it and 113 of those are too warm for rule 1 -- so the branch can be
tested rather than inferred.

Byte equality
-------------

Not attainable: `EXP` is applied to all three results and `jnp.exp` is not the
platform libm (issue #28). The polynomials themselves use only `LOG`, which is
bit-identical. `tests/test_binapara.py` asserts that substituting libm's `exp`
restores exact agreement on every point, which is what makes the remaining gap
attributable rather than merely small.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.physics._binapara_literals import (
    CLAMPS,
    JVEH_TERMS,
    LIMITS,
    NTOT_TERMS,
    RC_TERMS,
    TERMX_TERMS,
)

__all__ = ["CLAMPS", "LIMITS", "binapara", "clamp_inputs", "log_polynomials"]


def _evaluate(terms, table: dict[str, Array], shape_like: Array) -> Array:
    """One expression, summed in the Fortran's own order.

    `total - value` rather than `total + (-value)` for a minus sign: the two
    are identical in IEEE, and writing the subtraction keeps the generated
    table readable against the source it came from.
    """
    total = None
    for sign, coeff, factors, over in terms:
        value = jnp.full_like(shape_like, coeff)
        for name in factors:
            value = value * table[name]
        if over:
            value = value / table["termx(jl)"]
        if total is None:
            total = value if sign == "+" else -value
        else:
            total = total + value if sign == "+" else total - value
    return total


def clamp_inputs(t: Array, rh: Array, h2so4: Array) -> tuple[Array, Array, Array]:
    """`:106-118`. Returned rather than applied in place because the *clipped*
    temperature is what the cold rule at `:251` reads, and a caller of this
    module has no other way to see that."""
    return (
        jnp.clip(jnp.asarray(t), *CLAMPS["t"]),
        jnp.clip(jnp.asarray(rh), *CLAMPS["rh"]),
        jnp.clip(jnp.asarray(h2so4), *CLAMPS["h2so4"]),
    )


def log_polynomials(t: Array, rh: Array, h2so4: Array) -> tuple[Array, Array, Array, Array]:
    """`(termx, log_jveh, log_ntot, log_rc)` from **already clipped** inputs.

    Split out because these carry no exponential and are therefore byte-equal
    to the Fortran, while everything downstream of `exp_v` is not (issue #28).
    Gating them separately is what localises the gap.
    """
    logrh = jnp.log(rh)
    logh2so4 = jnp.log(h2so4)
    table: dict[str, Array] = {
        "tdegk": t,
        "tdegk2": t * t,
        "tdegk3": t * t * t,
        "logrh": logrh,
        "logrh2": logrh * logrh,
        "logrh3": logrh * logrh * logrh,
        "logh2so4": logh2so4,
        "logh2so42": logh2so4 * logh2so4,
        "logh2so43": logh2so4 * logh2so4 * logh2so4,
    }
    termx = _evaluate(TERMX_TERMS, table, t)
    table["termx(jl)"] = termx
    log_jveh = _evaluate(JVEH_TERMS, table, t)
    log_ntot = _evaluate(NTOT_TERMS, table, t)
    table["ntot(jl)"] = log_ntot
    log_rc = _evaluate(RC_TERMS, table, t)
    return termx, log_jveh, log_ntot, log_rc


def binapara(t: Array, rh: Array, h2so4: Array) -> tuple[Array, Array, Array]:
    """`(jveh, rc, ntot)` -- nucleation rate in cm-3 s-1, critical radius in nm,
    and the critical cluster size the Fortran keeps to itself.

    `ntot` is returned because the `< 4` rule at `:249` is otherwise
    unobservable wherever the temperature rule does not also fire, and a branch
    nothing can see is a branch nothing can test.
    """
    t, rh, h2so4 = clamp_inputs(t, rh, h2so4)
    _termx, log_jveh, log_ntot, log_rc = log_polynomials(t, rh, h2so4)

    ntot = jnp.exp(log_ntot)
    jveh = jnp.exp(log_jveh)
    rc = jnp.exp(log_rc)

    # `:249-283`, in order. 1e5 lies between the floor and the ceiling, so a
    # value set by the cold rule survives both of the rules that follow it --
    # which is why they cannot be reordered or fused.
    cold = (ntot < LIMITS["ntot_min"]) & (t < LIMITS["t_cold"])
    jveh = jnp.where(cold, LIMITS["j_cold"], jveh)
    jveh = jnp.where(jveh < LIMITS["j_floor"], 0.0, jveh)
    jveh = jnp.where(jveh > LIMITS["j_ceiling"], LIMITS["j_ceiling"], jveh)

    return jveh, rc, ntot
