"""`ukca_calcnucrate`: how much H2SO4 nucleation removes in one `dtz`.

Ported from `fortran/src/ukca/ukca_calcnucrate.F90`. Wraps `binapara` for the
binary rate, adds a boundary-layer term, and rewrites the H2SO4 concentration.

Only the Vehkamaki path is here. `:256-257` assigns
`i_bhn_method = i_bhn_method_vekhamaki` unconditionally, so the Kulmala (1998)
branch that occupies `:308-360` is unreachable and is not ported
(`docs/unsupported.md`). It is a third of the routine.

Two logicals, not two switches
------------------------------

`:302` and `:304` decide which terms run, and neither is a namelist variable::

    l1 = (i_nuc_method == 2) and (height > zbl or bln_on == 0)
    l2 = (i_nuc_method == 3) and (ibln == 3)

Binary nucleation runs where `l1 or l2`; boundary-layer nucleation runs where
`not l1`. `zbl = min(htpblg, 6000)` (`:265-267`), so the height test is against
a *filtered* boundary-layer height and a column with a 12 km boundary layer
finds it capped at 6 km.

The consequence worth stating: with `i_nuc_method = 3` and `ibln = 3`, **both**
terms run in the same box, binary first. No shipped namelist reaches that --
`bln_on` is off in all five -- so it exists only in the leaf golden.

The order inside a box is load-bearing
--------------------------------------

Binary nucleation updates `h2so4` and then boundary-layer nucleation reads the
**updated** value, both in its guard (`:397`) and in its rate. Two masked
updates applied in sequence, not one combined rate: computing both from the
original concentration would over-deplete wherever both run.

`jrate` accumulates across the two terms but is a local in the Fortran and
reaches no output. It is kept here because the binary update at `:372` divides
by `jrate` rather than by `japp` -- identical while `jrate` starts at zero, and
a structure a future edit could break silently.

What comes back
---------------

`(h2so4, delh2so4_nucl)`. `delh2so4_nucl` accumulates `h2so4old - h2so4` over
both terms, so it equals the total drop exactly -- asserted bit for bit against
the Fortran in the fixture, because two accumulations that should agree and do
not is the sort of thing a tolerance hides.

Byte equality is not attainable: `EXP` appears in both rate expressions and in
`binapara` beneath them (issue #28).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.core.constants import CONC_EPS, NMOL
from glomap_jax.physics.binapara import binapara

__all__ = ["AFAC_ACT", "AFAC_KIN", "AFAC_PNA", "DPBLN", "J_MIN", "ZMAXBLN", "calcnucrate"]

#: `:207`. Boundary-layer nucleation is not applied above this height even if
#: the reported boundary layer is deeper.
ZMAXBLN = 6000.0

#: `:209-211`. Metzger et al (2010), the activation coefficient and the
#: kinetic coefficient, selected by `ibln` 3, 1 and 2 respectively.
AFAC_PNA = 5.0e-13
AFAC_ACT = 5.0e-7
AFAC_KIN = 4.0e-13

#: `:271`. The diameter in nm at which the boundary-layer rate is quoted. Set
#: only when `ibln` is in 1..3, which is why an out-of-range `ibln` reads it
#: never having been assigned -- and why this port refuses one.
DPBLN = 1.5

#: `:369` and `:414`. Below this the rate is discarded rather than applied.
J_MIN = 1.0e-3


def _apparent_factor(diameter_term: Array, s_cond_s: Array, h2so4: Array) -> Array:
    """`EXP(0.23*(1/3 - 1/d)*s_cond_s/(1e6*h2so4*1e-13))`, `:370` and `:413`.

    The Kerminen-Kulmala correction from the critical cluster to the quoted
    diameter. Written with the Fortran's own association: the `0.23*(...)`
    product multiplies `s_cond_s` and the result is divided by
    `(1e6*h2so4)*1e-13`, which is a product of arrays and needs no
    `true_divide`.
    """
    return jnp.exp(0.23 * diameter_term * s_cond_s / (1.0e6 * h2so4 * 1.0e-13))


def _deplete(h2so4: Array, taken: Array, active: Array) -> tuple[Array, Array]:
    """One nucleation term's effect on the concentration.

    `:374-379` and `:417-421`, which are the same three lines twice: subtract,
    refuse an increase, refuse a negative. The increase guard is not dead --
    `taken` is a rate times `nmol*dtz` and nothing bounds it below zero, so a
    negative rate would raise the concentration.
    """
    before = h2so4
    after = h2so4 - taken
    after = jnp.where(after > before, before, after)
    after = jnp.where(after < 0.0, 0.0, after)
    after = jnp.where(active, after, before)
    return after, jnp.where(active, before - after, 0.0)


def calcnucrate(
    t: Array,
    s: Array,
    rh: Array,
    aird: Array,
    h2so4: Array,
    sec_org: Array,
    height: Array,
    htpblg: Array,
    s_cond_s: Array,
    *,
    dtz: float,
    bln_on: int,
    ibln: int,
    i_nuc_method: int,
) -> tuple[Array, Array]:
    """`(h2so4, delh2so4_nucl)` after one nucleation step.

    `s` and `aird` are in the signature and reach no output: they feed the
    Kulmala water-vapour concentration at `:326`, which is in the dead branch.
    Kept so the call site matches the Fortran's, and asserted inert by the
    fixture.
    """
    if i_nuc_method not in (2, 3):
        raise ValueError(
            f"i_nuc_method={i_nuc_method} must be 2 or 3 "
            "(ukca_calcnucrate.F90:288-293 ereports outside that range)"
        )
    if ibln not in (1, 2, 3):
        raise ValueError(
            f"ibln={ibln} must be 1, 2 or 3: dpbln is assigned only inside that "
            "range (:271) and both rate expressions read it"
        )

    t, rh = jnp.asarray(t), jnp.asarray(rh)
    h2so4 = jnp.asarray(h2so4)
    sec_org, s_cond_s = jnp.asarray(sec_org), jnp.asarray(s_cond_s)
    height, htpblg = jnp.asarray(height), jnp.asarray(htpblg)
    del s, aird  # dead outside the Kulmala branch; see the docstring

    # `:265-267`, then `:302-306`.
    zbl = jnp.minimum(htpblg, ZMAXBLN)
    l1 = (height > zbl) | (bln_on == 0) if i_nuc_method == 2 else jnp.zeros_like(height, dtype=bool)
    l2 = bool(i_nuc_method == 3 and ibln == 3)

    delh2so4 = jnp.zeros_like(h2so4)
    jrate = jnp.zeros_like(h2so4)

    # --- binary homogeneous nucleation, `:363-386` ---
    # binapara sees the ORIGINAL concentration: `:262` is called before the box
    # loop, so nothing has depleted it yet.
    jveh, rc, _ntot = binapara(t, rh, h2so4)

    bhn = (l1 | l2) & (h2so4 > CONC_EPS) & (jveh > 0.0) & (s_cond_s > 0.0) & (rc > 0.0)
    # `2*rc`, an integer literal times a real, and `1.0/3.0` exactly as written.
    safe_rc = jnp.where(bhn, rc, 1.0)
    japp = jveh * _apparent_factor(
        1.0 / 3.0 - 1.0 / (2 * safe_rc), s_cond_s, jnp.where(bhn, h2so4, 1.0)
    )
    bhn = bhn & (japp > J_MIN)
    # `:371`: the update at `:373` divides by `jrate`, not by `japp`. They are
    # equal here only because `jrate` starts at zero.
    jrate = jnp.where(bhn, jrate + japp, jrate)
    h2so4, taken = _deplete(h2so4, jrate * NMOL * dtz, bhn)
    delh2so4 = delh2so4 + taken

    # --- boundary-layer nucleation, `:390-424` ---
    # Reads the h2so4 the binary term left behind, in its guard and its rate.
    bln = (~l1) & (h2so4 > CONC_EPS)
    if ibln == 1:
        jbln = AFAC_ACT * h2so4
    elif ibln == 2:
        jbln = AFAC_KIN * h2so4 * h2so4
    else:
        jbln = AFAC_PNA * h2so4 * sec_org
    japp_bln = jbln * _apparent_factor(
        1.0 / 3.0 - 1.0 / DPBLN, s_cond_s, jnp.where(bln, h2so4, 1.0)
    )
    bln = bln & (japp_bln > J_MIN)
    # `:416` subtracts `japp_bln`, not the accumulated `jrate` -- the asymmetry
    # with `:373` is the Fortran's and is reproduced, not tidied.
    h2so4, taken = _deplete(h2so4, japp_bln * NMOL * dtz, bln)
    delh2so4 = delh2so4 + taken

    return h2so4, delh2so4
