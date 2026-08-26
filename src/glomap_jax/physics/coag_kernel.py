"""`ukca_calc_coag_kernel`: the coagulation kernel over every mode pair.

Ported from `fortran/src/ukca/ukca_calc_coag_kernel.F90`, byte-equal to the
compiled routine over `tests/goldens/coag_kernel.f64.leaf.npz` on all seven
supported setups.

A driver, not a formula. Every number it returns comes from `coag_coff`, which
task 49 already pinned. What this module adds is the *slot map* -- which
`(imode, jmode)` entries of `kij_arr` are written and which stay at zero -- and
the argument routing, which is not uniform.

The slot map is the part nothing downstream can check
-----------------------------------------------------

`coag_coff` is byte-symmetric under an `(i, j)` swap, and `coag_mode` is
symmetric on all 64 entries (phase C). So a transposed convention here writes
the **right number into the wrong slot** and every value comparison in the
project still passes. The only discriminator is which entries are non-zero,
which is why `tests/test_coag_kernel_fixtures.py` compares the filled set
against a re-derivation of the loop bounds rather than comparing values.

Three families, and they are not one triangle:

* `:243-262` soluble `imode`, soluble `jmode > imode` -- upper triangle of 1..4
* `:276-291` soluble `imode`, insoluble `jmode >= imode + 4` -- upper, across
  the split
* `:308-322` **insoluble** `imode`, soluble `jmode >= imode - 2` -- **lower**

With every mode active that is 19 slots, and no `(i, j)` is filled in both
orders. Three soluble/insoluble pairs are filled in neither: `(2,5)`, `(3,6)`
and `(4,7)` in the Fortran's 1-based indices -- the same-size-class pairs. That
is not a gap this port has to work around: `ukca_coagwithnucl` reads `kij_arr`
with the same three loop bounds, so it never asks for them either. It reads
*less*, in fact -- its insoluble inner loop stops at `topmode`, which is
`mode_ait_insol` unless `l_dust_mp_ageing` is on, so by default nine of the
slots this routine fills are never consumed.

The argument routing is not uniform, and the fixture measures it
----------------------------------------------------------------

For a soluble `imode` the partner's size is selected by `modesol` (`:281-286`):
a soluble `jmode` enters at its **wet** size, an insoluble one at its **dry**
size. But the insoluble-`imode` loop takes `wetdp`/`wvol` unconditionally
(`:297-299`), so the *same* insoluble mode enters at its wet size when it is
`i` and at its dry size when it is `j`.

In the model those coincide -- an insoluble mode carries no water -- so the
inconsistency is inert. It is reproduced rather than tidied, and the leaf grid
feeds `drydp = 0.7*wetdp` for every mode precisely so that the two conventions
give different numbers and the golden can say which one each slot took: 8 dry,
43 wet, over the seven setups.

Why plain Python loops
----------------------

`nmodes` is a PARAMETER and the loop bounds are all compile-time constants, so
the 19 slots are a static, fully unrolled set. Nothing is loop-carried: each
call writes a distinct slot and reads none. This is not one of CLAUDE.md's five
`lax.scan` routines and does not need `jnp.where` over mode indices -- the
indices are Python ints, not traced values.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from glomap_jax.physics.coag_coff import coag_coff
from glomap_jax.physics.modes import (
    MODE_AIT_INSOL,
    MODE_COR_INSOL,
    MODE_COR_SOL,
    MODE_NUC_SOL,
    MODE_SUP_INSOL,
    NMODES,
)


def slot_map(mode: Array, modesol: Array) -> tuple[list[int], list[tuple[int, int]]]:
    """The `kii` modes and `(imode, jmode)` slots the three loops fill.

    Zero-based throughout, unlike the Fortran. Returned rather than inlined so
    a test can compare it against what the compiled routine filled -- the only
    check there is on the `(i, j)` convention.

    `modesol` is unused here and is an argument anyway: `:281` branches on it
    when choosing the partner's *size*, not when choosing the slot, and keeping
    the two decisions visibly separate is what stops a later edit from
    conflating them.
    """
    del modesol
    active = [bool(m) for m in mode]
    kii: list[int] = []
    kij: list[tuple[int, int]] = []

    for imode in range(MODE_NUC_SOL, MODE_COR_SOL + 1):  # `:243`
        if not active[imode]:
            continue
        kii.append(imode)
        for jmode in range(imode + 1, MODE_COR_SOL + 1):  # `:262`
            if active[jmode]:
                kij.append((imode, jmode))
        for jmode in range(imode + 4, MODE_SUP_INSOL + 1):  # `:276`
            if active[jmode]:
                kij.append((imode, jmode))

    for imode in range(MODE_AIT_INSOL, MODE_SUP_INSOL + 1):  # `:295`
        if not active[imode]:
            continue
        kii.append(imode)
        if imode < MODE_COR_INSOL:  # `:308`
            for jmode in range(imode - 2, MODE_COR_SOL + 1):  # `:309`
                if active[jmode]:
                    kij.append((imode, jmode))

    return kii, kij


def calc_coag_kernel(
    mode: Array,
    modesol: Array,
    drydp: Array,
    dvol: Array,
    wetdp: Array,
    wvol: Array,
    rhopar: Array,
    mfpa: Array,
    dvisc: Array,
    t: Array,
    *,
    coag_on: int,
    icoag: int,
) -> tuple[Array, Array]:
    """`(kii_arr, kij_arr)`, shapes `(nbox, nmodes)` and `(nbox, nmodes, nmodes)`.

    `mask1` and `mask2` are `.TRUE.` for every box in the Fortran (`:224-225`)
    and are never changed, so the kernel's mask is inert in this caller and no
    mask argument is exposed. That is a property of the driver, not of
    `coag_coff`, which is why the mask is swept in the coefficient's own
    fixture and not here.
    """
    drydp, dvol = jnp.asarray(drydp), jnp.asarray(dvol)
    wetdp, wvol = jnp.asarray(wetdp), jnp.asarray(wvol)
    rhopar = jnp.asarray(rhopar)
    mfpa, dvisc, t = jnp.asarray(mfpa), jnp.asarray(dvisc), jnp.asarray(t)

    nbox = wetdp.shape[0]
    kii_arr = jnp.zeros((nbox, NMODES), dtype=wetdp.dtype)
    kij_arr = jnp.zeros((nbox, NMODES, NMODES), dtype=wetdp.dtype)
    if coag_on == 0:
        # `:239-247` zeroes both and `ukca_coag_coff_v` returns zeros anyway.
        return kii_arr, kij_arr

    ones = jnp.ones(nbox, dtype=bool)
    kii_modes, kij_slots = slot_map(mode, modesol)

    def partner(j: int) -> tuple[Array, Array]:
        """`:281-286`: a soluble partner enters wet, an insoluble one dry."""
        if int(modesol[j]) == 1:
            return wetdp[:, j] / 2.0, wvol[:, j]
        return drydp[:, j] / 2.0, dvol[:, j]

    def kernel(ri, rj, vi, vj, rhoi, rhoj):
        return coag_coff(
            ones, ri, rj, vi, vj, rhoi, rhoj, mfpa, dvisc, t, coag_on=coag_on, icoag=icoag
        )

    # `imode` always enters at its WET size, soluble or not (`:245-247`,
    # `:297-299`). Only the partner is selected by `modesol`.
    for imode in kii_modes:
        ri = wetdp[:, imode] / 2.0
        vi = wvol[:, imode]
        rhoi = rhopar[:, imode]
        kii_arr = kii_arr.at[:, imode].set(kernel(ri, ri, vi, vi, rhoi, rhoi))

    for imode, jmode in kij_slots:
        ri = wetdp[:, imode] / 2.0
        vi = wvol[:, imode]
        rj, vj = partner(jmode)
        kij_arr = kij_arr.at[:, imode, jmode].set(
            kernel(ri, rj, vi, vj, rhopar[:, imode], rhopar[:, jmode])
        )

    return kii_arr, kij_arr
