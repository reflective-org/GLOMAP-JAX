"""The ZSR ion tables, and the two index spaces they live in (task 39).

`ukca_water_content_v` uses **two different negative-lower-bound index spaces**
that look alike and are not, which is the whole reason this module exists
rather than the tables being inlined where they are used:

    cl(nv, -nanion:ncation)      ions,  -4 .. +3   -- 8 slots, anions negative,
                                                      cations positive
    y(ncation, -nanion:-1, 0:7)  table,  1 .. 3  x  -4 .. -1

An ion concentration is indexed by a signed species number running from -4 to
+3. A table row is indexed by a *pair*: a cation in 1..3 and an anion in
-4..-1. Both are "negative-indexed Fortran arrays", both are naturally rebased
by adding something, and the something is different. Rebasing the pair with the
ion offset silently shifts every electrolyte by four rows and produces water
contents that are wrong but finite.

So the offsets are named, used in exactly one place each, and tested against
each other:

    ION_OFFSET  = nanion      = 4    signed species  -> 0-based ion slot
    CATION_BASE = 1                  cation 1..3     -> 0-based row
    ANION_OFFSET = nanion            anion -4..-1    -> 0-based column 0..3

## Two tables, not one switch

`BASE` and `FIXED` differ in exactly one of 96 coefficients: `y(1,-3)[6]`,
which `l_fix_ukca_water_content` replaces with a value ten times larger. The
port holds both as frozen arrays and selects between them, rather than
reproducing the Fortran's in-place patch.

That is not stylistic. In the Fortran `y` is DATA-initialised and
`THREADPRIVATE`, hence implicitly `SAVE`, and `ukca_water_content_v.F90:235`
writes it **in place with no restore** -- so a process that has ever run with
the flag on keeps the patched coefficient after the flag goes off. A one-way
latch (issue #22) is a property of mutable module state; two immutable arrays
cannot have it.

## Reachability, measured rather than assumed

Pair (1,-3) -- the *only* pair the flag touches -- is **dead through the box
model's caller**. `cl(:,-3)` is identically zero because `ncp = 6` in all seven
supported setups while `cp_no3 = 7`, so the nitrate block at
`ukca_volume_mode.F90:402-419` never runs. The flag therefore has no effect on
any trajectory this repository can validate, and its difference is observable
only through a leaf driver that sets `ions(:,-3)` directly.

Recorded here because it decides what a both-settings test can honestly claim:
the tables differ, and the *model* does not, in every configuration we have.
"""

from __future__ import annotations

import numpy as np

from ._water_literals import BASE, FIXED, LIMITS, PATCHED_ENTRY

NCATION = 3
NANION = 4
NCOEFF = 8

# Signed ion species number -> 0-based slot in an (n, 8) ion array.
ION_OFFSET = NANION
# (cation, anion) -> 0-based (row, column) in a (3, 4, 8) coefficient table.
CATION_BASE = 1
ANION_OFFSET = NANION

__all__ = [
    "ANION_OFFSET",
    "CATION_BASE",
    "ION_OFFSET",
    "LIMITS_TABLE",
    "NANION",
    "NCATION",
    "NCOEFF",
    "PATCHED_ENTRY",
    "Y_BASE",
    "Y_FIXED",
    "coefficients",
    "ion_slot",
    "pair_index",
]


def ion_slot(species: int) -> int:
    """Signed ion species (-4..+3) to its column in an `(n, 8)` ion array.

    Anions are negative and cations positive, so the whole range shifts by
    `nanion`. This is *not* the same mapping as `pair_index`; see the module
    docstring for why they are easy to confuse and expensive to swap.
    """
    if not -NANION <= species <= NCATION:
        raise IndexError(f"ion species {species} outside -{NANION}..{NCATION}")
    return species + ION_OFFSET


def pair_index(cation: int, anion: int) -> tuple[int, int]:
    """`(cation, anion)` to `(row, column)` in the coefficient table."""
    if not 1 <= cation <= NCATION:
        raise IndexError(f"cation {cation} outside 1..{NCATION}")
    if not -NANION <= anion <= -1:
        raise IndexError(f"anion {anion} outside -{NANION}..-1")
    return cation - CATION_BASE, anion + ANION_OFFSET


def _dense(table: dict[tuple[int, int], tuple[float, ...]]) -> np.ndarray:
    out = np.zeros((NCATION, NANION, NCOEFF), dtype=np.float64)
    for (cation, anion), values in table.items():
        row, col = pair_index(cation, anion)
        out[row, col] = values
    out.flags.writeable = False
    return out


def _dense_limits() -> np.ndarray:
    out = np.zeros((NCATION, NANION, 2), dtype=np.float64)
    for (cation, anion), values in LIMITS.items():
        row, col = pair_index(cation, anion)
        out[row, col] = values
    out.flags.writeable = False
    return out


Y_BASE = _dense(BASE)
"""The DATA literals, `l_fix_ukca_water_content` off."""

Y_FIXED = _dense(FIXED)
"""The DATA literals with `y(1,-3)[6]` replaced, the flag on."""

LIMITS_TABLE = _dense_limits()
"""`(rh_min, molal_max)` per pair. `rh_min` is in **percent**, not fraction --
`ukca_water_content_v.F90:281` compares against `rh_min/1.0e2`."""


def coefficients(fix_water_content: bool) -> np.ndarray:
    """The table this flag setting selects. Frozen; callers must not mutate."""
    return Y_FIXED if fix_water_content else Y_BASE
