"""`coag_mode` — where the products of a coagulating mode pair land — and the
mask carrier `ukca_coagwithnucl` needs in order to be traceable.

Two things live here because they are the two halves of the same problem: the
final block of `ukca_coagwithnucl` is an **indexed scatter-add**, and every
input to it is either a compile-time index (this table) or a per-box predicate
(the masks). Neither is physics. Both decide what the physics is allowed to do.

`ukca_mode_setup.F90:174-183`::

    INTEGER, PARAMETER :: coag_mode( nmodes, nmodes ) = RESHAPE( [ ... ] )

consumed once, at `ukca_coagwithnucl.F90:534-535`::

    mtrantoi(:,coag_mode(imode,jmode),icp) =
         mtrantoi(:,coag_mode(imode,jmode),icp) + mtran(:,icp,imode,jmode)

Read that as: `mtran(:,icp,i,j)` is mass leaving mode `i` because `i`
coagulated with `j`; `coag_mode(i,j)` is the mode it arrives in.

Setup-independence
------------------

Unlike every other mode table in this package, `coag_mode` is a `PARAMETER` at
module scope and is **not** a member of `glomap_variables_type`. There is
exactly one assignment to the name in the whole vendored tree (the declaration
itself) and none of the `ukca_mode_*` setup routines touches it, so it
is identical for every `i_mode_setup` and every switch combination.

That claim is not left to grep. `tests/goldens/coagmode.f64.tables.npz` holds
the table read out of the built extension **once with no init at all** — legal
only because it is a `PARAMETER` — and then once per `i_mode_setup` in its own
process, both before and after `wrap_init`. Every read is byte-equal, and the
same archive carries `mode`/`topmode`/`ncp` per setup, which are *not*, so a
capture that silently ran one setup seven times fails rather than confirming
invariance by accident. `validation/capture_coag_mode.py`.

What *is* setup-dependent is which entries are ever read: `mode(imode)` gates
the loops, so setup 1 reaches 6 of the 64 entries and setup 6 reaches none at
all (its two active modes are both insoluble, so no `mtran` is ever written).
See `source_pairs`.

The literal, and why the reshape order is provably immaterial
-------------------------------------------------------------

Fortran `RESHAPE` fills column-major, so the *first* line of the literal is
`coag_mode(1:8, 1)` — a column, not a row. `_RESHAPE_SOURCE_ORDER` below is
emitted one Python tuple per source line, so it can be diffed against the
Fortran by eye, and the column-major fill is then written out explicitly rather
than assumed.

It happens not to matter: the table is symmetric, `coag_mode(i,j) ==
coag_mode(j,i)` for all 64 entries. That is a fact about the data, not a
guarantee, so it is asserted rather than relied on. It is also the one thing
neither mechanical reading can check: a transposed table is byte-equal to the
correct one, so no capture and no re-parse could tell them apart. The
`(imode, jmode)` order is established by the call site's subscripts and by
`wrap_coag_dest`'s indexed read, not by the data.

What a JAX scatter for this table must preserve
-----------------------------------------------

`mtrantoi` is accumulated with `+`, and **many source pairs collide on one
destination**. Mode indices below are 0-based, as everywhere else in this
package; add one for the Fortran's. Over the full table, the 64
`(imode, jmode)` entries land on only eight destinations
(`full_table_census`)::

    dest   0   1   2   3   4   5   6   7
    pairs  1   7  13  27   1   3   5   7

and restricted to the pairs that actually write `mtran` (`source_pairs`), the
box model's reachable configurations collide like this
(`destination_census`)::

    setup 1/3/5  (4 soluble modes)          6 pairs   -> {1: 1, 2: 2, 3: 3}
    setup 2/4/8  (+ Aitken insoluble)       9 pairs   -> {1: 2, 2: 3, 3: 4}
    setup 8, l_dust_mp_ageing              15 pairs   -> {1: 2, 2: 5, 3: 8}
    setup 6      (dust only, no soluble)    0 pairs   -> {}

So up to eight `mtran` terms are summed into `mtrantoi[:, 3, icp]` — four in
any configuration the box model can actually run — and float addition is not
associative: the *order* of those eight adds is part of the
answer, not an implementation detail. `tests/test_coag_mode.py` adds eight
representative magnitudes in Fortran order and in reverse and gets two
different doubles, so this is measured here, not quoted from a textbook.

The Fortran order is fixed by its loop nest (`imode` outer, then `icp`, then
`jmode`), which for a fixed destination and component means increasing
`(imode, jmode)` lexicographically — `accumulation_order` returns exactly
that.

Three consequences for the port, in decreasing order of how easy they are to
get wrong:

1. **Accumulate with `.at[dest].add(...)`, never `.at[dest].set(...)`.** The
   destination is written more than once and `set` would keep only the last.

2. **Emit the adds one at a time, with `dest` a Python `int`.** Both loop
   indices are static — `imode` and `jmode` come from the mode tables, which
   are frozen configuration, not traced state — so a vectorised
   `zeros.at[dest_array].add(all_terms)` is available and is the wrong choice:
   XLA's scatter with duplicate indices does not promise an accumulation order,
   and on GPU it is genuinely nondeterministic. Unrolling with static indices
   puts the order back under our control, at no cost, because the unrolled form
   is what XLA would have to produce anyway.

3. **`jnp.sum` over the `jmode` axis is not a shortcut.** A pairwise reduction
   re-associates; the Fortran runs a sequential accumulation. Same trap as
   `rhommav` in `physics/modes.py`, and the same fix: write the loop.

`accumulation_order` lists only the pairs in `source_pairs`. The Fortran loop
also visits every other active `(imode, jmode)`, but `mtran` is exactly `0.0`
there (it is zeroed once at entry and only the pairs in `source_pairs` are ever
written), and `x + 0.0 == x` for every double except `-0.0 + 0.0`, which cannot
arise: `mtran` is `mdold * ndold * (1 - EXP(-xxx))` with `xxx >= 0`, so it is
non-negative, and `mtrantoi` starts at `+0.0`.

Masks as static-shape traced bools
----------------------------------

`ukca_coagwithnucl` carries five `LOGICAL(nbox)` arrays, and they are the
routine's entire control flow — there is no scalar `IF` on a computed value
anywhere in it. `CoagMasks` carries them as `jnp` bool arrays of a shape fixed
at trace time. Two reasons that shape has to be static, both of which produce a
tracer error rather than a wrong number, which is the good kind of failure:

* **Under `vmap` a per-parcel trip count degenerates.** Any construct whose
  *size* depends on a mask — `jnp.nonzero`, boolean indexing, "loop over the
  boxes where `mask1` is true" — needs a concrete count, and under `vmap` each
  parcel would want a different one. The masks must therefore select *values*
  (`jnp.where`) over the full static extent, never trip counts. The loops over
  modes and components stay Python loops precisely because their bounds come
  from `ModeTables`, which is frozen and static.

* **`mask1` is loop-carried, so it has to be a `lax.scan` carry.** At
  `ukca_coagwithnucl.F90:567-570` the `icp` loop narrows `mask1` *in place*
  (`mask1(:)=.FALSE.` inside a `WHERE`, "set false so not used for other icp
  values"), which is why CLAUDE.md lists this `icp` loop among the five that
  need a sequential scan. `lax.scan` requires the carry's shape and dtype to be
  identical on every iteration, so a mask that changed length as boxes dropped
  out would not even trace. `CoagMasks` is registered as a pytree so it can be
  that carry directly, and `narrow_mask1` is the in-place narrowing.

The stale-`bterm` trap in `mask4`
---------------------------------

`mask4` is `ABS(-bterm*dtz) > xxx_eps`, but `bterm` is assigned only under
`IF (interoff /= 1)` inside `WHERE (mask2)`. Where `mask2` is false, `bterm`
still holds the *previous* `jmode`'s value and `mask4` is computed from it.
The science never uses that value — every consumer is `mask4 .AND. mask2` — but
the gate-0 dump records `mask4` raw, so a port that evaluates `mask4` per mode
pair records something different and gate 0 reports a disagreement with no
consequence. Hence `effective_mask4`, which is what the physics may use, and
the rule in `docs/harness.md`: compare `mask4` only where `mask2` is true.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glomap_jax.physics.modes import (
    MODE_AIT_INSOL,
    MODE_COR_INSOL,
    MODE_COR_SOL,
    NMODES,
    ModeTables,
)

# ---------------------------------------------------------------------------
# The table. `ukca_mode_setup.F90:174-183`, one tuple per source line.
#
# GENERATED, not typed. Reproduce with::
#
#     python validation/capture_coag_mode.py --emit-literal
#
# and `tests/test_coag_mode.py` re-runs that parse and compares, so the block
# cannot drift from the vendored tree and cannot be hand-edited without the
# test noticing. Sixty-four single-digit integers is small enough to look
# typeable, which is exactly why it is not: `core/constants.py` and
# `_mode_literals.py` already established that a number a human retyped from
# Fortran is a number nobody can vouch for.
#
# RESHAPE fills column-major, so each line below is a COLUMN of the result:
# line k is coag_mode(1:8, k).
# ---------------------------------------------------------------------------

_RESHAPE_SOURCE_ORDER: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 2, 3, 4, 4),
    (2, 2, 3, 4, 2, 3, 4, 4),
    (3, 3, 3, 4, 3, 3, 4, 4),
    (4, 4, 4, 4, 4, 4, 4, 4),
    (2, 2, 3, 4, 5, 6, 7, 8),
    (3, 3, 3, 4, 6, 6, 7, 8),
    (4, 4, 4, 4, 7, 7, 7, 8),
    (4, 4, 4, 4, 8, 8, 8, 8),
)

RESHAPE_SOURCE_LITERAL: tuple[int, ...] = tuple(v for line in _RESHAPE_SOURCE_ORDER for v in line)
"""The 64 integers in the order the Fortran source lists them."""

COAG_MODE_FORTRAN: np.ndarray = np.array(RESHAPE_SOURCE_LITERAL, dtype=np.int32).reshape(
    (NMODES, NMODES), order="F"
)
"""`coag_mode` with Fortran's own 1-based values, indexed `[imode, jmode]`
0-based. Kept alongside the 0-based table because the branch dumps, the
budget indices and every line number in the Fortran are 1-based, and silently
converting in both directions is how an off-by-one survives review."""

COAG_MODE: np.ndarray = COAG_MODE_FORTRAN - 1
"""`coag_mode` as 0-based destination indices: `COAG_MODE[i, j]` is the mode
that receives the mass leaving mode `i` when `i` coagulates with `j`."""

COAG_MODE_FORTRAN.flags.writeable = False
COAG_MODE.flags.writeable = False


def destination(imode: int, jmode: int) -> int:
    """The 0-based mode that `mtran(:, :, imode, jmode)` is scattered into."""
    return int(COAG_MODE[imode, jmode])


def source_pairs(tables: ModeTables) -> tuple[tuple[int, int], ...]:
    """The 0-based `(imode, jmode)` pairs for which `mtran` is ever written.

    This mirrors `ukca_coagwithnucl`'s loop nest exactly, in execution order —
    it is index bookkeeping, not physics, and it is what makes the collision
    census in the module docstring a measured number rather than an estimate.
    Three blocks:

    * `:304-340` soluble `imode`, `jmode` in `imode+1 .. mode_cor_sol`;
    * `:342-374` the same `imode`, `jmode` in `imode+4 .. topmode`;
    * `:464-498` insoluble `imode` below `mode_cor_insol`, `jmode` in
      `imode-2 .. mode_cor_sol`.

    Note the third block runs `jmode` *down*-mode: an insoluble particle
    coagulating with a larger soluble one transfers its own mass, so `imode` is
    the insoluble mode even though it is the larger index.

    The first two blocks are nested inside one `DO imode`, so the returned
    order is `(imode, jmode)` lexicographic overall — which happens to make it
    the same as `accumulation_order`. Do not rely on that when adding a block;
    `accumulation_order` sorts explicitly.

    Verified against the committed branch goldens, which record one dump per
    `(site, imode, jmode)` actually visited.
    """
    mode = tables.mode
    topmode = tables.topmode  # 1-based, as ukca_mode_setup defines it
    pairs: list[tuple[int, int]] = []

    for imode in range(0, MODE_COR_SOL + 1):
        if not mode[imode]:
            continue
        # sol-sol, with larger soluble modes
        for jmode in range(imode + 1, MODE_COR_SOL + 1):
            if mode[jmode]:
                pairs.append((imode, jmode))
        # sol-insol, with larger insoluble modes
        for jmode in range(imode + 4, topmode):
            if mode[jmode]:
                pairs.append((imode, jmode))

    for imode in range(MODE_AIT_INSOL, topmode):
        if not mode[imode] or imode >= MODE_COR_INSOL:
            continue
        # insol-sol, with soluble modes from imode-2 upwards
        for jmode in range(imode - 2, MODE_COR_SOL + 1):
            if mode[jmode]:
                pairs.append((imode, jmode))

    return tuple(pairs)


def accumulation_order(tables: ModeTables) -> dict[int, tuple[tuple[int, int], ...]]:
    """Destination mode -> the source pairs added into it, in Fortran order.

    The order is the answer, not a detail: see the module docstring. Fortran's
    loop nest is `imode` outer, `icp`, then `jmode`, so for a fixed destination
    and component the terms arrive in `(imode, jmode)` lexicographic order.

    Destinations that receive nothing are absent, so `len(v) for v in
    .values()` is the collision census.
    """
    plan: dict[int, list[tuple[int, int]]] = {}
    for imode, jmode in sorted(source_pairs(tables)):
        plan.setdefault(destination(imode, jmode), []).append((imode, jmode))
    return {dest: tuple(pairs) for dest, pairs in sorted(plan.items())}


def destination_census(tables: ModeTables) -> dict[int, int]:
    """How many source pairs collide on each destination mode, for one setup."""
    return {dest: len(pairs) for dest, pairs in accumulation_order(tables).items()}


def full_table_census() -> dict[int, int]:
    """How many of the 64 table entries name each destination, ignoring which
    modes are active. A property of the table itself, so setup-independent."""
    values, counts = np.unique(COAG_MODE, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


# ---------------------------------------------------------------------------
# The mask carrier.
# ---------------------------------------------------------------------------

_MASK_FIELDS = ("mask1", "mask1a", "mask2", "mask3", "mask4")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoagMasks:
    """`ukca_coagwithnucl`'s five `LOGICAL(nbox)` arrays, as one pytree.

    A pytree because `mask1` is loop-carried across the `icp` loop and so has
    to be a `lax.scan` carry; static-shape bools because under `vmap` anything
    sized by a mask degenerates. Both arguments are in the module docstring.

    All five fields hold the same shape and `bool` dtype: `(nbox,)` in the
    eager driver, `()` inside a `vmap` over boxes. Nothing here is `None` and
    nothing is optional — a field that is sometimes absent changes the pytree
    structure, and `lax.scan` rejects a carry whose structure moves.

    ``validate`` is a method rather than a ``__post_init__`` hook on purpose:
    ``register_dataclass`` unflattens by calling ``__init__``, and JAX
    unflattens with sentinels and abstract values in places where a dtype check
    would be wrong to run.
    """

    mask1: Array
    """`ndold(:,imode) > num_eps(imode)`. `:298`, `:456`. Also re-derived from
    the *updated* `nd` at `:548` for the mass-reset block, and then narrowed in
    place across `icp` (`:569`) — `narrow_mask1`."""

    mask1a: Array
    """`mask1 .OR. (imode == nuc_sol .AND. delh2so4_nucl > conc_eps)`, `:393`.
    What `ukca_solvecoagnucl_v` is actually gated on for the soluble modes —
    the nucleation mode must be solved even with no pre-existing particles.
    The insoluble block passes plain `mask1` instead (`:502`)."""

    mask2: Array
    """Both modes of the pair are populated: `ndold(:,imode) > num_eps(imode)
    .AND. ndold(:,jmode) > num_eps(jmode)`. `:311`, `:349`, `:471`."""

    mask3: Array
    """`mask1 .AND. mdcpnew >= 0`, `:572`. Gates the `md`/`mdt` rewrite, and is
    re-derived per component *after* `mask1` has been narrowed.

    The Fortran then *reassigns* the same variable at `:587` to
    `mask1 .AND. md(:,imode,icp) >= 0`, a different predicate gating the budget
    accumulation, and only the first is in the gate-0 dump. Carrying one field
    for both would be faithful to the Fortran's variable reuse and wrong for
    the comparison; the second belongs to the budget block, not here."""

    mask4: Array
    """`ABS(-bterm*dtz) > xxx_eps`, `:321`, `:362`, `:484` — the "is the
    exponential worth evaluating" test. Carries a stale `bterm` where `mask2`
    is false; use `effective_mask4`, and compare it to the gate-0 dump only
    where `mask2` holds."""

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.shape(self.mask1)

    def validate(self, nbox: int | None = None) -> CoagMasks:
        """Raise unless all five are bool and share one static shape.

        Returns `self`, so it can wrap a construction site. Shapes and dtypes
        are static even for tracers, so this is safe under `jit` and `vmap`.
        """
        shapes = {name: jnp.shape(getattr(self, name)) for name in _MASK_FIELDS}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"CoagMasks fields must share one shape, got {shapes}")
        dtypes = {name: jnp.result_type(getattr(self, name)) for name in _MASK_FIELDS}
        wrong = {n: str(d) for n, d in dtypes.items() if d != jnp.bool_}
        if wrong:
            raise ValueError(f"CoagMasks fields must be bool, got {wrong}")
        if nbox is not None and self.shape != (nbox,):
            raise ValueError(f"CoagMasks shape {self.shape} is not ({nbox},)")
        return self

    @classmethod
    def false(cls, nbox: int) -> CoagMasks:
        """All five masks false, shape `(nbox,)` — the state at routine entry."""
        off = jnp.zeros((nbox,), dtype=bool)
        return cls(mask1=off, mask1a=off, mask2=off, mask3=off, mask4=off)

    @property
    def effective_mask4(self) -> Array:
        """`mask4 .AND. mask2` — the only form the science ever uses.

        Every consumer of `mask4` in the Fortran is written `WHERE (mask4 .AND.
        mask2)` or `WHERE ((.NOT. mask4) .AND. mask2)`, which is what makes the
        stale `bterm` harmless there and visible in the gate-0 dump.
        """
        return jnp.logical_and(self.mask4, self.mask2)

    def narrow_mask1(self, keep: Array) -> CoagMasks:
        """`mask1 = mask1 .AND. keep`, the in-place narrowing at `:567-570`.

        Only ever narrows. A box whose `mdcpnew` goes negative for one
        component has its number zeroed and is excluded from *every later*
        component in the same `icp` loop — which is precisely why that loop is
        a sequential `lax.scan` and not a broadcast.
        """
        return dataclasses.replace(self, mask1=jnp.logical_and(self.mask1, keep))


BRANCH_TAGS: dict[str, tuple[tuple[str, str], ...]] = {
    "mask1": (
        ("coag_sol", "mask1"),
        ("coag_insol", "mask1"),
        ("coag_reset", "mask1_entry"),
        ("coag_reset", "mask1_after"),
    ),
    "mask1a": (("coag_sol_solve", "mask1a"),),
    "mask2": (
        ("coag_sol_sol", "mask2"),
        ("coag_sol_insol", "mask2"),
        ("coag_insol_insol", "mask2"),
    ),
    "mask3": (("coag_reset", "mask3"),),
    "mask4": (
        ("coag_sol_sol", "mask4"),
        ("coag_sol_insol", "mask4"),
        ("coag_insol_insol", "mask4"),
    ),
}
"""`CoagMasks` field -> the `(site, tag)` keys it appears under in the gate-0
branch dump (`validation/patches/0004-dump-branches.patch`).

`mask1` appears four times because the Fortran reuses the name for three
different predicates, and `mask1_entry`/`mask1_after` straddle the in-place
narrowing — the dump emits both so the narrowing itself is observable.

`coag_insol_insol` is the ins-**sol** block despite the name; the site label
comes from the committed overlay and is not corrected here, because renaming it
would invalidate every committed branch golden.
"""

DERIVED_BRANCH_TAGS: tuple[tuple[str, str], ...] = (("coag_reset", "mdcp_neg"),)
"""Dumped predicates that are not `CoagMasks` fields: `mdcp_neg` is
`mask1 .AND. (mdcpnew < 0)`, the narrowing condition, i.e. `.NOT. keep` in
`narrow_mask1`."""

SOLVECOAGNUCL_BRANCH_TAGS: tuple[str, ...] = ("form", "mask", "logic3", "sqd_clamp", "tan_pole")
"""Tags dumped at the `coag_*_solve` sites that belong to
`ukca_solvecoagnucl_v`, not to `ukca_coagwithnucl`. Listed so the completeness
test over the goldens can exclude them by name rather than by guessing."""


__all__ = [
    "BRANCH_TAGS",
    "COAG_MODE",
    "COAG_MODE_FORTRAN",
    "DERIVED_BRANCH_TAGS",
    "RESHAPE_SOURCE_LITERAL",
    "SOLVECOAGNUCL_BRANCH_TAGS",
    "CoagMasks",
    "accumulation_order",
    "destination",
    "destination_census",
    "full_table_census",
    "source_pairs",
]
