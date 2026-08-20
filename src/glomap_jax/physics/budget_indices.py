"""The aerosol mass-budget slot index map — the port of `nmas*` (task 32).

`ukca_setup_indices` declares 283 `nmas*` INTEGER scalars. Each is the second
index into `bud_aer_mas(nbox, 0:nbudaer)` for one (process, component, mode)
mass flux, and each `ukca_indices_*` routine assigns a different subset, so the
map is a per-setup table. `nbudaer` is per-setup too: 8, 46, 89, 104, 107, 123
and 138 across setups 6, 1, 3, 5, 2, 4 and 8 — seven setups, seven distinct
widths.

Three properties this module exists to preserve, all of them measured against
the Fortran rather than assumed (`tests/test_budget_indices.py`, and the golden
`tests/goldens/budidx.f64.tables.npz` captured through the gate-A binding):

**Slot 0 is a hole, not a null sink.** All 344 writes in the vendored tree are
guarded by `IF (nmasxxx > 0)`, so `bud_aer_mas(:, 0)` is allocated and never
written — asserted empirically by `test_budget_slot_zero_is_never_written` on
the committed budget goldens. A port that clamps an unset index to 0 and does
`.at[0].add()` would turn the hole into an accumulator and change what the
diagnostic means.

**The index needs no rebasing.** The Fortran dimension is `0:nbudaer`, so the
Fortran slot number and the 0-based Python column index are the same number.
This is the reason `NOT_CARRIED = 0` is safe as a sentinel and `-1` is not:
`jnp.zeros(5).at[-1].add(1.0)` wraps to the last element (measured, under every
scatter mode), so a -1 sentinel would silently accumulate every unused flux
into the highest budget slot. The sentinel stays inside the array, aimed at the one column nothing
reads.

**A 0 in the table means "not carried in this setup", never "slot 0".** Look a
name up that this setup does not carry and you get 0; look up a name that does
not exist and you get `KeyError`. The two must not be the same answer — a typo
that reads as "not carried here" is how a whole process's budget goes missing
without a test noticing.

Traced, not static: `slots` is data passed into a jitted kernel, not a set of
Python ints baked into the trace. See ADR-008 for the measurement behind that,
and `apply_deltas` for the write pattern it implies.

Machine-extracted, never retyped — 283 names x 7 setups. The literals come from
`validation/capture_budget_indices.py --emit-literals`, which parses the
vendored source, and the same script's capture cross-checks them against the
compiled Fortran.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from glomap_jax.physics._budget_index_literals import (
    BUDGET_NAMES,
    SETUP_NBUDAER,
    SETUP_SLOTS,
)

NOT_CARRIED = 0
"""The value a `nmas*` scalar has when its flux is not carried in this setup.

Also the index of the hole. Both facts are load-bearing and neither is a
coincidence: the Fortran leaves the scalar at 0 and then guards every write
with `IF (nmasxxx > 0)`, so an unguarded port that scattered anyway would land
in the one column the reference never touches."""

NBUDAER_MAX = max(SETUP_NBUDAER.values())
"""138, in setup 8. ADR-002 pads the per-setup sizes to their maxima so one
compiled kernel serves all seven setups; `PADDED_WIDTH` is that padding applied
to the budget array's own dimension."""

PADDED_WIDTH = NBUDAER_MAX + 1
"""139 columns: slot 0 (the hole) plus slots 1..138. A setup with a smaller
`nbudaer` simply never writes the tail, because no name maps there."""


@dataclass(frozen=True)
class BudgetIndexMap:
    """The `nmas*` map for one `i_mode_setup`.

    Frozen and numpy-backed, like `ModeTables`: this is configuration fixed at
    build time, and it may be a static argument to `jax.jit`. What crosses into
    a traced computation is `slots`/`carried` as *data* — see `apply_deltas`.
    """

    setup: int
    nbudaer: int
    width: int
    names: tuple[str, ...]
    slots: np.ndarray
    """int32, shape (283,), aligned to `names`. Fortran slot number, which is
    also the 0-based column index. 0 means not carried."""
    carried: np.ndarray
    """bool, shape (283,). `slots > 0`, precomputed because it is the mask
    every budget write needs and recomputing it per site invites someone to
    write `slots != 0` instead, which is the same thing until a negative index
    appears."""

    def slot(self, name: str) -> int:
        """The slot for one name; 0 if this setup does not carry it.

        `KeyError` for a name that is not one of the 283, which is the whole
        point of not just returning 0 for anything unknown.
        """
        return int(self.slots[self._position(name)])

    def is_carried(self, name: str) -> bool:
        return bool(self.carried[self._position(name)])

    def carried_names(self) -> tuple[str, ...]:
        """The names this setup carries, in slot order — so `carried_names()[k]`
        is the name of slot `k + 1`.

        The sort is a no-op on every table this port ships: in all seven
        supported setups each routine assigns its slots in declaration order,
        so `self.names[self.carried]` is already sorted
        (`test_declaration_order_is_already_slot_order_in_all_seven_setups`
        measures it). It is kept anyway, because the guarantee in the first
        line is what labels a budget column — `name_of` and the column checks
        in `tests/test_budget_indices.py` both read `carried_names()[k - 1]` as
        the name of column `k` — and a vendored update that assigned out of
        declaration order would otherwise mislabel every column from the first
        divergence on, silently. Deleting the sort leaves the whole suite green
        on today's data, so it is pinned on a deliberately permuted map instead
        (`test_carried_names_is_slot_order_not_declaration_order`).
        """
        order = np.argsort(self.slots[self.carried], kind="stable")
        return tuple(np.asarray(self.names)[self.carried][order])

    def name_of(self, slot: int) -> str:
        """Reverse lookup. `ValueError` for slot 0 (the hole) and for any slot
        this setup does not carry, because there is no name to return and a
        placeholder would be indistinguishable from a real one."""
        if slot <= 0 or slot > self.nbudaer:
            raise ValueError(f"slot {slot} is outside 1..{self.nbudaer} for setup {self.setup}")
        hits = [n for n, s in zip(self.names, self.slots, strict=True) if s == slot]
        if not hits:
            raise ValueError(f"setup {self.setup} carries no name at slot {slot}")
        if len(hits) > 1:
            raise ValueError(f"setup {self.setup}: slot {slot} maps from {hits}")
        return hits[0]

    def _position(self, name: str) -> int:
        try:
            return _POSITION[name]
        except KeyError:
            raise KeyError(f"{name!r} is not one of the {len(self.names)} nmas* names") from None


_POSITION = {name: i for i, name in enumerate(BUDGET_NAMES)}
"""Name -> position in `BUDGET_NAMES`. Module level because the name list is
the same in every setup; the per-setup part is the value, not the position."""


def supported_setups() -> tuple[int, ...]:
    """1, 2, 3, 4, 5, 6, 8. Setups 10-13 exist in UKCA but the box model's
    `init_indices` ereports on them, so they have no reference."""
    return tuple(sorted(SETUP_SLOTS))


def build(setup: int, *, padded: bool = False) -> BudgetIndexMap:
    """The index map for one `i_mode_setup`.

    `padded` sets the array width to `PADDED_WIDTH` instead of this setup's own
    `nbudaer + 1`, which is what a single kernel over all seven setups needs
    (ADR-002). It changes the allocation only: the map itself is unchanged, and
    the padding columns stay zero because no name points at them.
    """
    if setup not in SETUP_SLOTS:
        raise NotImplementedError(
            f"i_mode_setup = {setup} has no budget index map; have {supported_setups()}"
        )
    slots = np.array(SETUP_SLOTS[setup], dtype=np.int32)
    nbudaer = int(SETUP_NBUDAER[setup])

    # The map is not trusted just because it was generated. These three hold in
    # all seven routines and each fails on a different way of getting the table
    # wrong: a truncated extraction, an off-by-one rebasing, and a duplicated or
    # skipped slot.
    if len(slots) != len(BUDGET_NAMES):
        raise ValueError(f"setup {setup}: {len(slots)} slots for {len(BUDGET_NAMES)} names")
    if slots.min() < 0 or slots.max() != nbudaer:
        raise ValueError(f"setup {setup}: slots run to {slots.max()}, nbudaer is {nbudaer}")
    if sorted(int(s) for s in slots[slots > 0]) != list(range(1, nbudaer + 1)):
        raise ValueError(f"setup {setup}: carried slots are not exactly 1..{nbudaer}")

    return BudgetIndexMap(
        setup=setup,
        nbudaer=nbudaer,
        width=PADDED_WIDTH if padded else nbudaer + 1,
        names=tuple(BUDGET_NAMES),
        slots=slots,
        carried=slots > 0,
    )


def apply_deltas(bud, slots, deltas, carried=None):
    """Accumulate a batch of fluxes into `bud`, the way the Fortran does.

    `bud` is `(nbox, width)`, `deltas` is `(nbox, nsites)`, and `slots` is the
    `(nsites,)` int array of destination slots — normally `map.slots[[...]]`
    for the sites one routine writes. One fused scatter-add, not one update per
    site.

    Two things this does that a naive scatter does not:

    * **Masks before scattering, never multiplies.** `0.0 * inf` is `NaN`, and
      an uncarried flux is exactly the place an unmasked term can be garbage.
    * **Preserves the hole exactly.** Uncarried sites still scatter, into slot
      0, but they scatter a bit-exact `0.0`, so `bud[:, 0]` stays zero. That is
      cheaper than filtering the sites out and, unlike filtering, it does not
      need the mask to be known at trace time — which is what keeps one kernel
      serving all seven setups.

    `mode="drop"` on the scatter is belt and braces: every slot is in range by
    construction, and an out-of-range one is dropped by default anyway. What it
    does NOT protect against is a negative index, which wraps under every mode
    — see `test_a_negative_sentinel_would_silently_corrupt_the_last_slot`.

    ACCUMULATE, which is 340 of the 344 sites. The other four —
    `nmasclprsuaccsol1/2` and `nmasclprsucorsol1/2` in `ukca_aero_step.F90:750`
    onwards — **overwrite**: `bud_aer_mas(:, n) = frac * delso2 * ...`, with no
    read of the old value. Routing those through here would silently turn a
    per-step cloud-processing flux into a running total. Pinned by
    `test_four_write_sites_overwrite_rather_than_accumulate`.
    """
    slots = jnp.asarray(slots)
    deltas = jnp.asarray(deltas)
    if carried is None:
        carried = slots > 0
    return bud.at[:, slots].add(jnp.where(jnp.asarray(carried)[None, :], deltas, 0.0), mode="drop")
