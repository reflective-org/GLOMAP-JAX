#!/usr/bin/env python3
"""Sweep `ukca_vapour` at both `l_fix_neg_pvol_wat` settings (task 35b).

    python validation/capture_vapour_leaf.py --dry-run
    python validation/capture_vapour_leaf.py      # writes tests/goldens/vapour.f64.leaf.npz

`ukca_vapour` is setup-independent -- it takes no `glomap_variables` argument
and reads no per-setup table -- so the configuration axis is just the one
fidelity flag, and one subprocess per setting is enough. Two children, each
running `wrap_init` on the shipped `boundary_layer.nml` (`i_mode_setup = 1`),
then `wrap_set_fix_neg_pvol_wat`, then reading the flag back OUT OF THE FORTRAN
before it sweeps anything (`leaf_common.CHILD_PREAMBLE`).

Why these grids
---------------

The routine is 50 lines of arithmetic with eight decisions in it, and a
trajectory reaches almost none of them: `bh2o` sits between its two clamps, `d`
stays positive, `xsb` stays inside (0, 1), and `wts` stays inside the lookup
table. Every grid below exists to land on one of the eight.

`wts` and `rhosol_strat` depend on `(t, bh2o)` alone, and `bh2o` is
`1.609*s*(pmid/p0)` clamped to `[2e-8, 2e-6]` (`:140-142`). So the temperature
sweep is run at three *atmospheres* -- one clamped to `bminatm`, one strictly
between the clamps, one clamped to `bmaxatm` -- and the (`pmid`, `s`) plane is
swept separately at four fixed temperatures. A full cross product would be
150,000 rows of which all but a few thousand say the same thing.

The dense clusters sit on branch boundaries, and each boundary is a root of a
transcendental equation in `t`:

* `d_cold`, `d_hot` -- `IF (d < 0.0) d=0.0` at `:170`, which has TWO roots and
  not one. Below the cold root `2b < 0`, so the clamped `xsb` is negative and
  `:176` fires as well. Above the hot root `2b > 0`, `xsb = -a/(2b)` is greater
  than 1, `msb` changes sign and `wts` runs past 99 with `:176` never firing.
  Those are different branch states; the plan's fixture list had only the cold
  one.
* `xsb_eps` -- `IF (xsb(jl) < xsb_eps)` at `:176`, released where `c` changes
  sign.
* `ws41.0` -- the `MAX(41.0, ...)` floor, common to both arms.
* `ws42.5` through `ws92.5` -- every step of `(NINT(wts/5))*5` at `:226`, so
  every row of the `percent` table the density is looked up in.
* `ws97.5` -- the last of those steps. `round` becomes 100, `percent` stops at
  95 (`:90`), and the density falls through to the 1300.0 of `:223`.
* `ws99.0` -- the only place the two flag settings differ: `:184` caps, `:188`
  does not.
* `xsb_one` -- `msb = 55.51*xsb/(1.0-xsb)` at `:178`, an unguarded pole.
* `b_zero` -- `2.0*b` at `:175`, an unguarded pole of the other kind, and not
  the one the plan described. See below.

The roots are re-derived, not transcribed
-----------------------------------------

`capture_leaf`'s rule -- no grid point may come from a libm call -- cannot hold
here: these abscissae are roots of an equation containing `LOG`, and there is
no decimal literal that is one. What can hold, and does, is that no root is
**typed in**. Each is found at capture time by bisecting the *same predicate the
Fortran evaluates*, in double space, until the bracket is two adjacent doubles;
the archive stores the bracket and the residual either side of it. A root
transcribed to sixteen digits and then drifting by one ulp stops being on the
edge it was chosen for, silently -- which is the whole reason those points are
in the grid. A re-derived one lands on the edge wherever it is run.

The bisection uses `numpy.log`, which is bit-identical to gfortran's on the
capture platform (`docs/porting-notes.md`), and the transcription it bisects is
checked against the Fortran on every row of the sweep: `check_transcription`
requires the predicted `wts` and `rhosol_strat` to be **byte-identical** to what
the driver returned, all N rows, both flag settings. That check is what makes
the branch-hit counts below evidence rather than commentary -- they are counted
off the transcription, because `d`, `xsb` and `bh2o` never leave the routine.

The exception is `b_zero`, which is a division and so *is* exact:
`b = ks3 + ks4*(1/t)` vanishes at `t = 15732.0/51.81`, and at that double it
vanishes **exactly**.

Both poles are hit, and neither behaves the way the plan said
-------------------------------------------------------------

Nine of the 5,797 rows come out of `:179` with `ws` NaN, by two different
routes, and on every one of them the routine returns `wts = 41.0`.

**`b = 0`, three rows.** The plan recorded `xsb = 0.849354` there, from
`numerator = -2.8985e-10` over `2b = -3.4126e-10`. That is the *limit* -- as
`b -> 0`, `d -> a*a`, so `xsb -> -c/a` -- and it is what you get about 2e-9 K
away. At the double `15732.0/51.81` itself, measured on the pinned toolchain,
`b` is 0.0 exactly, `d` is `a*a` exactly, `SQRT(a*a)` is `-a` exactly, so the
numerator is 0.0 exactly and `xsb` is `0/0`, NaN.

**`xsb = 1`, six rows.** The plan said to take the two doubles either side and
"do not expect to hit it". It is hit, and not by luck: near that root `xsb - 1`
is a cancellation of order 1e-15, so `xsb` moves in plateaus of about twelve
ulps and one of those plateaus is the exact value 1.0, two consecutive doubles
wide. Bisecting `xsb > 1.0` therefore lands *on* the pole rather than beside
it, in two of the three atmospheres, and `msb = 55.51*xsb/(1.0-xsb)` at `:178`
is `+inf`, so `ws = inf/inf` is NaN at `:179`.

From there both routes are the same: `IF (xsb < xsb_eps)` is false for a NaN,
and `MAX(41.0, NaN)` in the compiled `ukca_vapour` returns **41.0** -- so
`wts = 41.0`, `round = 40`, and `rhosol_strat` is a *table* value, 1293.28 at
the `b = 0` pole, not the 1300.0 fall-through. That is codegen, not language:
gfortran's MAX with a NaN operand is not defined by the standard, and the same
expression compiled outside a loop on the same compiler propagates the NaN
instead. It is in the golden because it is what the routine being ported does,
and `NAN_MAX_IS_41` below is the rule the transcription needs to reproduce it.
A port whose maximum propagates NaN -- `jnp.maximum` does -- differs on those
nine rows and nowhere else.

The `rp` argument is dead, and the sweep proves it
--------------------------------------------------

`rp` enters at `:198` and feeds `kelvin`, `kelvin_out`, `muh2so4` and `ph2so4`,
none of which reaches either INTENT(OUT). The probe re-runs 64 rows at four
`rp` including **0.0** -- which makes `:198` a division by zero -- and requires
the outputs to be byte-identical. If the chain were live, that row could not
come back finite and unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from capture_leaf import DECADE_4, _decade_grid
from leaf_common import NAMELISTS, REPO, check_varied, run_child

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "vapour.f64.leaf.npz"
NAMELIST = NAMELISTS / "boundary_layer.nml"
SETUP = 1

# ukca_vapour.F90:46-56, as the Fortran spells them. Repeated here rather than
# imported because this module has to reproduce the routine's arithmetic
# operation for operation -- see check_transcription.
KS1, KS2, KS3, KS4 = -21.661, 2724.2, 51.81, -15732.0
KS5, KS6, KS7 = 47.004, -6969.0, -4.6183
XSB_EPS = 1.0e-6
P0 = 101325.0
BMINATM = 2.0e-8
BMAXATM = 2.0e-6

# The density lookup, ukca_vapour.F90:90-100. `percent` stopping at 95 is the
# reason every `round >= 100` falls through to the 1300.0 of :223.
PERCENT = (40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)
K_DIFF = (0.80, 0.82, 0.84, 0.86, 0.87, 0.89, 0.92, 0.95, 0.98, 1.02, 1.06, 1.10)
DATA253 = (
    1333.8, 1381.1, 1431.0, 1483.3, 1537.3, 1592.4,
    1647.6, 1701.6, 1753.0, 1800.1, 1840.9, 1873.2,
)  # fmt: skip

#: `MAX(41.0, NaN)` returns 41.0 in the compiled `ukca_vapour`, so a NaN `ws`
#: leaves `wts` on the floor rather than poisoning it. Measured, not assumed;
#: see the module docstring. Nine rows of the sweep reach it, from the `b = 0`
#: pole and from `xsb = 1`; a port using `jnp.maximum`, which propagates, gives
#: NaN on those nine and agrees everywhere else.
NAN_MAX_IS_41 = True

#: `b = ks3 + ks4*(1/t)` vanishes here, exactly, at this double. A division of
#: two literals rather than a transcribed decimal, so it is reproducible
#: wherever it is evaluated; `check_b_zero_is_exact` re-checks the residual.
T_B_ZERO = 15732.0 / 51.81

#: Keep the root scans away from `T_B_ZERO`, where `xsb` is dominated by the
#: cancellation of two ~1e-14 quantities and crosses everything several times.
#: No branch root of interest is within 1 K of it -- the nearest are 301.72 and
#: 305.82 -- and `_root` asserts it finds exactly one crossing, so a cluster
#: swallowed by the guard band fails loudly rather than quietly.
POLE_GUARD = 1.0

#: Three mantissas per decade, as decimal STRINGS: `10**(k/3)` to three
#: significant figures. Same rule as `capture_leaf.DECADE_30` -- a literal is
#: converted to binary by a correctly-rounded conversion, `np.logspace` is a
#: libm `pow` per point and is not.
DECADE_3 = ("1.00", "2.15", "4.64")

#: Temperatures the (pmid, s) plane is swept at. 200 K is below every `ws*100
#: = 41` root, so the floor is active whatever the humidity; 253.0 is where
#: `t_diff` at :229 is exactly zero; 300 K is inside the table; 330 K is past
#: every `ws*100 = 99` root, where the two flag settings differ.
ANCHOR_T = (200.0, 253.0, 300.0, 330.0)

#: The coarse temperature sweep: 150 to 340 K at 0.5 K, plus the two
#: temperatures that are special to the *dead* half of the routine -- 253.0 is
#: `t_diff = 0` at :229 and 360.0 is the Kulmala and Laaksonen reference at
#: :214, where `360.0/t` is 1 and its log is 0.
T_COARSE_LO, T_COARSE_HI, T_COARSE_STEP = 150.0, 340.0, 0.5
T_EXTRA = (253.0, 360.0)

#: rp for the main sweep, and the four the invariance probe uses. 0.0 is one of
#: them on purpose: it makes :198 a division by zero, so an rp that reached an
#: output could not come back byte-identical to the others.
RP_MAIN = 1.0e-7
RP_PROBE = (0.0, 1.0e-9, 1.0e-7, 1.0e-5)
RP_PROBE_ROWS = 64

#: `(NINT(x/5))*5` ties, fed to `leaf_vapour_round` directly. Reaching an exact
#: tie through the LOG/SQRT chain is not practical -- no row of the sweep lands
#: on one -- so the idiom is exercised through the shim instead, at every tie
#: the live `wts` range can produce and at the two doubles either side.
TIE_WTS = tuple(42.5 + 5.0 * k for k in range(12))

#: The one collision that is a finding about the Fortran rather than a capture
#: bug. The two arms differ only where `ws*100 > 99`; there the clamped arm
#: gives `wts = 99` and the other gives more, and `(NINT(wts/5))*5` sends both
#: to 100 or above while `percent` stops at 95 -- so both fall through to
#: `rhosol_strat = 1300.0`. Asserted, not assumed: `check_records` requires the
#: two settings to be byte-identical on `rhosol_strat` and to differ on `wts`.
IDENTICAL_ON_RHOSOL = (("fix_neg_pvol_0", "fix_neg_pvol_1"),)

#: One entry per decision in the routine, counted off the transcription and
#: stored in the archive. Criterion 3 of task 35: every one asserted > 0, or
#: named in EXPECTED_ZERO with the reason it cannot fire.
BRANCH_NAMES = (
    "bh2o_below_bminatm",  # :141 fires
    "bh2o_above_bmaxatm",  # :142 fires
    "bh2o_between_limits",  # neither fires
    "s_nonpositive",  # the floor at :141 is what keeps LOG at :149 defined
    "d_negative_cold",  # :170 fires below the b = 0 pole
    "d_negative_hot",  # :170 fires above it -- a different branch state
    "d_positive",  # :170 does not fire
    "xsb_below_eps",  # :176 fires
    "xsb_above_one",  # msb changes sign at :178
    "xsb_is_nan",  # b == 0 exactly, so :175 is 0/0
    "xsb_exactly_one",  # msb is +inf at :178
    "ws_is_nan",  # either pole, and the input to MAX(41.0, NaN)
    "wts_floor_41",  # MAX picked the floor, :184 or :188
    "wts_cap_99",  # MIN picked the cap, :184 only
    "wts_above_99",  # no ceiling, :188 only
    "round_ge_100",  # no percent match, so :223's 1300.0 survives
    *(f"round_{p}" for p in PERCENT),
)

#: (flag, branch) pairs that must be exactly zero, with the reason. Both are
#: structural: the cap exists only in the `l_fix_neg_pvol_wat` arm, and the
#: uncapped `wts` exists only in the other.
EXPECTED_ZERO = {
    (0, "wts_cap_99"): "ukca_vapour.F90:188 has no ceiling",
    (1, "wts_above_99"): "ukca_vapour.F90:184 caps at 99",
}


# ---------------------------------------------------------------------------
# The transcription: ukca_vapour.F90:136-234, operation for operation.
# ---------------------------------------------------------------------------


def bh2o_of(pmid: np.ndarray, s: np.ndarray) -> np.ndarray:
    """`:136-142`. The two clamps, in the order the Fortran applies them."""
    b = 1.609 * s * (pmid / P0)
    b = np.where(b < BMINATM, BMINATM, b)
    return np.where(b > BMAXATM, BMAXATM, b)


def intermediates(t: np.ndarray, bh2o: np.ndarray) -> dict[str, np.ndarray]:
    """`:143-179`. Everything the routine computes and does not return.

    Note `ust = 1/t` then `ks2*ust`, not `ks2/t`: the two differ in the last
    place and the roots below are roots of the first one.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ust = 1.0 / t
        tlog = np.log(t)
        pwlog = np.log(bh2o)
        a = KS1 + KS2 * ust
        b = KS3 + KS4 * ust
        c = KS5 + KS6 * ust + KS7 * tlog - pwlog
        d_raw = a * a - 4.0 * b * c
        d = np.where(d_raw < 0.0, 0.0, d_raw)
        xsb_raw = (-a - np.sqrt(d)) / (2.0 * b)
        xsb = np.where(xsb_raw < XSB_EPS, XSB_EPS, xsb_raw)
        msb = 55.51 * xsb / (1.0 - xsb)
        ws = msb * 0.098076 / (1.0 + msb * 0.098076)
    return {
        "a": a, "b": b, "c": c, "d_raw": d_raw, "d": d,
        "xsb_raw": xsb_raw, "xsb": xsb, "msb": msb, "ws": ws,
    }  # fmt: skip


def wts_of(t: np.ndarray, bh2o: np.ndarray, fix: int) -> np.ndarray:
    """`:182-189`, including gfortran's MAX-with-a-NaN (see NAN_MAX_IS_41)."""
    ws100 = intermediates(t, bh2o)["ws"] * 100.0
    floor = np.maximum(41.0, ws100)
    if NAN_MAX_IS_41:
        floor = np.where(np.isnan(ws100), 41.0, floor)
    return np.minimum(99.0, floor) if fix else floor


def nint(x: np.ndarray) -> np.ndarray:
    """Fortran NINT: half AWAY FROM ZERO, where numpy rounds half to even."""
    return np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5))


def rhosol_of(t: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """`:223-236`. 1300.0 unless `(NINT(wts/5))*5` matches a `percent` entry."""
    out = np.full(np.shape(t), 1300.0)
    t_diff = 253.0 - t
    rounded = nint(wts / 5.0) * 5.0
    for k, pct in enumerate(PERCENT):
        out = np.where(rounded == pct, DATA253[k] + K_DIFF[k] * t_diff, out)
    return out


# ---------------------------------------------------------------------------
# Roots. Re-derived every run; never transcribed.
# ---------------------------------------------------------------------------

#: Predicates in the form the Fortran writes them. Each is True on one side of
#: its root and False on the other, so bisecting on the boolean lands on the
#: branch edge rather than near it.
PREDICATES = {
    "d_cold": lambda it: it["d_raw"] < 0.0,
    "d_hot": lambda it: it["d_raw"] < 0.0,
    "xsb_eps": lambda it: it["xsb_raw"] < XSB_EPS,
    "xsb_one": lambda it: it["xsb"] > 1.0,
}
for _target in (41.0, *TIE_WTS, 99.0):
    PREDICATES[f"ws{_target}"] = (lambda tgt: lambda it: it["ws"] * 100.0 < tgt)(_target)

#: Clusters that get the full ulp ladder, and the ones that get the short one.
#: The eleven interior table steps are branch edges too, but they are steps
#: between two lookup rows rather than between two branch states, so they get
#: fifteen points each instead of ninety-one.
WIDE_CLUSTERS = ("d_cold", "xsb_eps", "ws41.0", "ws97.5", "ws99.0", "xsb_one", "d_hot")
NARROW_CLUSTERS = tuple(f"ws{v}" for v in TIE_WTS if v != 97.5)


def _ulp_ladder(wide: bool) -> np.ndarray:
    """Offsets in ulps: every one out to 8, then doubling to about 0.05 K.

    "±0.05 K at ulp density" is 1.8e12 points at 300 K, so the density thins
    geometrically with distance. What matters is that the two doubles either
    side of the edge are both in, and that nothing between them can be.
    """
    if wide:
        steps = list(range(1, 9)) + [1 << j for j in range(4, 41)]
    else:
        steps = [1, 2, 3, 4, 1 << 20, 1 << 30, 1 << 36]
    offsets = sorted({0} | {s for s in steps} | {-s for s in steps})
    return np.array(offsets, dtype=np.int64)


def _shift_ulps(x: float, offsets: np.ndarray) -> np.ndarray:
    """`x` displaced by whole ulps, through the bit pattern.

    Exact and libm-free. Only valid for x > 0, where the int64 ordering of the
    bit pattern is the float ordering; every temperature here is.
    """
    if not x > 0.0:
        raise SystemExit(f"_shift_ulps needs a positive centre, got {x!r}")
    bits = np.array(x, dtype=np.float64).view(np.int64)
    return (bits + offsets).view(np.float64)


def _sign_changes(pred, bh2o: float, lo: float, hi: float, n: int = 20001) -> list:
    values = np.linspace(lo, hi, n)
    flags = pred(intermediates(values, np.full(n, bh2o)))
    idx = np.nonzero(flags[:-1] != flags[1:])[0]
    return [(values[i], values[i + 1]) for i in idx]


def _bisect(pred, bh2o: float, lo: float, hi: float) -> tuple[float, float]:
    """Narrow a bracket to two adjacent doubles. Exact: only comparison and
    the midpoint, which is `lo + (hi-lo)/2` and stays inside the bracket."""

    def at(x):
        return bool(pred(intermediates(np.array([x]), np.array([bh2o])))[0])

    below = at(lo)
    while np.nextafter(lo, hi) != hi:
        mid = lo + (hi - lo) / 2.0
        if mid == lo or mid == hi:
            break
        if at(mid) == below:
            lo = mid
        else:
            hi = mid
    return lo, hi


def roots(bh2o: float) -> dict[str, tuple[float, float]]:
    """Every branch edge in temperature, at one water vapour pressure.

    Each is returned as the two adjacent doubles the branch flips between.
    Raises if a predicate does not change sign exactly once in the searched
    range, which is what would happen if the guard band swallowed a root or a
    grid change moved one.
    """
    found: dict[str, tuple[float, float]] = {}
    for name in ("d_cold", "d_hot"):
        lo, hi = (
            (T_COARSE_LO, T_B_ZERO - POLE_GUARD)
            if name == "d_cold"
            else (T_B_ZERO + POLE_GUARD, 400.0)
        )
        changes = _sign_changes(PREDICATES[name], bh2o, lo, hi)
        if len(changes) != 1:
            raise SystemExit(
                f"bh2o={bh2o!r}: {name} changes sign {len(changes)} times in "
                f"[{lo}, {hi}], expected once -- the cluster would not be on an edge"
            )
        found[name] = _bisect(PREDICATES[name], bh2o, *changes[0])

    # Everything else is searched below the b = 0 pole and above it up to the
    # hot d = 0 root. Past that root `xsb = -a/(2b)` falls back through 1 and
    # `ws*100` back through 99, 97.5 and 92.5 -- around 350 to 380 K, above the
    # swept range -- so a search that ran to 400 K would find two roots and
    # this would refuse to write.
    hot = found["d_hot"][0]
    for name, pred in PREDICATES.items():
        if name in ("d_cold", "d_hot"):
            continue
        changes = _sign_changes(pred, bh2o, T_COARSE_LO, T_B_ZERO - POLE_GUARD)
        changes += _sign_changes(pred, bh2o, T_B_ZERO + POLE_GUARD, hot)
        if len(changes) != 1:
            raise SystemExit(
                f"bh2o={bh2o!r}: {name} changes sign {len(changes)} times below "
                f"{hot!r}, expected once"
            )
        found[name] = _bisect(pred, bh2o, *changes[0])
    return found


def check_b_zero_is_exact() -> tuple[float, float]:
    """`b` must vanish exactly at T_B_ZERO, or the cluster is not on the pole.

    Returns the residual either side, which goes into the archive: one ulp
    away in each direction `b` is +-1.4e-14, and that is the cancellation the
    port has to reproduce.
    """
    t = np.array([np.nextafter(T_B_ZERO, 0.0), T_B_ZERO, np.nextafter(T_B_ZERO, 400.0)])
    b = intermediates(t, np.full(3, BMAXATM))["b"]
    if b[1] != 0.0:
        raise SystemExit(
            f"b = {b[1]!r} at T_B_ZERO = {T_B_ZERO!r}, expected exactly 0.0 -- "
            "the 0/0 that makes xsb NaN is no longer reachable at this double"
        )
    return float(b[0]), float(b[2])


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def atmospheres() -> dict[str, tuple[float, float]]:
    """(pmid, s) triples that put `bh2o` on each side of its two clamps.

    `bminatm` and `bmaxatm` are reached by clamping, so those two are exact
    whatever `s` is; the middle one uses `pmid = p0`, where `patm` is exactly
    1, and `s = 2e-7/1.609`, a division and so correctly rounded.
    """
    return {
        "bmin": (1.0e5, 1.0e-9),
        "mid": (P0, 2.0e-7 / 1.609),
        "bmax": (1.0e5, 1.0e-3),
    }


def pmid_grid() -> np.ndarray:
    """Thirteen decade points 1e2 to 1e5, plus 1.05e5, plus `putls` and its two
    neighbours -- `ukca_volume_mode.F90:258` cuts the UTLS branch at 1.5e4, and
    a paired volume_mode fixture has to straddle it on the same rows."""
    extra = [1.05e5, 1.5e4, np.nextafter(1.5e4, 0.0), np.nextafter(1.5e4, 1e9)]
    return np.unique(np.concatenate([_decade_grid(2, 5, DECADE_4), extra]))


def s_grid(pmid: float) -> np.ndarray:
    """Twenty-one decade points 1e-8 to 4.64e-2, the two specific humidities
    that put `bh2o` exactly on each clamp at this pressure with their
    neighbours, and the two non-positive values.

    `s <= 0` is the point: `bh2o` goes to zero or negative and the floor at
    `:141` is the only thing standing between `LOG` at `:149` and -inf or NaN.
    """
    decade = np.concatenate([_decade_grid(-8, -2, DECADE_3), [2.15e-2, 4.64e-2]])
    onto = []
    for target in (BMINATM, BMAXATM):
        s = target / (1.609 * (pmid / P0))
        onto += [s, np.nextafter(s, 0.0), np.nextafter(s, 1.0)]
    return np.unique(np.concatenate([decade, onto, [0.0, -1.0e-6]]))


def _t_axis(bh2o: float) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """The temperature axis at one water vapour pressure, plus its roots."""
    n = round((T_COARSE_HI - T_COARSE_LO) / T_COARSE_STEP) + 1
    parts = [np.linspace(T_COARSE_LO, T_COARSE_HI, n), np.array(T_EXTRA)]
    edges = roots(bh2o)
    for name, (lo, _hi) in edges.items():
        parts.append(_shift_ulps(lo, _ulp_ladder(name in WIDE_CLUSTERS)))
    parts.append(_shift_ulps(T_B_ZERO, _ulp_ladder(True)))
    return np.unique(np.concatenate(parts)), edges


def grid() -> dict:
    """Every row of the sweep, and the metadata that says where it came from."""
    t_parts, p_parts, s_parts, block = [], [], [], []
    roots_by: dict[str, dict[str, tuple[float, float]]] = {}

    for name, (pmid, s) in atmospheres().items():
        bh2o = float(bh2o_of(np.array(pmid), np.array(s)))
        axis, edges = _t_axis(bh2o)
        roots_by[name] = edges
        t_parts.append(axis)
        p_parts.append(np.full(axis.shape, pmid))
        s_parts.append(np.full(axis.shape, s))
        block += [f"t_{name}"] * len(axis)

    for t in ANCHOR_T:
        for pmid in pmid_grid():
            s = s_grid(pmid)
            t_parts.append(np.full(s.shape, t))
            p_parts.append(np.full(s.shape, pmid))
            s_parts.append(s)
            block += [f"ps_{t:g}"] * len(s)

    return {
        "t": np.concatenate(t_parts),
        "pmid": np.concatenate(p_parts),
        "s": np.concatenate(s_parts),
        "block": np.array(block, dtype=np.str_),
        "roots": roots_by,
    }


def branch_hits(t: np.ndarray, pmid: np.ndarray, s: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """How many rows reach each decision in BRANCH_NAMES, at one flag setting."""
    bh2o = bh2o_of(pmid, s)
    raw = 1.609 * s * (pmid / P0)
    it = intermediates(t, bh2o)
    ws100 = it["ws"] * 100.0
    rounded = nint(wts / 5.0) * 5.0
    hits = {
        "bh2o_below_bminatm": raw < BMINATM,
        "bh2o_above_bmaxatm": raw > BMAXATM,
        "bh2o_between_limits": (raw >= BMINATM) & (raw <= BMAXATM),
        "s_nonpositive": s <= 0.0,
        "d_negative_cold": (it["d_raw"] < 0.0) & (t < T_B_ZERO),
        "d_negative_hot": (it["d_raw"] < 0.0) & (t > T_B_ZERO),
        "d_positive": it["d_raw"] >= 0.0,
        "xsb_below_eps": it["xsb_raw"] < XSB_EPS,
        "xsb_above_one": it["xsb"] > 1.0,
        "xsb_is_nan": np.isnan(it["xsb_raw"]),
        "xsb_exactly_one": it["xsb"] == 1.0,
        "ws_is_nan": np.isnan(ws100),
        "wts_floor_41": wts == 41.0,
        "wts_cap_99": (wts == 99.0) & (ws100 > 99.0),
        "wts_above_99": wts > 99.0,
        "round_ge_100": rounded >= 100.0,
    }
    for pct in PERCENT:
        hits[f"round_{pct}"] = rounded == pct
    return np.array([int(hits[name].sum()) for name in BRANCH_NAMES], dtype=np.int64)


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------

# Not indented: `run_child` dedents the body, and a dedent of the prologue plus
# this would find no common prefix and leave the body indented.
_CHILD_BODY = """
sys.path.insert(0, _VALIDATION)
from leaf_common import bind_call

call = bind_call(g)
grid = np.load(_INPUTS)
t, pmid, s = grid["t"], grid["pmid"], grid["s"]
rp = np.full(t.shape, float(grid["rp_main"]))

wts, rhosol, _ = call("leaf_vapour(sweep)", g.leaf_vapour, t, pmid, s, rp)

# rp invariance. Four values including 0.0, which makes :198 a division by
# zero; if rp reached an output this could not come back unchanged.
probe_w, probe_r = [], []
pt, pp, ps = grid["probe_t"], grid["probe_pmid"], grid["probe_s"]
for value in grid["rp_probe"]:
    rpv = np.full(pt.shape, float(value))
    w, r, _ = call("leaf_vapour(rp probe)", g.leaf_vapour, pt, pp, ps, rpv)
    probe_w.append(w.tolist())
    probe_r.append(r.tolist())

# nbox > 1 against nbox = 1. The routine is three loops with two whole-array
# sections between them (:194-199); a one-row fixture cannot see a mis-
# vectorised port.
rows = grid["nbox_rows"]
many_w, many_r, _ = call(
    "leaf_vapour(nbox=8)", g.leaf_vapour, t[rows], pmid[rows], s[rows], rp[rows]
)
one_w, one_r = [], []
for i in rows:
    j = int(i)
    w, r, _ = call(
        "leaf_vapour(nbox=1)", g.leaf_vapour, t[j:j+1], pmid[j:j+1], s[j:j+1], rp[j:j+1]
    )
    one_w.append(float(w[0]))
    one_r.append(float(r[0]))

# The NINT tie idiom, through the shim rather than through the LOG/SQRT chain.
tie_y = call("leaf_vapour_round", g.leaf_vapour_round, grid["tie_x"])

result = {
    "wts": wts.tolist(),
    "rhosol_strat": rhosol.tolist(),
    "probe_wts": probe_w,
    "probe_rhosol": probe_r,
    "nbox8_wts": many_w.tolist(),
    "nbox8_rhosol": many_r.tolist(),
    "nbox1_wts": one_w,
    "nbox1_rhosol": one_r,
    "tie_y": tie_y.tolist(),
    "flags": [int(_fw), int(_fn), int(_got_setup)],
}
print("@@RESULT@@" + json.dumps(result))
"""


def run_flag(flag: int, inputs: Path) -> dict:
    """One subprocess, one `l_fix_neg_pvol_wat` setting."""
    prologue = f"_INPUTS = {str(inputs)!r}\n_VALIDATION = {str(REPO / 'validation')!r}\n"
    rec = run_child(
        prologue + _CHILD_BODY,
        namelist_text=NAMELIST.read_text(encoding="utf-8"),
        setup=SETUP,
        fix_neg_pvol=flag,
        label=f"l_fix_neg_pvol_wat={flag}",
    )
    # The preamble already refused a mismatch from inside the child; this is
    # the parent refusing to believe a record that says otherwise.
    if rec["flags"] != [1, flag, SETUP]:
        raise SystemExit(
            f"l_fix_neg_pvol_wat={flag}: child reported flags {rec['flags']}, "
            f"wanted [1, {flag}, {SETUP}]"
        )
    return rec


# ---------------------------------------------------------------------------
# Anti-collapse, all of it before np.savez_compressed
# ---------------------------------------------------------------------------


def check_transcription(g: dict, records: dict[int, dict]) -> None:
    """The transcription must reproduce the Fortran byte for byte, every row.

    This is what makes the branch counts evidence: `bh2o`, `d` and `xsb` never
    leave the routine, so they can only be counted off a re-implementation, and
    a re-implementation nobody checked is a guess. Byte equality on both
    outputs at both flag settings is the check.
    """
    for flag, rec in records.items():
        want_wts = wts_of(g["t"], bh2o_of(g["pmid"], g["s"]), flag)
        got_wts = np.array(rec["wts"], dtype=np.float64)
        bad = np.nonzero(want_wts.view(np.int64) != got_wts.view(np.int64))[0]
        if bad.size:
            i = int(bad[0])
            raise SystemExit(
                f"flag {flag}: the transcription disagrees with ukca_vapour on "
                f"{bad.size} of {len(got_wts)} rows -- first at t={g['t'][i]!r}, "
                f"pmid={g['pmid'][i]!r}, s={g['s'][i]!r}: "
                f"wts {got_wts[i]!r} from the Fortran, {want_wts[i]!r} here"
            )
        want_rho = rhosol_of(g["t"], got_wts)
        got_rho = np.array(rec["rhosol_strat"], dtype=np.float64)
        bad = np.nonzero(want_rho.view(np.int64) != got_rho.view(np.int64))[0]
        if bad.size:
            i = int(bad[0])
            raise SystemExit(
                f"flag {flag}: the density lookup disagrees on {bad.size} rows -- "
                f"first at t={g['t'][i]!r}: {got_rho[i]!r} from the Fortran, "
                f"{want_rho[i]!r} here"
            )


def check_records(g: dict, records: dict[int, dict]) -> dict[str, int]:
    """Every anti-collapse rule for this archive. Returns what it witnessed."""
    named = {f"fix_neg_pvol_{flag}": rec for flag, rec in records.items()}

    # (i) the two settings are not the same record ...
    check_varied(
        {k: {"wts": v["wts"], "rhosol": v["rhosol_strat"]} for k, v in named.items()},
        what="l_fix_neg_pvol_wat settings",
    )
    # ... and (ii) they are the same record on rhosol_strat, which is the
    # provable half and so is recorded as an expected collision.
    check_varied(
        {k: {"rhosol": v["rhosol_strat"]} for k, v in named.items()},
        expected_identical=IDENTICAL_ON_RHOSOL,
        what="rhosol_strat across l_fix_neg_pvol_wat",
    )

    w0 = np.array(records[0]["wts"], dtype=np.float64)
    w1 = np.array(records[1]["wts"], dtype=np.float64)
    r0 = np.array(records[0]["rhosol_strat"], dtype=np.float64)
    r1 = np.array(records[1]["rhosol_strat"], dtype=np.float64)

    if r0.tobytes() != r1.tobytes():
        raise SystemExit(
            "rhosol_strat is NOT byte-identical across l_fix_neg_pvol_wat. "
            "docs/porting-notes.md says it must be -- the two arms differ only "
            "where ws*100 > 99, and (NINT(wts/5))*5 sends 99 and anything above "
            "it alike to >= 100, past the end of `percent`. Investigate before "
            "re-blessing; this is a finding about the Fortran, not a tolerance"
        )

    differs = w0.view(np.int64) != w1.view(np.int64)
    ws100 = intermediates(g["t"], bh2o_of(g["pmid"], g["s"]))["ws"] * 100.0
    if not np.array_equal(differs, ws100 > 99.0):
        n = int((differs != (ws100 > 99.0)).sum())
        raise SystemExit(
            f"the two settings differ on {int(differs.sum())} rows but ws*100 > 99 "
            f"on a different set ({n} rows disagree) -- :184 and :188 differ "
            "exactly where the cap binds and nowhere else"
        )
    if not differs.any():
        raise SystemExit("the two settings produced identical wts -- the flag never took")
    hot = differs & (g["t"] > 310.54)
    if not hot.any():
        raise SystemExit(
            "no row above 310.54 K distinguishes the two settings, so the "
            "bmaxatm arm of the ws*100 = 99 cluster is missing"
        )
    if not (np.isfinite(w0).all() and np.isfinite(w1).all()):
        raise SystemExit("wts is not finite everywhere; a NaN cannot be compared byte-wise")
    if not (np.isfinite(r0).all() and np.isfinite(r1).all()):
        raise SystemExit("rhosol_strat is not finite everywhere")

    # rp invariance. Byte equality, not closeness: if this fails the dead-chain
    # analysis behind task 38's scope is wrong.
    for flag, rec in records.items():
        base_w = np.array(rec["probe_wts"][0], dtype=np.float64)
        base_r = np.array(rec["probe_rhosol"][0], dtype=np.float64)
        for k, value in enumerate(RP_PROBE[1:], start=1):
            got_w = np.array(rec["probe_wts"][k], dtype=np.float64)
            got_r = np.array(rec["probe_rhosol"][k], dtype=np.float64)
            if base_w.tobytes() != got_w.tobytes() or base_r.tobytes() != got_r.tobytes():
                raise SystemExit(
                    f"flag {flag}: rp = {value!r} changed the output. rp feeds "
                    "kelvin/muh2so4/ph2so4 at :198-216 and none of those reaches "
                    "an INTENT(OUT) -- if it does, task 38's scope changes"
                )

    # nbox > 1 against nbox = 1, byte-wise.
    for flag, rec in records.items():
        for field in ("wts", "rhosol"):
            many = np.array(rec[f"nbox8_{field}"], dtype=np.float64)
            one = np.array(rec[f"nbox1_{field}"], dtype=np.float64)
            if many.tobytes() != one.tobytes():
                raise SystemExit(
                    f"flag {flag}: the nbox=8 call disagrees with eight nbox=1 calls "
                    f"on {field} -- ukca_vapour's whole-array sections at :194-199 "
                    "are not row-independent after all"
                )

    return {"rows_flag_differs": int(differs.sum()), "rows_above_310_54": int(hot.sum())}


def check_branches(hits: np.ndarray) -> None:
    """Every predicate reached, or explicitly expected not to be."""
    for flag in (0, 1):
        for k, name in enumerate(BRANCH_NAMES):
            count = int(hits[flag, k])
            reason = EXPECTED_ZERO.get((flag, name))
            if reason is not None:
                if count != 0:
                    raise SystemExit(
                        f"flag {flag}: {name} fired {count} times but cannot -- {reason}"
                    )
            elif count == 0:
                raise SystemExit(
                    f"flag {flag}: no row reaches {name}. A grid edit dropped a branch; "
                    "the archive would look complete and cover one decision fewer"
                )


# ---------------------------------------------------------------------------


def _probe_rows(n: int) -> np.ndarray:
    """`RP_PROBE_ROWS` rows spread evenly across the sweep, so the invariance
    probe covers the branch mix rather than one corner of it."""
    return np.unique(np.linspace(0, n - 1, RP_PROBE_ROWS).astype(np.int64))


def _nbox_rows(t, pmid, s, wts) -> np.ndarray:
    """Eight rows, one per branch state that can be reached independently."""
    bh2o = bh2o_of(pmid, s)
    raw = 1.609 * s * (pmid / P0)
    it = intermediates(t, bh2o)
    wanted = {
        "bminatm": raw < BMINATM,
        "bmaxatm": raw > BMAXATM,
        "d_cold": (it["d_raw"] < 0.0) & (t < T_B_ZERO),
        "d_hot": (it["d_raw"] < 0.0) & (t > T_B_ZERO),
        "xsb_eps": it["xsb_raw"] < XSB_EPS,
        "floor_41": wts == 41.0,
        "above_99": it["ws"] * 100.0 > 99.0,
        "round_100": nint(wts / 5.0) * 5.0 >= 100.0,
    }
    rows = []
    for name, mask in wanted.items():
        idx = np.nonzero(mask)[0]
        if not idx.size:
            raise SystemExit(f"the nbox block wants a {name} row and the grid has none")
        rows.append(int(idx[len(idx) // 2]))
    return np.array(rows, dtype=np.int64)


def tie_grid() -> np.ndarray:
    parts = [np.array(TIE_WTS)]
    parts.append(np.array([np.nextafter(v, 0.0) for v in TIE_WTS]))
    parts.append(np.array([np.nextafter(v, 1e9) for v in TIE_WTS]))
    return np.unique(np.concatenate(parts))


def capture(out_dir: Path, quiet: bool = False) -> Path:
    g = grid()
    n = len(g["t"])
    b_lo, b_hi = check_b_zero_is_exact()

    # A first pass at the unclamped arm, only to choose the nbox rows: they
    # have to be rows that reach a named branch, and that is a property of the
    # transcription, which is checked against the Fortran below.
    provisional = wts_of(g["t"], bh2o_of(g["pmid"], g["s"]), 0)
    nbox_rows = _nbox_rows(g["t"], g["pmid"], g["s"], provisional)
    probe_rows = _probe_rows(n)
    ties = tie_grid()

    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "vapour_inputs.npz"
        np.savez(
            inputs,
            t=g["t"],
            pmid=g["pmid"],
            s=g["s"],
            rp_main=np.array(RP_MAIN),
            rp_probe=np.array(RP_PROBE),
            probe_t=g["t"][probe_rows],
            probe_pmid=g["pmid"][probe_rows],
            probe_s=g["s"][probe_rows],
            nbox_rows=nbox_rows,
            tie_x=ties,
        )
        records = {}
        for flag in (0, 1):
            records[flag] = run_flag(flag, inputs)
            if not quiet:
                print(f"  l_fix_neg_pvol_wat={flag}: {n:,} rows swept")

    check_transcription(g, records)
    witness = check_records(g, records)
    hits = np.stack(
        [branch_hits(g["t"], g["pmid"], g["s"], np.array(records[f]["wts"])) for f in (0, 1)]
    )
    check_branches(hits)
    if not quiet:
        print(
            f"  witness : {witness['rows_flag_differs']:,} rows separate the two "
            f"settings ({witness['rows_above_310_54']:,} of them above 310.54 K); "
            "rhosol_strat byte-identical, as the algebra requires"
        )
        for k, name in enumerate(BRANCH_NAMES):
            print(f"    {name:<22} {hits[0, k]:>6,} {hits[1, k]:>6,}")

    root_labels, root_lo, root_hi, root_res, root_bh2o = [], [], [], [], []
    for atmos, edges in g["roots"].items():
        bh2o = float(bh2o_of(*(np.array(v) for v in atmospheres()[atmos])))
        for name, (lo, hi) in sorted(edges.items()):
            it = intermediates(np.array([lo, hi]), np.full(2, bh2o))
            root_labels.append(f"{atmos}/{name}")
            root_lo.append(lo)
            root_hi.append(hi)
            root_bh2o.append(bh2o)
            # The quantity whose sign the branch turns on, either side.
            key = {"d_cold": "d_raw", "d_hot": "d_raw", "xsb_eps": "xsb_raw"}.get(name, "ws")
            root_res.append([float(it[key][0]), float(it[key][1])])

    arrays = {
        "t": g["t"],
        "pmid": g["pmid"],
        "s": g["s"],
        "block": g["block"],
        "rp": np.array(RP_MAIN),
        "flag_values": np.array([0, 1], dtype=np.int32),
        "wts": np.stack([np.array(records[f]["wts"]) for f in (0, 1)]),
        "rhosol_strat": np.stack([np.array(records[f]["rhosol_strat"]) for f in (0, 1)]),
        "branch_names": np.array(BRANCH_NAMES, dtype=np.str_),
        "branch_hits": hits,
        "branch_expected_zero": np.array(
            [[(f, nm) in EXPECTED_ZERO for nm in BRANCH_NAMES] for f in (0, 1)]
        ),
        "root_labels": np.array(root_labels, dtype=np.str_),
        "root_lo": np.array(root_lo),
        "root_hi": np.array(root_hi),
        "root_bh2o": np.array(root_bh2o),
        "root_residual": np.array(root_res),
        "t_b_zero": np.array(T_B_ZERO),
        "b_residual": np.array([b_lo, 0.0, b_hi]),
        "rp_probe_values": np.array(RP_PROBE),
        "rp_probe_rows": probe_rows,
        "rp_probe_wts": np.stack([np.array(records[f]["probe_wts"]) for f in (0, 1)]),
        "rp_probe_rhosol": np.stack([np.array(records[f]["probe_rhosol"]) for f in (0, 1)]),
        "nbox_rows": nbox_rows,
        "nbox8_wts": np.stack([np.array(records[f]["nbox8_wts"]) for f in (0, 1)]),
        "nbox1_wts": np.stack([np.array(records[f]["nbox1_wts"]) for f in (0, 1)]),
        "nbox8_rhosol": np.stack([np.array(records[f]["nbox8_rhosol"]) for f in (0, 1)]),
        "nbox1_rhosol": np.stack([np.array(records[f]["nbox1_rhosol"]) for f in (0, 1)]),
        "tie_x": ties,
        "tie_y": np.array(records[0]["tie_y"]),
        "_case": np.array("vapour"),
        "_mode": np.array("leaf"),
        "_variant": np.array("f64"),
        "_rows": np.array(n),
        "_namelist_sha256": np.array(
            hashlib.sha256(NAMELIST.read_bytes()).hexdigest(),
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grid and stop")
    args = parser.parse_args(argv)

    if args.dry_run:
        g = grid()
        print(f"ukca_vapour sweep -> {args.out / ARCHIVE}")
        print(f"  b = 0 exactly at T = {T_B_ZERO!r} (residual {check_b_zero_is_exact()})")
        for atmos, (pmid, s) in atmospheres().items():
            bh2o = float(bh2o_of(np.array(pmid), np.array(s)))
            rows = int((g["block"] == f"t_{atmos}").sum())
            print(f"  {atmos:<5} pmid={pmid:<10g} s={s:<12.6g} bh2o={bh2o:.6g}  {rows:,} rows")
            for name, (lo, hi) in sorted(g["roots"][atmos].items()):
                print(f"      {name:<10} {lo!r}  ..{hi!r}")
        for t in ANCHOR_T:
            print(f"  T={t:<6g} {int((g['block'] == f'ps_{t:g}').sum()):,} (pmid, s) rows")
        print(f"  {len(g['t']):,} rows total, {len(tie_grid())} tie points")
        return 0

    print(f"sweeping ukca_vapour at both l_fix_neg_pvol_wat settings -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
