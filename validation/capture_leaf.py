#!/usr/bin/env python3
"""Sweep the leaf reference drivers over dense grids and capture the result.

    python validation/capture_leaf.py --dry-run
    python validation/capture_leaf.py            # writes tests/goldens/*.npz

A trajectory fixture only ever exercises the inputs a trajectory produces, and
the branch dump showed how narrow that is: half of `ukca_solvecoagnucl_v`'s
closed forms are unreachable from any shipped namelist, and `ukca_remode` never
merges at all. A leaf driver reaches the inputs the *physics* can reach, by
calling one routine in-process with chosen arguments.

This is the numerics sweep (task 21), the input to the transcendental compat
layer (task 34). The grids are here rather than in Fortran because choosing
which inputs matter is a judgement about the physics, and it should be readable
and changeable without a recompile.

Three of these are known hazards where gfortran and XLA need not agree, and
each grid is built to land on the hazard rather than near it:

* **ERF** feeds `ukca_remode`'s `FRAC_N`, cut at 0.5 — that is, at
  `erf(x) = 0`. That clamp is continuous at the boundary, so the consequence
  of a disagreement is smaller than the plan assumed (merging itself is gated
  on `drydp`, not on `erf`), but zero is still the point where the transfer
  fraction is decided, so the grid is dense through it at several scales.
* **`x ** (1.0/3.0)`** is what `cubrt_v` literally computes. It is not a cube
  root function, and the constant `1.0/3.0` is itself not exactly a third. The
  grid includes exact cubes, where any honest cube root returns an integer and
  the power form need not.
* **NINT** rounds half away from zero; numpy and `jnp.round` round half to
  even. `ukca_vapour.F90:226` computes `(NINT(wts/5))*5`, so
  `wts = 42.5, 47.5, …` land exactly on ties, and the result indexes a lookup
  table — a tie going the wrong way selects a different row. Both the bare
  intrinsic and the live idiom are swept.

  `wts` is floored at 41 in both arms but is **not** clamped to `[41, 99]`, as
  this said: only the `l_fix_neg_pvol_wat` arm has the ceiling (`:184`), and
  the default arm reaches 103.8.

The grids are exactly reproducible
----------------------------------

The *results* of this sweep are platform-dependent — that is what it measures.
The *inputs* must not be, or a re-capture on another machine moves the sample
points as well as the answers, and the drift gate can no longer tell the two
apart. So no grid point may come from a libm call.

`np.logspace(a, b, n)` is `10.0 ** np.linspace(a, b, n)`, i.e. a libm `pow` per
point, and it is not correctly rounded: on the arm64 capture platform 4 of the
1801 points of the old `cubrt` grid came back 1 ulp away from the correctly
rounded value, so a host with a correctly-rounded `pow` (glibc ≥ 2.28) would
have produced different abscissae. The log-spaced grids are therefore built by
`_decade_grid` out of decimal literals, whose conversion to binary IEEE 754
requires to be correctly rounded, and the exact cubes are built with integer
arithmetic. `linspace`, `arange`, `nextafter` and `finfo` are all exact or
correctly-rounded elementary operations and are used as they were.

REGENERATION NEEDED: `tests/goldens/numerics.f64.leaf.npz` was captured before
that change, with the `logspace` abscissae, so the golden and this script no
longer describe the same grid — re-run `validation/capture_leaf.py` on the
pinned toolchain and re-write the manifest. Expect a drift report naming
`cubrt`, `log`, `oneover` and `pow` (whose grid is `cubrt`'s), `_x` and `_y`
alike, and nothing else: `erf`, `exp`, `nint` and `vapour_round` are built from
`linspace` and `arange` and do not move. Every grid keeps its point count and
every point moved by less than 0.4%, so a drift larger than that, or in another
array, is a different change and not this one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "numerics.f64.leaf.npz"

# Exponents worth sweeping `**` at. 1/3 and 2/3 because volume-to-diameter
# conversions use them; 0.5 because it has an exact hardware alternative that a
# port might reach for; 3.0 as the inverse of the first; -1.0 because
# `oneover_v` exists separately and the two should be compared.
POW_EXPONENTS = (1.0 / 3.0, 2.0 / 3.0, 0.5, 3.0, -1.0)

# Mantissas for a decade-based log grid, as decimal STRINGS: `10 ** (i/30)` and
# `10 ** (i/4)` rounded to three significant figures. Strings, because the point
# is to keep libm out of the abscissae -- `float("1.08e-30")` is a
# decimal-to-binary conversion, which IEEE 754 requires to be correctly rounded,
# so every machine parses it to the same double. See the module docstring.
DECADE_30 = (
    "1.00", "1.08", "1.17", "1.26", "1.36", "1.47", "1.58", "1.71", "1.85", "2.00",
    "2.15", "2.33", "2.51", "2.71", "2.93", "3.16", "3.41", "3.69", "3.98", "4.30",
    "4.64", "5.01", "5.41", "5.84", "6.31", "6.81", "7.36", "7.94", "8.58", "9.26",
)  # fmt: skip
DECADE_4 = ("1.00", "1.78", "3.16", "5.62")

# The log-spaced grids, as (first decade, last decade, mantissas). Named here
# rather than inline so `tests/test_capture_scripts.py` can check every point of
# every one of them against the short-decimal property.
LOG_GRIDS: dict[str, tuple[int, int, tuple[str, ...]]] = {
    "cubrt": (-30, 30, DECADE_30),
    "log": (-300, 300, DECADE_4),
    "oneover": (-150, 150, DECADE_4),
}


def _decade_grid(lo: int, hi: int, mantissas: tuple[str, ...]) -> np.ndarray:
    """`len(mantissas)` points per decade from `1e{lo}` to `1e{hi}` inclusive.

    The replacement for `np.logspace`, which reaches libm `pow` for every point
    and so makes the *inputs* of the sweep platform-dependent. Every value here
    is `float("<decimal literal>")`.
    """
    values = [float(f"{m}e{k}") for k in range(lo, hi) for m in mantissas]
    values.append(float(f"1e{hi}"))
    grid = np.array(values, dtype=np.float64)
    if not (np.diff(grid) > 0).all():
        raise AssertionError(f"decade grid 1e{lo}..1e{hi} is not strictly increasing")
    return grid


def _exact_cubes() -> np.ndarray:
    """1, 8, 27, … 64**3, cubed in *integer* arithmetic.

    `float(k) ** 3` is a libm `pow` call. It happens to be exact for these 64
    values on every libm anyone has, but the whole point of this grid is that
    an honest cube root returns an integer here, so the input had better be one
    for a reason and not by luck.
    """
    return np.array([float(k**3) for k in range(1, 65)], dtype=np.float64)


def _dense_through_zero() -> np.ndarray:
    """Zero, and approaches to it at every scale that matters.

    `erf` near zero is where remode decides whether a mode merges, so a grid
    that merely passes near zero is not good enough — it has to include exact
    zero and the smallest representable neighbourhoods.
    """
    scales = [1e-300, 1e-30, 1e-16, 1e-8, 1e-4, 1e-2]
    near = np.concatenate([[-s, s] for s in scales])
    return np.unique(np.concatenate([[0.0, -0.0], near]))


def grids() -> dict[str, np.ndarray]:
    exact_cubes = _exact_cubes()
    return {
        "erf": np.unique(
            np.concatenate(
                [
                    np.linspace(-6.0, 6.0, 2401),
                    np.linspace(-1.0, 1.0, 2001),
                    _dense_through_zero(),
                ]
            )
        ),
        # x ** (1/3). Strictly positive: dvol and ddpcub are non-negative
        # everywhere the Fortran calls cubrt_v, and negatives are probed
        # separately below because they are a different question.
        "cubrt": np.unique(
            np.concatenate(
                [
                    _decade_grid(*LOG_GRIDS["cubrt"]),
                    exact_cubes,
                    [np.finfo(np.float64).tiny, np.finfo(np.float64).max],
                ]
            )
        ),
        # exp_v. Capped below the float64 overflow threshold (~709.78) so the
        # golden stays finite; the 50.0 point is the hard clamp in
        # ukca_solvecoagnucl_v, which is a branch and not just a large value.
        "exp": np.unique(
            np.concatenate(
                [
                    np.linspace(-700.0, 700.0, 2801),
                    np.linspace(-1.0, 1.0, 401),
                    [0.0, 50.0, np.nextafter(50.0, 0.0), np.nextafter(50.0, 100.0)],
                ]
            )
        ),
        "log": np.unique(np.concatenate([_decade_grid(*LOG_GRIDS["log"]), [1.0], exact_cubes])),
        "oneover": np.unique(
            np.concatenate(
                [_decade_grid(*LOG_GRIDS["oneover"]), -_decade_grid(*LOG_GRIDS["oneover"])]
            )
        ),
        # Ties at every half-integer, and the two representable neighbours of
        # each tie -- a port that gets the tie right by accident but the
        # neighbourhood wrong should still fail.
        "nint": np.unique(
            np.concatenate(
                [
                    np.arange(-64.0, 64.5, 0.5),
                    [np.nextafter(v, 0.0) for v in np.arange(-64.0, 64.5, 0.5)],
                    [np.nextafter(v, 1e9) for v in np.arange(-64.0, 64.5, 0.5)],
                    [0.0, -0.0],
                ]
            )
        ),
        # Only the l_fix_neg_pvol_wat arm caps wts at 99 (ukca_vapour.F90:184);
        # the default arm is MAX(41.0, ws*100) with no ceiling (:188) and
        # reaches 103.8. The grid runs past 99 for that reason. What it is
        # really pinning is the tie behaviour of (NINT(wts/5))*5, whose result
        # is matched against `percent` (:90), which stops at 95 -- so every
        # round >= 100 falls through to rhosol_strat = 1300.0.
        "vapour_round": np.unique(np.arange(40.0, 110.0 + 0.125, 0.125)),
    }


def check_no_ereport(before: tuple, after: tuple, what: str, last: tuple | None = None) -> None:
    """The rule from `docs/harness.md`: every leaf driver checks the shim.

    The shim returns where the real `ereport` would `STOP 1`, so a driver that
    hit an error path returns a number rather than a crash, and the number is
    meaningless. `wrap_init` and `wrap_step` already record the fatal count
    before the call and compare after; leaf drivers must do the same.

    All three counters, not just `fatal`: a warning during a sweep of pure
    intrinsics would mean the driver is not calling what this script thinks it
    is, which is exactly as much of a finding.

    Split out from `capture` so it is testable without the built extension.
    """
    labels = ("fatal", "warning", "info")
    moved = [
        f"{name} +{int(a) - int(b)}"
        for name, b, a in zip(labels, before, after)
        if int(a) != int(b)
    ]
    if not moved:
        return
    detail = ""
    if last is not None:
        status, routine, message = last
        for part in (routine, message):
            text = part.decode() if isinstance(part, bytes) else str(part)
            detail += f"\n  {text.strip()}"
        detail = f"\n  status {int(status)}" + detail
    raise SystemExit(f"{what} reached ereport ({', '.join(moved)}) -- the sweep is void{detail}")


def capture(out_dir: Path, quiet: bool = False) -> Path:
    sys.path.insert(0, str(F2PY_DIR))
    try:
        import glomap_f2py as g
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise SystemExit(
            f"cannot import the binding ({exc}). Run: ./validation/build_f2py.sh"
        ) from None

    def call(what: str, fn, *args):
        """One driver call, with the shim counted either side of it."""
        before = tuple(int(v) for v in g.wrap_ereport_count())
        y = fn(*args)
        after = tuple(int(v) for v in g.wrap_ereport_count())
        check_no_ereport(before, after, what, g.wrap_ereport_last())
        return y

    g.wrap_ereport_reset()

    arrays: dict[str, np.ndarray] = {}
    for name, x in grids().items():
        driver = getattr(g, f"leaf_{name}")
        arrays[f"{name}_x"] = x
        arrays[f"{name}_y"] = call(f"leaf_{name}", driver, x)
        if not quiet:
            print(f"  {name:<14} {len(x):>6,} points")

    # `**` at several fixed exponents. powr_v takes a SCALAR exponent -- it
    # raises a whole array to one power rather than doing elementwise pairs --
    # so each exponent is a separate sweep, which is also how the Fortran uses
    # it.
    pow_x = grids()["cubrt"]
    arrays["pow_x"] = pow_x
    arrays["pow_exponents"] = np.array(POW_EXPONENTS)
    arrays["pow_y"] = np.stack(
        [call(f"leaf_pow(p={p!r})", g.leaf_pow, pow_x, p) for p in POW_EXPONENTS]
    )
    if not quiet:
        print(f"  {'pow':<14} {len(pow_x):>6,} points x {len(POW_EXPONENTS)} exponents")

    # A negative-input probe, kept apart from the main grid because the answer
    # is not a number. `x ** (1.0/3.0)` is undefined for x < 0 -- a non-integer
    # power of a negative -- while np.cbrt and most cube-root functions return
    # the real root. The Fortran never reaches it (dvol >= 0 everywhere
    # cubrt_v is called), but a port that reaches for np.cbrt would silently
    # differ if it ever did.
    negatives = np.array([-1.0, -8.0, -1e-30, -1e30])
    arrays["cubrt_negative_x"] = negatives
    arrays["cubrt_negative_y"] = call("leaf_cubrt(negative)", g.leaf_cubrt, negatives)

    arrays["_case"] = np.array("numerics")
    arrays["_mode"] = np.array("leaf")
    arrays["_variant"] = np.array("f64")
    arrays["_rows"] = np.array(sum(len(v) for k, v in arrays.items() if k.endswith("_x")))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the grids and stop")
    args = parser.parse_args(argv)

    if args.dry_run:
        total = 0
        print(f"leaf numerics sweep -> {args.out / ARCHIVE}")
        for name, x in grids().items():
            total += len(x)
            print(f"  {name:<14} {len(x):>6,} points   [{x.min():.3e}, {x.max():.3e}]")
        print(f"  {'pow':<14} {len(grids()['cubrt']):>6,} points x {len(POW_EXPONENTS)} exponents")
        print(f"  total {total:,} points")
        return 0

    print(f"sweeping the leaf drivers -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
