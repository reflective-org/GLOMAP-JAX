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
  even. `ukca_vapour.F90:226` computes `(NINT(wts/5))*5` with `wts` clamped to
  `[41, 99]`, so `wts = 42.5, 47.5, …` land exactly on ties. Both the bare
  intrinsic and the live idiom are swept.
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
    exact_cubes = np.array([float(k) ** 3 for k in range(1, 65)])
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
                    np.logspace(-30.0, 30.0, 1801),
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
        "log": np.unique(np.concatenate([np.logspace(-300.0, 300.0, 2401), [1.0], exact_cubes])),
        "oneover": np.unique(
            np.concatenate([np.logspace(-150.0, 150.0, 1201), -np.logspace(-150.0, 150.0, 1201)])
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
        # ukca_vapour clamps wts to [41, 99] before dividing by 5, so this is
        # the full reachable domain of that lookup, at a step fine enough to
        # straddle every tie.
        "vapour_round": np.unique(np.arange(40.0, 100.0 + 0.125, 0.125)),
    }


def capture(out_dir: Path, quiet: bool = False) -> Path:
    sys.path.insert(0, str(F2PY_DIR))
    try:
        import glomap_f2py as g
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise SystemExit(
            f"cannot import the binding ({exc}). Run: ./validation/build_f2py.sh"
        ) from None

    arrays: dict[str, np.ndarray] = {}
    for name, x in grids().items():
        driver = getattr(g, f"leaf_{name}")
        arrays[f"{name}_x"] = x
        arrays[f"{name}_y"] = driver(x)
        if not quiet:
            print(f"  {name:<14} {len(x):>6,} points")

    # `**` at several fixed exponents. powr_v takes a SCALAR exponent -- it
    # raises a whole array to one power rather than doing elementwise pairs --
    # so each exponent is a separate sweep, which is also how the Fortran uses
    # it.
    pow_x = grids()["cubrt"]
    arrays["pow_x"] = pow_x
    arrays["pow_exponents"] = np.array(POW_EXPONENTS)
    arrays["pow_y"] = np.stack([g.leaf_pow(pow_x, p) for p in POW_EXPONENTS])
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
    arrays["cubrt_negative_y"] = g.leaf_cubrt(negatives)

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
