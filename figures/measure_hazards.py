#!/usr/bin/env python3
"""Measure the numerical hazards this port has hit, into `figures/hazards.json`.

    python figures/measure_hazards.py

Unlike `extract_figure_data.py`, nothing here comes from a golden. These are
properties of the **running JAX and this CPU**, re-measured on every run,
because that is exactly the point: three of them are version- or
platform-dependent, and a figure that pinned them as constants would be wrong
the next time somebody upgraded.

Each entry answers "on what fraction of a realistic sample does the obvious
spelling give a different double from the Fortran's?" -- which is the question
byte equality actually asks.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import glomap_jax  # noqa: F401  -- enables x64 in the one place it is set
from glomap_jax.core import numerics

OUT = Path(__file__).resolve().parent / "hazards.json"
N = 200_000


def _frac(a, b) -> float:
    return float(np.mean(np.asarray(a) != np.asarray(b)))


def fma_contraction() -> dict:
    """`a*b + c` under `jit`: XLA fuses, gfortran (-ffp-contract=off) does not."""
    rng = np.random.default_rng(0)
    a, b, c = (rng.standard_normal(N) for _ in range(3))

    def fn(x, y, z):
        return x * y + z

    eager = fn(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c))
    jitted = jax.jit(fn)(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c))
    return {"a*b + c under jit": _frac(eager, jitted)}


def division_rewrite() -> dict:
    """`x / c` for scalar constant `c`: XLA rewrites to `x * (1/c)`, eagerly."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 1e6, N)
    jx = jnp.asarray(x)
    out = {}
    for name, c in (
        ("x / f_ao", 0.150 / 0.0168),
        ("x / 5.0", 5.0),
        ("x / p0", 101325.0),
        ("x / avogadro", 6.022e23),
        ("x / 2.0", 2.0),
    ):
        out[name] = _frac(jx / c, x / c)
    # The fix, on the worst one, so the figure shows the remedy beside the fault.
    out["true_divide(x, f_ao)"] = _frac(
        numerics.true_divide(jx, 0.150 / 0.0168), x / (0.150 / 0.0168)
    )
    return out


def integer_powers() -> dict:
    """`x**k` vs the powi chain gfortran expands an integer literal exponent to."""
    x = jnp.asarray(np.linspace(0.1, 0.95, N))
    p2 = x * x
    p3 = x * p2
    p4 = p2 * p2
    chain = {2: p2, 3: p3, 4: p4, 5: p2 * p3, 6: p3 * p3, 7: p3 * p4}
    return {f"x**{k}": _frac(x**k, v) for k, v in chain.items()}


def cbrt_and_rounding() -> dict:
    """The two rules the compat layer exists for, against the swept goldens."""
    sweep = np.load(Path(__file__).resolve().parents[1] / "tests/goldens/numerics.f64.leaf.npz")
    x = sweep["cubrt_x"]
    out = {"jnp.cbrt vs x**(1/3)": _frac(jnp.cbrt(x), numerics.cbrt(x))}
    r = sweep["vapour_round_x"]
    out["jnp.round vs Fortran NINT"] = _frac(jnp.round(r / 5.0) * 5.0, sweep["vapour_round_y"])
    return out


def main() -> int:
    data = {
        "environment": {
            "jax": jax.__version__,
            "numpy": np.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "samples": N,
        },
        "fma_contraction": fma_contraction(),
        "division_rewrite": division_rewrite(),
        "integer_powers": integer_powers(),
        "cbrt_and_rounding": cbrt_and_rounding(),
    }
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}  ({data['environment']['jax']}, {data['environment']['platform']})")
    for section in ("fma_contraction", "division_rewrite", "integer_powers", "cbrt_and_rounding"):
        for name, frac in data[section].items():
            print(f"  {name:<28} {frac:7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
