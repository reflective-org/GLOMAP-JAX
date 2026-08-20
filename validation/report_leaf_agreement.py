#!/usr/bin/env python3
"""How closely JAX matches gfortran on *this* machine, primitive by primitive.

    python validation/report_leaf_agreement.py

Run it after `capture_leaf.py`, so both sides come from the same box. The
committed sweep is a Darwin arm64 artefact; comparing it against JAX anywhere
else measures the distance between two machines, not between the port and the
reference.

Why this is a report and not an assertion. Bit equality between JAX and
gfortran was demonstrated on arm64 and does not reproduce on x86_64 -- and not
merely in the "different platform" sense. XLA-CPU lowers float64 ``erf`` and
``pow`` to the host libm, whose selected code path depends on the CPU features
of the machine it runs on, so the *same* code against the *same* freshly built
reference on the *same* runner image passed, failed and passed again across
three consecutive CI runs. A gate demanding bit equality there measures the
scheduler. The bounds in `tests/conftest.py` are the gate; this prints what was
actually achieved, so the number is visible on every run instead of being
rediscovered the next time something goes red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "goldens" / "numerics.f64.leaf.npz"
sys.path.insert(0, str(REPO / "tests"))


def ulp_gap(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Distance in ulp, using the spacing at the expected value.

    Exact within a binade and off by at most a factor of two across one, which
    is fine for a report -- the pass/fail decision uses the exact nextafter
    window in `conftest.assert_matches_reference`.
    """
    both_nan = np.isnan(actual) & np.isnan(expected)
    gap = np.abs(actual - expected) / np.spacing(np.abs(expected))
    return np.where(both_nan | (actual == expected), 0.0, gap)


def main() -> int:
    import platform

    import jax
    import jax.numpy as jnp

    import conftest
    from glomap_jax.core import numerics

    if not GOLDEN.is_file():
        raise SystemExit(f"{GOLDEN} is missing -- run validation/capture_leaf.py first")
    sweep = np.load(GOLDEN, allow_pickle=False)

    cases = {
        "erf": (numerics.erf(sweep["erf_x"]), sweep["erf_y"]),
        "cubrt": (jnp.asarray(sweep["cubrt_x"]) ** (1.0 / 3.0), sweep["cubrt_y"]),
        "log": (jnp.log(jnp.asarray(sweep["log_x"])), sweep["log_y"]),
        "oneover": (1.0 / jnp.asarray(sweep["oneover_x"]), sweep["oneover_y"]),
        "nint": (numerics.nint(sweep["nint_x"]), sweep["nint_y"]),
        "vapour_round": (numerics.vapour_round(sweep["vapour_round_x"]), sweep["vapour_round_y"]),
    }

    print(f"{platform.system()} {platform.machine()}, jax {jax.__version__}")
    print(f"goldens captured on: {conftest.capture_platform()}")
    print(f"{'primitive':<14} {'points':>7} {'differ':>7} {'worst ulp':>10} {'bound':>6}")
    worst_over_bound = []
    for name, (got, want) in cases.items():
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        gap = ulp_gap(got, want)
        bound = (
            0
            if name in ("nint", "vapour_round")
            else conftest.CROSS_PLATFORM_ULP_BY_PRIMITIVE.get(name, conftest.CROSS_PLATFORM_ULP)
        )
        n_differ = int((gap > 0).sum())
        print(f"{name:<14} {gap.size:>7} {n_differ:>7} {gap.max():>10.1f} {bound:>6}")
        if gap.max() > bound:
            worst_over_bound.append(f"{name}: {gap.max():.1f} ulp against a bound of {bound}")

    if worst_over_bound:
        print("\nover the recorded bound:")
        for line in worst_over_bound:
            print(f"  {line}")
        print("Re-measure and update conftest.CROSS_PLATFORM_ULP_BY_PRIMITIVE with the finding.")
        return 1
    print("\nall primitives within their recorded bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
