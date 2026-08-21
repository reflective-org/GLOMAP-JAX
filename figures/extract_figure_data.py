#!/usr/bin/env python3
"""Extract the numbers behind the project's headline claims into one JSON.

    python figures/extract_figure_data.py            # writes figures/data.json
    python figures/extract_figure_data.py --print    # and dump a summary

Every series here comes from a **committed golden**, so a figure drawn from it
is reproducible without a Fortran toolchain and moves when the goldens move.
That is the point: `docs/` and `PROGRESS.md` are full of measured numbers, and
until now none of them could be looked at.

Four datasets, one per claim:

* `precision_floor` -- the f32-vs-f64 trajectory gap per case. The claim that
  `ref-f32` is a *different trajectory* rather than a noisier one (ADR-001).
* `aitins_collapse` -- the mechanism behind `marine_bcoc`'s 0.80 gap:
  ageing depletes the Aitken insoluble mode, f32 loses the residual, and the
  mean dry diameter collapses while the number stops decaying.
* `branch_coverage` -- every predicate the gate-0 dump records, and the ones
  no shipped namelist ever reaches. The count is derived, not written down
  here; stale counts are a recurring defect in this repository.
* `golden_sizes` -- the committed fixture set against ADR-007's budgets.

The numerical-hazard figures (FMA contraction, the division rewrite, the powi
chain, cross-platform ulp) are *measurements against the running JAX*, not
goldens, so they live in `measure_hazards.py` beside this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"
OUT = Path(__file__).resolve().parent / "data.json"

CASES = ("boundary_layer", "free_troposphere", "bl_nmts3", "marine_bcoc")

# ADR-007, asserted in tests/test_goldens_manifest.py.
MAX_ARCHIVE_MB = 5.0
MAX_TOTAL_MB = 25.0


def _trajectory(case: str, variant: str):
    d = np.load(GOLDENS / f"{case}.{variant}.trajectory.npz")
    return [str(c) for c in d["columns"]], d["values"]


def precision_floor() -> dict:
    """Column-scaled relative gap between the f32 and f64 references, per step.

    Scaled by each column's own f64 range rather than by its value: the state
    spans twenty orders of magnitude, so an unscaled relative error is
    dominated by whichever column is nearest zero.
    """
    out = {}
    for case in CASES:
        cols, f64 = _trajectory(case, "f64")
        _, f32 = _trajectory(case, "f32")
        # Skip the two time columns; they are inputs, not state.
        state = slice(2, None)
        a, b = f64[:, state], f32[:, state]
        scale = np.maximum(np.abs(a).max(axis=0), 1e-300)
        gap = np.abs(a - b) / scale
        out[case] = {
            "hours": f64[:, cols.index("time_h")].tolist(),
            "gap": gap.max(axis=1).tolist(),
            # Both, because docs/ quotes the peak and a reader looking at the
            # last point of the curve would otherwise think the docs wrong.
            "final": float(gap.max(axis=1)[-1]),
            "peak": float(gap.max()),
        }
    return out


def aitins_collapse() -> dict:
    """Why `marine_bcoc` is 0.80 and the setup-1 cases are 1e-3.

    Two quantities, two precisions, one case. Kept as separate series rather
    than a ratio: the point is that the f32 *number* stops decaying and turns
    upward while the f64 one keeps going, and a ratio hides that.
    """
    cols, f64 = _trajectory("marine_bcoc", "f64")
    _, f32 = _trajectory("marine_bcoc", "f32")
    hours = f64[:, cols.index("time_h")].tolist()
    out = {"hours": hours}
    for field in ("Ddry_aitins_nm", "N_aitins_cm3"):
        i = cols.index(field)
        out[field] = {"f64": f64[:, i].tolist(), "f32": f32[:, i].tolist()}
    return out


def branch_coverage() -> dict:
    """Every gate-0 predicate, and how often it was true.

    A predicate at 0 is one no shipped namelist reaches -- it has no reference
    data at all, and a port could get it arbitrarily wrong without any gate
    noticing. That is the figure's whole subject.
    """
    total: dict[str, int] = {}
    hits: dict[str, int] = {}
    for case in CASES:
        d = np.load(GOLDENS / f"{case}.f64.branches.npz")
        tags = d["tag_levels"][d["tag"]]
        value = d["value"]
        for tag in np.unique(tags):
            m = tags == tag
            key = str(tag)
            total[key] = total.get(key, 0) + int(m.sum())
            hits[key] = hits.get(key, 0) + int((value[m] != 0).sum())
    return {
        tag: {"records": total[tag], "true": hits[tag], "fraction": hits[tag] / total[tag]}
        for tag in sorted(total)
    }


def golden_sizes() -> dict:
    archives = sorted(GOLDENS.glob("*.npz"))
    sizes = {p.name: p.stat().st_size / 1e6 for p in archives}
    return {
        "archives": sizes,
        "total_mb": sum(sizes.values()),
        "max_archive_mb": MAX_ARCHIVE_MB,
        "max_total_mb": MAX_TOTAL_MB,
    }


def build() -> dict:
    return {
        "precision_floor": precision_floor(),
        "aitins_collapse": aitins_collapse(),
        "branch_coverage": branch_coverage(),
        "golden_sizes": golden_sizes(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", action="store_true", dest="show")
    args = parser.parse_args(argv)

    data = build()
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e3:.0f} kB)")

    if args.show:
        print("\nprecision floor, final step:")
        for case, rec in data["precision_floor"].items():
            print(f"  {case:<18} {rec['final']:.3e}")
        never = [t for t, r in data["branch_coverage"].items() if r["true"] == 0]
        print(f"\nbranch predicates: {len(data['branch_coverage'])}, never true: {len(never)}")
        print("  " + ", ".join(never))
        print(
            f"\ngoldens: {len(data['golden_sizes']['archives'])} archives, "
            f"{data['golden_sizes']['total_mb']:.2f} MB of {MAX_TOTAL_MB} MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
