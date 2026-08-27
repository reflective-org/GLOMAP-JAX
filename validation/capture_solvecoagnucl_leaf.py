#!/usr/bin/env python3
"""Sweep `ukca_solvecoagnucl_v` over all eight branch codes (task 55).

    python validation/capture_solvecoagnucl_leaf.py --dry-run
    python validation/capture_solvecoagnucl_leaf.py   # writes tests/goldens/

`ukca_solvecoagnucl_v` solves `dN/dt = A*N^2 + B*N + C` analytically. `A` is
intra-modal coagulation, `B` inter-modal, `C` new particle formation, and the
routine picks between five closed forms and one error case by the sign of
`A` and of the discriminant `D = 4AC - B^2`.

**Issue #13 is what this file is for.** The shipped fixtures reach 4 of the 8
branch codes; the other four cannot be reached from a trajectory at all,
because a trajectory does not get to choose `A`, `B` and `C`. A constructed
grid does. `verify_branch_coverage` refuses a golden that leaves any of the
eight unexercised, counted separately.

The eight codes, in the source's own logical names
--------------------------------------------------

| code | condition | closed form |
|---|---|---|
| `1a_ok` | `A != 0`, `D < -eps_d`, `term3` and `term4` both usable | `:245` |
| `1a_term3` | as above but `\\|term3\\| <= eps_ab` | falls through, `ndnew = nd` |
| `1a_term4` | as above but `\\|term4\\| <= eps_ab` | falls through, `ndnew = nd` |
| `1b` | `A != 0`, `D > eps_d` | `ATAN`/`TAN`, `:251` |
| `1ca` | `A != 0`, `\\|D\\| <= eps_d`, `B != 0` | **`ierr = 1`, fatal `ereport`** |
| `1cb` | `A != 0`, `\\|D\\| <= eps_d`, `B == 0` | `1/(1/nd - 3*A*dtz)`, `:259` |
| `2a` | `A == 0`, `B != 0` | `EXP(B*dtz)`, `:264` |
| `2b` | `A == 0`, `B == 0` | `nd + C*dtz`, `:270` |

`1cb` carries UP-1's spurious factor 3 -- the analytic solution of
`dN/dt = A*N^2` is `1/(1/N0 - A*t)` and `:259` writes `3.0*a(:)*dtz`. Gate 0
already established that this is the branch the top soluble mode takes every
substep of every shipped namelist, because it has no larger mode to coagulate
into and no nucleation, so `B = C = 0`.

A ninth branch code, which nobody wrote down
--------------------------------------------

`:198-199` reads

    logic1 = mask .AND. ((a > eps_ab) .OR. (a < -eps_ab))
    logic2 = mask .AND. ((a < eps_ab) .AND. (a > -eps_ab))

Both tests are **strict** and both use `eps_ab`, so `|a| == eps_ab` exactly
satisfies neither. A row there takes no closed form at all: `ndnew` keeps the
`nd` it was initialised with at `:190`, and `deln` comes back exactly zero --
the solver reporting "nothing coagulated" rather than "no branch applied".
`logic3` is written `<=` / `>=`, so `b` has no equivalent gap.

This capture found it rather than the source reading did: the `2b` invariant
fired on a row the classifier called `2b` and the Fortran had left alone. The
grid now carries `a = +/- 1.0e-20` exactly and the golden records the zero.

Whether it can happen in a run is a separate question -- `a` is `-0.5*kii`, so
it needs a coagulation kernel of exactly `2e-20` -- but "vanishingly unlikely"
is what UP-5 and issue #19 were also called.

`1ca` is fatal and is probed, not swept
---------------------------------------

`:293` calls `ereport`, which does `STOP 1` in the real build. Under this
binding it is the shim, which returns -- so the routine would come back with
`ndnew` still at its initialised `nd` and hand out a zero increment that reads
as "nothing coagulated" rather than "the solver gave up". `bind_call` counts
ereports around every call, so `capture` runs that branch in its own call and
requires the count to rise; every other call must leave it at zero.

Reaching `1ca` at all takes construction: `D == 0` exactly with `B != 0` means
`B^2 == 4AC` in double precision, which is why the probe solves for `C` rather
than sampling.

The `sqd*dtz > 50` clamp
------------------------

`:225-228` caps the exponent because `EXP(x)` overflows above about 709 and the
comment says 50. It is a **silent** clamp -- no diagnostic -- so CLAUDE.md's
rule about surfacing caps applies to the port, and the grid straddles it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_modes import render_namelist
from leaf_common import CHILD_PREAMBLE, F2PY_DIR, NAMELISTS, REPO, check_varied, run_child

DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "solvecoagnucl.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_solvecoagnucl_v.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 4)

EPS_AB = 1.0e-20
EPS_D = EPS_AB * EPS_AB
SQD_CLAMP = 50.0

#: Several timesteps, because `dtz` enters every closed form and the clamp.
DTZ_VALUES = tuple(float(s) for s in ("1.0", "60.0", "300.0", "1800.0"))

CODES = (
    "1a_ok",
    "1a_term3",
    "1a_term4",
    "1b",
    "1ca",
    "1cb",
    "2a",
    "2b",
    # A NINTH code, found by the capture rather than by reading the source.
    "gap_a",
    "masked_off",
)


def _code(name: str) -> int:
    return CODES.index(name)


def classify(a: float, b: float, c: float, nd: float, dtz: float) -> str:
    """Which branch a row takes, re-derived from `:196-213`.

    Written from the source rather than inferred from the answer, so a grid
    that fails to reach a branch says so at build time instead of producing a
    golden whose coverage nobody checked.
    """
    d = 4.0 * a * c - b * b
    # `:198-199`. BOTH tests are strict and BOTH use eps_ab, so
    # `|a| == eps_ab` exactly satisfies neither: `logic1` needs `a > eps_ab`
    # and `logic2` needs `a < eps_ab`. A row there takes no branch at all,
    # `ndnew` keeps its initialised `nd`, and `deln` comes back exactly zero --
    # the solver reporting "nothing happened" rather than failing. `logic3`
    # is inclusive (`<=` / `>=`) so `b` has no such gap.
    #
    # Found by this capture: the 2b invariant fired on a row the classifier
    # had called 2b and the Fortran had left alone.
    logic1 = a > EPS_AB or a < -EPS_AB
    logic2 = a < EPS_AB and a > -EPS_AB
    logic3 = -EPS_AB <= b <= EPS_AB
    if not logic1 and not logic2:
        return "gap_a"
    if logic1:
        if d < -EPS_D:
            sqd = np.sqrt(-d)
            t1 = 2.0 * a * nd + b + sqd
            t2 = 2.0 * a * nd + b - sqd
            term3 = t1 / t2 if t2 != 0.0 else np.inf
            if not (term3 > EPS_AB or term3 < -EPS_AB):
                return "1a_term3"
            term4 = 1.0 - np.exp(min(sqd * dtz, SQD_CLAMP)) / term3
            if not (term4 > EPS_AB or term4 < -EPS_AB):
                return "1a_term4"
            return "1a_ok"
        if d > EPS_D:
            return "1b"
        return "1cb" if logic3 else "1ca"
    return "2b" if logic3 else "2a"


def build_term4_rows(limit: int = 6) -> list[tuple[float, float, float, float, float]]:
    """`(a, b, c, nd, dtz)` rows that land in `1a_term4`, solved for.

    `term4 = 1 - EXP(sqd*dtz)/term3` and the branch needs `|term4| <= 1e-20`,
    which in double precision means the division must round to **exactly**
    `1.0`. Sampling will not find that: the tolerance is four orders below the
    representable spacing, so the only rows that qualify are the ones where
    `term3` agrees with `EXP(sqd*dtz)` to within half an ulp.

    So it is solved. With `c = 0` and `b > 0`, `sqd = b` and
    `term3 = 1 + b/(a*nd)`, giving `a = b/(nd*(EXP(b*dtz) - 1))`. Rounding
    means that value does not always land, so the search walks a few ulps
    either side and keeps the ones that do -- and raises if none do, rather
    than quietly returning a grid that misses the branch.
    """
    rows: list[tuple[float, float, float, float, float]] = []
    for dtz in DTZ_VALUES:
        for b in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 0.05, 0.1):
            for nd in (1.0e2, 1.0e3, 1.0e4):
                x = min(b * dtz, SQD_CLAMP)
                target = float(np.exp(x))
                if target == 1.0:
                    continue
                a0 = b / (nd * (target - 1.0))
                for step in range(-60, 61):
                    a = a0
                    for _ in range(abs(step)):
                        a = float(np.nextafter(a, np.inf if step > 0 else -np.inf))
                    if classify(a, b, 0.0, nd, dtz) == "1a_term4":
                        rows.append((a, b, 0.0, nd, dtz))
                        break
                if len(rows) >= limit:
                    return rows
    if not rows:
        raise SystemExit(
            "no (a, b, nd, dtz) row lands in 1a_term4; the branch would be "
            "unvalidated and a port that omitted it would pass"
        )
    return rows


def build_grid() -> dict[str, np.ndarray]:
    """Rows constructed per branch, not sampled and hoped over."""
    rows: list[tuple[float, float, float, float, int]] = []

    def add(a, b, c, nd, mask=1):
        rows.append((a, b, c, nd, mask))

    # 1b: A != 0, D > 0 needs 4AC > B^2. Take B = 0 and A, C the same sign.
    for a in (-1.0e-8, -1.0e-12, 1.0e-12, 1.0e-8):
        for c in (1.0e-2, 1.0e2, 1.0e6):
            add(a, 0.0, c if a > 0 else -c, 1.0e3)
    # 1a: A != 0, D < 0. B large enough that B^2 > 4AC; C = 0 makes D = -B^2.
    for a in (-1.0e-8, -1.0e-10, 1.0e-10, 1.0e-8):
        for b in (-1.0e-2, -1.0e-5, 1.0e-5, 1.0e-2):
            for nd in (1.0e2, 1.0e5):
                add(a, b, 0.0, nd)
    # 1a with a large sqd*dtz, to straddle the 50 clamp.
    for b in (-2.0, -0.1, 0.1, 2.0):
        add(1.0e-9, b, 0.0, 1.0e4)
    # 1cb: A != 0, B = 0, D = 0 needs C = 0. UP-1's branch.
    for a in (-1.0e-6, -1.0e-9, 1.0e-9, 1.0e-6, 1.0e-3):
        for nd in (1.0e2, 1.0e4, 1.0e6):
            add(a, 0.0, 0.0, nd)
    # 2a: A = 0 (below eps_ab), B != 0.
    for b in (-1.0e-2, -1.0e-4, 1.0e-4, 1.0e-2):
        for c in (0.0, 1.0e-2, 1.0e3):
            add(0.0, b, c, 1.0e3)
    # 2b: A = 0, B = 0 -- pure nucleation.
    for c in (0.0, 1.0e-4, 1.0, 1.0e4):
        add(0.0, 0.0, c, 1.0e3)
    # Across the eps_ab window on A and B, where "zero" is decided -- and
    # including `a = +/- eps_ab` EXACTLY, which is the gap above.
    for a in (0.0, 5.0e-21, -5.0e-21, 1.0e-20, -1.0e-20, 2.0e-20):
        for b in (0.0, 5.0e-21, 1.0e-20, 1.0e-3):
            add(a, b, 1.0e-3, 1.0e3)
    # `1a_term4`, which cannot be sampled -- see build_term4_rows. Each row
    # carries its own `dtz`, so they are swept at that timestep alone and the
    # main grid's timesteps do not have to hit the same coincidence.
    for a, b, c, nd, _dtz in build_term4_rows():
        add(a, b, c, nd)

    # Masked-off rows carrying poison: `deln` must come back exactly 0.
    for poison in (0.0, float("inf"), float("-inf"), float("nan"), 1.0e308):
        add(poison, poison, poison, poison, mask=0)

    arr = np.array(rows, dtype=np.float64)
    return {
        "a": arr[:, 0],
        "b": arr[:, 1],
        "c": arr[:, 2],
        "nd": arr[:, 3],
        "mask": arr[:, 4].astype(np.int32),
    }


def build_error_probe() -> dict[str, np.ndarray]:
    """Rows in `1ca`: `A != 0`, `B != 0`, and `D` exactly zero.

    Solved for rather than sampled. `D = 4AC - B^2 == 0` in double precision
    needs `C = B*B/(4*A)` and then a check that the round trip really lands on
    zero -- which it does not for every triple, so the probe keeps only the
    ones that do.
    """
    a_vals = (1.0e-8, -1.0e-8, 1.0e-4, 2.0e-6)
    b_vals = (1.0e-3, -1.0e-3, 4.0e-2)
    a_out, b_out, c_out = [], [], []
    for a in a_vals:
        for b in b_vals:
            c = b * b / (4.0 * a)
            if 4.0 * a * c - b * b == 0.0:
                a_out.append(a)
                b_out.append(b)
                c_out.append(c)
    if not a_out:
        raise SystemExit("no (a, b, c) triple gives D exactly zero; the probe cannot be built")
    n = len(a_out)
    return {
        "a": np.array(a_out),
        "b": np.array(b_out),
        "c": np.array(c_out),
        "nd": np.full(n, 1.0e3),
        "mask": np.ones(n, dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_solvecoagnucl_leaf import build_grid, build_error_probe, DTZ_VALUES\n"
    "from leaf_common import bind_call, check_no_ereport\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "rows = []\n"
    "for _dtz in DTZ_VALUES:\n"
    "    _deln, _e = call('solve dtz=%g' % _dtz, g.leaf_solvecoagnucl,\n"
    "                     grid['mask'], grid['a'], grid['b'], grid['c'],\n"
    "                     grid['nd'], _dtz)\n"
    "    rows.append(np.asarray(_deln))\n"
    "\n"
    "# The 1ca branch calls ereport, which the shim lets return. Run it in its\n"
    "# own call and read the counter DELIBERATELY, instead of letting\n"
    "# bind_call's check turn a reached branch into a failed capture.\n"
    "probe = build_error_probe()\n"
    "_before = tuple(int(v) for v in g.wrap_ereport_count())\n"
    "_pdeln, _pe = g.leaf_solvecoagnucl(probe['mask'], probe['a'], probe['b'],\n"
    "                                   probe['c'], probe['nd'], 60.0)\n"
    "_after = tuple(int(v) for v in g.wrap_ereport_count())\n"
    "g.wrap_ereport_reset()\n"
    "\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, deln=np.stack(rows), probe_deln=np.asarray(_pdeln),\n"
    "         probe_ierr=np.int64(_pe),\n"
    "         ereport_before=np.array(_before), ereport_after=np.array(_after))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- verification -----------------------------------------------------------


def verify_source_literals() -> dict[str, int]:
    text = SOURCE.read_text(encoding="utf-8").replace(" ", "")
    required = {
        "eps_ab": "eps_ab=1.0e-20" in text,
        "eps_d": "eps_d=eps_ab*eps_ab" in text,
        "clamp": "sqd_tms_dtz(i)=50.0" in text,
        "clamp test": "sqd_tms_dtz(i)>50.0" in text,
        "up1 factor 3": "ndnew(:)=1.0/(1.0/nd(:)-3.0*a(:)*dtz)" in text,
        "1ca sets ierr": "WHERE(logic1ca(:))ierr(:)=1" in text,
        "ereport": "CALLereport('UKCA_SOLVECOAGNUCL_V'" in text,
        "2b form": "WHERE(logic2b(:))ndnew(:)=nd(:)+c(:)*dtz" in text,
    }
    missing = sorted(k for k, ok in required.items() if not ok)
    if missing:
        raise SystemExit("the vendored routine no longer contains: " + ", ".join(missing))
    return {"checked": len(required)}


def branch_census(grid: dict) -> dict[str, int]:
    """How many (row, dtz) pairs each branch code takes, across the sweep."""
    census = dict.fromkeys(CODES, 0)
    for dtz in DTZ_VALUES:
        for i in range(grid["a"].size):
            if not grid["mask"][i]:
                census["masked_off"] += 1
                continue
            census[
                classify(
                    float(grid["a"][i]),
                    float(grid["b"][i]),
                    float(grid["c"][i]),
                    float(grid["nd"][i]),
                    dtz,
                )
            ] += 1
    return census


def verify_branch_coverage(grid: dict) -> dict[str, int]:
    census = branch_census(grid)
    # 1ca is unreachable from the main grid by construction -- it needs D
    # exactly zero with B non-zero -- and is probed separately.
    for code in CODES:
        if code == "1ca":
            continue
        if census[code] == 0:
            raise SystemExit(
                f"branch `{code}` is never taken; issue #13 is about exactly this, "
                "and a port that got it wrong would pass every comparison"
            )
    if census["1ca"] != 0:
        raise SystemExit(
            "the main grid reaches 1ca, which calls a fatal ereport -- move those "
            "rows into the probe or the capture will record a poisoned result"
        )
    return census


def verify_outputs(deln: np.ndarray, data: dict, grid: dict) -> dict:
    if deln.shape != (len(SETUPS), len(DTZ_VALUES), grid["a"].size):
        raise SystemExit(f"deln has shape {deln.shape}")
    if not np.array_equal(deln[0], deln[1]):
        raise SystemExit("the two mode setups disagree; the solver is not setup-independent")

    off = grid["mask"] == 0
    if np.any(deln[:, :, off] != 0.0):
        raise SystemExit("deln is non-zero on a masked-off row")
    live = grid["mask"] == 1
    if not np.all(np.isfinite(deln[:, :, live])):
        n = int((~np.isfinite(deln[:, :, live])).sum())
        raise SystemExit(f"deln is not finite on {n} live entries")

    # 2b is `ndnew = nd + c*dtz` and `deln = ndnew - nd`, so `deln` is
    # `(nd + c*dtz) - nd` -- which is NOT `c*dtz`. At nd = 1e3 and c*dtz = 1e-4
    # the sum is not representable and the difference comes back
    # 9.999999997489795e-05. This check asserted `c*dtz` on its first draft and
    # the capture refused the golden, correctly: the cancellation is part of
    # the closed form and a port that computed `deln` directly as `c*dtz` would
    # disagree with the Fortran on every row where `nd` dominates.
    for k, dtz in enumerate(DTZ_VALUES):
        for i in range(grid["a"].size):
            if not grid["mask"][i]:
                continue
            if (
                classify(
                    float(grid["a"][i]),
                    float(grid["b"][i]),
                    float(grid["c"][i]),
                    float(grid["nd"][i]),
                    dtz,
                )
                == "2b"
            ):
                nd_i = float(grid["nd"][i])
                want = (nd_i + float(grid["c"][i]) * dtz) - nd_i
                if deln[0, k, i] != want:
                    raise SystemExit(
                        f"2b row {i} at dtz={dtz}: deln is {deln[0, k, i]!r}, "
                        f"expected (nd + c*dtz) - nd = {want!r}"
                    )

    # The gap rows must come back EXACTLY zero: no branch ran, so `ndnew` kept
    # its initialised `nd`. This is the ninth branch code, and it is the one
    # nobody wrote down.
    for k, dtz in enumerate(DTZ_VALUES):
        for i in range(grid["a"].size):
            if not grid["mask"][i]:
                continue
            if (
                classify(
                    float(grid["a"][i]),
                    float(grid["b"][i]),
                    float(grid["c"][i]),
                    float(grid["nd"][i]),
                    dtz,
                )
                == "gap_a"
            ):
                if deln[0, k, i] != 0.0:
                    raise SystemExit(
                        f"gap row {i} at dtz={dtz}: deln is {deln[0, k, i]!r}, expected "
                        "exactly 0.0 -- |a| == eps_ab satisfies neither logic1 nor logic2, "
                        "so no closed form runs"
                    )

    # The error probe must have fired an ereport, and nothing else may have.
    before = data["ereport_before"]
    after = data["ereport_after"]
    if not np.any(after > before):
        raise SystemExit(
            "the 1ca probe raised no ereport, so it did not reach the error branch "
            "-- D was not exactly zero after all"
        )
    return {"ereports": int((after - before).sum())}


def capture(out_dir: Path, quiet: bool = False) -> Path:
    verify_source_literals()
    grid = build_grid()
    census = verify_branch_coverage(grid)
    probe = build_error_probe()

    stacks: dict[str, list] = {}
    paths: list[str] = []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"solvecoagnucl setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as d:
                for k in d.files:
                    stacks.setdefault(k, []).append(d[k])
            if not quiet:
                print(
                    f"  i_mode_setup = {setup}: {len(DTZ_VALUES)} timesteps x {grid['a'].size} rows"
                )
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    data = {k: np.stack(v) for k, v in stacks.items()}
    deln = data["deln"]

    check_varied(
        {f"dtz_{d}": {"deln": deln[0, k].tolist()} for k, d in enumerate(DTZ_VALUES)},
        expected_identical=[],
        what="solvecoagnucl timesteps",
    )
    report = verify_outputs(deln, {k: v[0] for k, v in data.items()}, grid)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        deln=deln,
        probe_deln=data["probe_deln"],
        probe_ereports=np.int64(report["ereports"]),
        setups=np.array(SETUPS, dtype=np.int64),
        dtz=np.array(DTZ_VALUES, dtype=np.float64),
        codes=np.array(CODES, dtype="U12"),
        **{k: grid[k] for k in ("a", "b", "c", "nd", "mask")},
        **{f"probe_{k}": probe[k] for k in ("a", "b", "c", "nd")},
        **{f"_hits_{k}": np.int64(v) for k, v in census.items()},
    )
    if not quiet:
        print(f"  wrote {path.name}: deln {deln.shape}")
        print("  branch hits: " + ", ".join(f"{k}={v}" for k, v in census.items() if v))
        print(f"  1ca probe: {probe['a'].size} rows, {report['ereports']} ereport(s)")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    grid = build_grid()
    if args.dry_run:
        verify_source_literals()
        census = verify_branch_coverage(grid)
        probe = build_error_probe()
        print(f"leaf solvecoagnucl sweep -> {args.out / ARCHIVE}")
        print(f"  rows        {grid['a'].size:>6,} x {len(DTZ_VALUES)} timesteps")
        print(f"  probe rows  {probe['a'].size:>6,}  (1ca, fatal ereport)")
        for code in CODES:
            print(f"    {code:<12}{census[code]:>6,}")
        return 0

    print(f"sweeping leaf_solvecoagnucl -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
