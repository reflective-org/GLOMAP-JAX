#!/usr/bin/env python3
"""Sweep `ukca_binapara` over the inputs the physics can reach (task 51).

    python validation/capture_binapara_leaf.py --dry-run
    python validation/capture_binapara_leaf.py       # writes tests/goldens/

`ukca_binapara` is the Vehkamaki (2002) binary homogeneous nucleation
parameterisation: three temperature/humidity/H2SO4 polynomials, 113 literal
coefficients, and an exponential on each of the three results. It is the whole
of the binary nucleation rate -- `ukca_calcnucrate` hard-codes
`i_bhn_method = i_bhn_method_vekhamaki` at `:256-257`, so the Kulmala branch
beside it is dead code (`docs/unsupported.md`).

What this sweep is built to reach
---------------------------------

**The clamps, and the fact that they are applied to the routine's own copies.**
`:106-118` clips `t` to [190.15, 300.15], `rh` to [0.0001, 1.0] and `h2so4` to
[1e4, 1e11] -- and then every later reference, including the `t(jl) < 195.15`
test at `:251`, reads the *clipped* value. So a caller passing 150 K gets the
190.15 K answer and does **not** take the cold branch, because 190.15 is not
below 195.15. That is easy to port backwards, and the grid straddles all six
bounds with their neighbouring doubles.

**The three post-exponential rules at `:249-283`, in their own order.**

* `ntot_out < 4` **and** clipped `t < 195.15` sets `jveh = 1e5`;
* then `jveh < 1e-7` sets it to 0;
* then `jveh > 1e10` sets it to 1e10.

The cold-branch value 1e5 sits strictly between the floor and the ceiling, so
it survives both later rules -- and, measured, the three therefore commute.
That is a coincidence of the constants rather than a property of the code, and
the sweep records enough to show it: 182 points exceed the ceiling before rule
1 and 74 after. A port that applied the cold rule to the *unclipped*
temperature would disagree only below 190.15 K, which is why the grid goes
there.

**The `ntot < 4` region at all, which needs cold and dry and dilute together.**
`verify_branch_coverage` refuses a grid that does not reach it, and separately
refuses one that does not reach it *with* `t < 195.15` -- the two halves of the
cold rule are counted apart, because a sweep that only ever hit them together
could not tell which one gates.

Why the polynomials are the interesting part
--------------------------------------------

`termx` divides the other two: 10 of `jveh`'s 50 terms and 10 of `ntot`'s
carry `/termx(jl)`. So `termx` near zero is a pole, and it is not obviously
unreachable -- it is a mole fraction, not a normalised quantity, and nothing in
the routine guards it. The grid is dense where `termx` is small so the golden
records what the compiled routine actually returns there.

`log_v` and `exp_v` are plain `LOG`/`EXP` loops (`ukca_um_legacy_mod.F90:394`,
`:479`), so the port has no wrapper to reproduce -- but `EXP` is the primitive
issue #28 is about, and this routine applies it to all three results. Byte
equality is therefore not expected on `jveh` or `rc`; what is expected is that
substituting libm's `exp` restores it, which is what the port's tests assert.
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
ARCHIVE = "binapara.f64.leaf.npz"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_binapara_mod.F90"
NAMELIST_TEXT = (NAMELISTS / "boundary_layer.nml").read_text(encoding="utf-8")

SETUPS: tuple[int, ...] = (1, 4)

T_LO, T_HI = 190.15, 300.15
RH_LO, RH_HI = 0.0001, 1.0
H_LO, H_HI = 1.0e4, 1.0e11
T_COLD = 195.15


def _straddle(x: float) -> tuple[float, float, float]:
    """A clamp bound and both neighbouring doubles.

    The tests are strict `<` and `>`, so the bound itself must come out
    unclamped and the two neighbours must fall on opposite sides. A grid that
    sampled only round numbers could not tell `<` from `<=`.
    """
    return float(np.nextafter(x, -np.inf)), x, float(np.nextafter(x, np.inf))


T_AXIS = tuple(
    sorted(
        set(
            [
                float(s)
                for s in (
                    "150.0",
                    "180.0",
                    "192.0",
                    "196.0",
                    "210.0",
                    "230.0",
                    "250.0",
                    "273.15",
                    "290.0",
                    "298.0",
                    "310.0",
                    "350.0",
                )
            ]
            + list(_straddle(T_LO))
            + list(_straddle(T_HI))
            + list(_straddle(T_COLD))
        )
    )
)

RH_AXIS = tuple(
    sorted(
        set(
            [
                float(s)
                for s in (
                    "0.0",
                    "0.00001",
                    "0.001",
                    "0.01",
                    "0.1",
                    "0.3",
                    "0.5",
                    "0.7",
                    "0.9",
                    "0.99",
                    "1.5",
                )
            ]
            + list(_straddle(RH_LO))
            + list(_straddle(RH_HI))
        )
    )
)

H2SO4_AXIS = tuple(
    sorted(
        set(
            [
                float(s)
                for s in (
                    "0.0",
                    "1.0e2",
                    "1.0e3",
                    "1.0e5",
                    "1.0e6",
                    "1.0e7",
                    "1.0e8",
                    "1.0e9",
                    "1.0e10",
                    "1.0e12",
                )
            ]
            + list(_straddle(H_LO))
            + list(_straddle(H_HI))
        )
    )
)

BASE_T, BASE_RH, BASE_H = 273.15, 0.5, 1.0e7

BLOCKS = ("main", "t", "rh", "h2so4", "cold_dilute")


def _block_code(name: str) -> int:
    return BLOCKS.index(name)


def build_grid() -> dict[str, np.ndarray]:
    t: list[float] = []
    rh: list[float] = []
    h: list[float] = []
    block: list[int] = []

    def add(tt, rr, hh, blk):
        t.append(tt)
        rh.append(rr)
        h.append(hh)
        block.append(_block_code(blk))

    # A coarse product over all three, because the polynomials are trivariate
    # and an axis sweep at a fixed base cannot reach the cross terms -- 26 of
    # jveh's 50 terms carry two or three factors.
    for tt in T_AXIS[::2]:
        for rr in RH_AXIS[::2]:
            for hh in H2SO4_AXIS[::2]:
                add(tt, rr, hh, "main")

    for tt in T_AXIS:
        add(tt, BASE_RH, BASE_H, "t")
    for rr in RH_AXIS:
        add(BASE_T, rr, BASE_H, "rh")
    for hh in H2SO4_AXIS:
        add(BASE_T, BASE_RH, hh, "h2so4")

    # Cold, dry and dilute together: the corner where the critical cluster
    # falls below four molecules and the `t < 195.15` rule can fire.
    for tt in (150.0, 185.0, 190.15, 192.0, 194.0, T_COLD, 196.0, 200.0, 210.0):
        for rr in (0.0001, 0.001, 0.01, 0.05):
            for hh in (1.0e4, 1.0e5, 1.0e6, 1.0e8):
                add(tt, rr, hh, "cold_dilute")

    return {
        "t": np.array(t, dtype=np.float64),
        "rh": np.array(rh, dtype=np.float64),
        "h2so4": np.array(h, dtype=np.float64),
        "block": np.array(block, dtype=np.int32),
    }


CHILD_BODY = (
    "\nimport tempfile\n"
    f"sys.path.insert(0, {str(REPO / 'validation')!r})\n"
    "from capture_binapara_leaf import build_grid\n"
    "from leaf_common import bind_call\n"
    "\n"
    "call = bind_call(g)\n"
    "grid = build_grid()\n"
    "jveh, rc, _e = call('leaf_binapara', g.leaf_binapara,\n"
    "                    grid['t'], grid['rh'], grid['h2so4'])\n"
    "_tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)\n"
    "_tmp.close()\n"
    "np.savez(_tmp.name, jveh=np.asarray(jveh), rc=np.asarray(rc))\n"
    "print('@@RESULT@@' + json.dumps({'npz_path': _tmp.name, 'setup': _setup}))\n"
)

_CHILD = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + CHILD_BODY


# --- an independent numpy reconstruction, used to count what the outputs hide -


def simulate(t: np.ndarray, rh: np.ndarray, h2so4: np.ndarray) -> dict[str, np.ndarray]:
    """`ukca_binapara` again, in numpy, from the extracted literals.

    The same trick `capture_water_leaf.py` plays: an independent computation
    that the capture can cross-check the Fortran against before writing a
    golden. Two things it buys that the outputs alone cannot.

    First, **`ntot` is not returned**, so "the critical cluster fell below four
    molecules" is invisible in `jveh` unless the temperature rule also fired.
    Counting that branch needs the intermediate.

    Second, it is a check on the *extraction*. `numpy.exp` reaches the same
    libm gfortran does (issue #28), so this reconstruction should agree with
    the compiled routine to the last bit -- and if it does not, a coefficient
    was parsed wrongly, which is the failure mode 113 machine-extracted
    literals exist to prevent.

    Deliberately numpy and deliberately not importing the port: a
    reconstruction that shared code with the thing it checks would only
    confirm that the code equals itself.
    """
    sys.path.insert(0, str(REPO / "src"))
    from glomap_jax.physics._binapara_literals import (
        CLAMPS,
        JVEH_TERMS,
        LIMITS,
        NTOT_TERMS,
        RC_TERMS,
        TERMX_TERMS,
    )

    t = np.clip(np.asarray(t, dtype=np.float64), *CLAMPS["t"])
    rh = np.clip(np.asarray(rh, dtype=np.float64), *CLAMPS["rh"])
    h2so4 = np.clip(np.asarray(h2so4, dtype=np.float64), *CLAMPS["h2so4"])

    logrh = np.log(rh)
    logh2so4 = np.log(h2so4)
    env = {
        "tdegk": t,
        "tdegk2": t * t,
        "tdegk3": t * t * t,
        "logrh": logrh,
        "logrh2": logrh * logrh,
        "logrh3": logrh * logrh * logrh,
        "logh2so4": logh2so4,
        "logh2so42": logh2so4 * logh2so4,
        "logh2so43": logh2so4 * logh2so4 * logh2so4,
    }

    def evaluate(terms, extra=None):
        table = dict(env, **(extra or {}))
        total = np.zeros_like(t)
        for i, (sign, coeff, factors, over) in enumerate(terms):
            value = np.full_like(t, coeff)
            for f in factors:
                value = value * table[f]
            if over:
                value = value / table["termx(jl)"]
            total = (
                value.copy()
                if i == 0 and sign == "+"
                else (-value if i == 0 else (total + value if sign == "+" else total - value))
            )
        return total

    termx = evaluate(TERMX_TERMS)
    jveh_log = evaluate(JVEH_TERMS, {"termx(jl)": termx})
    ntot_log = evaluate(NTOT_TERMS, {"termx(jl)": termx})
    rc_log = evaluate(RC_TERMS, {"termx(jl)": termx, "ntot(jl)": ntot_log})

    ntot = np.exp(ntot_log)
    jveh = np.exp(jveh_log)
    rc = np.exp(rc_log)

    cold = (ntot < LIMITS["ntot_min"]) & (t < LIMITS["t_cold"])
    jveh = np.where(cold, LIMITS["j_cold"], jveh)
    jveh = np.where(jveh < LIMITS["j_floor"], 0.0, jveh)
    jveh = np.where(jveh > LIMITS["j_ceiling"], LIMITS["j_ceiling"], jveh)

    return {
        "termx": termx,
        "ntot": ntot,
        "jveh": jveh,
        "rc": rc,
        "cold": cold,
        "small_cluster": ntot < LIMITS["ntot_min"],
        "floored": np.exp(jveh_log) < LIMITS["j_floor"],
        "ceilinged": np.exp(jveh_log) > LIMITS["j_ceiling"],
    }


def verify_branch_coverage(sim: dict) -> dict[str, int]:
    """Refuse a grid that leaves any of the four rules unexercised."""
    counts = {
        "small_cluster": int(sim["small_cluster"].sum()),
        "cold_and_small": int(sim["cold"].sum()),
        "small_but_warm": int((sim["small_cluster"] & ~sim["cold"]).sum()),
        "floored": int(sim["floored"].sum()),
        "ceilinged": int(sim["ceilinged"].sum()),
    }
    for name, n in counts.items():
        if n == 0:
            raise SystemExit(
                f"no grid point reaches `{name}`; the rule it gates is unvalidated "
                "and a port that omitted it would pass every comparison"
            )
    return counts


def verify_outputs(jveh: np.ndarray, rc: np.ndarray, grid: dict, sim: dict) -> dict:
    """Everything the golden claims, before it is written."""
    if jveh.shape != (len(SETUPS), grid["t"].size):
        raise SystemExit(f"jveh has shape {jveh.shape}")

    if not (np.array_equal(jveh[0], jveh[1]) and np.array_equal(rc[0], rc[1])):
        raise SystemExit("the two mode setups disagree; ukca_binapara is not setup-independent")

    if not np.all(np.isfinite(jveh)) or not np.all(np.isfinite(rc)):
        raise SystemExit("non-finite output on a grid every point of which is clamped into range")
    if np.any(jveh < 0.0) or np.any(rc <= 0.0):
        raise SystemExit("jveh must be >= 0 and rc > 0; rc is exp() of a polynomial")

    counts = verify_branch_coverage(sim)

    # The reconstruction must reproduce the compiled routine bit for bit.
    # numpy.exp reaches the same libm gfortran does, so any disagreement is a
    # mis-parsed coefficient and not issue #28.
    for name, got, want in (("jveh", sim["jveh"], jveh[0]), ("rc", sim["rc"], rc[0])):
        bad = np.flatnonzero(got != want)
        if bad.size:
            worst = bad[np.argmax(np.abs(got[bad] - want[bad]) / np.abs(want[bad]))]
            raise SystemExit(
                f"the numpy reconstruction of {name} differs from the compiled routine on "
                f"{bad.size} of {want.size} points -- worst at t={grid['t'][worst]}, "
                f"rh={grid['rh'][worst]}, h2so4={grid['h2so4'][worst]}: "
                f"{got[worst]!r} vs {want[worst]!r}. A coefficient is mis-parsed."
            )

    # The three post-exponential rules must be visible in the golden itself.
    from glomap_jax.physics._binapara_literals import LIMITS

    seen = {
        "j_cold": int((jveh[0] == LIMITS["j_cold"]).sum()),
        "j_floor": int((jveh[0] == 0.0).sum()),
        "j_ceiling": int((jveh[0] == LIMITS["j_ceiling"]).sum()),
    }
    for name, n in seen.items():
        if n == 0:
            raise SystemExit(f"the {name} rule left no trace in the golden")
    return {**counts, **seen}


def capture(out_dir: Path, quiet: bool = False) -> Path:
    grid = build_grid()
    sim = simulate(grid["t"], grid["rh"], grid["h2so4"])
    verify_branch_coverage(sim)

    stacks: dict[str, list] = {"jveh": [], "rc": []}
    paths: list[str] = []
    try:
        for setup in SETUPS:
            record = run_child(
                CHILD_BODY,
                namelist_text=render_namelist(NAMELIST_TEXT, setup, "default"),
                setup=setup,
                label=f"binapara setup {setup}",
            )
            paths.append(record["npz_path"])
            with np.load(record["npz_path"]) as d:
                stacks["jveh"].append(d["jveh"])
                stacks["rc"].append(d["rc"])
            if not quiet:
                print(f"  i_mode_setup = {setup}: {grid['t'].size} points")
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    jveh = np.stack(stacks["jveh"])
    rc = np.stack(stacks["rc"])

    check_varied(
        {"setup_1": {"jveh": jveh[0].tolist()}, "setup_4": {"jveh": jveh[1].tolist()}},
        expected_identical=[("setup_1", "setup_4")],
        what="binapara setups",
    )

    report = verify_outputs(jveh, rc, grid, sim)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ARCHIVE
    np.savez_compressed(
        path,
        jveh=jveh,
        rc=rc,
        setups=np.array(SETUPS, dtype=np.int64),
        blocks=np.array(BLOCKS, dtype="U16"),
        **{k: grid[k] for k in ("t", "rh", "h2so4", "block")},
        **{f"_{k}": np.int64(v) for k, v in report.items()},
    )
    if not quiet:
        print(f"  wrote {path.name}: {jveh.shape}")
        print("  branch hits: " + ", ".join(f"{k}={v}" for k, v in report.items()))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    grid = build_grid()
    if args.dry_run:
        sim = simulate(grid["t"], grid["rh"], grid["h2so4"])
        counts = verify_branch_coverage(sim)
        print(f"leaf binapara sweep -> {args.out / ARCHIVE}")
        print(f"  points      {grid['t'].size:>6,}")
        for b in BLOCKS:
            print(f"    {b:<14}{int((grid['block'] == _block_code(b)).sum()):>6,}")
        print(f"  setups      {len(SETUPS):>6,}  {SETUPS}")
        print("  branch hits: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"  termx range: [{sim['termx'].min():.4g}, {sim['termx'].max():.4g}]")
        return 0

    print(f"sweeping leaf_binapara -> {args.out}")
    capture(args.out)
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
