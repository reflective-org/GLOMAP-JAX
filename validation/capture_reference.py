#!/usr/bin/env python3
"""Capture Fortran reference output as `.npz` goldens.

    python validation/capture_reference.py --dry-run
    python validation/capture_reference.py --mode branches --case marine_bcoc
    python validation/capture_reference.py --steps 3 --out /tmp/smoke

The reference binary emits four independent streams, each written only when its
namelist key is set to a non-empty path (see docs/REFERENCE_BUILD.md):

    trajectory   output_file   the wide per-step state table
    budgets      budget_file   the 283 per-process mass fluxes
    state        state_file    a snapshot after each of the 13 process calls
    branches     branch_file   the predicates the science branches on (gate 0)

`--mode` selects among them. `all` is not a cross product: `ref-f32` exists to
measure the precision floor against `ref-f64`, and the floor is a property of
the trajectory. Capturing budgets or per-substep dumps in single precision would
produce large fixtures that no test can use, so the matrix pairs the trajectory
with both variants and everything else with `f64` only. `--dry-run` prints that
matrix rather than describing it, because the shape of the matrix is the design
decision here and it should be inspectable without a Fortran toolchain.

Storage. The two wide streams become one float64 array plus a column-name array.
The two long-format streams are stored columnar with their string columns
factorised into a codebook -- `site`, `field` and `tag` repeat over hundreds of
thousands of rows, and storing them as strings costs more than the numbers do.
`branches` values are small integers and are stored as int8. Task 18 decides the
LFS question; this script reports what it wrote so that decision has numbers
behind it.

Provenance travels with the data: every archive carries the case, mode, variant,
step count and a hash of the exact namelist that produced it, so task 17's
manifest gate can tell a stale golden from a regenerated one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FORTRAN = REPO / "fortran"
NAMELIST_DIRS = (FORTRAN / "namelists", REPO / "validation" / "namelists")
DEFAULT_OUT = REPO / "tests" / "goldens"

# mode -> (namelist key, which variants are worth capturing)
MODES: dict[str, tuple[str, tuple[str, ...]]] = {
    "trajectory": ("output_file", ("f32", "f64")),
    "budgets": ("budget_file", ("f64",)),
    "state": ("state_file", ("f64",)),
    "branches": ("branch_file", ("f64",)),
}

# Long-format streams: which columns are text (factorised) and which are numeric.
LONG_SCHEMA: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    # mode: (text columns, integer columns, value column dtype)
    "state": (("site", "field"), ("step", "imts", "izts", "imode", "icp"), "f8"),
    "branches": (
        ("site", "tag"),
        ("step", "imts", "izts", "i1", "i2", "ibox"),
        "i1",
    ),
}


@dataclass(frozen=True)
class Job:
    case: str
    mode: str
    variant: str
    steps: int | None

    @property
    def stem(self) -> str:
        return f"{self.case}.{self.variant}.{self.mode}"


def discover_cases() -> dict[str, Path]:
    """Namelists from both directories, keyed by stem.

    `fortran/namelists/` are the shipped cases; `validation/namelists/` are ones
    this repo added to cover structure the shipped set misses (`bl_nmts3` is the
    only case with `nmts > 1`, so it is the only one that exercises the nested
    outer/inner scan at all).
    """
    cases: dict[str, Path] = {}
    for d in NAMELIST_DIRS:
        for nml in sorted(d.glob("*.nml")):
            cases[nml.stem] = nml
    return cases


def build_matrix(cases: list[str], modes: list[str], variants: list[str], steps: int | None):
    jobs = []
    for case in cases:
        for mode in modes:
            for variant in MODES[mode][1]:
                if variant in variants:
                    jobs.append(Job(case, mode, variant, steps))
    return jobs


def _rewrite_namelist(source: Path, key: str, target: Path, steps: int | None) -> str:
    """Point one output key at `target`, silence the others, and set nsteps.

    Every key is rewritten explicitly rather than left alone: a namelist that
    already carries, say, `state_file` would otherwise write a second stream on
    every capture, which is slow and confusing rather than wrong.
    """
    text = source.read_text(encoding="utf-8")

    for mode, (mode_key, _) in MODES.items():
        if mode_key == key:
            value = str(target)
        elif mode_key == MODES["trajectory"][0]:
            # The driver always opens output_file; an empty name aborts it. Even
            # when the trajectory is not what we are capturing, it has to go
            # somewhere, so it goes to the scratch directory and is discarded.
            value = str(target.with_name("trajectory.csv"))
        else:
            value = ""
        pattern = re.compile(rf"^(\s*){mode_key}(\s*)=\s*'[^']*'\s*$", re.MULTILINE)
        line = f"  {mode_key} = '{value}'"
        if pattern.search(text):
            text = pattern.sub(line, text, count=1)
        else:
            # Not present: insert at the top of &box_run.
            text = re.sub(r"^(&box_run\s*)$", rf"\1\n{line}", text, count=1, flags=re.MULTILINE)

    if steps is not None:
        text = re.sub(
            r"^(\s*nsteps\s*=\s*)\d+", rf"\g<1>{steps}", text, count=1, flags=re.MULTILINE
        )
    return text


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty -- did the reference write anything?")
    return [c.strip() for c in rows[0]], rows[1:]


def _pack_wide(header: list[str], rows: list[list[str]]) -> dict[str, np.ndarray]:
    values = np.array([[float(c) for c in r] for r in rows], dtype=np.float64)
    return {"columns": np.array(header, dtype=np.str_), "values": values}


def _pack_long(mode: str, header: list[str], rows: list[list[str]]) -> dict[str, np.ndarray]:
    text_cols, int_cols, value_dtype = LONG_SCHEMA[mode]
    idx = {name: header.index(name) for name in header}
    out: dict[str, np.ndarray] = {}

    for name in text_cols:
        raw = [r[idx[name]] for r in rows]
        codes, uniques = _factorise(raw)
        out[name] = codes
        out[f"{name}_levels"] = uniques

    for name in int_cols:
        out[name] = np.array([int(r[idx[name]]) for r in rows], dtype=np.int32)

    raw_values = [r[idx["value"]] for r in rows]
    if value_dtype == "i1":
        out["value"] = np.array([int(v) for v in raw_values], dtype=np.int8)
    else:
        out["value"] = np.array([float(v) for v in raw_values], dtype=np.float64)
    return out


def _factorise(values: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map repeated strings to small integer codes plus a level table.

    `np.unique(..., return_inverse=True)` would do this, but it sorts, and a
    stable first-appearance order makes a diff between two archives readable.
    """
    levels: dict[str, int] = {}
    codes = np.empty(len(values), dtype=np.int16)
    for i, v in enumerate(values):
        codes[i] = levels.setdefault(v, len(levels))
    return codes, np.array(list(levels), dtype=np.str_)


def capture(job: Job, out_dir: Path, quiet: bool = False) -> Path:
    exe = FORTRAN / f"bin-ref-{job.variant}" / "glomap_box"
    if not exe.is_file():
        raise SystemExit(f"{exe} not found. Run: ./validation/build_reference.sh {job.variant}")
    source = discover_cases()[job.case]
    key = MODES[job.mode][0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        target = tmp_path / "stream.csv"
        nml_text = _rewrite_namelist(source, key, target, job.steps)
        nml = tmp_path / f"{job.case}.nml"
        nml.write_text(nml_text, encoding="utf-8")

        result = subprocess.run([str(exe), str(nml)], cwd=tmp_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"reference failed for {job.stem} (exit {result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )
        if not target.is_file():
            raise SystemExit(f"{job.stem}: the reference wrote no {job.mode} stream")

        header, rows = _read_csv(target)
        if job.mode in LONG_SCHEMA:
            arrays = _pack_long(job.mode, header, rows)
        else:
            arrays = _pack_wide(header, rows)

        arrays["_case"] = np.array(job.case)
        arrays["_mode"] = np.array(job.mode)
        arrays["_variant"] = np.array(job.variant)
        arrays["_rows"] = np.array(len(rows), dtype=np.int64)
        arrays["_namelist_sha256"] = np.array(hashlib.sha256(nml_text.encode()).hexdigest())

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{job.stem}.npz"
    np.savez_compressed(path, **arrays)
    if not quiet:
        print(f"  {path.name:<40} {len(rows):>9,} rows  {path.stat().st_size / 1e6:6.2f} MB")
    return path


def print_matrix(jobs: list[Job], out_dir: Path) -> None:
    print(f"{len(jobs)} capture(s) -> {out_dir}")
    print(f"  {'case':<18} {'mode':<12} {'variant':<8} {'steps':<7} archive")
    for job in jobs:
        steps = "namelist" if job.steps is None else str(job.steps)
        print(f"  {job.case:<18} {job.mode:<12} {job.variant:<8} {steps:<7} {job.stem}.npz")
    missing = sorted(
        {j.variant for j in jobs if not (FORTRAN / f"bin-ref-{j.variant}" / "glomap_box").is_file()}
    )
    if missing:
        print(
            f"\nnot built: {', '.join(missing)} -- run "
            f"./validation/build_reference.sh {' '.join(missing)}"
        )


def main(argv: list[str] | None = None) -> int:
    cases = discover_cases()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        action="append",
        choices=[*MODES, "all"],
        help="stream to capture; repeatable (default: all)",
    )
    parser.add_argument(
        "--case", action="append", choices=sorted(cases), help="namelist; repeatable (default: all)"
    )
    parser.add_argument("--variant", action="append", choices=["f32", "f64", "both"])
    parser.add_argument("--steps", type=int, help="override nsteps (for smoke captures)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and stop")
    args = parser.parse_args(argv)

    modes = list(MODES) if not args.mode or "all" in args.mode else args.mode
    variants = ["f32", "f64"]
    if args.variant and "both" not in args.variant:
        variants = args.variant
    jobs = build_matrix(args.case or sorted(cases), modes, variants, args.steps)

    if args.dry_run:
        print_matrix(jobs, args.out)
        return 0

    if shutil.which("gfortran") is None and not any(
        (FORTRAN / f"bin-ref-{v}" / "glomap_box").is_file() for v in variants
    ):
        raise SystemExit("no reference binary and no gfortran; nothing to capture")

    print(f"capturing {len(jobs)} archive(s) -> {args.out}")
    total = 0
    for job in jobs:
        total += capture(job, args.out).stat().st_size
    print(f"total {total / 1e6:.2f} MB")
    # Deliberately not automatic. Auto-blessing a capture would make the drift
    # gate report nothing the one time it matters.
    print("record them with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
