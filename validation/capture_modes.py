#!/usr/bin/env python3
"""Capture the mode and component tables for every supported setup (task 24).

    python validation/capture_modes.py --dry-run
    python validation/capture_modes.py

`common_mode_setup_interface` turns `i_mode_setup` plus five density and
hygroscopicity switches into `glomap_variables_type` — the mode and component
tables that every process routine reads. Phase C ports them, and its acceptance
is **byte equality** rather than a tolerance, because these are the inputs to
everything downstream: a mode diameter that is one ulp off is not a small error
in a later result, it is a different model.

ONE SUBPROCESS PER SETUP, and that is not a stylistic choice. `ukca_mode_setup`
allocates under `IF (.NOT. ALLOCATED)` and never deallocates, and the 283
`nmas*` budget indices have no initialiser, so a second `init_ukca_for_box` in
one process leaves stale indices — and since `nbudaer` also changes (8 vs 138)
a stale index can be out of bounds. The binding refuses a second init; this
script gives each setup its own process.

Setups: 1, 2, 3, 4, 5, 6, 8. UKCA also defines 10, 11, 12 and 13, but
`glomap_box_config_mod`'s `init_indices` has no CASE for them and ereports
instead — so they have no reference and cannot be captured. Note the
consequence for mode coverage: slot 8, `mode_sup_insol`, is active only in
setups 12 and 13, so it is off in every configuration this port can validate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "modes.f64.tables.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)

MODE_REAL = (
    "fracbcem",
    "fracocem",
    "ddplim0",
    "ddpmid",
    "ddplim1",
    "mmid",
    "mlo",
    "mhi",
    "num_eps",
    "sigmag",
    "x",
)
MODE_INT = ("mode_choice", "modesol", "mode")
CP_REAL = ("mm", "rhocomp", "no_ions")
CP_INT = ("component_choice", "soluble_choice", "soluble")
MODE_CP_REAL = ("mfrac_0",)
MODE_CP_INT = ("component_mode", "component")


def capture_one(setup: int) -> dict:
    """Run one setup in its own process and return every table as lists."""
    script = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(F2PY_DIR)!r})
        import glomap_f2py as g

        # Any namelist works: the tables depend on i_mode_setup and the five
        # switches, not on the meteorology, so the case is overridden below.
        src = {str(NAMELISTS)!r} + '/boundary_layer.nml'
        text = open(src).read()
        import re, tempfile, pathlib
        text = re.sub(r'^(\\s*i_mode_setup\\s*=\\s*)\\d+', r'\\g<1>{setup}',
                      text, count=1, flags=re.MULTILINE)
        d = pathlib.Path(tempfile.mkdtemp())
        nml = d / 'setup.nml'
        nml.write_text(text)

        ierr = g.wrap_init(str(nml))
        if ierr != 0:
            print('@@FAIL@@' + json.dumps({{'setup': {setup}, 'ierr': int(ierr)}}))
            raise SystemExit(0)

        nbox, nmodes, ncp = g.wrap_sizes()[:3]
        out = {{'setup': {setup}, 'nmodes': int(nmodes), 'ncp': int(ncp)}}
        out['topmode'] = int(g.wrap_topmode()[0])

        for f in {MODE_REAL!r}:
            v, e = g.wrap_mode_real(f, nmodes); assert e == 0, (f, e)
            out[f] = v.tolist()
        for f in {MODE_INT!r}:
            v, e = g.wrap_mode_int(f, nmodes); assert e == 0, (f, e)
            out[f] = [int(x) for x in v]
        for f in {CP_REAL!r}:
            v, e = g.wrap_cp_real(f, ncp); assert e == 0, (f, e)
            out[f] = v.tolist()
        for f in {CP_INT!r}:
            v, e = g.wrap_cp_int(f, ncp); assert e == 0, (f, e)
            out[f] = [int(x) for x in v]
        for f in {MODE_CP_REAL!r}:
            v, e = g.wrap_mode_cp_real(f, nmodes, ncp); assert e == 0, (f, e)
            out[f] = v.tolist()
        for f in {MODE_CP_INT!r}:
            v, e = g.wrap_mode_cp_int(f, nmodes, ncp); assert e == 0, (f, e)
            out[f] = [[int(x) for x in row] for row in v]

        names, e = g.wrap_component_names(ncp); assert e == 0, e
        s = names.decode() if isinstance(names, bytes) else names
        out['component_names'] = [s[7*i:7*(i+1)].strip() for i in range(ncp)]

        print('@@RESULT@@' + json.dumps(out))
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"setup {setup} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"setup {setup}: wrap_init returned ierr={payload['ierr']}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"{len(SETUPS)} setups -> {args.out / ARCHIVE}")
        for s in SETUPS:
            print(f"  i_mode_setup = {s}")
        fields = (
            MODE_REAL
            + MODE_INT
            + CP_REAL
            + CP_INT
            + MODE_CP_REAL
            + MODE_CP_INT
            + ("component_names", "topmode")
        )
        print(f"  {len(fields)} fields per setup: {', '.join(sorted(fields))}")
        return 0

    arrays: dict[str, np.ndarray] = {}
    for setup in SETUPS:
        rec = capture_one(setup)
        print(
            f"  setup {setup}: ncp={rec['ncp']} topmode={rec['topmode']} "
            f"active={sum(rec['mode'])} modes"
        )
        for key, value in rec.items():
            if key == "setup":
                continue
            name = f"s{setup}_{key}"
            if key == "component_names":
                arrays[name] = np.array(value, dtype=np.str_)
            elif key in MODE_INT + CP_INT + MODE_CP_INT or key in ("nmodes", "ncp", "topmode"):
                arrays[name] = np.array(value, dtype=np.int32)
            else:
                arrays[name] = np.array(value, dtype=np.float64)

    arrays["_case"] = np.array("modes")
    arrays["_mode"] = np.array("tables")
    arrays["_variant"] = np.array("f64")
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_rows"] = np.array(len(arrays), dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ARCHIVE
    np.savez_compressed(path, **arrays)
    print(f"wrote {path.name}  {path.stat().st_size / 1e3:.0f} kB")
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
