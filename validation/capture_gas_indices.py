#!/usr/bin/env python3
"""Capture the gas-phase index tables for every supported setup (task 31).

    python validation/capture_gas_indices.py --dry-run
    python validation/capture_gas_indices.py

`glomap_box_config_mod`'s `init_indices` calls TWO routines per setup. Task 24
captured what the mode one builds; this captures what the gas one builds — the
174 integer scalars and four length-50 arrays of `ukca_setup_indices` that say
where each gas species sits in `s0g`, which of them condense, into which
aerosol component, and with what molar mass and molecular diameter.

Byte equality, not a tolerance: these are indices and molar masses feeding
every process routine.

ONE SUBPROCESS PER SETUP, for the same reason `capture_modes.py` needs one:
`ukca_mode_setup` allocates under `IF (.NOT. ALLOCATED)` and never deallocates,
the 283 `nmas*` indices have no initialiser, and `wrap_init` refuses a second
init that would need the tables rebuilt.

Setups 1, 2, 3, 4, 5, 6, 8. The gas side collapses to FOUR distinct routines::

    1        -> ukca_indices_sv1
    2, 3, 8  -> ukca_indices_orgv1_soto3
    4, 5     -> ukca_indices_orgv1_soto6
    6        -> ukca_indices_nochem

which is why the variation check below is written per *group* and not "all
seven differ" — three of them are identical by construction, and a check that
demanded otherwise would have to be weakened until it proved nothing.

**The capture is asserted to be real.** Two independent ways, because this repo
has already shipped a golden that was silently the default configuration seven
times over — a string replacement no-op'd, the setup never changed, and every
byte-equality test passed:

1. every subprocess reads `i_mode_setup` back out of the Fortran through
   `wrap_sizes` and refuses to return a record for the wrong setup, and the
   namelist edit itself is a `re.subn` whose match count is asserted;
2. `main` asserts the four routine groups differ pairwise AND that setups
   inside a group agree exactly. If the injection no-op'd, all seven would be
   setup 1 and the first of those fails.
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
sys.path.insert(0, str(REPO / "validation"))
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "gasidx.f64.tables.npz"

import extract_gas_literals as extractor  # noqa: E402

SETUPS = (1, 2, 3, 4, 5, 6, 8)

# The four gas routines, as setup groups. From init_indices' SELECT CASE via
# the extractor, so the capture cannot disagree with the port about which
# setups are supposed to share a table.
GROUPS: dict[str, tuple[int, ...]] = {}
for _setup in SETUPS:
    _routine = extractor.SETUP_ROUTINE[_setup]
    GROUPS[_routine] = (*GROUPS.get(_routine, ()), _setup)

_literals = extractor.extract()
NCHEMGMAX = _literals["nchemgmax"]

# Names in a fixed order, taken from the extraction rather than retyped, so
# the captured set, the ported set and the Fortran accessor's SELECT CASE all
# come from one place. `nadvg` and `ntrag` are derived in the Fortran and the
# port recomputes them — captured anyway, so the recomputation is gated too.
GAS_SCALARS: tuple[str, ...] = tuple(
    _literals["groups"]["count"]
    + ["nadvg", "ntrag"]
    + _literals["groups"]["s0"]
    + _literals["groups"]["st"]
    + _literals["groups"]["budget"]
    + _literals["groups"]["reaction"]
)

# Set by the MODE-side routine, not the gas one, so they belong to the aerosol
# budget map (task 32). Captured here because task 31's acceptance names them
# and two integers cost nothing; `nbudaer` is cross-checked against wrap_sizes.
MODE_SIDE_SCALARS = ("ntraer", "nbudaer")

REAL_ARRAYS = ("mm_gas", "dimen")
INT_ARRAYS = ("condensable_choice", "condensable")

# Declared in `ukca_setup_indices` and never given a value on any box-model
# path: `budget`, `nbudget`, `traqu` and `ntraqu` are assigned only in
# `ukca_indices_traqu38`/`ukca_indices_traqu9`, which `init_indices` does not
# call, and the other three are assigned nowhere in the tree. Reading them
# would capture whatever was left in memory. `tests/test_gas_indices.py`
# re-parses the Fortran and fails if any of them gains an initialiser.
NEVER_INITIALISED = (
    "budget",
    "nbudget",
    "traqu",
    "ntraqu",
    "idustdep",
    "ndustdep",
    "nbudaertot",
)


def capture_one(setup: int) -> dict:
    """Run one setup in its own process and return every gas table."""
    script = textwrap.dedent(f"""
        import json, sys, re, tempfile, pathlib
        sys.path.insert(0, {str(F2PY_DIR)!r})
        import glomap_f2py as g

        # Any namelist works: these tables depend on i_mode_setup alone -- not
        # on the meteorology, and not on any of the five switches that reach
        # the mode side.
        src = {str(NAMELISTS)!r} + '/boundary_layer.nml'
        text = open(src).read()
        text, n = re.subn(r'^(\\s*i_mode_setup\\s*=\\s*)\\d+', r'\\g<1>{setup}',
                          text, count=1, flags=re.MULTILINE)
        # The failure this repo has already shipped: a no-op replacement, seven
        # identical captures, and every byte-equality test green against the
        # default.
        assert n == 1, 'failed to inject i_mode_setup'
        d = pathlib.Path(tempfile.mkdtemp())
        nml = d / 'setup.nml'
        nml.write_text(text)

        ierr = g.wrap_init(str(nml))
        if ierr != 0:
            print('@@FAIL@@' + json.dumps({{'setup': {setup}, 'ierr': int(ierr)}}))
            raise SystemExit(0)

        sizes = g.wrap_sizes()
        assert sizes[-1] == 0, ('wrap_sizes', sizes[-1])
        nbox, nmodes, ncp, nchemg, nadvg, nbudaer, nsteps, i_mode_setup = sizes[:8]
        # The Fortran's own opinion of which setup it is running. Without this
        # a failed namelist edit is indistinguishable from a successful one.
        assert int(i_mode_setup) == {setup}, ('wrong setup', int(i_mode_setup))

        nmax, e = g.wrap_nchemgmax(); assert e == 0, e
        assert int(nmax) == {NCHEMGMAX}, ('nchemgmax moved', int(nmax))

        out = {{'setup': {setup}, 'nchemgmax': int(nmax)}}
        for f in {(GAS_SCALARS + MODE_SIDE_SCALARS)!r}:
            v, e = g.wrap_gas_scalar(f); assert e == 0, (f, e)
            out[f] = int(v)
        for f in {REAL_ARRAYS!r}:
            v, e = g.wrap_gas_real(f, nmax); assert e == 0, (f, e)
            out[f] = v.tolist()
        for f in {INT_ARRAYS!r}:
            v, e = g.wrap_gas_int(f, nmax); assert e == 0, (f, e)
            out[f] = [int(x) for x in v]

        # wrap_sizes reads the same three module variables through a different
        # entry point. Disagreeing would mean the dispatch table is wired to
        # the wrong variable, which no comparison against the port would catch.
        for name, other in (('nchemg', nchemg), ('nadvg', nadvg),
                            ('nbudaer', nbudaer)):
            assert out[name] == int(other), (name, out[name], int(other))

        print('@@RESULT@@' + json.dumps(out))
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"setup {setup} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"setup {setup}: wrap_init returned ierr={payload['ierr']}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


def _check_groups(records: dict[int, dict]) -> None:
    """The variation check. See the module docstring for why it exists.

    Within a routine group every field must be identical; across groups at
    least one field must differ for every pair. A capture that silently ran
    the same setup seven times passes the first and fails the second.

    `ntraer` and `nbudaer` are excluded from both halves. They are set by the
    MODE-side routine, so they differ between setups 2, 3 and 8 even though
    those three share a gas table — including them would break the first half
    for a legitimate reason and make the second half pass for a wrong one.
    """
    skip = {"setup", *MODE_SIDE_SCALARS}
    fields = [k for k in records[SETUPS[0]] if k not in skip]

    for routine, members in GROUPS.items():
        first = records[members[0]]
        for setup in members[1:]:
            for f in fields:
                assert records[setup][f] == first[f], (
                    f"{routine}: setups {members[0]} and {setup} should share "
                    f"a gas table but {f} differs"
                )

    reps = {r: m[0] for r, m in GROUPS.items()}
    names = sorted(reps)
    assert len(names) == 4, f"expected 4 gas routines, got {names}"
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            differs = [f for f in fields if records[reps[a]][f] != records[reps[b]][f]]
            assert differs, (
                f"{a} (setup {reps[a]}) and {b} (setup {reps[b]}) captured "
                f"identical tables -- the setup injection did not take effect"
            )


def _store(arrays: dict, setup: int, rec: dict) -> None:
    """One record into the archive.

    The 176 scalars go in as a SINGLE vector per setup, aligned to
    `_scalar_fields`, not as 176 separate keys. An `.npz` is a zip, so each key
    carries a member header; one key per scalar made the archive 245 kB of
    almost entirely header for 1232 integers. Read them back with
    `dict(zip(data["_scalar_fields"], data[f"s{setup}_scalars"]))`.
    """
    names = GAS_SCALARS + MODE_SIDE_SCALARS
    arrays[f"s{setup}_scalars"] = np.array([rec[n] for n in names], dtype=np.int32)
    arrays[f"s{setup}_nchemgmax"] = np.array(rec["nchemgmax"], dtype=np.int32)
    for key in REAL_ARRAYS:
        arrays[f"s{setup}_{key}"] = np.array(rec[key], dtype=np.float64)
    for key in INT_ARRAYS:
        arrays[f"s{setup}_{key}"] = np.array(rec[key], dtype=np.int32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    n_fields = len(GAS_SCALARS) + len(MODE_SIDE_SCALARS) + len(REAL_ARRAYS) + len(INT_ARRAYS)
    if args.dry_run:
        print(f"{len(SETUPS)} setups -> {args.out / ARCHIVE}")
        for routine, members in sorted(GROUPS.items()):
            print(f"  {routine:<26} setups {', '.join(str(s) for s in members)}")
        print(f"  {len(GAS_SCALARS)} gas scalars + {len(MODE_SIDE_SCALARS)} mode-side")
        print(f"  {len(REAL_ARRAYS) + len(INT_ARRAYS)} arrays of {NCHEMGMAX}")
        print(f"  {n_fields} fields per setup")
        print(f"  not captured (never initialised): {', '.join(NEVER_INITIALISED)}")
        return 0

    records = {}
    for setup in SETUPS:
        rec = capture_one(setup)
        records[setup] = rec
        print(
            f"  setup {setup}: {extractor.SETUP_ROUTINE[setup]:<26} "
            f"nchemg={rec['nchemg']} nadvg={rec['nadvg']} ntrag={rec['ntrag']} "
            f"condensable={sum(rec['condensable'])}"
        )

    _check_groups(records)
    print(f"  {len(GROUPS)} distinct gas tables, pairwise different, groups internally equal")

    arrays: dict[str, np.ndarray] = {}
    for setup in SETUPS:
        _store(arrays, setup, records[setup])

    arrays["_case"] = np.array("gasidx")
    arrays["_mode"] = np.array("tables")
    arrays["_variant"] = np.array("f64")
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_nchemgmax"] = np.array(NCHEMGMAX, dtype=np.int32)
    arrays["_scalar_fields"] = np.array(GAS_SCALARS + MODE_SIDE_SCALARS, dtype=np.str_)
    arrays["_never_initialised"] = np.array(NEVER_INITIALISED, dtype=np.str_)
    arrays["_rows"] = np.array(len(arrays), dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ARCHIVE
    np.savez_compressed(path, **arrays)
    print(f"wrote {path.name}  {path.stat().st_size / 1e3:.0f} kB")
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
