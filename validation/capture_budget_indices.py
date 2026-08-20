#!/usr/bin/env python3
"""Capture the aerosol mass-budget slot index map for every setup (task 32).

    python validation/capture_budget_indices.py --dry-run
    python validation/capture_budget_indices.py            # capture + golden
    python validation/capture_budget_indices.py --emit-literals
    python validation/capture_budget_indices.py --check-literals

`ukca_setup_indices` declares 283 `nmas*` INTEGER scalars. Each one is the
second index into `bud_aer_mas(nbox, 0:nbudaer)` for one (process, component,
mode) mass flux, and each `ukca_indices_*` routine assigns a different subset —
so the map is a per-setup table of 283 small integers, and `nbudaer` itself
takes a different value in each of the seven supported setups.

TWO INDEPENDENT ROUTES TO THE SAME MAP, which is the point of this script.

  * `extract()` parses the assignments out of the vendored source text. No
    toolchain, so `tests/test_budget_indices.py` and the port can use it.
  * `capture_one()` runs the compiled Fortran through the gate-A binding and
    reads the module scalars back after `init_ukca_for_box`.

They are cross-checked against each other on every capture. A text parser can
misread a continuation or miss a routine, and a capture can silently read a
setup it did not ask for; neither failure survives having to agree with the
other. Where they *legitimately* differ is itself a finding — see
`_UNASSIGNED_ARE_ZERO` below.

ONE SUBPROCESS PER SETUP, and that is not a stylistic choice. `ukca_mode_setup`
allocates under `IF (.NOT. ALLOCATED)` and never deallocates, and the 283
`nmas*` scalars have no initialiser, so a second `init_ukca_for_box` in one
process leaves stale indices — and since `nbudaer` also changes (8 vs 138) a
stale index can be out of bounds. The binding refuses a second init; this
script gives each setup its own process, exactly as `capture_modes.py` does.

Setups: 1, 2, 3, 4, 5, 6, 8. UKCA defines 10-13 as well, but the box model's
`init_indices` has no CASE for them and ereports instead, so they have no
reference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"
SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_setup_indices.F90"
ACCESSOR = REPO / "validation" / "f2py" / "glomap_budidx_mod.F90"
LITERALS = REPO / "src" / "glomap_jax" / "physics" / "_budget_index_literals.py"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "budidx.f64.tables.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)

# i_mode_setup -> the aerosol index routine it dispatches to. From the
# SELECT CASE in glomap_box_config_mod.F90's init_indices, which pairs a gas
# routine with an aerosol one; only the aerosol routine touches nmas*/nbudaer.
ROUTINES = {
    1: "ukca_indices_suss_4mode",
    2: "ukca_indices_sussbcoc_5mode",
    3: "ukca_indices_sussbcoc_4mode",
    4: "ukca_indices_sussbcocso_5mode",
    5: "ukca_indices_sussbcocso_4mode",
    6: "ukca_indices_duonly_2mode",
    8: "ukca_indices_sussbcocdu_7mode",
}

# The 38 nmas*mp* names are assigned by ukca_indices_sussbcocdump_8mode alone
# (setup 14, which the box model does not implement), so in all seven supported
# setups they are READ WITHOUT EVER HAVING BEEN ASSIGNED -- 34 of them from a
# live `IF (nmasxxx > 0)` guard. Module scalars have static storage, so gfortran
# puts them in .bss and they read as 0, which makes every one of those guards
# false. The standard does not promise it. The capture measures it rather than
# assuming it: check_capture() asserts every unassigned name comes back 0, so
# a build (or a compiler) where they do not is a failed capture rather than a
# golden full of garbage indices.
UNASSIGNED_MUST_BE_ZERO = True


# ---------------------------------------------------------------------------
# Route 1: the vendored source text
# ---------------------------------------------------------------------------
def declared_names() -> list[str]:
    """The 283 `nmas*` scalars, in declaration order.

    Declaration order, not sorted order: it is the order the accessor's name
    blob uses and the order the golden's value arrays are aligned to, and it
    groups the map by process the way the source does.
    """
    names = [m.lower() for m in re.findall(r"^INTEGER\s*::\s*(nmas\w+)", SOURCE.read_text(), re.M)]
    assert names, f"no nmas* declarations found in {SOURCE.name}"
    assert len(names) == len(set(names)), "duplicate nmas* declaration"
    return names


def _routine_body(source: str, name: str) -> list[str]:
    """One routine's lines, comments dropped.

    Comments are dropped before anything is matched because
    `ukca_ddepaer_mod.F90` carries commented-out `nmasddepntnucsol` blocks, and
    a parser that counted those would report names the live code never uses.
    """
    match = re.search(
        rf"^SUBROUTINE {name}\b.*?^END SUBROUTINE {name}\b",
        source,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    assert match, f"routine {name} not found in {SOURCE.name}"
    return [ln.strip() for ln in match.group(0).split("\n") if not ln.strip().startswith("!")]


def extract() -> dict:
    """Parse the per-setup map out of the source. No toolchain needed.

    Returns {setup: {"routine", "nbudaer", "slots": {name: value}}}, where
    `slots` carries all 283 names with an explicit 0 for every one the routine
    leaves unassigned — that 0 is the port's own decision, recorded here rather
    than left implicit, and the capture is what says the Fortran agrees.
    """
    source = SOURCE.read_text()
    names = declared_names()
    out = {}
    for setup, routine in ROUTINES.items():
        body = _routine_body(source, routine)
        slots = dict.fromkeys(names, 0)
        assigned = set()
        nbudaer = None
        for line in body:
            m = re.match(r"(nmas\w+)\s*=\s*(\d+)", line, re.I)
            if m:
                name, value = m.group(1).lower(), int(m.group(2))
                assert name in slots, f"{routine} assigns undeclared {name}"
                # Every routine assigns `nmasprocntintr23 = 0` and
                # `nmasprocnhintr23 = 0` twice, verbatim. Harmless, so a repeat
                # of the SAME value is accepted -- but a repeat of a DIFFERENT
                # one would make the map depend on statement order, which is
                # exactly the kind of thing a text parser gets wrong silently,
                # so that is an error.
                if name in assigned:
                    assert slots[name] == value, (
                        f"{routine} assigns {name} twice with different values: "
                        f"{slots[name]} then {value}"
                    )
                assigned.add(name)
                slots[name] = value
                continue
            m = re.match(r"nbudaer\s*=\s*(\d+)", line, re.I)
            if m:
                assert nbudaer is None, f"{routine} sets nbudaer twice"
                nbudaer = int(m.group(1))
        assert nbudaer is not None, f"{routine} never sets nbudaer"
        assert assigned, f"{routine} assigns no nmas* at all"
        out[setup] = {
            "routine": routine,
            "nbudaer": nbudaer,
            "assigned": sorted(assigned),
            "slots": slots,
        }
    return out


# ---------------------------------------------------------------------------
# Route 2: the compiled Fortran, through the gate-A binding
# ---------------------------------------------------------------------------
def capture_one(setup: int) -> dict:
    """Run one setup in its own process and read the map back out."""
    script = textwrap.dedent(f"""
        import json, re, sys, tempfile, pathlib
        sys.path.insert(0, {str(F2PY_DIR)!r})
        import glomap_f2py as g

        # Any namelist works: the index map depends on i_mode_setup alone, not
        # on the meteorology, so the case is overridden below.
        text = open({str(NAMELISTS)!r} + '/boundary_layer.nml').read()
        text, nsub = re.subn(r'^(\\s*i_mode_setup\\s*=\\s*)\\d+', r'\\g<1>{setup}',
                             text, count=1, flags=re.MULTILINE)
        # ASSERT THE SUBSTITUTION MATCHED. A replacement that silently no-ops
        # gives seven captures of setup 1 under seven different names, and
        # every byte-equality test downstream passes against the wrong data.
        # That has happened in this repo; it is not a hypothetical.
        assert nsub == 1, 'i_mode_setup substitution matched %d times' % nsub
        assert re.search(r'^\\s*i_mode_setup\\s*=\\s*{setup}\\s*$', text,
                         re.MULTILINE), 'setup {setup} not in the rewritten namelist'

        d = pathlib.Path(tempfile.mkdtemp())
        nml = d / 'setup.nml'
        nml.write_text(text)

        ierr = g.wrap_init(str(nml))
        if ierr != 0:
            print('@@FAIL@@' + json.dumps({{'setup': {setup}, 'ierr': int(ierr)}}))
            raise SystemExit(0)

        # The binding's own view of which setup it initialised, so a capture
        # keyed 'setup N' that actually ran something else is an error here
        # rather than a mystery in the golden.
        sizes = g.wrap_sizes()
        nbudaer_sizes, setup_seen = int(sizes[5]), int(sizes[7])
        assert setup_seen == {setup}, (setup_seen, {setup})

        nnames, namelen, e = g.wrap_bud_count(); assert e == 0, e
        blob, e = g.wrap_bud_names(nnames, namelen); assert e == 0, e
        s = blob.decode() if isinstance(blob, bytes) else blob
        names = [s[namelen*i:namelen*(i+1)].strip() for i in range(nnames)]

        values, nbudaer, e = g.wrap_bud_values(nnames); assert e == 0, e
        assert nbudaer == nbudaer_sizes, (nbudaer, nbudaer_sizes)

        # Prove the blob order IS the value order rather than assuming it.
        # wrap_bud_index looks a name up by string and reports both its slot
        # and its position; if the two arrays were misaligned by even one
        # entry this disagrees. Every name, not a sample: a spot check on the
        # first and last would pass through a rotation of the middle.
        for i, nm in enumerate(names):
            v, pos, e = g.wrap_bud_index(nm)
            assert e == 0, (nm, e)
            assert pos == i + 1, (nm, pos, i + 1)
            assert v == int(values[i]), (nm, v, int(values[i]))
        v, pos, e = g.wrap_bud_index('nmasnosuchname')
        assert (e, v, pos) == (3, 0, 0), (e, v, pos)

        print('@@RESULT@@' + json.dumps({{
            'setup': {setup},
            'nbudaer': int(nbudaer),
            'names': names,
            'values': [int(x) for x in values],
        }}))
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"setup {setup} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"setup {setup}: wrap_init returned ierr={payload['ierr']}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


# ---------------------------------------------------------------------------
# Cross-checks. Each one names the failure it exists to catch.
# ---------------------------------------------------------------------------
def check_capture(records: dict, parsed: dict) -> list[str]:
    """Everything the capture must satisfy before it is written as a golden.

    Returns a list of complaints; an empty list is a pass. Written as a
    returned list rather than bare asserts so a broken capture reports all of
    its problems at once instead of the first.
    """
    problems = []
    names = declared_names()

    for setup, rec in sorted(records.items()):
        if rec["names"] != names:
            problems.append(f"setup {setup}: accessor name blob != the source declarations")
        if rec["nbudaer"] != parsed[setup]["nbudaer"]:
            problems.append(
                f"setup {setup}: nbudaer {rec['nbudaer']} from the binding, "
                f"{parsed[setup]['nbudaer']} from the source"
            )
        slots = parsed[setup]["slots"]
        for name, value in zip(rec["names"], rec["values"], strict=True):
            if value != slots[name]:
                problems.append(
                    f"setup {setup}: {name} = {value} from the binding, {slots[name]} "
                    f"from the source"
                )
        if UNASSIGNED_MUST_BE_ZERO:
            # The names this routine never assigns must still read back 0. They
            # are uninitialised module scalars, so this is a statement about
            # what gfortran did with .bss, not about what the standard says --
            # which is exactly why it is measured on every capture.
            unassigned = set(rec["names"]) - set(parsed[setup]["assigned"])
            bad = [
                n
                for n, v in zip(rec["names"], rec["values"], strict=True)
                if n in unassigned and v != 0
            ]
            if bad:
                problems.append(
                    f"setup {setup}: {len(bad)} never-assigned names read back nonzero, "
                    f"first {bad[0]}"
                )

        nonzero = [v for v in rec["values"] if v > 0]
        # The nonzero slots are a bijection onto 1..nbudaer in every routine.
        # It is the strongest statement that can be made about the map without
        # naming numbers, and it fails on a duplicate, a gap or an overrun --
        # the three ways a hand-maintained index table goes wrong.
        if sorted(nonzero) != list(range(1, rec["nbudaer"] + 1)):
            problems.append(f"setup {setup}: the nonzero slots are not exactly 1..{rec['nbudaer']}")
        if min(rec["values"]) < 0:
            problems.append(f"setup {setup}: negative slot index")

    # A capture that returns the same thing seven times is the failure mode
    # this whole script is exposed to: seven subprocesses, one namelist, one
    # substitution. If the substitution no-ops, everything below still looks
    # like a clean run.
    fingerprints = {setup: tuple(rec["values"]) for setup, rec in records.items()}
    if len(set(fingerprints.values())) != len(records):
        problems.append("the seven setups did not produce seven distinct index maps")
    if len({rec["nbudaer"] for rec in records.values()}) != len(records):
        problems.append("the seven setups did not produce seven distinct nbudaer")
    return problems


# ---------------------------------------------------------------------------
# The generated literals the port imports
# ---------------------------------------------------------------------------
def render_literals(parsed: dict) -> str:
    import pprint

    names = declared_names()
    slots = {setup: [rec["slots"][n] for n in names] for setup, rec in sorted(parsed.items())}
    nbudaer = {setup: rec["nbudaer"] for setup, rec in sorted(parsed.items())}
    return (
        '"""Per-setup aerosol mass-budget slot indices, extracted from\n'
        "`ukca_setup_indices.F90`.\n"
        "\n"
        "GENERATED -- do not edit. Regenerate with::\n"
        "\n"
        "    python validation/capture_budget_indices.py --emit-literals\n"
        "\n"
        "`tests/test_budget_indices.py` re-runs the extraction and compares, and also\n"
        "compares against `tests/goldens/budidx.f64.tables.npz`, which was read out of\n"
        "the compiled Fortran. An edit here fails twice over.\n"
        "\n"
        "`BUDGET_NAMES` is in declaration order and `SETUP_SLOTS[setup]` is aligned to\n"
        "it. A 0 means the flux is not carried in that setup -- it does not mean slot\n"
        "0, which is a hole the Fortran never writes.\n"
        '"""\n\n'
        f"BUDGET_NAMES = {pprint.pformat(tuple(names), width=88)}\n\n"
        f"SETUP_NBUDAER = {pprint.pformat(nbudaer, width=88, sort_dicts=True)}\n\n"
        f"SETUP_SLOTS = {pprint.pformat(slots, width=88, sort_dicts=True)}\n"
    )


def emit_literals(parsed: dict) -> None:
    LITERALS.write_text(render_literals(parsed), encoding="utf-8")
    print(
        f"wrote {LITERALS.relative_to(REPO)} ({len(parsed)} setups, {len(declared_names())} names)"
    )


def check_literals(parsed: dict) -> int:
    """Compare the DATA, not the bytes: `ruff format` reformats the generated
    file after it is written, so a byte comparison reports every formatted file
    as stale, which is a property of the formatter and not of the table."""
    if not LITERALS.is_file():
        print(f"{LITERALS.relative_to(REPO)} does not exist; generate it")
        return 1
    namespace: dict = {}
    exec(compile(LITERALS.read_text(encoding="utf-8"), str(LITERALS), "exec"), namespace)
    names = declared_names()
    fresh_slots = {s: [rec["slots"][n] for n in names] for s, rec in parsed.items()}
    fresh_nbudaer = {s: rec["nbudaer"] for s, rec in parsed.items()}
    if (
        list(namespace.get("BUDGET_NAMES", ())) != names
        or namespace.get("SETUP_NBUDAER") != fresh_nbudaer
        or {k: list(v) for k, v in namespace.get("SETUP_SLOTS", {}).items()} != fresh_slots
    ):
        print(f"{LITERALS.relative_to(REPO)} disagrees with the Fortran; regenerate it")
        return 1
    print("up to date")
    return 0


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--emit-literals", action="store_true")
    parser.add_argument("--check-literals", action="store_true")
    args = parser.parse_args(argv)

    parsed = extract()

    if args.check_literals:
        return check_literals(parsed)
    if args.emit_literals:
        emit_literals(parsed)
        return 0
    if args.dry_run:
        print(f"{len(SETUPS)} setups -> {args.out / ARCHIVE}")
        for setup in SETUPS:
            rec = parsed[setup]
            print(
                f"  i_mode_setup = {setup:<2} {rec['routine']:<32} "
                f"nbudaer={rec['nbudaer']:<4} assigned={len(rec['assigned'])}"
            )
        print(f"  {len(declared_names())} nmas* names per setup, one subprocess each")
        return 0

    records = {}
    for setup in SETUPS:
        rec = capture_one(setup)
        records[setup] = rec
        nonzero = sum(1 for v in rec["values"] if v > 0)
        print(
            f"  setup {setup}: nbudaer={rec['nbudaer']:<4} "
            f"{nonzero} of {len(rec['values'])} names carried"
        )

    problems = check_capture(records, parsed)
    if problems:
        for line in problems:
            print(f"  FAIL {line}")
        raise SystemExit(f"{len(problems)} problem(s); nothing written")

    names = declared_names()
    arrays: dict[str, np.ndarray] = {"names": np.array(names, dtype=np.str_)}
    for setup, rec in records.items():
        arrays[f"s{setup}_values"] = np.array(rec["values"], dtype=np.int32)
        arrays[f"s{setup}_nbudaer"] = np.array(rec["nbudaer"], dtype=np.int32)
        arrays[f"s{setup}_routine"] = np.array(parsed[setup]["routine"])

    arrays["_case"] = np.array("budidx")
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
