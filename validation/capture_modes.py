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

**The capture is asserted to be real**, three ways, because the failure this
repo has already shipped is a capture whose namelist edit no-op'd: every setup
then holds the default configuration, and every byte-equality test passes
against it. A bad golden must not be *written*, so all three checks run before
`np.savez_compressed`, not in `pytest` afterwards:

1. `render_namelist` asserts the match count of BOTH substitutions — the
   `i_mode_setup` one as well as the switch injection — and then re-reads
   `i_mode_setup` out of the text it produced;
2. every subprocess reads `i_mode_setup` back out of the *Fortran*, through
   `wrap_sizes`, and refuses to return a record for the wrong setup. That is
   the one check a text bug cannot fool;
3. `check_capture_varied` requires the captured records to differ where the
   Fortran says they must: all 21 setup pairs differ (`component` differs for
   every one of them, and four other fields for most), and the eight
   switch combinations collapse to exactly seven distinct records per setup —
   `bc_oob` is identical to `default` by construction, which is the point of
   capturing it, and nothing else is.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"
DEFAULT_OUT = REPO / "tests" / "goldens"
ARCHIVE = "modes.f64.tables.npz"

SETUPS = (1, 2, 3, 4, 5, 6, 8)

# Switch combinations. Not a cross product: the five switches touch only
# rhocomp, no_ions and topmode, so what is worth capturing is each switch moved
# off its default, plus the one pair that interacts.
#
# `i_tune_bc` has exactly TWO named values, 1 (tuned) and 2 (mg_mix), and the
# SELECT CASE that reads them has **no CASE DEFAULT** -- so a third, unnamed
# case exists: any other value silently leaves rhocomp(cp_bc) at its literal.
# `oob` captures that, because a silent fall-through is a behaviour the port
# has to reproduce whether or not it is intended. (Same shape as UP-5's
# unchecked icoag.)
#
# Note i_tune_bc is inert unless l_radaer is on, which the box model defaults
# off -- so the two BC combinations both set it.
COMBOS: dict[str, dict] = {
    "default": {},
    "nacl_off": {"l_fix_nacl_density": False},
    "hygro_off": {"l_fix_ukca_hygroscopicities": False},
    "both_off": {"l_fix_nacl_density": False, "l_fix_ukca_hygroscopicities": False},
    "bc_tuned": {"l_radaer": True, "i_tune_bc": 1},
    "bc_mg_mix": {"l_radaer": True, "i_tune_bc": 2},
    "bc_oob": {"l_radaer": True, "i_tune_bc": 3},
    "dust_ageing": {"l_dust_mp_ageing": True},
}

SWITCH_DEFAULTS = {
    "l_radaer": ".FALSE.",
    "i_tune_bc": "1",
    "l_fix_nacl_density": ".TRUE.",
    "l_fix_ukca_hygroscopicities": ".TRUE.",
    "l_dust_mp_ageing": ".FALSE.",
}

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


def _switch_lines(overrides: dict) -> str:
    """Namelist lines for one combination, written explicitly.

    Every switch is emitted, not just the overridden ones: relying on a
    namelist default means the capture silently changes if the default ever
    does, and the golden would move with no diff to explain it.
    """
    values = dict(SWITCH_DEFAULTS)
    for key, value in overrides.items():
        values[key] = (
            str(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else (".TRUE." if value else ".FALSE.")
        )
    return "".join(f"\n  {k} = {v}" for k, v in values.items())


_SETUP_RE = re.compile(r"^(\s*i_mode_setup\s*=\s*)(\d+)", re.MULTILINE)
# Anchored at both ends: `^(&box_aerosol)` alone also matches the opening of a
# group merely named like it, and would inject five switches into the wrong one.
_GROUP_RE = re.compile(r"^(&box_aerosol)[ \t]*$", re.MULTILINE)


def render_namelist(text: str, setup: int, combo: str) -> str:
    """The namelist for one (setup, combination), with every edit asserted.

    Both substitutions, not one. The switch injection already counted its
    matches; the `i_mode_setup` one did not, and that is precisely the edit
    whose silent failure produced a golden holding the default configuration
    seven times over, byte-equal and green.

    Text rewriting only — no extension, no subprocess — so
    `tests/test_capture_scripts.py` can exercise it directly, including the
    no-op cases that must raise.
    """
    if combo not in COMBOS:
        raise SystemExit(f"unknown switch combination {combo!r}")

    found = _SETUP_RE.findall(text)
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly 1 i_mode_setup line in the namelist, found {len(found)} -- "
            "count=1 would silently edit the first of them"
        )

    text, n = _SETUP_RE.subn(lambda m: m.group(1) + str(setup), text, count=1)
    if n != 1:
        raise SystemExit(f"failed to set i_mode_setup = {setup} in the namelist")

    # Every switch is written explicitly into &box_aerosol, so the capture
    # never depends on a namelist default that might change.
    switches = _switch_lines(COMBOS[combo])
    text, n = _GROUP_RE.subn(lambda m: m.group(1) + switches, text, count=1)
    if n != 1:
        raise SystemExit("failed to inject switches into &box_aerosol")

    # The substitution counted its match; this checks what the match produced.
    written = _SETUP_RE.findall(text)
    if len(written) != 1 or int(written[0][1]) != setup:
        raise SystemExit(f"rendered namelist reads i_mode_setup = {written}, wanted {setup}")
    return text


# The child, built once at import so `tests/test_capture_scripts.py` can compile
# it without a built extension: a syntax error in here would otherwise surface
# only during a capture, seven subprocesses deep.
_CHILD = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {str(F2PY_DIR)!r})
        import glomap_f2py as g

        nml, setup = sys.argv[1], int(sys.argv[2])

        ierr = g.wrap_init(nml)
        if ierr != 0:
            print('@@FAIL@@' + json.dumps({{'setup': setup, 'ierr': int(ierr)}}))
            raise SystemExit(0)

        sizes = g.wrap_sizes()
        assert sizes[-1] == 0, ('wrap_sizes', sizes[-1])
        nbox, nmodes, ncp, nchemg, nadvg, nbudaer, nsteps, i_mode_setup = sizes[:8]
        # The Fortran's own opinion of which setup it is running. Every check on
        # the namelist text is a check on the text; this is the one that fails
        # if the text was fine and the setup still did not take.
        assert int(i_mode_setup) == setup, ('wrong setup', int(i_mode_setup), setup)

        out = {{'setup': setup, 'nmodes': int(nmodes), 'ncp': int(ncp)}}
        topmode, e = g.wrap_topmode(); assert e == 0, ('topmode', e)
        out['topmode'] = int(topmode)

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


def capture_one(setup: int, combo: str = "default") -> dict:
    """Run one setup in its own process and return every table as lists."""
    # Any namelist works: the tables depend on i_mode_setup and the five
    # switches, not on the meteorology, so the case is overridden here. Rendered
    # in the parent rather than in the child so the edits are asserted by code
    # that a test can call.
    source = (NAMELISTS / "boundary_layer.nml").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        nml = Path(tmp) / "setup.nml"
        nml.write_text(render_namelist(source, setup, combo))
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, str(nml), str(setup)],
            capture_output=True,
            text=True,
        )
    label = f"setup {setup} ({combo})"
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"{label}: wrap_init returned ierr={payload['ierr']}")
    if "@@RESULT@@" not in proc.stdout:
        raise SystemExit(f"{label}: child produced no result:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


# The one pair of records in the whole matrix that is allowed to be identical.
# `bc_oob` sets i_tune_bc to a value the SELECT CASE does not name, and that
# CASE has no DEFAULT, so rhocomp(cp_bc) keeps the literal it already had --
# which is what `default` produces as well. Capturing the fall-through is the
# point of the combination; writing the collision down is what keeps the guard
# below falsifiable instead of tuned to whatever came out.
IDENTICAL_COMBOS = frozenset({frozenset({"default", "bc_oob"})})


def _fingerprint(rec: dict) -> tuple:
    """A hashable form of one captured record, ignoring the setup label itself."""

    def freeze(value):
        return tuple(freeze(v) for v in value) if isinstance(value, list) else value

    return tuple((k, freeze(v)) for k, v in sorted(rec.items()) if k != "setup")


def check_setups_differ(by_setup: dict[int, dict], combo: str = "default") -> None:
    """Refuse a capture in which two setups produced the same tables.

    All 21 pairs of the seven supported setups differ in the Fortran —
    `component` in every pair, and `component_choice`, `mode`, `mode_choice`,
    `fracbcem`, `fracocem` in most — so any collision means the capture did not
    vary `i_mode_setup`, whatever the tables look like.
    """
    prints = {setup: _fingerprint(rec) for setup, rec in by_setup.items()}
    same = [(a, b) for a, b in itertools.combinations(sorted(prints), 2) if prints[a] == prints[b]]
    if same:
        pairs = ", ".join(f"{a}=={b}" for a, b in same)
        raise SystemExit(
            f"combination {combo!r}: setups {pairs} captured identical tables -- "
            "i_mode_setup did not vary, so the golden would hold one setup several "
            "times over and every byte-equality test would pass against it"
        )


def check_combos_differ(by_combo: dict[str, dict], setup: int) -> None:
    """Refuse a capture in which the switch combinations did not take effect.

    Exactly one collision is expected and it is named in `IDENTICAL_COMBOS`;
    an unexpected one means a switch never reached the Fortran, and a *missing*
    one means the fall-through this capture exists to record has stopped
    happening. Both are findings, so both raise.
    """
    missing = sorted(set(COMBOS) - set(by_combo))
    if missing:
        raise SystemExit(f"setup {setup}: no record for combination(s) {missing}")
    prints = {combo: _fingerprint(rec) for combo, rec in by_combo.items()}
    collided = {
        frozenset({a, b})
        for a, b in itertools.combinations(sorted(prints), 2)
        if prints[a] == prints[b]
    }
    if collided != IDENTICAL_COMBOS:

        def show(pairs):
            return ", ".join("==".join(sorted(p)) for p in sorted(map(sorted, pairs))) or "none"

        raise SystemExit(
            f"setup {setup}: switch combinations collided as [{show(collided)}], "
            f"expected exactly [{show(IDENTICAL_COMBOS)}] -- "
            "an extra collision means a switch never reached the Fortran; a missing "
            "one means the i_tune_bc fall-through changed"
        )


def check_capture_varied(records: dict[str, dict[int, dict]]) -> None:
    """Every anti-collapse check, run before anything is written."""
    for combo, by_setup in records.items():
        check_setups_differ(by_setup, combo)
    for setup in SETUPS:
        check_combos_differ({c: recs[setup] for c, recs in records.items()}, setup)


def _store(arrays: dict, setup: int, combo: str, rec: dict) -> None:
    """Default combination keeps the bare `s<setup>_<field>` key so the
    existing goldens and tests are unchanged; variants are prefixed."""
    prefix = f"s{setup}_" if combo == "default" else f"v_{combo}_s{setup}_"
    for key, value in rec.items():
        if key == "setup":
            continue
        name = prefix + key
        if key == "component_names":
            arrays[name] = np.array(value, dtype=np.str_)
        elif key in MODE_INT + CP_INT + MODE_CP_INT or key in ("nmodes", "ncp", "topmode"):
            arrays[name] = np.array(value, dtype=np.int32)
        else:
            arrays[name] = np.array(value, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"{len(SETUPS)} setups x {len(COMBOS)} switch combinations -> {args.out / ARCHIVE}")
        for s in SETUPS:
            print(f"  i_mode_setup = {s}")
        for c, o in COMBOS.items():
            print(f"  {c:<12} {o or '(box model defaults)'}")
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
    records: dict[str, dict[int, dict]] = {}
    for combo in COMBOS:
        records[combo] = {}
        for setup in SETUPS:
            rec = capture_one(setup, combo)
            records[combo][setup] = rec
            if combo == "default":
                print(
                    f"  setup {setup}: ncp={rec['ncp']} topmode={rec['topmode']} "
                    f"active={sum(rec['mode'])} modes"
                )
            _store(arrays, setup, combo, rec)
        if combo != "default":
            print(f"  {combo:<12} captured for {len(SETUPS)} setups")

    # Before anything is written: a collapsed golden must not reach the disk,
    # because once it is there every byte-equality test agrees with it.
    check_capture_varied(records)
    print(
        f"  witness : {len(SETUPS)} setups pairwise distinct in every combination; "
        f"{len(COMBOS) - len(IDENTICAL_COMBOS)} distinct combinations per setup"
    )

    arrays["_case"] = np.array("modes")
    arrays["_mode"] = np.array("tables")
    arrays["_variant"] = np.array("f64")
    arrays["_setups"] = np.array(SETUPS, dtype=np.int32)
    arrays["_combos"] = np.array(list(COMBOS), dtype=np.str_)
    arrays["_rows"] = np.array(len(arrays), dtype=np.int64)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / ARCHIVE
    np.savez_compressed(path, **arrays)
    print(f"wrote {path.name}  {path.stat().st_size / 1e3:.0f} kB")
    print("record it with: python validation/goldens_manifest.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
