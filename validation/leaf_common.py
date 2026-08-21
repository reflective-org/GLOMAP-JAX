#!/usr/bin/env python3
"""Shared machinery for the phase-D physics capture scripts.

The four capture scripts (`capture_vapour_leaf.py`, `capture_water_leaf.py`,
`capture_drydiam_leaf.py`, `capture_volume_mode_leaf.py`) differ only in their
grids and their anti-collapse expectations. Everything else -- bracketing every
driver call with the ereport shim, running one child per configuration,
confirming from the Fortran that the configuration took, refusing to write a
collapsed golden -- is the same, and is here.

It is one module rather than four copies because the harness rules it encodes
are the ones this repository has already broken once each:

* a substitution that silently matched nothing produced a golden holding
  identical data for all seven setups, and every byte-equality test passed
  against it (hence `check_varied` and `CHILD_PREAMBLE`);
* `ukca_mode_setup` never deallocates, so a second init in the same process
  returns the first setup's tables (hence `run_child`);
* `ukca_water_content_v.F90:235` patches its own SAVEd table in place and never
  restores it, so a flag swept in one process compares the patched table
  against itself (hence `run_child` again, and issue #22).

`capture_leaf.py` predates this module and keeps its own copies; `check_no_ereport`
is imported from there rather than forked, so there is exactly one definition of
what "the sweep is void" means.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from capture_leaf import check_no_ereport

REPO = Path(__file__).resolve().parents[1]
F2PY_DIR = REPO / "validation" / "f2py"
NAMELISTS = REPO / "fortran" / "namelists"

__all__ = [
    "CHILD_PREAMBLE",
    "F2PY_DIR",
    "NAMELISTS",
    "REPO",
    "bind_call",
    "check_no_ereport",
    "check_varied",
    "fingerprint",
    "run_child",
]


def bind_call(g) -> Callable:
    """Return a `call(what, fn, *args)` that brackets every driver call.

    The shim returns where the real `ereport` would `STOP 1`, so a driver that
    took an error path hands back a number rather than crashing -- and the
    number is meaningless. Counting either side is the only way to notice.

    Bound to the extension module rather than taking it per call, because the
    one thing worse than an unchecked driver call is one that looks checked.
    """

    def call(what: str, fn, *args):
        before = tuple(int(v) for v in g.wrap_ereport_count())
        result = fn(*args)
        after = tuple(int(v) for v in g.wrap_ereport_count())
        check_no_ereport(before, after, what, g.wrap_ereport_last())
        ierr = result[-1] if isinstance(result, tuple) else None
        if ierr is not None and int(ierr) != 0:
            raise SystemExit(
                f"{what} returned ierr={int(ierr)} -- "
                "1 the process is poisoned, 2 a shape disagrees with the module, "
                "4 wrap_init has not run"
            )
        return result

    return call


# Prepended to every child script. Confirms the configuration from the Fortran
# rather than from the text that was meant to set it: `wrap_get_config_flags`
# reads back i_mode_setup and both phase-D flags, so a namelist edit that
# silently matched nothing fails here instead of producing a plausible golden.
CHILD_PREAMBLE = textwrap.dedent("""
    import json, sys
    import numpy as np
    sys.path.insert(0, {f2py!r})
    import glomap_f2py as g

    _nml, _setup, _fix_water, _fix_neg_pvol = sys.argv[1:5]
    _setup = int(_setup)
    _fix_water = int(_fix_water)
    _fix_neg_pvol = int(_fix_neg_pvol)

    g.wrap_ereport_reset()
    _ierr = int(g.wrap_init(_nml))
    if _ierr != 0:
        print("@@FAIL@@" + json.dumps({{"stage": "wrap_init", "ierr": _ierr}}))
        raise SystemExit(0)

    for _setter, _value in ((g.wrap_set_fix_water_content, _fix_water),
                            (g.wrap_set_fix_neg_pvol_wat, _fix_neg_pvol)):
        if _value >= 0:
            _e = int(_setter(_value))
            if _e != 0:
                print("@@FAIL@@" + json.dumps({{"stage": _setter.__name__, "ierr": _e}}))
                raise SystemExit(0)

    # Read back, from the Fortran, what this process actually holds.
    _fw, _fn, _got_setup, _e = g.wrap_get_config_flags()
    if int(_e) != 0:
        print("@@FAIL@@" + json.dumps({{"stage": "wrap_get_config_flags", "ierr": int(_e)}}))
        raise SystemExit(0)
    if int(_got_setup) != _setup:
        print("@@FAIL@@" + json.dumps({{
            "stage": "setup readback", "want": _setup, "got": int(_got_setup)}}))
        raise SystemExit(0)
    for _name, _want, _got in (("fix_water", _fix_water, int(_fw)),
                               ("fix_neg_pvol", _fix_neg_pvol, int(_fn))):
        if _want >= 0 and _want != _got:
            print("@@FAIL@@" + json.dumps({{
                "stage": f"{{_name}} readback", "want": _want, "got": _got}}))
            raise SystemExit(0)
""")


def run_child(
    body: str,
    *,
    namelist_text: str,
    setup: int,
    fix_water: int = -1,
    fix_neg_pvol: int = -1,
    label: str | None = None,
) -> dict:
    """Run one configuration in its own process and return its `@@RESULT@@`.

    One process per configuration is not tidiness. `ukca_mode_setup` never
    deallocates, and `ukca_water_content_v`'s coefficient table is a one-way
    latch (#22) -- either one, swept in a single process, silently compares a
    configuration against itself.

    `fix_water` and `fix_neg_pvol` are -1 for "leave alone", 0 or 1 to set and
    then confirm from the Fortran.
    """
    label = label or f"setup {setup} (fix_water={fix_water}, fix_neg_pvol={fix_neg_pvol})"
    script = CHILD_PREAMBLE.format(f2py=str(F2PY_DIR)) + textwrap.dedent(body)
    with tempfile.TemporaryDirectory() as tmp:
        nml = Path(tmp) / "case.nml"
        nml.write_text(namelist_text, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-c", script, str(nml), str(setup), str(fix_water), str(fix_neg_pvol)],
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed:\n{proc.stdout}\n{proc.stderr}")
    if "@@FAIL@@" in proc.stdout:
        payload = json.loads(proc.stdout[proc.stdout.rindex("@@FAIL@@") + 8 :])
        raise SystemExit(f"{label}: {payload}")
    if "@@RESULT@@" not in proc.stdout:
        raise SystemExit(f"{label}: child produced no result:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout[proc.stdout.rindex("@@RESULT@@") + 10 :])


def fingerprint(record: Mapping) -> tuple:
    """A hashable summary of one captured record, for collision detection."""

    def freeze(value):
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return value

    return tuple(sorted((k, freeze(v)) for k, v in record.items()))


def check_varied(
    records: Mapping[str, Mapping],
    *,
    expected_identical: Iterable[Sequence[str]] = (),
    what: str = "configurations",
) -> None:
    """Refuse to write a golden in which configurations that must differ did not.

    Run this **before** `savez_compressed`, never as a test afterwards. The
    incident this exists for wrote a collapsed golden, committed it, and passed
    every byte-equality test against it; a guard that lives in the test suite
    catches that one run too late.

    `expected_identical` names the collisions that are real findings about the
    Fortran rather than capture bugs -- an unexpected collision means a
    configuration never reached the Fortran, and a *missing* expected one means
    the behaviour being recorded has changed. Both raise, because both are
    findings.
    """
    expected = {frozenset(pair) for pair in expected_identical}
    prints = {name: fingerprint(rec) for name, rec in records.items()}
    collided = {
        frozenset({a, b})
        for a, b in itertools.combinations(sorted(prints), 2)
        if prints[a] == prints[b]
    }
    if collided == expected:
        return

    def show(pairs):
        return ", ".join("==".join(sorted(p)) for p in sorted(map(sorted, pairs))) or "none"

    raise SystemExit(
        f"{what} collided as [{show(collided)}], expected exactly [{show(expected)}] -- "
        "an unexpected collision means a configuration never reached the Fortran, so the "
        "golden would hold one configuration several times over and every byte-equality "
        "test would pass against it; a missing one means the behaviour being recorded "
        "has changed"
    )
