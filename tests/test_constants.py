"""Constants are extracted from the Fortran, never retyped.

`mam4-jax`'s convention, and the reason for it is arithmetic: UKCA carries
`avogadro = 6.022e23` where CODATA says `6.02214076e23`. That is 2.3e-5
relative — eight orders of magnitude above `RTOL_ALGEBRAIC`. A single digit
typed from memory instead of from the source silently invalidates every golden
comparison downstream of a concentration conversion, and the failure appears in
whichever routine happens to use it first.

So this file re-parses the vendored Fortran on every run and compares. A
constant cannot drift, cannot be "corrected" to the modern value, and cannot
survive an update to the vendored tree that changes it.

No `fortran` marker: reads source text, needs no toolchain.
"""

import re
from pathlib import Path

import pytest

from glomap_jax.core import constants

REPO = Path(__file__).resolve().parents[1]
UKCA = REPO / "fortran" / "src" / "ukca"

# name in glomap_jax -> (Fortran file, Fortran identifier)
EXTRACTED = {
    "AVOGADRO": ("ukca_config_constants_mod.F90", "avogadro"),
    "BOLTZMANN": ("ukca_config_constants_mod.F90", "boltzmann"),
    "RMOL": ("ukca_config_constants_mod.F90", "rmol"),
    "RHO_SO4": ("ukca_config_constants_mod.F90", "rho_so4"),
    "RHO_WATER": ("ukca_config_constants_mod.F90", "rho_water"),
    "PI": ("ukca_constants.F90", "pi"),
    "ZERODEGC": ("ukca_constants.F90", "zerodegc"),
    "MMSUL": ("ukca_constants.F90", "mmsul"),
    "MMW": ("ukca_constants.F90", "mmw"),
    "NMOL": ("ukca_constants.F90", "nmol"),
    "CONC_EPS": ("ukca_constants.F90", "conc_eps"),
    "DN_EPS": ("ukca_constants.F90", "dn_eps"),
}


def _fortran_value(filename: str, identifier: str) -> float:
    """Last assignment of `identifier` in `filename`, as a float.

    Last rather than first: ukca_config_constants_mod declares each as
    `REAL, SAVE :: x = rmdi` (a missing-data sentinel) and assigns the real
    value later in init_config_constants.
    """
    text = (UKCA / filename).read_text(encoding="utf-8")
    pattern = re.compile(rf"^\s*(?:REAL[^:]*::\s*)?{identifier}\s*=\s*([0-9.eE+-]+)", re.MULTILINE)
    matches = [m for m in pattern.findall(text) if m.lower() not in ("rmdi",)]
    assert matches, f"{identifier} not found in {filename}"
    return float(matches[-1])


@pytest.mark.parametrize("name", sorted(EXTRACTED))
def test_constant_matches_the_vendored_fortran(name):
    filename, identifier = EXTRACTED[name]
    assert getattr(constants, name) == _fortran_value(filename, identifier), (
        f"{name} disagrees with {identifier} in {filename}. Do not edit the "
        f"Python value to match: check whether the vendored tree changed."
    )


def test_the_reference_does_not_use_codata_values():
    """Stated as a test because it is the mistake most likely to be made in good
    faith, by someone who notices the values look dated and 'fixes' them."""
    assert constants.AVOGADRO == 6.022e23, "not CODATA 6.02214076e23 — see the module docstring"
    assert constants.BOLTZMANN == 1.3804e-23, "not CODATA 1.380649e-23"
    codata_shift = abs(6.02214076e23 - constants.AVOGADRO) / constants.AVOGADRO
    assert codata_shift > 1e-13, "the gap has closed; re-derive the argument"


def test_eps_d_is_formed_as_the_product_the_fortran_forms():
    """`ukca_solvecoagnucl_v:178` computes `eps_d = eps_ab*eps_ab`. Writing
    `1.0e-40` directly is not required to give the same double."""
    assert constants.EPS_D == constants.EPS_AB * constants.EPS_AB


@pytest.mark.parametrize(
    ("name", "expected", "where"),
    [
        ("EPS_AB", 1.0e-20, "ukca_solvecoagnucl_v.F90"),
        ("SQD_CLAMP", 50.0, "ukca_solvecoagnucl_v.F90"),
        ("XXX_EPS", 1.0e-3, "ukca_coagwithnucl.F90"),
        ("J_EPS", 1.0e-3, "ukca_calcnucrate.F90"),
    ],
)
def test_inline_threshold_still_appears_in_its_routine(name, expected, where):
    """These are locals in the Fortran, not module constants, so they cannot be
    extracted by name. Assert the literal is still present in the routine that
    owns it — enough to catch an upstream change to the value."""
    assert getattr(constants, name) == expected
    rendered = {1.0e-20: "1.0e-20", 50.0: "50.0", 1.0e-3: "1.0e-3"}[expected]
    text = (UKCA / where).read_text(encoding="utf-8")
    assert rendered in text, f"{rendered} no longer appears in {where}"
