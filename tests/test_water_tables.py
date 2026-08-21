"""Task 39 — the two ZSR ion tables, and the index space that is easy to get wrong.

Byte equality against a re-extraction of the Fortran, plus the thing the task's
acceptance criterion actually names: the negative-index remap.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))

import extract_water_literals as ex  # noqa: E402

from glomap_jax.physics import water_tables as wt  # noqa: E402
from glomap_jax.physics._water_literals import BASE, FIXED, LIMITS  # noqa: E402

SOURCE = REPO / "fortran" / "src" / "ukca" / "ukca_water_content_v.F90"
PAIRS = ex.PAIRS


def test_the_committed_literals_are_not_stale():
    """Re-extract and compare. The file is generated, so the only way it can be
    wrong is by drifting from the vendored tree or being hand-edited."""
    assert ex.main(["--check"]) == 0


def test_the_check_mode_rejects_a_doctored_file(tmp_path, monkeypatch, capsys):
    """The passing path above is true of a gate that compares nothing. This is
    the one that fails when the gate is a stub."""
    doctored = tmp_path / "_water_literals.py"
    doctored.write_text("BASE = {}\nFIXED = {}\nLIMITS = {}\n", encoding="utf-8")
    monkeypatch.setattr(ex, "TARGET", doctored)
    assert ex.main(["--check"]) == 1
    assert "regenerate" in capsys.readouterr().out


@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}{p[1]}")
def test_every_pair_is_byte_equal_to_a_fresh_parse(pair):
    """`assert_array_equal`, not `allclose`. These are eleven-significant-digit
    fit coefficients; a single wrong digit gives a plausible water content."""
    fresh = ex.extract()
    np.testing.assert_array_equal(np.array(BASE[pair]), np.array(fresh["base"][pair]))
    np.testing.assert_array_equal(np.array(FIXED[pair]), np.array(fresh["fixed"][pair]))
    np.testing.assert_array_equal(np.array(LIMITS[pair]), np.array(fresh["limits"][pair]))


def test_the_two_tables_differ_in_exactly_one_coefficient():
    """The whole content of `l_fix_ukca_water_content`. If they ever stopped
    differing, every both-settings test downstream would go quietly vacuous --
    which has happened before in this repo, to a flag that was removed for it.
    """
    differ = np.flatnonzero(wt.Y_BASE.ravel() != wt.Y_FIXED.ravel())
    assert differ.size == 1, f"{differ.size} coefficients differ, expected exactly 1"
    (cation, anion), index, value = wt.PATCHED_ENTRY
    row, col = wt.pair_index(cation, anion)
    assert np.unravel_index(differ[0], wt.Y_BASE.shape) == (row, col, index)
    assert wt.Y_FIXED[row, col, index] == value


def test_the_patch_is_a_factor_of_ten_not_a_rounding():
    """Recorded because it says what kind of upstream defect this is. The
    source comment above the block calls the DATA value incorrect."""
    (cation, anion), index, _ = wt.PATCHED_ENTRY
    row, col = wt.pair_index(cation, anion)
    assert wt.Y_FIXED[row, col, index] == pytest.approx(10.0 * wt.Y_BASE[row, col, index])


def test_the_two_index_spaces_are_not_the_same_mapping():
    """The trap this module exists for.

    Ion arrays are indexed by a signed species number over -4..+3; the
    coefficient table by a (cation, anion) pair. Both are negative-lower-bound
    Fortran arrays, both rebase by adding something, and the somethings differ.
    Using the ion offset on a pair shifts every electrolyte by four rows and
    gives water contents that are wrong and finite.
    """
    # Anion -1 is the LAST table column but the fourth ion slot.
    assert wt.pair_index(1, -1) == (0, 3)
    assert wt.ion_slot(-1) == 3
    # ... and anion -4 is the FIRST table column and the zeroth ion slot,
    # which is the coincidence that makes the two look interchangeable.
    assert wt.pair_index(1, -4) == (0, 0)
    assert wt.ion_slot(-4) == 0
    # They part company on cations, which have no table column at all.
    assert wt.ion_slot(3) == 7
    with pytest.raises(IndexError):
        wt.pair_index(1, 3)


def test_the_ion_slots_span_exactly_the_eight_columns_the_driver_passes():
    """`leaf_water_content` marshals `cl` and `ions` as `(n, 8)`. If these ever
    disagree the driver silently reads the wrong column."""
    slots = [wt.ion_slot(s) for s in range(-wt.NANION, wt.NCATION + 1)]
    assert slots == list(range(8))


@pytest.mark.parametrize("bad", [-5, 4])
def test_an_out_of_range_species_raises_rather_than_wrapping(bad):
    """A negative index that wraps is the failure mode here: Python would
    happily read `array[-1]` and return the wrong ion."""
    with pytest.raises(IndexError):
        wt.ion_slot(bad)


def test_the_tables_are_frozen():
    """Two immutable arrays instead of one mutable module array is the whole
    reason the port cannot reproduce the Fortran's one-way latch (#22)."""
    for table in (wt.Y_BASE, wt.Y_FIXED, wt.LIMITS_TABLE):
        assert not table.flags.writeable
        with pytest.raises(ValueError):
            table[0, 0, 0] = 1.0


def test_coefficients_selects_by_flag_and_returns_the_frozen_array():
    assert wt.coefficients(False) is wt.Y_BASE
    assert wt.coefficients(True) is wt.Y_FIXED


def test_the_first_two_pairs_share_coefficients_in_the_fortran():
    """Not a bug and not an extraction artefact: `(1,-1)` H+ HSO4- and `(1,-2)`
    2H+ SO42- carry byte-identical DATA in the source. Pinned so that a future
    "all twelve pairs must differ" check is recognised as wrong before it is
    added, and so that a capture collapsing two rows is still detectable
    against the other ten."""
    text = SOURCE.read_text(encoding="utf-8")
    blocks = re.findall(r"DATA \(y\(1,(-[12]),j\),j=0,7\)/(.*?)/", text, re.DOTALL)
    assert len(blocks) == 2
    assert ex._numbers(blocks[0][1]) == ex._numbers(blocks[1][1])
    np.testing.assert_array_equal(np.array(BASE[(1, -1)]), np.array(BASE[(1, -2)]))
    # The other ten are genuinely distinct, so a collapsed capture is visible.
    rest = [BASE[p] for p in PAIRS if p not in {(1, -1), (1, -2)}]
    assert len({tuple(v) for v in rest}) == len(rest)


def test_rh_min_is_percent_and_molal_max_is_not():
    """`ukca_water_content_v.F90:281` compares `aw` against `rh_min/1.0e2`, so
    `rh_min` is in percent. Reading it as a fraction would put every floor at
    a hundredth of its real value and the clamp would never fire."""
    rh_min = wt.LIMITS_TABLE[..., 0]
    molal_max = wt.LIMITS_TABLE[..., 1]
    assert rh_min.max() > 1.0, "rh_min looks like a fraction, not a percent"
    assert rh_min.max() <= 100.0
    assert molal_max.min() > 1.0
    assert "rh_min(ic,ia)/1.0e2" in SOURCE.read_text(encoding="utf-8").replace(" ", "")
