"""Task 31: the gas-phase index tables, for all seven supported setups.

Three things are being checked here, and they fail for different reasons:

1. **the generated literals match the vendored Fortran** — the extraction is
   re-run and compared, so `_gas_literals.py` cannot drift or be hand-edited;
2. **the port matches the compiled Fortran, byte for byte** — every scalar,
   every array, every setup, against `gasidx.f64.tables.npz`;
3. **the 1-based-to-0-based conversion is the one described** — asserted on
   values whose Fortran side is known, so a sentinel handled the other way
   fails here rather than in whichever process routine indexes with it first.

No `fortran` marker anywhere: the golden is committed and everything else
reads source text, so all of this runs in CI without a toolchain.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "validation"))
import extract_gas_literals as extractor  # noqa: E402

from glomap_jax.physics import gas_indices as gas  # noqa: E402
from glomap_jax.physics._gas_literals import GAS_LITERALS  # noqa: E402

GOLDEN = REPO / "tests" / "goldens" / "gasidx.f64.tables.npz"
ACCESSOR = REPO / "validation" / "f2py" / "glomap_gasidx_mod.F90"
FORTRAN = REPO / "fortran" / "src" / "ukca" / "ukca_setup_indices.F90"
SETUPS = (1, 2, 3, 4, 5, 6, 8)

# The four gas routines, as setup groups. Written out rather than derived, so
# a change to `init_indices`'s pairing has to be made in two places.
ROUTINE_GROUPS = {
    "ukca_indices_sv1": (1,),
    "ukca_indices_orgv1_soto3": (2, 3, 8),
    "ukca_indices_orgv1_soto6": (4, 5),
    "ukca_indices_nochem": (6,),
}

REAL_ARRAYS = ("mm_gas", "dimen")
INT_ARRAYS = ("condensable_choice", "condensable")


@pytest.fixture(scope="module")
def tables():
    assert GOLDEN.is_file(), "run `python validation/capture_gas_indices.py`"
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def scalar_names(tables):
    return [str(n) for n in tables["_scalar_fields"]]


def golden_scalars(tables, names, setup):
    return dict(zip(names, (int(v) for v in tables[f"s{setup}_scalars"])))


# ---------------------------------------------------------------------------
# 1. The generated literals against the vendored Fortran
# ---------------------------------------------------------------------------


def test_the_committed_literals_are_not_stale():
    assert GAS_LITERALS == extractor.extract(), (
        "src/glomap_jax/physics/_gas_literals.py disagrees with the Fortran. "
        "Regenerate: python validation/extract_gas_literals.py"
    )


def test_the_check_mode_agrees(capsys):
    assert extractor.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_continuations_are_joined_before_parsing():
    """`mm_gas` spans six source lines. A line-based parser silently truncates
    it to the eight values on the first line, and every index past 8 is then
    wrong by a table that looks fine."""
    for routine in ROUTINE_GROUPS:
        arrays = GAS_LITERALS["routines"][routine]["arrays"]
        for field in ("mm_gas", "dimen", "condensable_choice"):
            assert len(arrays[field]) == 50, f"{routine}: {field}"


def test_every_gas_routine_assigns_the_same_variables():
    """All four assign an identical set of 178 names. That is what makes a
    single dataclass shape correct for every setup — if one routine left a
    variable alone it would keep the previous setup's value, which in a
    one-init-per-process world is the previous *run's*."""
    sizes = {r: set(v["scalars"]) | set(v["arrays"]) for r, v in GAS_LITERALS["routines"].items()}
    first = next(iter(sizes.values()))
    # 178 assigned names, less the three the extractor drops as derived
    # (nadvg, ntrag, condensable), leaves 172 scalars and 3 array literals.
    assert len(first) == 175
    for routine, names in sizes.items():
        assert names == first, routine


# ---------------------------------------------------------------------------
# 2. The port against the compiled Fortran, byte for byte
# ---------------------------------------------------------------------------


def test_every_supported_setup_was_captured(tables):
    assert list(tables["_setups"]) == list(SETUPS)


def test_unsupported_setups_are_absent(tables):
    """10-13 exist in UKCA but `init_indices` ereports on them, so there is no
    reference to capture."""
    captured = {int(k.split("_")[0][1:]) for k in tables if k.startswith("s")}
    assert captured & {10, 11, 12, 13} == set()


@pytest.mark.parametrize("setup", SETUPS)
def test_counts_match(tables, scalar_names, setup):
    g = golden_scalars(tables, scalar_names, setup)
    p = gas.build(setup)
    for name in ("nchemg", "ichem", "noffox", "nbudchem", "gasbudget", "ngasbudget"):
        assert getattr(p, name) == g[name], name


@pytest.mark.parametrize("setup", SETUPS)
def test_derived_counts_are_recomputed_not_copied(tables, scalar_names, setup):
    """`nadvg = 2 + nchemg` and `ntrag = nadvg + noffox` are computed in
    `gas_indices.build`, and captured from the Fortran separately. Copying them
    out of the literals would make this test compare a value with itself."""
    g = golden_scalars(tables, scalar_names, setup)
    p = gas.build(setup)
    assert p.nadvg == g["nadvg"] == 2 + g["nchemg"]
    assert p.ntrag == g["ntrag"] == g["nadvg"] + g["noffox"]
    assert "nadvg" not in GAS_LITERALS["routines"][p.routine]["scalars"]
    assert "ntrag" not in GAS_LITERALS["routines"][p.routine]["scalars"]


@pytest.mark.parametrize("setup", SETUPS)
def test_every_index_matches_after_conversion(tables, scalar_names, setup):
    """The 166 named indices, converted back to Fortran's convention here —
    `0` for absent, otherwise `+1` — and compared to what the Fortran held.

    The conversion is written out in this file rather than imported from the
    module under test. Calling the module's own converter in both directions
    would make the assertion true for any converter at all.
    """
    g = golden_scalars(tables, scalar_names, setup)
    p = gas.build(setup)
    checked = 0
    for group in (p.s0, p.st, p.budget, p.reaction):
        for name, value in group.items():
            back = 0 if value == gas.ABSENT else value + 1
            assert back == g[name], f"setup {setup}: {name} port={value} fortran={g[name]}"
            checked += 1
    assert checked == 55 + 77 + 26 + 8


@pytest.mark.parametrize("setup", SETUPS)
@pytest.mark.parametrize("field", REAL_ARRAYS)
def test_real_arrays_are_byte_equal(tables, setup, field):
    """`mm_gas` and `dimen` feed `ukca_cond_coff_v` directly. Byte equality,
    not `allclose`: nothing here is the result of a calculation."""
    np.testing.assert_array_equal(getattr(gas.build(setup), field), tables[f"s{setup}_{field}"])


@pytest.mark.parametrize("setup", SETUPS)
def test_condensable_choice_matches_after_conversion(tables, setup):
    raw = tables[f"s{setup}_condensable_choice"]
    expected = np.where(raw == 0, gas.ABSENT, raw - 1).astype(np.int32)
    np.testing.assert_array_equal(gas.build(setup).condensable_choice, expected)


@pytest.mark.parametrize("setup", SETUPS)
def test_condensable_matches(tables, setup):
    got = gas.build(setup).condensable
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got.astype(np.int32), tables[f"s{setup}_condensable"])


@pytest.mark.parametrize("setup", SETUPS)
def test_arrays_stay_full_width(tables, setup):
    """`dimension(nchemgmax)` = 50 in every setup, including `nochem` where
    `nchemg` is 0. Trimming to `nchemg` would renumber every index."""
    p = gas.build(setup)
    assert int(tables[f"s{setup}_nchemgmax"]) == gas.NCHEMGMAX == 50
    for field in REAL_ARRAYS + INT_ARRAYS:
        assert getattr(p, field).shape == (50,), field
        assert tables[f"s{setup}_{field}"].shape == (50,), field


# ---------------------------------------------------------------------------
# 3. The 1-based to 0-based conversion, and the sentinel
# ---------------------------------------------------------------------------


def test_absent_is_not_a_usable_index():
    assert gas.ABSENT == -1
    assert gas.ABSENT != 0


@pytest.mark.parametrize("setup", (1, 2, 3, 4, 5, 8))
def test_h2so4_is_the_third_advected_gas_everywhere_it_exists(tables, scalar_names, setup):
    """`mh2so4 = 3` in sv1, soto3 and soto6 alike, so 0-based it is 2 — the
    single most load-bearing conversion in the table, and the one a shift
    applied to the wrong side would move to 3 or 4."""
    assert golden_scalars(tables, scalar_names, setup)["mh2so4"] == 3
    assert gas.build(setup).mh2so4 == 2


def test_nochem_carries_no_gases_at_all(tables, scalar_names):
    """Setup 6 is dust only: `nchemg = 0`, every `m*` index 0, and the only two
    non-zero entries in the whole S0 group are `mq3d` and `mpt` — the water
    vapour and potential temperature slots, which are not chemistry."""
    p = gas.build(6)
    assert p.nchemg == 0 and p.ichem == 0 and p.nadvg == 2
    present = {n: v for n, v in p.s0.items() if v != gas.ABSENT}
    assert present == {"mq3d": 0, "mpt": 1}
    assert p.mh2so4 == gas.ABSENT
    assert not p.condensable.any()
    assert p.condensable_species() == ()


def test_condensable_is_derived_before_the_shift():
    """The trap. `condensable = (condensable_choice > 0)` is a test on the
    1-BASED array: H2SO4's component index is 1, so shifting first and then
    testing `> 0` drops sulphate from every setup that has it.

    Asserted against the value, not against the golden, so it stays a
    statement about the semantics rather than about the capture.
    """
    p = gas.build(1)
    (h2so4,) = np.nonzero(p.condensable)[0]
    assert int(h2so4) == p.mh2so4
    # The shifted array holds 0 there -- CP_SU, the first aerosol component.
    assert p.condensable_choice[h2so4] == 0
    # And the naive derivation would have found nothing.
    assert not (p.condensable_choice > 0)[h2so4]


def test_soto3_and_soto6_differ_in_exactly_one_number():
    """Measured, not assumed: across all 174 scalars and all four arrays, the
    two organic routines differ in a SINGLE entry — `condensable_choice` at the
    `Sec_Org` slot, which is aerosol component 3 (CP_OC) under soto3 and 6
    (CP_SO) under soto6. Both are 1-based component indices, so 0-based they
    are 2 and 5.

    Worth pinning exhaustively rather than by inspection: it means setups 4 and
    5 exercise no gas-phase code path that 2, 3 and 8 do not, so anything the
    port gets wrong about `soto6` it gets wrong about `soto3` too — except
    which component the organic mass lands in.
    """
    three, six = gas.build(2), gas.build(4)
    assert three.raw == six.raw, "no gas scalar differs between soto3 and soto6"
    np.testing.assert_array_equal(three.mm_gas, six.mm_gas)
    np.testing.assert_array_equal(three.dimen, six.dimen)
    np.testing.assert_array_equal(three.condensable, six.condensable)

    slot = three.msec_org
    assert slot == six.msec_org
    differing = np.nonzero(three.condensable_choice != six.condensable_choice)[0]
    assert list(differing) == [slot]
    assert three.condensable_choice[slot] == 2
    assert six.condensable_choice[slot] == 5


def test_four_of_the_live_indices_are_absent_in_every_supported_setup():
    """A coverage fact, not a curiosity, and the gas-phase twin of
    `test_mode_sup_insol_is_never_active`.

    `msec_orgi`, `mh2o2`, `mhno3` and `mnh3` are read by the vendored tree —
    `ukca_conden`'s isoprene-SOA block, `ukca_aero_step`'s
    `IF (mh2o2 > 0)` wet-oxidation path, and the fine/coarse nitrate modules —
    and every one of them is 0 in all four gas routines. So those code paths
    have NO reference in any configuration this port can validate, in either
    setting of any fidelity flag, and a port of them cannot be gated.

    Wet oxidation is not dead, though: `mh2o2f`, the semi-prognostic H2O2, is
    present in all six chemistry setups. It is `mh2o2`, the ASAD tracer, that
    is never wired up.
    """
    never = {"msec_orgi", "mh2o2", "mhno3", "mnh3"}
    for setup in SETUPS:
        p = gas.build(setup)
        for name in never:
            assert p.s0[name] == gas.ABSENT, f"setup {setup}: {name}"
    for setup in (1, 2, 3, 4, 5, 8):
        assert gas.build(setup).mh2o2f != gas.ABSENT


def test_so2_is_index_zero_which_is_why_presence_is_not_a_positivity_test():
    """`msotwo = 1` in every chemistry setup, so 0-based it is 0. Any port that
    kept the Fortran's `IF (mxxx > 0)` idiom after converting would drop SO2
    from wet oxidation entirely — silently, and only in the setups that have
    it."""
    for setup in (1, 2, 3, 4, 5, 8):
        p = gas.build(setup)
        assert p.msotwo == 0
        assert p.msotwo != gas.ABSENT
        assert not p.msotwo > 0  # the idiom that must NOT be carried over


def test_counts_are_never_shifted(tables, scalar_names):
    """`nchemg` and friends are counts, not indices. Shifting them is the
    mirror-image mistake to not shifting an index, and it produces an
    off-by-one in an array bound rather than a wrong element."""
    for setup in SETUPS:
        g = golden_scalars(tables, scalar_names, setup)
        p = gas.build(setup)
        assert p.nchemg == g["nchemg"]
        assert p.condensable_species() == tuple(i for i in range(p.nchemg) if p.condensable[i])


def test_an_unsupported_setup_raises():
    with pytest.raises(NotImplementedError, match="i_mode_setup = 12"):
        gas.build(12)
    assert gas.supported_setups() == SETUPS


# ---------------------------------------------------------------------------
# The routine groups, and the two things that make them easy to get wrong
# ---------------------------------------------------------------------------


def test_setups_that_share_a_gas_routine_have_identical_tables():
    for routine, members in ROUTINE_GROUPS.items():
        first = gas.build(members[0])
        assert first.routine == routine
        for setup in members[1:]:
            other = gas.build(setup)
            assert other.routine == routine
            assert other.raw == first.raw, f"{routine}: setups {members[0]} and {setup}"


def test_the_four_gas_routines_are_pairwise_different(tables, scalar_names):
    """The check that catches a capture which silently ran one setup seven
    times. Written on the GOLDEN, not on the port: the port would agree with
    itself whatever the Fortran did."""
    reps = [m[0] for m in ROUTINE_GROUPS.values()]
    seen = {}
    for setup in reps:
        key = tuple(int(v) for v in tables[f"s{setup}_scalars"])
        assert key not in seen, f"setups {seen.get(key)} and {setup} captured the same table"
        seen[key] = setup
    assert len(seen) == 4


def test_no_switch_reaches_the_gas_tables():
    """`build` takes the setup and nothing else. The five switches that reshape
    the mode tables — `l_radaer`, `i_tune_bc`, `l_fix_nacl_density`,
    `l_fix_ukca_hygroscopicities`, `l_dust_mp_ageing` — appear nowhere in
    `ukca_setup_indices.F90`."""
    source = FORTRAN.read_text(encoding="utf-8").lower()
    for switch in (
        "l_radaer",
        "i_tune_bc",
        "l_fix_nacl_density",
        "l_fix_ukca_hygroscopicities",
        "l_dust_mp_ageing",
    ):
        assert switch not in source, switch


# ---------------------------------------------------------------------------
# The Fortran accessor, and the variables that have no value at all
# ---------------------------------------------------------------------------


def test_the_accessor_reads_the_variable_it_names():
    """`CASE ('mox'); out = mnox` compiles, links, and returns a plausible
    integer. Nothing numeric would catch it, on either side: the golden would
    record the wrong variable and the port would be compared against it.

    So the dispatch table is parsed and every label checked against its own
    right-hand side. The file was generated from the extraction for the same
    reason; this is the check that it stays that way.
    """
    text = ACCESSOR.read_text(encoding="utf-8")
    cases = re.findall(r"CASE \('(\w+)'\);\s*out = (\w+)\b", text)
    assert len(cases) >= 176
    mismatched = [(a, b) for a, b in cases if a != b and b != "MERGE"]
    assert mismatched == [], mismatched


def test_the_accessor_covers_exactly_the_extracted_field_set():
    """A name dropped from the SELECT CASE returns ierr = 3, which the capture
    asserts on — but a name dropped from BOTH the accessor and the capture list
    would just quietly stop being validated."""
    text = ACCESSOR.read_text(encoding="utf-8")
    labels = {m for m, _ in re.findall(r"CASE \('(\w+)'\);\s*out = (\w+)\b", text)}
    groups = GAS_LITERALS["groups"]
    expected = set(
        groups["count"]
        + groups["s0"]
        + groups["st"]
        + groups["budget"]
        + groups["reaction"]
        + ["nadvg", "ntrag", "ntraer", "nbudaer"]
        + list(REAL_ARRAYS)
        + list(INT_ARRAYS)
    )
    assert labels == expected


def test_the_accessor_declares_real_kind_8():
    """A bare `REAL` here maps to C float under f2py regardless of
    -fdefault-real-8, which feeds float32 buffers into real(8) dummies.
    `mm_gas` and `dimen` are the only reals that cross this boundary."""
    text = ACCESSOR.read_text(encoding="utf-8")
    assert "REAL(KIND=8),     INTENT(OUT)" in text
    assert not re.search(r"^REAL,", text, re.MULTILINE)


@pytest.mark.parametrize(
    "name", ("budget", "nbudget", "traqu", "ntraqu", "idustdep", "ndustdep", "nbudaertot")
)
def test_the_uninitialised_module_variables_stay_uninitialised(name):
    """`ukca_setup_indices` declares these and no routine `init_indices` calls
    assigns any of them. `budget`, `nbudget`, `traqu` and `ntraqu` are set only
    in `ukca_indices_traqu38` / `ukca_indices_traqu9`, which the box model
    never calls; the other three are set nowhere in the tree at all.

    They are therefore NOT captured and NOT ported — a golden of whatever the
    loader left in memory is worse than no golden. This test fails if the
    vendored tree ever gives one of them a value, which is the point at which
    it should be captured instead.
    """
    source = FORTRAN.read_text(encoding="utf-8")
    assignments = []
    for match in re.finditer(rf"^\s*{name}\s*=[^=]", source, re.MULTILINE):
        line = source.count("\n", 0, match.start()) + 1
        assignments.append(line)

    traqu = [
        (m.start(), m.end())
        for m in re.finditer(
            r"^SUBROUTINE ukca_indices_traqu\d+\b.*?^END SUBROUTINE ukca_indices_traqu\d+\b",
            source,
            re.MULTILINE | re.DOTALL,
        )
    ]
    assert len(traqu) == 2

    def inside_traqu(offset):
        return any(lo <= offset < hi for lo, hi in traqu)

    stray = [
        source.count("\n", 0, m.start()) + 1
        for m in re.finditer(rf"^\s*{name}\s*=[^=]", source, re.MULTILINE)
        if not inside_traqu(m.start())
    ]
    assert stray == [], f"{name} is now assigned at {stray}; capture it"

    # And nothing outside this module assigns it either.
    for path in sorted((REPO / "fortran" / "src").rglob("*.F90")):
        if path.name == FORTRAN.name:
            continue
        assert not re.search(rf"^\s*{name}\s*=[^=]", path.read_text(encoding="utf-8"), re.M), path


def test_init_indices_never_calls_a_traqu_routine():
    """The reason the four variables above have no value. If this changes they
    acquire one, and the test above becomes the one that fails."""
    config = (REPO / "fortran" / "src" / "box" / "glomap_box_config_mod.F90").read_text()
    assert "traqu" not in config.lower()


def test_init_indices_pairs_the_routines_this_module_assumes():
    """`gas_indices.build` maps setup -> gas routine from a table. This is the
    only check that the table still describes `init_indices`."""
    config = (REPO / "fortran" / "src" / "box" / "glomap_box_config_mod.F90").read_text()
    body = re.search(r"SUBROUTINE init_indices.*?END SUBROUTINE init_indices", config, re.DOTALL)
    assert body
    pairs = re.findall(
        r"CASE \(i_(\w+)\)\s*CALL (ukca_indices_\w+)\s*CALL ukca_indices_\w+",
        re.sub(r"\n\s*", "\n", body.group(0)).replace("\n", " ").replace("  ", " "),
    )
    named = {
        "suss_4mode": 1,
        "sussbcoc_5mode": 2,
        "sussbcoc_4mode": 3,
        "sussbcocso_5mode": 4,
        "sussbcocso_4mode": 5,
        "du_2mode": 6,
        "sussbcocdu_7mode": 8,
    }
    assert len(pairs) == 7, pairs
    for case, routine in pairs:
        setup = named[case]
        assert extractor.SETUP_ROUTINE[setup] == routine.lower(), (setup, routine)
