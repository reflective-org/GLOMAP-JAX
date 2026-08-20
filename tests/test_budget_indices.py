"""The 283 `nmas*` budget slot indices, for all seven setups (task 32).

Three sources have to agree, and the point of this file is that they are
genuinely independent:

1. the **declarations and assignments** in `fortran/src/ukca/ukca_setup_indices.F90`,
   re-parsed here from source text;
2. the **generated literals** the port imports,
   `src/glomap_jax/physics/_budget_index_literals.py`;
3. the **compiled Fortran**, read back through the gate-A binding into
   `tests/goldens/budidx.f64.tables.npz`.

Byte equality, not a tolerance — these are integers. `assert_array_equal`
throughout.

No `fortran` marker anywhere: everything here reads committed text or a
committed archive, so it runs in CI where there is no gfortran.

Two mutations every test in here was checked against, because a test that
cannot fail is worse than none:

* **the capture returns the same data for all seven setups** (a no-op namelist
  substitution — this repo has shipped that exact bug once). Caught by
  `test_the_seven_setups_are_genuinely_different` and, independently, by the
  nbudaer and bijection tests.
* **a name is dropped from the middle of the table**, shifting every slot after
  it. Caught by the declaration-order comparisons in
  `test_the_literal_names_are_the_vendored_declarations` and
  `test_the_f90_accessor_lists_exactly_the_declared_names`.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GOLDENS = REPO / "tests" / "goldens"
UKCA = REPO / "fortran" / "src" / "ukca"
ACCESSOR = REPO / "validation" / "f2py" / "glomap_budidx_mod.F90"
ARCHIVE = GOLDENS / "budidx.f64.tables.npz"

sys.path.insert(0, str(REPO / "validation"))
import capture_budget_indices as capture  # noqa: E402

from glomap_jax.physics import budget_indices as bi  # noqa: E402
from glomap_jax.physics._budget_index_literals import (  # noqa: E402
    BUDGET_NAMES,
    SETUP_NBUDAER,
    SETUP_SLOTS,
)

SETUPS = (1, 2, 3, 4, 5, 6, 8)

# The acceptance criterion, written out. Measured from the compiled Fortran and
# asserted against the golden below, not copied from a plan: seven setups, seven
# distinct widths.
EXPECTED_NBUDAER = {1: 46, 2: 107, 3: 89, 4: 123, 5: 104, 6: 8, 8: 138}

# Which case in tests/goldens/ runs which setup. From the shipped namelists'
# `i_mode_setup`; the budget goldens' column count is what ties the two.
CASE_SETUP = {
    "bl_nmts3": 1,
    "boundary_layer": 1,
    "free_troposphere": 1,
    "marine_bcoc": 2,
}


@pytest.fixture(scope="module")
def golden():
    assert ARCHIVE.is_file(), f"{ARCHIVE.name} missing -- run `make goldens`"
    with np.load(ARCHIVE, allow_pickle=False) as data:
        yield {k: data[k] for k in data.files}


@pytest.fixture(scope="module")
def parsed():
    """The map re-parsed from the vendored source, with no toolchain."""
    return capture.extract()


# ---------------------------------------------------------------------------
# The name table
# ---------------------------------------------------------------------------
def test_the_literal_names_are_the_vendored_declarations():
    """In declaration order, which is what every value array is aligned to.
    Sorting here would hide exactly the failure that matters: a name dropped
    from the middle, which shifts the map by one from that point on."""
    assert list(BUDGET_NAMES) == capture.declared_names()
    assert len(BUDGET_NAMES) == 283
    assert len(set(BUDGET_NAMES)) == len(BUDGET_NAMES)


def test_the_f90_accessor_lists_exactly_the_declared_names():
    """`glomap_budidx_mod.F90` carries the name table twice over — once as the
    string blob and once as the USE list feeding the value array. Both are
    generated; if a vendored-tree update added a `nmas*` scalar, this is what
    says the accessor is now short one, rather than the capture quietly
    returning a 283-long prefix of a longer list."""
    text = ACCESSOR.read_text()
    blob = [m.lower() for m in re.findall(r"'(nmas\w+)\s*'", text)]
    used = re.search(r"USE ukca_setup_indices, ONLY:(.*?)\nIMPLICIT NONE", text, re.S)
    assert used, "the USE list feeding wrap_bud_values was not found"
    use_names = [m.lower() for m in re.findall(r"\b(nmas\w+)\b", used.group(1))]

    declared = capture.declared_names()
    assert blob == declared
    assert use_names == declared


# ---------------------------------------------------------------------------
# Source text vs generated literals
# ---------------------------------------------------------------------------
def test_the_committed_literals_are_not_stale(parsed):
    names = capture.declared_names()
    for setup in SETUPS:
        assert SETUP_NBUDAER[setup] == parsed[setup]["nbudaer"]
        assert list(SETUP_SLOTS[setup]) == [parsed[setup]["slots"][n] for n in names]


def test_the_check_mode_agrees(capsys):
    """What a future CI job would run. Compares the DATA, not the bytes: `ruff
    format` reformats the generated file after it is written."""
    assert capture.main(["--check-literals"]) == 0
    assert "up to date" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Generated literals vs the compiled Fortran
# ---------------------------------------------------------------------------
def test_the_golden_carries_every_setup(golden):
    assert list(golden["_setups"]) == list(SETUPS)
    assert str(golden["_case"]) == "budidx"
    np.testing.assert_array_equal(golden["names"], np.array(BUDGET_NAMES))


@pytest.mark.parametrize("setup", SETUPS)
def test_the_literals_match_the_compiled_fortran_byte_for_byte(golden, setup):
    """The acceptance criterion: 283 names x 7 setups, integers, exact.

    The two sides reached the same numbers by genuinely different routes — a
    regex over the source text, and `init_ukca_for_box` running in a
    subprocess and the scalars read back out of the module — so agreement is
    evidence and not a tautology.
    """
    np.testing.assert_array_equal(
        np.array(SETUP_SLOTS[setup], dtype=np.int32), golden[f"s{setup}_values"]
    )
    assert int(golden[f"s{setup}_nbudaer"]) == SETUP_NBUDAER[setup]


@pytest.mark.parametrize("setup", SETUPS)
def test_nbudaer_is_the_measured_value(golden, setup):
    assert int(golden[f"s{setup}_nbudaer"]) == EXPECTED_NBUDAER[setup]


def test_there_are_seven_distinct_nbudaer(golden):
    """8, 46, 89, 104, 107, 123, 138. Stated as distinctness rather than as a
    list so it also fails if two setups collapse onto one value."""
    measured = {int(golden[f"s{s}_nbudaer"]) for s in SETUPS}
    assert measured == {8, 46, 89, 104, 107, 123, 138}
    assert len(measured) == len(SETUPS)


def test_the_seven_setups_are_genuinely_different(golden):
    """The failure this whole capture is exposed to: seven subprocesses, one
    namelist, one string substitution. If the substitution no-ops, every
    byte-equality test above still passes — against seven copies of setup 1.

    So: seven distinct value vectors, and no two setups agreeing on every name.
    """
    vectors = {s: tuple(int(v) for v in golden[f"s{s}_values"]) for s in SETUPS}
    assert len(set(vectors.values())) == len(SETUPS)
    for a in SETUPS:
        for b in SETUPS:
            if a < b:
                assert vectors[a] != vectors[b], f"setups {a} and {b} have the same map"


def test_setup_6_is_the_sparse_one_and_setup_8_the_full_one(golden):
    """A shape check with content in it: dust-only carries 8 fluxes and the
    7-mode setup carries 138, so a capture that returned a plausible-looking
    but wrong setup for either would show up here rather than in a later
    physics test."""
    assert int((golden["s6_values"] > 0).sum()) == 8
    assert int((golden["s8_values"] > 0).sum()) == 138
    carried_6 = {n for n, v in zip(BUDGET_NAMES, golden["s6_values"], strict=True) if v > 0}
    assert all(n.startswith("nmas") and "du" in n for n in carried_6), sorted(carried_6)


# ---------------------------------------------------------------------------
# Structure of the map
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("setup", SETUPS)
def test_the_carried_slots_are_exactly_one_to_nbudaer(golden, setup):
    """A bijection onto 1..nbudaer: no duplicate, no gap, no overrun. It is the
    strongest statement available about the map without naming numbers, and it
    fails on the three ways a hand-maintained index table goes wrong."""
    values = [int(v) for v in golden[f"s{setup}_values"]]
    nbudaer = int(golden[f"s{setup}_nbudaer"])
    assert sorted(v for v in values if v > 0) == list(range(1, nbudaer + 1))
    assert min(values) == 0


@pytest.mark.parametrize("setup", SETUPS)
def test_no_name_maps_to_slot_zero(golden, setup):
    """Slot 0 is a hole. A name mapping *to* it would be a name whose flux is
    accumulated into the column the Fortran never writes — the exact confusion
    between "not carried" and "carried at index 0" that this map exists to keep
    apart."""
    m = bi.build(setup)
    for name in m.carried_names():
        assert m.slot(name) >= 1


def test_the_mp_names_are_carried_by_no_supported_setup(golden, parsed):
    """A finding, pinned so it cannot be "tidied" away.

    The 38 `nmas*mp*` names are assigned only by
    `ukca_indices_sussbcocdump_8mode`, which no supported setup dispatches to.
    In all seven they are therefore read without ever having been assigned —
    and 34 of them are read from a live `IF (nmasxxx > 0)` guard in
    `ukca_ageing`, `ukca_coagwithnucl`, `ukca_ddepaer*`, `ukca_impc_scav`,
    `ukca_rainout`, `ukca_remode` and `ukca_cloudproc`. Module scalars have
    static storage, so gfortran gives them a .bss zero and the guard is false;
    the standard does not promise it. The golden is what says this build really
    does return 0, for every one of them, in every setup.
    """
    names = capture.declared_names()
    unassigned = set(names) - set(parsed[1]["assigned"])
    assert len(unassigned) == 38
    assert all("mp" in n for n in unassigned)
    for setup in SETUPS:
        assert set(names) - set(parsed[setup]["assigned"]) == unassigned
        values = dict(zip(names, (int(v) for v in golden[f"s{setup}_values"]), strict=True))
        assert all(values[n] == 0 for n in unassigned)

    written, _ = _write_sites()
    assert len(unassigned & set(written)) == 34


# ---------------------------------------------------------------------------
# The Fortran's own write sites
# ---------------------------------------------------------------------------
def _write_sites():
    """Every `bud_aer_mas(..., nmasxxx) = ...` in the vendored tree, and
    whether an `nmasxxx > 0` guard is in scope above it.

    Comments are dropped first: `ukca_ddepaer_mod.F90` carries commented-out
    `nmasddepntnucsol` blocks, and counting those would report a write to a
    name the live code never touches.
    """
    written: dict[str, int] = {}
    unguarded = []
    for path in sorted(UKCA.glob("*.F90")):
        code = ["" if ln.lstrip().startswith("!") else ln for ln in path.read_text().split("\n")]
        for i, line in enumerate(code):
            m = re.search(r"bud_aer_mas\(([^)]*)\)\s*=", line)
            if not m:
                continue
            name = m.group(1).split(",")[-1].strip().lower()
            written[name] = written.get(name, 0) + 1
            window = code[max(0, i - 25) : i + 1]
            if not any(re.search(re.escape(name) + r"\s*>\s*0", ln, re.I) for ln in window):
                unguarded.append(f"{path.name}:{i + 1}")
    return written, unguarded


def test_every_write_site_is_guarded_on_its_own_index():
    """The source-level half of "slot 0 is never written".

    `tests/test_goldens.py` shows the column stays zero in four runs; this
    shows *why*, and it covers the sites those four runs never reach. An
    unguarded write would be a write to slot 0 in any setup that does not carry
    that flux — and it would be invisible in the goldens until someone ran the
    setup that reaches it.
    """
    written, unguarded = _write_sites()
    assert unguarded == []
    assert sum(written.values()) == 344, "write-site count moved; re-read the tree"
    assert len(written) == 258


def test_four_write_sites_overwrite_rather_than_accumulate():
    """A porting trap, machine-checked rather than described.

    340 of the 344 sites are `bud = bud + delta`. Four are not: the
    cloud-processing fluxes in `ukca_aero_step.F90` are assigned outright, so
    each step's value replaces the last. `apply_deltas` accumulates, so routing
    those four through it would turn a per-step flux into a running total —
    a wrong diagnostic that stays finite, stays positive and stays plausible.
    """
    setonly = []
    for path in sorted(UKCA.glob("*.F90")):
        lines = ["" if ln.lstrip().startswith("!") else ln for ln in path.read_text().split("\n")]
        for i, line in enumerate(lines):
            m = re.search(r"bud_aer_mas\(([^)]*)\)\s*=", line)
            if not m:
                continue
            name = m.group(1).split(",")[-1].strip().lower()
            statement = "\n".join(lines[i : i + 4]).lower()
            rhs = statement[statement.index("=", statement.index("bud_aer_mas")) + 1 :]
            if name not in rhs:
                setonly.append((path.name, i + 1, name))
    assert setonly == [
        ("ukca_aero_step.F90", 750, "nmasclprsuaccsol1"),
        ("ukca_aero_step.F90", 757, "nmasclprsuaccsol2"),
        ("ukca_aero_step.F90", 771, "nmasclprsucorsol1"),
        ("ukca_aero_step.F90", 778, "nmasclprsucorsol2"),
    ]


def test_every_written_index_is_in_the_map():
    """A name the code writes but the map does not carry would be a budget
    flux with nowhere to go."""
    written, _ = _write_sites()
    assert set(written) <= set(BUDGET_NAMES)


# ---------------------------------------------------------------------------
# The port's own object
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("setup", SETUPS)
def test_build_reproduces_the_golden(golden, setup):
    m = bi.build(setup)
    np.testing.assert_array_equal(m.slots, golden[f"s{setup}_values"])
    np.testing.assert_array_equal(m.carried, golden[f"s{setup}_values"] > 0)
    assert m.names == tuple(BUDGET_NAMES)
    assert m.nbudaer == int(golden[f"s{setup}_nbudaer"])
    assert m.width == m.nbudaer + 1
    assert bi.build(setup, padded=True).width == bi.PADDED_WIDTH == 139


def test_unsupported_setups_raise():
    for setup in (0, 7, 10, 12, 13, 99):
        with pytest.raises(NotImplementedError):
            bi.build(setup)
    assert bi.supported_setups() == SETUPS


def test_an_unknown_name_raises_and_an_uncarried_one_returns_zero():
    """The two must not be the same answer. A typo returning 0 reads as "this
    setup does not carry it", which is how a whole process's budget goes
    missing with every test still green."""
    m = bi.build(1)
    assert m.slot("nmasprimsuaitsol") == 1
    assert m.slot("nmasddepmpaccsol") == bi.NOT_CARRIED == 0
    assert not m.is_carried("nmasddepmpaccsol")
    with pytest.raises(KeyError):
        m.slot("nmasddepmpaccsoll")
    with pytest.raises(KeyError):
        m.is_carried("")


def test_name_of_is_the_inverse_of_slot():
    for setup in SETUPS:
        m = bi.build(setup)
        for k, name in enumerate(m.carried_names(), start=1):
            assert m.slot(name) == k
            assert m.name_of(k) == name
        with pytest.raises(ValueError):
            m.name_of(0)
        with pytest.raises(ValueError):
            m.name_of(m.nbudaer + 1)


# ---------------------------------------------------------------------------
# The write pattern the map implies
# ---------------------------------------------------------------------------
def test_apply_deltas_matches_a_guarded_reference_loop():
    """`apply_deltas` against a transcription of the Fortran's own loop:
    `IF (nmasxxx > 0) bud_aer_mas(:, nmasxxx) = bud_aer_mas(:, nmasxxx) + d`.
    Exact equality — these are sums of the same doubles in the same order per
    slot, and duplicate slots do not occur within one call."""
    import jax.numpy as jnp

    m = bi.build(8)
    rng = np.random.default_rng(32)
    deltas = rng.random((4, len(m.names)))

    reference = np.zeros((4, m.width))
    for j, slot in enumerate(m.slots):
        if slot > 0:
            reference[:, slot] += deltas[:, j]

    out = bi.apply_deltas(jnp.zeros((4, m.width)), m.slots, jnp.asarray(deltas))
    np.testing.assert_array_equal(np.asarray(out), reference)


def test_apply_deltas_leaves_the_hole_exactly_zero_even_with_garbage():
    """The uncarried sites still scatter — into slot 0 — so the mask is what
    keeps the hole a hole. Feeding them `inf` and `nan` is the mutation test:
    `mask * term` gives `0.0 * inf = NaN` and fails, `jnp.where` gives an exact
    zero and passes. Under `jit` and under `vmap`, because the mask has to
    survive both."""
    import jax
    import jax.numpy as jnp

    m = bi.build(1)
    deltas = np.where(m.carried[None, :], 1.0, np.inf)
    deltas[:, ~m.carried][:, ::2] = np.nan
    deltas = jnp.asarray(np.repeat(deltas, 4, axis=0))
    bud = jnp.zeros((4, m.width))

    for fn in (
        bi.apply_deltas,
        jax.jit(bi.apply_deltas),
    ):
        out = np.asarray(fn(bud, m.slots, deltas))
        assert out[:, 0].tolist() == [0.0] * 4
        assert np.isfinite(out).all()
        assert (out[:, 1:] == 1.0).all()

    def per_box(b, d):
        return bi.apply_deltas(b[None, :], m.slots, d[None, :])[0]

    out = np.asarray(jax.jit(jax.vmap(per_box))(bud, deltas))
    assert out[:, 0].tolist() == [0.0] * 4
    assert np.isfinite(out).all()


def test_a_negative_sentinel_would_silently_corrupt_the_last_slot():
    """Why `NOT_CARRIED` is 0 and not -1, pinned as a fact about JAX rather
    than an argument. Python's -1 wraps; the Fortran's 0-based lower bound
    means no rebasing is needed anyway, so the sentinel can stay inside the
    array pointed at the hole."""
    import jax.numpy as jnp

    for mode in ("drop", "clip", "promise_in_bounds"):
        wrapped = np.asarray(jnp.zeros(5).at[-1].add(1.0, mode=mode))
        assert wrapped.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0], f"-1 no longer wraps ({mode})"
    assert np.asarray(jnp.zeros(5).at[-1].add(1.0)).tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]

    # An out-of-RANGE index is the benign one: dropped by default, and only
    # clamped into a real slot if someone asks for mode="clip".
    assert np.asarray(jnp.zeros(5).at[9].add(1.0, mode="drop")).tolist() == [0.0] * 5
    assert np.asarray(jnp.zeros(5).at[9].add(1.0)).tolist() == [0.0] * 5
    assert np.asarray(jnp.zeros(5).at[9].add(1.0, mode="clip")).tolist() == [0.0] * 4 + [1.0]


# ---------------------------------------------------------------------------
# Ties to the trajectory goldens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case,setup", sorted(CASE_SETUP.items()))
def test_the_budget_goldens_are_nbudaer_plus_one_columns_wide(case, setup):
    """The committed budget dumps are `step, time_s, bud0 .. bud<nbudaer>`.
    Their width is an independent measurement of `nbudaer` — captured by the
    box binary writing a CSV, not by the f2py accessor — so it is worth
    checking against the map rather than trusting one route twice."""
    path = GOLDENS / f"{case}.f64.budgets.npz"
    with np.load(path, allow_pickle=False) as data:
        columns = list(data["columns"])
        values = data["values"]
    assert columns[:2] == ["step", "time_s"]
    assert columns[2:] == [f"bud{k}" for k in range(EXPECTED_NBUDAER[setup] + 1)]
    assert len(columns) - 2 == bi.build(setup).width

    # And the empirical half of the slot-0 claim, from a second angle to
    # test_goldens.py's: not just that bud0 is zero, but that every column that
    # is ever nonzero belongs to a name this setup carries.
    nonzero = {k for k in range(EXPECTED_NBUDAER[setup] + 1) if np.abs(values[:, 2 + k]).max() > 0}
    assert 0 not in nonzero
    m = bi.build(setup)
    for k in nonzero:
        assert m.name_of(k)
