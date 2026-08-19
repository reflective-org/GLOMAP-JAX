"""Task 8 acceptance: state pytrees and static config behave as documented."""

import jax
import jax.numpy as jnp
import pytest

from glomap_jax.config import SUPPORTED_MODE_SETUPS, FidelityConfig, ModelConfig
from glomap_jax.core import state as st


def _zeros(nbox=2):
    a = st.AerosolState(
        nd=jnp.zeros((nbox, st.NMODES)),
        md=jnp.zeros((nbox, st.NMODES, st.NCP_MAX)),
        mdt=jnp.zeros((nbox, st.NMODES)),
        mdwat=jnp.zeros((nbox, st.NMODES)),
    )
    return a


def test_static_extents_match_the_fortran_parameters():
    assert (st.NMODES, st.NCP_MAX) == (8, 10)
    assert st.NMODES_SOL == st.NMODES_INS == 4
    # Largest nbudaer across the seven supported setups, and nchemgmax.
    assert st.NBUDAER_MAX == 138
    assert st.NCHEMG_MAX == 50


def test_mode_and_component_indices_are_zero_based_and_ordered():
    assert (st.MODE_NUC_SOL, st.MODE_SUP_INS) == (0, 7)
    assert (st.CP_SU, st.CP_MP) == (0, 9)
    assert len(st.MODE_NAMES) == st.NMODES
    assert len(st.CP_NAMES) == st.NCP_MAX


def test_aerosol_state_is_a_pytree_that_round_trips():
    a = _zeros()
    leaves, treedef = jax.tree_util.tree_flatten(a)
    assert len(leaves) == 4
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, st.AerosolState)
    assert rebuilt.md.shape == (2, st.NMODES, st.NCP_MAX)


def test_state_survives_a_jit_boundary():
    @jax.jit
    def bump(a):
        return a._replace(nd=a.nd + 1.0)

    out = bump(_zeros())
    assert float(out.nd[0, 0]) == 1.0


@pytest.mark.parametrize("cls", [st.AerosolState, st.GasState, st.DerivedSize, st.Environment])
def test_every_field_carries_a_shape_and_unit_comment(cls):
    """Units are load-bearing here and easy to get wrong.

    md is molecules per particle rather than a mass; s0g is a mixing ratio times
    gridbox air mass rather than a concentration; s0g and s0g_dot sit in
    different index spaces. A field whose comment is deleted or wrong is a real
    hazard, so parse the source and require every field to carry one.

    (The previous version of this test ended in `or True` and could not fail.)
    """
    import inspect
    import re

    src = inspect.getsource(cls)
    fields = [line for line in src.splitlines() if re.match(r"\s+\w+: jnp\.ndarray", line)]
    assert fields, f"{cls.__name__} declares no array fields"
    for line in fields:
        name = line.strip().split(":")[0]
        assert "#" in line, f"{cls.__name__}.{name} has no shape/unit comment"
        comment = line.split("#", 1)[1]
        assert "(" in comment and ")" in comment, (
            f"{cls.__name__}.{name} comment has no shape: {comment.strip()!r}"
        )
        assert "[" in comment and "]" in comment, (
            f"{cls.__name__}.{name} comment has no units: {comment.strip()!r}"
        )


@pytest.mark.parametrize("setup", SUPPORTED_MODE_SETUPS)
def test_supported_mode_setups_construct(setup):
    assert ModelConfig(i_mode_setup=setup).i_mode_setup == setup


def test_config_is_hashable_so_it_can_be_a_static_jit_arg():
    hash(ModelConfig())
    hash(FidelityConfig())


@pytest.mark.parametrize("bad", [7, 10, 12, 13, 99])
def test_unsupported_mode_setup_is_rejected(bad):
    with pytest.raises(ValueError, match="not supported"):
        ModelConfig(i_mode_setup=bad)


def test_icoag_4_is_rejected_with_the_upstream_reason():
    # There is no correct reference to validate against, so silently accepting
    # it would produce garbage that looks like a result.
    with pytest.raises(NotImplementedError, match="UP-5"):
        ModelConfig(icoag=4)


def test_i_nuc_method_1_is_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        ModelConfig(i_nuc_method=1)


def test_iextra_checks_above_one_is_rejected():
    with pytest.raises(NotImplementedError, match="mode_check_mdt"):
        FidelityConfig(iextra_checks=2)
