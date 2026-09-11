"""Explicit dry configuration keeps native wet behavior as the default."""

import jax
import numpy as np
import pytest

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.core.constants import AVOGADRO
from glomap_jax.drivers.box import _update_size, box_env, init_state
from glomap_jax.physics import gas_indices, modes


@pytest.mark.parametrize("pressure", [5000.0, 14999.0, 15000.0, 15001.0, 20000.0])
def test_dry_volume_density_and_default_wet_are_scoped(pressure):
    tables = modes.build(1)
    gas = gas_indices.build(1)
    env = box_env(
        1,
        t=220.0,
        pmid=pressure,
        rh=0.1,
        spec_humid=-1.0,
        height=20000.0,
        pbl_height=1000.0,
        box_volume=1.0,
    )
    kwargs = dict(
        nbox=1,
        nadvg=int(gas.nadvg),
        nchemg=int(gas.nchemg),
        nd_init=[0.0, 1e4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dp_init=[0.0, 25e-9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mfrac_init=[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 8,
    )
    wet = init_state(tables, gas, env, **kwargs)
    dry = init_state(tables, gas, env, dry=True, **kwargs)
    wet_after = init_state(tables, gas, env, **kwargs)
    for a, b in zip(wet, wet_after):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(dry.mdwat, 0.0)
    np.testing.assert_array_equal(dry.pvol_wat, 0.0)
    np.testing.assert_array_equal(dry.wetdp, dry.drydp)
    np.testing.assert_array_equal(dry.wvol, dry.dvol)
    member = np.asarray(tables.component)[None, :, :]
    mass = np.where(member, np.asarray(dry.md) * np.asarray(tables.mm) / AVOGADRO, 0.0)
    active = np.asarray(tables.mode)
    np.testing.assert_allclose(
        np.asarray(dry.rhopar)[:, active],
        mass.sum(axis=-1)[:, active] / np.asarray(dry.dvol)[:, active],
        rtol=1e-13,
    )
    np.testing.assert_allclose(dry.pvol, mass / np.asarray(tables.rhocomp), rtol=1e-13)
    assert float(wet.mdwat[0, 1]) > 0

    # Closures give wet and dry distinct compiled configurations without caches
    # or mutable patches. Direct volume/size calls remain jit-compatible.
    def compiled(dry_flag):
        return jax.jit(
            lambda st: _update_size(
                tables, st.nd, st.md, st.mdt, env, FidelityConfig(), dry=dry_flag
            )
        )

    wet_fn, dry_fn = compiled(False), compiled(True)
    for fn, expected in ((wet_fn, wet), (dry_fn, dry), (wet_fn, wet), (dry_fn, dry)):
        out = fn(dry)
        np.testing.assert_allclose(out[4], expected.wetdp, rtol=1e-13)
