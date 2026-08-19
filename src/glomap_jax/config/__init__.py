"""Static configuration: what the model is, and how faithfully it reproduces it.

Split in two because the two answer different questions and change for
different reasons.

``model``    — what to run: mode setup, process switches, timestep structure.
``fidelity`` — where the port deliberately reproduces upstream behaviour that
               is wrong, each flag defaulting to what the Fortran does.

Both are frozen and hashable so they can be passed to ``jax.jit`` as static
arguments, which turns every flag into a compile-time Python branch with no
runtime cost.
"""

from glomap_jax.config.fidelity import FidelityConfig
from glomap_jax.config.model import SUPPORTED_MODE_SETUPS, ModelConfig

__all__ = ["SUPPORTED_MODE_SETUPS", "FidelityConfig", "ModelConfig"]
