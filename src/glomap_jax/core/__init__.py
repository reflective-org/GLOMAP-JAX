"""Machinery every process module needs, and that none of them owns.

``state``    — the traced pytrees (``NamedTuple``) carried through a step.
``numerics`` — the transcendental/rounding compatibility layer. Not a
               convenience wrapper: three primitives have to be written a
               specific way to match gfortran, and writing them the obvious
               way is silently wrong. See ``docs/porting-notes.md``.
"""

from glomap_jax.core import numerics, state

__all__ = ["numerics", "state"]
