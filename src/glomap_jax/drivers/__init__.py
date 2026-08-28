"""Timestepping. Two implementations that must agree to ``RTOL_JIT_VS_EAGER``.

``eager`` is the debugger and stays permanently — it can raise a real exception
where a scan can only return an error flag. ``scan`` is the one that goes fast.
The equivalence test between them is not a formality; it is what makes the
eager one trustworthy as a reference for the other.

`aero_step.py` is the eager one, and it is the first thing in this project that
checks the ported routines against each other rather than one at a time. At the
shipped substep count it is byte-equal to `ukca_aero_step` on every field of
every setup; at three times the substeps, 19 elements differ by at most 4.6e-10
relative, and substituting libm's `exp` in the eight modules that call it
removes every one. `scan` does not exist yet.

`box.py` is the model around it -- environment, initial state, and the
chemistry-step loop -- and with it **gate C is reached**: the ported model
reproduces the committed 48-step trajectory goldens to 5.4e-13 relative on
every aerosol diagnostic, and bit-identically once libm's `exp` is
substituted. The two gas columns are bounded by the goldens' own seven-digit
resolution rather than by the port; issue #32.
"""
