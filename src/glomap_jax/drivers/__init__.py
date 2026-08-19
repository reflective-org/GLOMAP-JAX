"""Timestepping. Two implementations that must agree to ``RTOL_JIT_VS_EAGER``.

``eager`` is the debugger and stays permanently — it can raise a real exception
where a scan can only return an error flag. ``scan`` is the one that goes fast.
The equivalence test between them is not a formality; it is what makes the
eager one trustworthy as a reference for the other.
"""
