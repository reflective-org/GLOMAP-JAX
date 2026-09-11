# Explicit dry aerosol experiments

`init_state(..., dry=True)` and `aero_step(..., dry=True)` use zero aerosol water,
unchanged dry diameter/volume, and dry-composition density in every size refresh.
The calculation bypasses both tropospheric hydration and the stratospheric
solution-density override. Mixed composition uses total dry mass divided by
additive component volume. `_update_size` accepts the same keyword for callers
that refresh diagnostic sizes after stepping.

Pass the same setting to initialization and every step. `dry` is a static Python
configuration when compiling JAX functions (capture it in a closure or mark it
static). It does not mutate a module or require clearing JAX caches. The default
is `False`, retaining the native wet calculation. This experimental dry protocol
is independent of `FidelityConfig`; in particular, `coag_intra_factor3=True`
remains the native default and its false control remains available.

The companion comparison-harness branch `fix/coag-readiness-20260910` exercises
initialization and compiled physical-kernel calls at 220 K and pressures below,
at and above 150 hPa, and compares alternating wet/dry runs with captured wet
baselines. Local `tests/test_dry_configuration.py` checks dry composition/volume,
pressure coverage and isolated compiled configurations. Original Fortran sources
are unchanged by this change.
