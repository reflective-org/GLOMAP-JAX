"""One module per UKCA science routine, in the order they must be ported.

The order is forced by the Fortran, not by preference:

    numerics  -> erf/cbrt/nint feed remode, volume_mode and the kernels alike
    drydiam   -> ukca_calc_drydiam, produces drydp
    vapour    -> ukca_volume_mode:287 calls it unconditionally
    water     -> ukca_water_content_v, needed by volume_mode's soluble branch
    volume    -> ukca_volume_mode
    coeffs    -> cond/coag coefficient kernels, need wetdp/wvol/rhopar
    nucleation   -> independent of the above; can proceed in parallel
    condensation -> produces ageterm1 and s_cond_s
    coagulation  -> produces ageterm2
    ageing       -> LAST among processes: consumes ageterm1 AND ageterm2
    remode       -> needs only drydp and erf

Phase C is complete: `modes.py` (the mode tables), `gas_indices.py`,
`budget_indices.py` and `coag_mode.py`, each with a generated `_*_literals`
module beside it.

Phase D is complete: `drydiam.py`, `vapour.py`, `water_content.py` (with
`water_tables.py`) and `volume_mode.py`, every one byte-equal to the compiled
routine.

Phase E is in progress: `cond_coff.py` (`ukca_cond_coff_v`), `coag_coff.py`
(`ukca_coag_coff_v`) and `coag_kernel.py` (`ukca_calc_coag_kernel`) are ported.

`coag_kernel.py` is where the project's byte-equality gate first runs out.
`ukca_coag_coff_v.F90:266` calls `EXP`, `jnp.exp` is XLA's own evaluation while
gfortran's goes to the platform libm, and 42 of the driver's 3,690 non-zero
outputs differ by up to 3 ulp because of it. Substituting `numpy.exp` removes
every one. `icoag = 3`, which reaches no exponential, stays byte-equal. Issue
#28.

`ukca_dcoff_par_av_k` and `ukca_vgrav_av_k` are NOT part of this phase despite
being coefficient kernels of the same shape. Their only callers are
`ukca_ddepaer_mod` and `ukca_ddepaer_incl_sedi_mod`, and the box model passes
`ddepaer = 0` and `sedi = 0` (`glomap_box.F90:144`), so no validated reference
exists for either. Porting them would be a physics commit with nothing to
validate it against; see `docs/unsupported.md`.

See PROGRESS.md.
"""
