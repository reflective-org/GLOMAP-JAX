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
module beside it. The process routines land in phases D onward; see
PROGRESS.md.
"""
