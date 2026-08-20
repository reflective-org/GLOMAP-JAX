! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Leaf reference drivers: one f2py entry point per Fortran routine, driven
!   with chosen inputs rather than with whatever a trajectory happens to reach.
!
!   THE PATTERN. A leaf driver is a thin `SUBROUTINE leaf_<name>(n, in..., out)`
!   that does nothing but call the vendored routine over an array of inputs.
!   The grids live in Python (validation/capture_leaf.py), because deciding
!   which inputs matter is a judgement about the physics and belongs where it
!   can be read and changed, not compiled into Fortran. Each driver adds one
!   subroutine here and one sweep there; nothing else in the harness changes.
!
!   Why this is worth the trouble: a trajectory fixture only ever exercises the
!   inputs a trajectory produces. The branch dump already showed how narrow
!   that is -- half of `ukca_solvecoagnucl_v`'s closed forms are unreachable
!   from any shipped namelist, and `ukca_remode` never merges at all. A leaf
!   driver reaches the inputs the physics can reach.
!
!   THIS FILE covers the numerics primitives (task 21, feeding the compat layer
!   at task 34). They come first because they are consumed by remode,
!   volume_mode, the coagulation kernels and binapara alike, and because three
!   of them are known hazards where gfortran and XLA need not agree:
!
!     * ERF feeds `ukca_remode`'s FRAC_N, cut at 0.5 -- i.e. at erf(x) = 0.
!       Note this is NOT what decides whether a mode merges: :234 does that
!       with a bare `dp > dp_thresh1` on drydp. erf sizes the transfer once
!       merging is already happening, and its clamps are continuous at the
!       boundary. Swept densely through zero anyway, because that is where the
!       transfer fraction is decided.
!
!     * cubrt_v is literally `x ** (1.0/3.0)`, NOT a cube root function. The
!       two are not the same computation and need not give the same bits, and
!       the constant 1.0/3.0 itself changes value under -fdefault-real-8.
!       Both forms are exposed so the port can be checked against the one the
!       Fortran actually performs.
!
!     * Fortran NINT rounds half AWAY FROM ZERO; numpy and jnp.round round half
!       to EVEN. `ukca_vapour.F90:226` computes `(NINT(wts/5))*5`, so
!       wts = 42.5, 47.5, ... land exactly on ties.
!
!       wts is NOT clamped to [41, 99], as this said. Only the
!       l_fix_neg_pvol_wat arm has the 99 ceiling (`:184`); the default arm is
!       MAX(41.0, ws*100) with no ceiling (`:188`), and reaches 103.8 at
!       T = 303.65, bh2o = 2e-8. The floor of 41 is common to both.
!       leaf_vapour_round exposes that idiom directly rather than NINT alone,
!       because the idiom is what the port has to reproduce.
!
!   Sizes are explicit leading integers, as everywhere in this binding; see the
!   header of glomap_f2py_mod.F90 for why, and for the signature asymmetry f2py
!   imposes on routines with input arrays.
!
! ---------------------------------------------------------------------------
SUBROUTINE leaf_erf(n, x, y)
! ukca_remode reaches ERF through umErf, so the driver does too -- wrapping the
! intrinsic directly would not prove the wrapper is transparent.
USE ukca_um_legacy_mod, ONLY: umerf
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
INTEGER :: i
DO i = 1, n
  y(i) = umerf(x(i))
END DO
END SUBROUTINE leaf_erf

! ---------------------------------------------------------------------------
SUBROUTINE leaf_cubrt(n, x, y)
! cubrt_v as the Fortran defines it: y = x ** (1.0/3.0).
USE ukca_um_legacy_mod, ONLY: cubrt_v
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
CALL cubrt_v(n, x, y)
END SUBROUTINE leaf_cubrt

! ---------------------------------------------------------------------------
SUBROUTINE leaf_pow(n, x, p, y)
! An arbitrary power, so the port can separate "x**(1/3) disagrees" from
! "the ** operator disagrees".
!
! Note the exponent is a SCALAR: powr_v raises a whole array to one power, it
! does not do elementwise pairs. Worth knowing before porting it -- an
! elementwise version would compile, run, and be a different routine.
USE ukca_um_legacy_mod, ONLY: powr_v
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(IN)  :: p
REAL(KIND=8), INTENT(OUT) :: y(n)
CALL powr_v(n, x, p, y)
END SUBROUTINE leaf_pow

! ---------------------------------------------------------------------------
SUBROUTINE leaf_exp(n, x, y)
USE ukca_um_legacy_mod, ONLY: exp_v
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
CALL exp_v(n, x, y)
END SUBROUTINE leaf_exp

! ---------------------------------------------------------------------------
SUBROUTINE leaf_log(n, x, y)
USE ukca_um_legacy_mod, ONLY: log_v
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
CALL log_v(n, x, y)
END SUBROUTINE leaf_log

! ---------------------------------------------------------------------------
SUBROUTINE leaf_oneover(n, x, y)
USE ukca_um_legacy_mod, ONLY: oneover_v
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
CALL oneover_v(n, x, y)
END SUBROUTINE leaf_oneover

! ---------------------------------------------------------------------------
SUBROUTINE leaf_nint(n, x, y)
! Returned as REAL rather than INTEGER so a tie that rounds the wrong way shows
! up as a value rather than as an overflow or a dtype argument.
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
INTEGER :: i
DO i = 1, n
  y(i) = REAL(NINT(x(i)), KIND=8)
END DO
END SUBROUTINE leaf_nint

! ---------------------------------------------------------------------------
SUBROUTINE leaf_vapour_round(n, x, y)
! ukca_vapour.F90:226 exactly: round = (NINT(wts/5))*5, which then indexes a
! lookup table. The idiom is the thing the port must reproduce, so it is the
! thing the driver exposes.
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: x(n)
REAL(KIND=8), INTENT(OUT) :: y(n)
INTEGER :: i
DO i = 1, n
  y(i) = REAL((NINT(x(i) / 5)) * 5, KIND=8)
END DO
END SUBROUTINE leaf_vapour_round

! ---------------------------------------------------------------------------
! ereport shim accessors (task 20b). The shim itself is
! glomap_ereport_shim.F90, linked in place of src/ukca/ereport_mod.F90 for
! this extension only; see docs/harness.md.
!
! ANY gate-A driver must call wrap_ereport_count after every call and discard
! the result if it is non-zero. The shim lets a caller continue past a fatal
! error so Python can see it, which means whatever the caller computed
! afterwards is meaningless -- and looks like a number.
! ---------------------------------------------------------------------------
SUBROUTINE wrap_ereport_count(fatal, warning, info)
USE ereport_mod, ONLY: ereport_shim_counts
IMPLICIT NONE
INTEGER, INTENT(OUT) :: fatal, warning, info
CALL ereport_shim_counts(fatal, warning, info)
END SUBROUTINE wrap_ereport_count

SUBROUTINE wrap_ereport_last(status, routine, message)
USE ereport_mod, ONLY: ereport_shim_last
IMPLICIT NONE
INTEGER,            INTENT(OUT) :: status
CHARACTER(LEN=256), INTENT(OUT) :: routine, message
CALL ereport_shim_last(status, routine, message)
END SUBROUTINE wrap_ereport_last

SUBROUTINE wrap_ereport_reset()
USE ereport_mod, ONLY: ereport_shim_reset
IMPLICIT NONE
CALL ereport_shim_reset()
END SUBROUTINE wrap_ereport_reset


! ---------------------------------------------------------------------------
! Phase D physics leaves (task 35a).
!
! These four differ from the numerics leaves above in one way that matters:
! the numerics leaves touch only PARAMETERs, so they are meaningful before
! wrap_init. These are not. `avogadro`, `rho_so4`, `rho_water` and `rmol` are
! `REAL, SAVE :: x = rmdi` in ukca_config_constants_mod and are assigned only
! by init_config_constants(), called from init_ukca_for_box
! (glomap_box_config_mod.F90:317). Called cold, these routines return
! plausible-looking numbers built from a missing-data sentinel. Hence the
! init guard on all four:
!
!   ierr = 0  fine
!        = 1  the process is poisoned; a previous init failed or the switches
!             changed, so nothing here can be trusted
!        = 2  a shape argument disagrees with the module's own extents
!        = 4  wrap_init has not run
!
! Logicals cross as INTEGER 0/1 rather than as LOGICAL, following
! glomap_modes_mod's convention: the callee wants LOGICAL(KIND=log_small),
! which is SELECTED_INT_KIND(1) -- one byte -- and f2py's notion of a Fortran
! logical is not something to depend on for a kind that narrow.
! ---------------------------------------------------------------------------

SUBROUTINE leaf_vapour(n, t, pmid, s, rp, wts, rhosol_strat, ierr)
! ukca_vapour is setup-independent: it takes no glomap_variables argument and
! reads no per-setup table, so one process can sweep it whole.
!
! `rp` is in the signature although the chain it feeds (ph2so4, muh2so4,
! kelvin, kelvin_out) reaches neither INTENT(OUT). Sweeping it and asserting
! the outputs do not move is the cheapest confirmation of that analysis, and
! it costs one argument.
USE ukca_vapour_mod,   ONLY: ukca_vapour
USE glomap_f2py_state, ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: t(n), pmid(n), s(n), rp(n)
REAL(KIND=8), INTENT(OUT) :: wts(n), rhosol_strat(n)
INTEGER,      INTENT(OUT) :: ierr

wts          = 0.0
rhosol_strat = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0
CALL ukca_vapour(n, t, pmid, s, rp, wts, rhosol_strat)
END SUBROUTINE leaf_vapour


SUBROUTINE leaf_water_content(n, mask_i, ions_i, cl, rh, wc, ierr)
! Also setup-independent -- ncation and nanion are PARAMETERs, not per-setup.
!
! Two things here are not cosmetic.
!
! `wc` is INTENT(OUT) but the callee writes only wc(idx(:m)) -- the compacted,
! masked rows. Unmasked rows would carry whatever was on the stack into a
! golden, so it is zeroed here before the call rather than trusted afterwards.
!
! `cl` and `ions` are declared (nv,-nanion:ncation) in the callee. f2py cannot
! express a negative lower bound, so they cross as (n,8) and are remapped
! here. The extents are asserted rather than assumed: if ncation and nanion
! ever change, +5 stops being the right offset and this must fail loudly
! instead of silently shifting every ion by one.
USE ukca_water_content_v_mod, ONLY: ukca_water_content_v
USE ukca_mode_setup,          ONLY: ncation, nanion
USE ukca_types_mod,           ONLY: log_small
USE glomap_f2py_state,        ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
INTEGER,      INTENT(IN)  :: mask_i(n)
INTEGER,      INTENT(IN)  :: ions_i(n, 8)
REAL(KIND=8), INTENT(IN)  :: cl(n, 8)
REAL(KIND=8), INTENT(IN)  :: rh(n)
REAL(KIND=8), INTENT(OUT) :: wc(n)
INTEGER,      INTENT(OUT) :: ierr

LOGICAL(KIND=log_small) :: mask(n)
LOGICAL(KIND=log_small) :: ions(n, -4:3)
REAL(KIND=8)            :: cl_l(n, -4:3)
INTEGER                 :: i, j

wc = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (ncation /= 3 .OR. nanion /= 4) THEN
  ierr = 2
  RETURN
END IF
ierr = 0

DO i = 1, n
  mask(i) = (mask_i(i) /= 0)
  DO j = -4, 3
    ions(i, j) = (ions_i(i, j + 5) /= 0)
    cl_l(i, j) = cl(i, j + 5)
  END DO
END DO

CALL ukca_water_content_v(n, mask, cl_l, rh, ions, wc)
END SUBROUTINE leaf_water_content


SUBROUTINE leaf_drydiam(n, nm, ncp_in, nd, md_in, mdt_in,                      &
                        drydp, dvol, md_out, mdt_out, ierr)
! Setup-DEPENDENT: takes glomap_variables_local, so one subprocess per
! i_mode_setup, and the module-level glomap_variables is what gets passed.
!
! md and mdt are INTENT(IN OUT) in the callee and are rewritten by the
! undersize reset at :253-256. They are NOT passed through as in-out here.
! f2py's copy-in/copy-out for INTENT(IN OUT) depends on the incoming array's
! dtype, order and contiguity: with a non-conforming array the mutation is
! silently dropped, and with a conforming one the caller's grid is silently
! overwritten so the next call is driven by the previous call's output. Both
! failures are invisible from Python. Copy into locals, return the results
! separately, and let the caller compare in_ against out.
USE ukca_calc_drydiam_mod,         ONLY: ukca_calc_drydiam
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in
REAL(KIND=8), INTENT(IN)  :: nd(n, nm)
REAL(KIND=8), INTENT(IN)  :: md_in(n, nm, ncp_in)
REAL(KIND=8), INTENT(IN)  :: mdt_in(n, nm)
REAL(KIND=8), INTENT(OUT) :: drydp(n, nm), dvol(n, nm)
REAL(KIND=8), INTENT(OUT) :: md_out(n, nm, ncp_in)
REAL(KIND=8), INTENT(OUT) :: mdt_out(n, nm)
INTEGER,      INTENT(OUT) :: ierr

drydp   = 0.0
dvol    = 0.0
md_out  = md_in
mdt_out = mdt_in
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF
ierr = 0

CALL ukca_calc_drydiam(n, glomap_variables, nd, md_out, mdt_out, drydp, dvol)
END SUBROUTINE leaf_drydiam


SUBROUTINE leaf_volume_mode(n, nm, ncp_in, nd, md, mdt, rh, dvol, drydp,       &
                            t, pmid, s, mdwat, wvol, wetdp, rhopar,            &
                            pvol, pvol_wat, ierr)
! Setup-dependent, same as leaf_drydiam.
!
! dvol and drydp are INPUTS here. Feed them from leaf_drydiam's outputs on the
! same (nd, md) rows: inventing them independently risks a zero that trips the
! five-way guard at :704-708 and voids the whole call.
USE ukca_volume_mode_mod,          ONLY: ukca_volume_mode
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in
REAL(KIND=8), INTENT(IN)  :: nd(n, nm), md(n, nm, ncp_in), mdt(n, nm)
REAL(KIND=8), INTENT(IN)  :: rh(n), dvol(n, nm), drydp(n, nm)
REAL(KIND=8), INTENT(IN)  :: t(n), pmid(n), s(n)
REAL(KIND=8), INTENT(OUT) :: mdwat(n, nm), wvol(n, nm), wetdp(n, nm)
REAL(KIND=8), INTENT(OUT) :: rhopar(n, nm), pvol(n, nm, ncp_in), pvol_wat(n, nm)
INTEGER,      INTENT(OUT) :: ierr

mdwat    = 0.0
wvol     = 0.0
wetdp    = 0.0
rhopar   = 0.0
pvol     = 0.0
pvol_wat = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF
ierr = 0

CALL ukca_volume_mode(glomap_variables, n, nd, md, mdt, rh, dvol, drydp,       &
                      t, pmid, s, mdwat, wvol, wetdp, rhopar, pvol, pvol_wat)
END SUBROUTINE leaf_volume_mode


! ---------------------------------------------------------------------------
! Config setters for the two phase-D fidelity flags.
!
! These write glomap_config AFTER wrap_init has run, which is the only way to
! sweep a flag whose effect is inside a science routine rather than inside the
! mode-table setup. Both are deliberately narrow: they touch one LOGICAL each
! and nothing derived from it.
!
! l_fix_ukca_water_content is a ONE-WAY LATCH in the callee and no setter can
! undo it. ukca_water_content_v.F90:235 patches its own SAVEd, DATA-initialised
! `y` table in place when the flag is on and never restores it, so a process
! that has ever seen .TRUE. keeps the patched coefficient for good. Setting it
! back to .FALSE. here changes the flag and NOT the table. Sweep it with one
! subprocess per setting. Issue #22, and CLAUDE.md's process-global state rule.
! ---------------------------------------------------------------------------

SUBROUTINE wrap_set_fix_water_content(v, ierr)
USE ukca_config_specification_mod, ONLY: glomap_config
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER, INTENT(IN)  :: v
INTEGER, INTENT(OUT) :: ierr
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0
glomap_config%l_fix_ukca_water_content = (v /= 0)
END SUBROUTINE wrap_set_fix_water_content


SUBROUTINE wrap_set_fix_neg_pvol_wat(v, ierr)
USE ukca_config_specification_mod, ONLY: glomap_config
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER, INTENT(IN)  :: v
INTEGER, INTENT(OUT) :: ierr
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0
glomap_config%l_fix_neg_pvol_wat = (v /= 0)
END SUBROUTINE wrap_set_fix_neg_pvol_wat


SUBROUTINE wrap_get_config_flags(fix_water, fix_neg_pvol, o_setup, ierr)
! Read-back, so a capture confirms what the Fortran actually holds rather than
! what the text that was meant to set it says. The mode-table captures learned
! this the hard way: a substitution that silently matched nothing produced a
! golden with identical data for all seven setups, and every byte-equality
! test passed against it.
USE ukca_config_specification_mod, ONLY: glomap_config
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER, INTENT(OUT) :: fix_water, fix_neg_pvol, o_setup, ierr
fix_water    = -1
fix_neg_pvol = -1
o_setup      = -1
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0
fix_water    = MERGE(1, 0, glomap_config%l_fix_ukca_water_content)
fix_neg_pvol = MERGE(1, 0, glomap_config%l_fix_neg_pvol_wat)
o_setup      = glomap_config%i_mode_setup
END SUBROUTINE wrap_get_config_flags
