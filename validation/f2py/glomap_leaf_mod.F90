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
!   THIS FILE opened with the numerics primitives (task 21, feeding the compat
!   layer at task 34) and has grown one driver per ported routine since:
!   vapour, water_content, drydiam and volume_mode in phase D, cond_coff and
!   coag_coff and binapara in phase E-F. The primitives came first because they
!   are consumed by remode, volume_mode, the coagulation kernels alike, and
!   because three of them are known hazards where gfortran and XLA need not
!   agree:
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
USE glomap_f2py_state,        ONLY: is_initialised, must_restart, water_flag_set
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
! Init OR an explicit flag. This routine reads no SAVEd constant that init
! populates, so requiring init would make the unfixed arm unreachable -- see
! wrap_set_fix_water_content. What it does need is a DEFINED flag: nothing
! initialises glomap_config%l_fix_ukca_water_content before init, and reading
! it undefined would silently select an arm at random.
IF (.NOT. (is_initialised .OR. water_flag_set)) THEN
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


SUBROUTINE leaf_cond_coff(n, mask_i, rp, tsqrt, airdm3, rhoa, pmid, t,         &
                          mmcg, se, dmol, difvol, ifuchs, idcmfp,              &
                          cc, sinkarr, ierr)
! ukca_cond_coff_v reads no per-setup table and no glomap_config -- only pi,
! rmol, boltzmann, avogadro and rgas -- so one process can sweep it whole.
! That is asserted rather than argued: the capture runs the same grid under two
! mode setups and requires the two results to be byte-equal, which is the
! treatment coag_mode got in phase C and for the same reason. "It takes no
! glomap_variables argument" is an argument about the signature, not a
! measurement of the answer.
!
! `tsqrt` and `t` are separate arguments in the callee and are NOT required to
! be consistent. idcmfp=1 reads only tsqrt; idcmfp=2 reads only t. The capture
! sweeps them decoupled in one block precisely to show which output moves with
! which, the same trick leaf_vapour plays with `rp`.
!
! REFUSING THE OUT-OF-RANGE SWITCHES IS THE POINT, not a nicety. Neither
! switch is validated anywhere upstream: not in ukca_cond_coff_v, not in
! ukca_conden, not in ukca_aero_step, and not in glomap_box_config_mod, which
! reads both from the namelist (`:155`). Out of range:
!
!   idcmfp /= 1,2  leaves `dcoff_cp` NEVER ASSIGNED, and both Fuchs branches
!                  read it. The output is whatever was on the stack -- stable
!                  enough within a process to be captured as a golden and to
!                  pass every byte-equality test written against it. That is
!                  exactly the failure mode of issue #19, and the reason task
!                  31's `budget` table was not captured.
!
!   ifuchs /= 1,2  leaves cc and sinkarr at the 0.0 the routine opens with, so
!                  condensation is silently switched off for every mode and
!                  every condensable, with no ereport and no diagnostic.
!
! Both are findings, and neither is reproduced. `ModelConfig.validate` already
! rejects both, so the port has no arm here to be faithful to.
USE ukca_cond_coff_v_mod, ONLY: ukca_cond_coff_v
USE ukca_types_mod,       ONLY: log_small
USE glomap_f2py_state,    ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, ifuchs, idcmfp
INTEGER,      INTENT(IN)  :: mask_i(n)
REAL(KIND=8), INTENT(IN)  :: rp(n), tsqrt(n), airdm3(n), rhoa(n), pmid(n), t(n)
REAL(KIND=8), INTENT(IN)  :: mmcg, se, dmol, difvol
REAL(KIND=8), INTENT(OUT) :: cc(n), sinkarr(n)
INTEGER,      INTENT(OUT) :: ierr

LOGICAL(KIND=log_small) :: mask(n)
INTEGER                 :: i

cc      = 0.0
sinkarr = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF ((ifuchs /= 1 .AND. ifuchs /= 2) .OR.                                       &
    (idcmfp /= 1 .AND. idcmfp /= 2)) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

DO i = 1, n
  mask(i) = (mask_i(i) /= 0)
END DO

CALL ukca_cond_coff_v(n, mask, rp, tsqrt, airdm3, rhoa, mmcg, se, dmol,        &
                      ifuchs, cc, sinkarr, pmid, t, difvol, idcmfp)
END SUBROUTINE leaf_cond_coff


SUBROUTINE leaf_coag_coff(n, mask_i, ri, rj, vi, vj, rhoi, rhoj, mfpa, dvisc,  &
                          t, coag_on, icoag, kij, ierr)
! ukca_coag_coff_v, like ukca_cond_coff_v, reads no per-setup table and no
! glomap_config -- only pi and boltzmann. Measured the same way: the grid runs
! under two mode setups and the results must be byte-equal.
!
! ICOAG = 4 IS REFUSED, and that refusal is UP-5's disposition made
! executable. `:339-340` reads mfppi and mfppj, which are assigned only inside
! the `IF (icoag == 1)` block at `:270`/`:281`. The four IFs are sequential and
! not exclusive, so icoag = 4 means block 1 did not run and both arrays are
! read never having been assigned. There is no correct reference to capture:
! the answer is whatever was on the stack, and it is repeatable enough within
! one process to pass every byte-equality test written against it. Same class
! as issue #19, and the reason `docs/unsupported.md` lists icoag = 4 as raising
! rather than producing plausible garbage.
!
! icoag outside {1,2,3,4} is refused too, for the reason ifuchs is: the four
! blocks are the only writers of kij after `:238` zeroes it, so an unknown
! value returns kij = 0 everywhere -- coagulation silently switched off, with
! no ereport. `ModelConfig.validate` already rejects both cases.
!
! coag_on IS NOT REFUSED. `:239-242` returns early with kij = 0 when it is
! zero, which is a real branch of a supported configuration (the box model has
! a `coag_on` namelist switch) and the only path on which kij is zero for
! *unmasked* rows. It is swept.
USE ukca_coag_coff_v_mod, ONLY: ukca_coag_coff_v
USE ukca_types_mod,       ONLY: log_small
USE glomap_f2py_state,    ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, coag_on, icoag
INTEGER,      INTENT(IN)  :: mask_i(n)
REAL(KIND=8), INTENT(IN)  :: ri(n), rj(n), vi(n), vj(n), rhoi(n), rhoj(n)
REAL(KIND=8), INTENT(IN)  :: mfpa(n), dvisc(n), t(n)
REAL(KIND=8), INTENT(OUT) :: kij(n)
INTEGER,      INTENT(OUT) :: ierr

LOGICAL(KIND=log_small) :: mask(n)
INTEGER                 :: i

kij = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (icoag /= 1 .AND. icoag /= 2 .AND. icoag /= 3) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

DO i = 1, n
  mask(i) = (mask_i(i) /= 0)
END DO

CALL ukca_coag_coff_v(n, mask, ri, rj, vi, vj, rhoi, rhoj, mfpa, dvisc, t,     &
                      kij, coag_on, icoag)
END SUBROUTINE leaf_coag_coff


SUBROUTINE leaf_calc_coag_kernel(n, nm, drydp, dvol, wetdp, wvol, rhopar,      &
                                 mfpa, dvisc, t, coag_on, icoag,               &
                                 kii_arr, kij_arr, ierr)
! Setup-DEPENDENT, unlike the two coefficient leaves. ukca_calc_coag_kernel
! reads glomap_variables%mode and %modesol from the module rather than from a
! dummy argument, so which mode pairs it visits is decided by i_mode_setup and
! the capture runs one subprocess per setup.
!
! What this driver is for is the SLOT MAP, not the numbers. The numbers come
! from ukca_coag_coff_v, which task 48 already pinned. What only this routine
! can settle is which (imode, jmode) entries of kij_arr get written and which
! are left at the 0.0 of `:239-247` -- and that cannot be checked from the
! values, because the kernel is byte-symmetric under an (i,j) swap. A
! transposed transcription writes the right number into the wrong slot, so the
! zero/non-zero pattern is the only discriminator there is.
!
! icoag is guarded here as in leaf_coag_coff: 4 reads never-assigned memory
! (UP-5) and anything outside 1-3 returns zeros with no diagnostic.
USE ukca_calc_coag_kernel_mod,     ONLY: ukca_calc_coag_kernel
USE ukca_mode_setup,               ONLY: nmodes
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, coag_on, icoag
REAL(KIND=8), INTENT(IN)  :: drydp(n, nm), dvol(n, nm), wetdp(n, nm)
REAL(KIND=8), INTENT(IN)  :: wvol(n, nm), rhopar(n, nm)
REAL(KIND=8), INTENT(IN)  :: mfpa(n), dvisc(n), t(n)
REAL(KIND=8), INTENT(OUT) :: kii_arr(n, nm), kij_arr(n, nm, nm)
INTEGER,      INTENT(OUT) :: ierr

kii_arr = 0.0
kij_arr = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes) THEN
  ierr = 2
  RETURN
END IF
IF (icoag /= 1 .AND. icoag /= 2 .AND. icoag /= 3) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

CALL ukca_calc_coag_kernel(n, kii_arr, kij_arr, drydp, dvol, wetdp, wvol,      &
                           rhopar, mfpa, dvisc, t, coag_on, icoag)
END SUBROUTINE leaf_calc_coag_kernel


SUBROUTINE leaf_binapara(n, t, rh, h2so4, jveh, rc, ierr)
! ukca_binapara is setup-independent -- it reads no glomap table and no config,
! only its own 113 literal coefficients -- so one process sweeps it whole.
! Asserted the same way as the two coefficient leaves: the grid runs under two
! mode setups and the results must be byte-equal.
!
! The routine CLAMPS its own inputs (`:106-118`) before doing anything, and
! then overwrites the local copies, so `t` inside the routine is the clamped
! temperature and the `t(jl) < 195.15` test at `:251` reads the clamped value
! rather than the caller's. That matters: a caller passing 150 K gets the
! 190.15 K answer and does NOT take the cold branch. Swept on both sides of
! every clamp for that reason.
USE ukca_binapara_mod, ONLY: ukca_binapara
USE glomap_f2py_state, ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
REAL(KIND=8), INTENT(IN)  :: t(n), rh(n), h2so4(n)
REAL(KIND=8), INTENT(OUT) :: jveh(n), rc(n)
INTEGER,      INTENT(OUT) :: ierr

jveh = 0.0
rc   = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0

CALL ukca_binapara(n, t, rh, h2so4, jveh, rc)
END SUBROUTINE leaf_binapara


SUBROUTINE leaf_calcnucrate(n, dtz, t, s, rh, aird, h2so4_in, sec_org,         &
                            height, htpblg, s_cond_s, bln_on, ibln,            &
                            i_nuc_method, h2so4_out, delh2so4_nucl, ierr)
! ukca_calcnucrate takes h2so4 as INTENT(IN OUT) and rewrites it in place, so
! the driver copies in and hands both the updated concentration and the
! reported change back out. A capture that passed the same array twice would
! record the routine's effect on its own input.
!
! Setup-independent: it reads no glomap table, only its own PARAMETERs and the
! two shared constants conc_eps and nmol. Measured as such, like the other
! leaves.
!
! i_nuc_method AND ibln ARE REFUSED OUT OF RANGE. The routine itself checks
! them at :288-293 and calls ereport -- but ereport under this binding is the
! shim, which RETURNS where the real one does STOP 1, so the routine would
! carry on into a branch with dpbln never assigned (:271 sets it only for
! ibln in 1..3) and produce a plausible number from uninitialised memory. The
! guard here is what stops that reaching a golden. ModelConfig rejects the
! same values.
!
! bln_on is NOT refused: 0 and 1 are both supported configurations and the
! switch selects a whole branch (l1 at :302), so both are swept.
USE ukca_calcnucrate_mod, ONLY: ukca_calcnucrate
USE glomap_f2py_state,    ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, bln_on, ibln, i_nuc_method
REAL(KIND=8), INTENT(IN)  :: dtz
REAL(KIND=8), INTENT(IN)  :: t(n), s(n), rh(n), aird(n), h2so4_in(n)
REAL(KIND=8), INTENT(IN)  :: sec_org(n), height(n), htpblg(n), s_cond_s(n)
REAL(KIND=8), INTENT(OUT) :: h2so4_out(n), delh2so4_nucl(n)
INTEGER,      INTENT(OUT) :: ierr

h2so4_out     = 0.0
delh2so4_nucl = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (i_nuc_method /= 2 .AND. i_nuc_method /= 3) THEN
  ierr = 3
  RETURN
END IF
IF (ibln < 1 .OR. ibln > 3) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

h2so4_out = h2so4_in
CALL ukca_calcnucrate(n, dtz, t, s, rh, aird, h2so4_out, delh2so4_nucl,        &
                      sec_org, bln_on, ibln, i_nuc_method, height, htpblg,     &
                      s_cond_s)
END SUBROUTINE leaf_calcnucrate


SUBROUTINE leaf_conden(n, nm, ncp_in, nchem, nbud1, nmins, ifuchs, idcmfp,     &
                       icondiam, dtz, nd, tsqrt, rhoa, airdm3, wetdp, pmid, t, &
                       md_in, mdt_in, gc_in, md_out, mdt_out, gc_out,          &
                       bud_out, delgc_cond, ageterm1, s_cond_s, ierr)
! Setup-DEPENDENT and the widest leaf in the project so far: ukca_conden reads
! mode, modesol, num_eps, sigmag and topmode from glomap_variables, and the gas
! and budget index tables through ukca_setup_indices, so one subprocess per
! i_mode_setup.
!
! md, mdt, gc AND bud_aer_mas are all INTENT(IN OUT). The driver copies each in
! and hands the result back separately, because a capture that passed one array
! for both would record the routine's effect on its own input and could not
! tell a field the routine left alone from one it wrote back unchanged.
!
! bud_aer_mas is declared `(nbox,0:nbudaer)` -- a ZERO lower bound, with slot 0
! the hole every unassigned index points at. f2py cannot express that, so it
! crosses as `(n, nbud1)` with `nbud1 = nbudaer+1` and column 1 IS slot 0. The
! extent is asserted rather than assumed: if nbudaer ever changes shape, +1
! stops being the right offset and this must fail loudly instead of shifting
! every budget field by one.
!
! bud_out is zeroed here rather than taking a caller value. Every write site is
! an accumulation onto whatever was there, so a non-zero input would make the
! golden record the sum of the routine's effect and the caller's choice -- and
! 4 of the 344 sites overwrite rather than accumulate (phase C), which only a
! zero start can distinguish.
USE ukca_conden_mod,               ONLY: ukca_conden
USE ukca_mode_setup,               ONLY: nmodes, nmodes_ins
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nbudaer
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in, nchem, nbud1, nmins
INTEGER,      INTENT(IN)  :: ifuchs, idcmfp, icondiam
REAL(KIND=8), INTENT(IN)  :: dtz
REAL(KIND=8), INTENT(IN)  :: nd(n, nm), tsqrt(n), rhoa(n), airdm3(n)
REAL(KIND=8), INTENT(IN)  :: wetdp(n, nm), pmid(n), t(n)
REAL(KIND=8), INTENT(IN)  :: md_in(n, nm, ncp_in), mdt_in(n, nm), gc_in(n, nchem)
REAL(KIND=8), INTENT(OUT) :: md_out(n, nm, ncp_in), mdt_out(n, nm), gc_out(n, nchem)
REAL(KIND=8), INTENT(OUT) :: bud_out(n, nbud1)
REAL(KIND=8), INTENT(OUT) :: delgc_cond(n, nchem), ageterm1(n, nmins, nchem)
REAL(KIND=8), INTENT(OUT) :: s_cond_s(n)
INTEGER,      INTENT(OUT) :: ierr

REAL(KIND=8) :: bud(n, 0:nbud1 - 1)
INTEGER      :: j

md_out     = md_in
mdt_out    = mdt_in
gc_out     = gc_in
bud_out    = 0.0
delgc_cond = 0.0
ageterm1   = 0.0
s_cond_s   = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp .OR. nchem /= nchemg      &
    .OR. nbud1 /= nbudaer + 1 .OR. nmins /= nmodes_ins) THEN
  ierr = 2
  RETURN
END IF
IF ((ifuchs /= 1 .AND. ifuchs /= 2) .OR. (idcmfp /= 1 .AND. idcmfp /= 2)       &
    .OR. (icondiam /= 1 .AND. icondiam /= 2)) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

bud = 0.0
CALL ukca_conden(n, nchem, nbudaer, ifuchs, idcmfp, icondiam,                  &
                 nd, tsqrt, rhoa, airdm3, dtz, wetdp, pmid, t,                 &
                 md_out, mdt_out, gc_out, bud,                                 &
                 delgc_cond, ageterm1, s_cond_s)
DO j = 0, nbud1 - 1
  bud_out(:, j + 1) = bud(:, j)
END DO
END SUBROUTINE leaf_conden


SUBROUTINE leaf_solvecoagnucl(n, mask_i, a, b, c, nd, dtz, deln, ierr)
! ukca_solvecoagnucl_v solves dN/dt = A*N^2 + B*N + C analytically, choosing
! between five closed forms and one error case. It reads no table and no
! config -- eps_ab and eps_d are locals -- so one process sweeps it whole.
!
! THE ERROR CASE IS FATAL AND THAT MATTERS HERE. `logic1ca` (A /= 0, D == 0,
! B /= 0) sets ierr = 1 and the routine then calls ereport at `:293`. The real
! ereport does STOP 1; the shim this binding links returns, so a grid that
! wandered into that branch would come back with a plausible `deln` computed
! from `ndnew` left at its initialised `nd` -- a zero increment that looks like
! "nothing coagulated" rather than "the solver gave up". `bind_call` counts
! ereports around every call for exactly this reason, so the capture sees it;
! the branch is probed deliberately rather than avoided.
!
! Issue #13: the shipped fixtures reach only 4 of the 8 branch codes. This
! driver exists to reach the other four, which is what a constructed leaf can
! do and a trajectory cannot.
USE ukca_solvecoagnucl_v_mod, ONLY: ukca_solvecoagnucl_v
! logical_32, not log_small: this routine is the one place in the vendored
! tree that declares its mask a 4-byte LOGICAL, and passing the 1-byte kind
! every other driver uses is a compile error rather than a silent
! reinterpretation. Worth the deviation being visible.
USE ukca_types_mod,           ONLY: logical_32
USE glomap_f2py_state,        ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n
INTEGER,      INTENT(IN)  :: mask_i(n)
REAL(KIND=8), INTENT(IN)  :: a(n), b(n), c(n), nd(n), dtz
REAL(KIND=8), INTENT(OUT) :: deln(n)
INTEGER,      INTENT(OUT) :: ierr

LOGICAL(KIND=logical_32) :: mask(n)
INTEGER                  :: i

deln = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
ierr = 0

DO i = 1, n
  mask(i) = (mask_i(i) /= 0)
END DO

CALL ukca_solvecoagnucl_v(n, mask, a, b, c, nd, dtz, deln)
END SUBROUTINE leaf_solvecoagnucl


SUBROUTINE leaf_coagwithnucl(n, nm, ncp_in, nchem, nbud1, nmsol, nmins,        &
                             intraoff, interoff, iextra_checks, dtz,           &
                             nd_in, md_in, mdt_in, delgc_nucl, kii_arr,        &
                             kij_arr, nd_out, md_out, mdt_out, bud_out,        &
                             ageterm2, ierr)
! Setup-DEPENDENT: ukca_coagwithnucl reads mode, component, mfrac_0, mmid,
! num_eps and topmode from glomap_variables, and coag_mode and the budget
! indices from ukca_setup_indices. One subprocess per i_mode_setup.
!
! nd, md, mdt and bud_aer_mas are all INTENT(IN OUT); each is copied in and
! handed back separately so a capture cannot mistake the routine's effect on
! its own input for a field it left alone.
!
! bud_aer_mas is (nbox,0:nbudaer) -- the same zero lower bound leaf_conden
! remaps, with column 1 carrying slot 0, the hole every uncarried index points
! at. Zeroed here rather than taken from the caller: every one of the 178
! nmascoag write sites accumulates, so a non-zero start would record the sum of
! the routine's effect and the caller's choice.
!
! iextra_checks IS REFUSED ABOVE 1. `:583` calls ukca_mode_check_mdt when it is
! 2, which zeroes number concentration for out-of-range modes and so changes
! mass budgets; docs/unsupported.md records that as not ported, and capturing
! it would put a reference in the goldens for code this project does not have.
USE ukca_coagwithnucl_mod,         ONLY: ukca_coagwithnucl
USE ukca_mode_setup,               ONLY: nmodes, nmodes_sol, nmodes_ins
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nbudaer
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in, nchem, nbud1, nmsol, nmins
INTEGER,      INTENT(IN)  :: intraoff, interoff, iextra_checks
REAL(KIND=8), INTENT(IN)  :: dtz
REAL(KIND=8), INTENT(IN)  :: nd_in(n, nm), md_in(n, nm, ncp_in), mdt_in(n, nm)
REAL(KIND=8), INTENT(IN)  :: delgc_nucl(n, nchem)
REAL(KIND=8), INTENT(IN)  :: kii_arr(n, nm), kij_arr(n, nm, nm)
REAL(KIND=8), INTENT(OUT) :: nd_out(n, nm), md_out(n, nm, ncp_in), mdt_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: bud_out(n, nbud1)
REAL(KIND=8), INTENT(OUT) :: ageterm2(n, nmsol, nmins, ncp_in)
INTEGER,      INTENT(OUT) :: ierr

REAL(KIND=8) :: bud(n, 0:nbud1 - 1)
INTEGER      :: j

nd_out   = nd_in
md_out   = md_in
mdt_out  = mdt_in
bud_out  = 0.0
ageterm2 = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp .OR. nchem /= nchemg      &
    .OR. nbud1 /= nbudaer + 1 .OR. nmsol /= nmodes_sol .OR. nmins /= nmodes_ins) THEN
  ierr = 2
  RETURN
END IF
IF (iextra_checks > 1) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

bud = 0.0
CALL ukca_coagwithnucl(n, nchem, nbudaer, nd_out, md_out, mdt_out,             &
                       delgc_nucl, dtz, ageterm2, intraoff, interoff, bud,     &
                       kii_arr, kij_arr, iextra_checks)
DO j = 0, nbud1 - 1
  bud_out(:, j + 1) = bud(:, j)
END DO
END SUBROUTINE leaf_coagwithnucl


SUBROUTINE leaf_ageing(n, nm, ncp_in, nchem, nbud1, nmsol, nmins,              &
                       nd_in, md_in, mdt_in, ageterm1, ageterm2, wetdp,        &
                       nd_out, md_out, mdt_out, bud_out, ierr)
! Setup-DEPENDENT: ukca_ageing reads component, mm, mode, num_eps and topmode
! from glomap_variables and condensable, condensable_choice, mm_gas and dimen
! from ukca_setup_indices. One subprocess per i_mode_setup.
!
! ageterm1 and ageterm2 are INPUTS here, produced by ukca_conden and
! ukca_coagwithnucl respectively -- both already ported and pinned, which is
! what makes it legitimate to construct them rather than run the chain.
! Constructing them is also the only way to reach the ageing branches at all:
! ageterm1 is zero unless condensation ran onto an insoluble mode, and
! ageterm2 unless a soluble mode coagulated into one.
USE ukca_ageing_mod,               ONLY: ukca_ageing
USE ukca_mode_setup,               ONLY: nmodes, nmodes_sol, nmodes_ins
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nbudaer
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in, nchem, nbud1, nmsol, nmins
REAL(KIND=8), INTENT(IN)  :: nd_in(n, nm), md_in(n, nm, ncp_in), mdt_in(n, nm)
REAL(KIND=8), INTENT(IN)  :: ageterm1(n, nmins, nchem)
REAL(KIND=8), INTENT(IN)  :: ageterm2(n, nmsol, nmins, ncp_in)
REAL(KIND=8), INTENT(IN)  :: wetdp(n, nm)
REAL(KIND=8), INTENT(OUT) :: nd_out(n, nm), md_out(n, nm, ncp_in), mdt_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: bud_out(n, nbud1)
INTEGER,      INTENT(OUT) :: ierr

REAL(KIND=8) :: bud(n, 0:nbud1 - 1)
INTEGER      :: j

nd_out  = nd_in
md_out  = md_in
mdt_out = mdt_in
bud_out = 0.0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp .OR. nchem /= nchemg      &
    .OR. nbud1 /= nbudaer + 1 .OR. nmsol /= nmodes_sol .OR. nmins /= nmodes_ins) THEN
  ierr = 2
  RETURN
END IF
ierr = 0

bud = 0.0
CALL ukca_ageing(n, nchem, nbudaer, nd_out, md_out, mdt_out, ageterm1,          &
                 ageterm2, wetdp, bud)
DO j = 0, nbud1 - 1
  bud_out(:, j + 1) = bud(:, j)
END DO
END SUBROUTINE leaf_ageing


SUBROUTINE leaf_remode(n, nm, ncp_in, nbud1, imerge, nd_in, md_in, mdt_in,     &
                       drydp, pmid, nd_out, md_out, mdt_out, bud_out,          &
                       n_merge, ierr)
! Setup-DEPENDENT. imerge is refused outside {1,2,3}: :215-231 assigns
! dp_thresh1 and dp_thresh2 only inside those three IFs, so any other value
! reads both never having been assigned and the merge criterion at :234 becomes
! whatever was on the stack. ModelConfig already rejects it.
USE ukca_remode_mod,               ONLY: ukca_remode
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nbudaer
USE ukca_types_mod,                ONLY: integer_32
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in, nbud1, imerge
REAL(KIND=8), INTENT(IN)  :: nd_in(n, nm), md_in(n, nm, ncp_in), mdt_in(n, nm)
REAL(KIND=8), INTENT(IN)  :: drydp(n, nm), pmid(n)
REAL(KIND=8), INTENT(OUT) :: nd_out(n, nm), md_out(n, nm, ncp_in), mdt_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: bud_out(n, nbud1)
INTEGER,      INTENT(OUT) :: n_merge(n, nm)
INTEGER,      INTENT(OUT) :: ierr

REAL(KIND=8)             :: bud(n, 0:nbud1 - 1)
INTEGER(KIND=integer_32) :: nm1d(n, nm)
INTEGER                  :: j

nd_out  = nd_in
md_out  = md_in
mdt_out = mdt_in
bud_out = 0.0
n_merge = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp .OR. nbud1 /= nbudaer + 1) THEN
  ierr = 2
  RETURN
END IF
IF (imerge < 1 .OR. imerge > 3) THEN
  ierr = 3
  RETURN
END IF
ierr = 0

bud  = 0.0
nm1d = 0
CALL ukca_remode(n, nbudaer, nd_out, md_out, mdt_out, drydp, imerge, bud,      &
                 nm1d, pmid)
DO j = 0, nbud1 - 1
  bud_out(:, j + 1) = bud(:, j)
END DO
n_merge = nm1d
END SUBROUTINE leaf_remode


SUBROUTINE leaf_aero_step(n, nm, ncp_in, nchem, nadv, nbud1,                   &
                          dtc, dtz, nmts, nzts,                                &
                          cond_on, nucl_on, coag_on, bln_on, icoag, imerge,    &
                          ifuchs, idcmfp, icondiam, ibln, i_nuc_method,        &
                          ichem, intraoff, interoff,                           &
                          nd_in, mdt_in, md_in, mdwat_in, s0g_in, drydp_in,    &
                          wetdp_in, rhopar_in, dvol_in, wvol_in,               &
                          sm, aird, airdm3, rhoa, mfpa, dvisc,                 &
                          t, tsqrt, rh, rh_clr, s, pmid, pupper, plower,       &
                          s0g_dot, height, htpblg,                             &
                          nd_out, mdt_out, md_out, mdwat_out, s0g_out,         &
                          drydp_out, wetdp_out, rhopar_out, dvol_out,          &
                          wvol_out, pvol_out, pvol_wat_out, bud_out,           &
                          n_merge, ierr)
! ukca_aero_step in the ONE configuration the box model runs, and the whole
! point of this driver is that it is a sequence rather than a routine: the
! twenty ported modules have each been checked against the reference on their
! own and never against each other.
!
! The routine takes 96 arguments. This wrapper exposes the ~40 the box model
! varies and bakes in the rest exactly as `glomap_box.F90:140-163` passes them
! -- every scavenging, deposition, cloud and nitrate switch off, dryox_in_aer
! = 1, wetox_in_aer = 0. That is not a simplification of the science: it is the
! configuration docs/unsupported.md records, and passing 50 zeros through f2py
! on every call would make the signature unreadable without making it more
! general.
!
! verbose = 0 deliberately. ukca_calcminmaxndmdt and ukca_calcminmaxgc write to
! umPrint on every process at verbose >= 2, which under this binding means
! thousands of lines through the shim per capture.
USE ukca_aero_step_mod,            ONLY: ukca_aero_step
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nadvg, nbudaer
USE ukca_types_mod,                ONLY: integer_32
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n, nm, ncp_in, nchem, nadv, nbud1
INTEGER,      INTENT(IN)  :: nmts, nzts
INTEGER,      INTENT(IN)  :: cond_on, nucl_on, coag_on, bln_on, icoag, imerge
INTEGER,      INTENT(IN)  :: ifuchs, idcmfp, icondiam, ibln, i_nuc_method
INTEGER,      INTENT(IN)  :: ichem, intraoff, interoff
REAL(KIND=8), INTENT(IN)  :: dtc, dtz
REAL(KIND=8), INTENT(IN)  :: nd_in(n, nm), mdt_in(n, nm), md_in(n, nm, ncp_in)
REAL(KIND=8), INTENT(IN)  :: mdwat_in(n, nm), s0g_in(n, nadv)
REAL(KIND=8), INTENT(IN)  :: drydp_in(n, nm), wetdp_in(n, nm), rhopar_in(n, nm)
REAL(KIND=8), INTENT(IN)  :: dvol_in(n, nm), wvol_in(n, nm)
REAL(KIND=8), INTENT(IN)  :: sm(n), aird(n), airdm3(n), rhoa(n), mfpa(n), dvisc(n)
REAL(KIND=8), INTENT(IN)  :: t(n), tsqrt(n), rh(n), rh_clr(n), s(n)
REAL(KIND=8), INTENT(IN)  :: pmid(n), pupper(n), plower(n)
REAL(KIND=8), INTENT(IN)  :: s0g_dot(n, nchem), height(n), htpblg(n)
REAL(KIND=8), INTENT(OUT) :: nd_out(n, nm), mdt_out(n, nm), md_out(n, nm, ncp_in)
REAL(KIND=8), INTENT(OUT) :: mdwat_out(n, nm), s0g_out(n, nadv)
REAL(KIND=8), INTENT(OUT) :: drydp_out(n, nm), wetdp_out(n, nm), rhopar_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: dvol_out(n, nm), wvol_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: pvol_out(n, nm, ncp_in), pvol_wat_out(n, nm)
REAL(KIND=8), INTENT(OUT) :: bud_out(n, nbud1)
INTEGER,      INTENT(OUT) :: n_merge(n, nm)
INTEGER,      INTENT(OUT) :: ierr

REAL(KIND=8)             :: bud(n, 0:nbud1 - 1)
REAL(KIND=8)             :: zeros(n)
REAL(KIND=8)             :: delso2(n), delso2_2(n)
INTEGER(KIND=integer_32) :: nm1d(n, nm)
INTEGER                  :: jlabove(n), ilscat(n), lday(n)
INTEGER                  :: j

nd_out = nd_in;  mdt_out = mdt_in;  md_out = md_in
mdwat_out = mdwat_in;  s0g_out = s0g_in
drydp_out = drydp_in;  wetdp_out = wetdp_in;  rhopar_out = rhopar_in
dvol_out = dvol_in;  wvol_out = wvol_in
pvol_out = 0.0;  pvol_wat_out = 0.0;  bud_out = 0.0;  n_merge = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (nm /= nmodes .OR. ncp_in /= glomap_variables%ncp .OR. nchem /= nchemg      &
    .OR. nadv /= nadvg .OR. nbud1 /= nbudaer + 1) THEN
  ierr = 2
  RETURN
END IF
ierr = 0

zeros = 0.0
delso2 = 0.0
delso2_2 = 0.0
bud = 0.0
nm1d = 0
jlabove = 1
ilscat = 1
lday = 1

CALL ukca_aero_step(n, nchem, nadv, nbudaer,                                   &
  nd_out, mdt_out, md_out, mdwat_out, s0g_out, drydp_out, wetdp_out,           &
  rhopar_out, dvol_out, wvol_out, sm,                                          &
  aird, airdm3, rhoa, mfpa, dvisc,                                             &
  t, tsqrt, rh, rh_clr, s,                                                     &
  pmid, pupper, plower,                                                        &
  zeros, zeros, zeros,                                                         &
  zeros, zeros, zeros,                                                         &
  zeros, zeros, zeros, zeros, zeros,                                           &
  zeros, zeros, zeros, zeros, zeros,                                           &
  ! 0.0 and not 0.0d0: -fdefault-real-8 promotes a plain literal to
  ! REAL(8), and `d0` promotes it again to REAL(16), which is a type
  ! mismatch rather than a silent widening.
  dtc, dtz, nmts, nzts, lday, 0.0, bud,                                      &
  0, 0,                                                                        &
  0, 0, 0, 0, 0,                                                               &
  1, 0, delso2, delso2_2,                                                      &
  cond_on, nucl_on, coag_on, bln_on, icoag,                                    &
  imerge, 0, 0, 0.0,                                                         &
  ifuchs, idcmfp, icondiam, ibln, i_nuc_method,                                &
  0, 1, 1, ichem, .FALSE., .FALSE.,                                            &
  0, 0, intraoff, interoff,                                                    &
  s0g_dot, zeros, zeros, pvol_out, pvol_wat_out,                               &
  jlabove, ilscat, nm1d, height, htpblg)

DO j = 0, nbud1 - 1
  bud_out(:, j + 1) = bud(:, j)
END DO
n_merge = nm1d
END SUBROUTINE leaf_aero_step


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
! Usable BEFORE wrap_init, unlike every other entry point here, and that is the
! whole point. init_ukca_for_box hardcodes this flag .TRUE.
! (glomap_box_config_mod.F90:322) and then init_state calls volume_mode ->
! water_content_v, which patches its own SAVEd table and never restores it.
! After init the unpatched table does not exist in this process. Setting the
! flag first, and calling leaf_water_content without init, is the only way to
! reach the unfixed arm at all.
USE ukca_config_specification_mod, ONLY: glomap_config
USE glomap_f2py_state,             ONLY: must_restart, water_flag_set
IMPLICIT NONE
INTEGER, INTENT(IN)  :: v
INTEGER, INTENT(OUT) :: ierr
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
ierr = 0
glomap_config%l_fix_ukca_water_content = (v /= 0)
water_flag_set = .TRUE.
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
