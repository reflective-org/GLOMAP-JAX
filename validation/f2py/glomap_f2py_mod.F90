! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   In-process binding to the vendored GLOMAP-mode box model (validation
!   gate A).
!
!   Gate A is the only mechanism in this repository that reaches ~1e-14. The
!   trajectory goldens are compared at RTOL_TRAJECTORY = 1e-9 and the
!   per-substep dumps at RTOL_STEP = 1e-11; both go through a text file and
!   both compare a whole call. Calling the Fortran in-process lets it be driven
!   with chosen inputs and read back at full double precision, with no file and
!   no accumulated trajectory in between.
!
!   FREE SUBROUTINES, NOT A MODULE. f2py exposes every module-level variable in
!   a file it cracks, so a module here would drag in the derived-type state and
!   fail (see glomap_f2py_state_mod.F90). File-scope subroutines avoid f2py's
!   f90mod_rules path altogether.
!
!   Three further constraints, each of which cost something to learn:
!
!   1. NO DERIVED TYPES CROSS THE BOUNDARY. f2py cannot marshal
!      TYPE(box_state_type) or TYPE(glomap_variables_type). State stays in
!      Fortran and moves across as plain arrays.
!
!   2. EVERY SIZE IS AN EXPLICIT LEADING INTEGER, AND IS CHECKED. f2py infers
!      array dimensions when it can, and an inferred dimension that disagrees
!      with the Fortran's own is a silent out-of-bounds read, not an error.
!
!      This makes the generated signatures ASYMMETRIC, and no f2py directive
!      undoes it (`required` is ignored here -- the dependency analysis wins):
!
!          out, ierr = wrap_get_2d(field, n1, n2)          sizes required
!          ierr       = wrap_set_2d(field, values[, n1, n2])  sizes inferred
!
!      For the getters the array is INTENT(OUT), so there is nothing to infer
!      from and the sizes stay required.
!
!      For the setters f2py derives the sizes from the array -- and that is
!      exactly why the Fortran-side check is load-bearing rather than
!      decorative. f2py only guarantees that n1/n2 agree with the array it was
!      handed; it knows nothing about nmodes or ncp, so an array of the wrong
!      width arrives as a perfectly self-consistent (n1, n2) pair and reaches
!      Fortran unchallenged. The comparison against the module's own sizes
!      below is the only thing standing between that and a wrong-shaped
!      assignment. Do not remove it, and do not "fix" the asymmetry by dropping
!      the size arguments: the getters genuinely need them.
!
!   3. REALS ARE REAL(KIND=8), NOT BARE REAL. The reference is built with
!      -fdefault-real-8, so a bare REAL here is real(8) -- but f2py maps the
!      *token* `real` to C float regardless of compiler flags, which would feed
!      float32 buffers into real(8) dummies and produce garbage rather than an
!      error. Being explicit sidesteps the .f2py_f2cmap entirely. The binding
!      is therefore f64-only by construction, which is right: ref-f32 is
!      diagnostic (ADR-001) and nothing is gated against it.
!
!   Not yet handled: `ereport` does STOP 1 in-process, which under f2py kills
!   the interpreter with no traceback, and there are twenty reachable call
!   sites. Until the shim lands (task 20b), drive this only with inputs known
!   not to abort.
!
!   Error codes, uniform across every entry point:
!       0  ok
!       1  already initialised with a different i_mode_setup; the process is
!          now unusable and must be restarted (see must_restart)
!       2  shape mismatch against the Fortran's own sizes
!       3  unknown field name
!       4  not initialised yet
!       5  a fatal ereport fired during the call; the result is meaningless
!
!   On 5: the ereport shim (task 20b, glomap_ereport_shim.F90) lets a caller
!   continue past a fatal error so Python can see it rather than inherit a
!   STOP. That is only safe if somebody checks, and leaving the checking to the
!   caller turns a loud crash into a silent wrong answer -- which is strictly
!   worse. So the entry points that run Fortran check for themselves: they
!   record the shim's fatal count before the call and return 5 if it moved.
!   wrap_ereport_last() then says which routine and why.
!
! ---------------------------------------------------------------------------
SUBROUTINE wrap_init(namelist_path, ierr)
! Read a namelist and run the same four-stage initialisation the box driver
! runs, so nothing about the configuration can drift between this .so and the
! binary that produced the goldens.
USE glomap_box_config_mod, ONLY: read_box_namelist, init_ukca_for_box,         &
    temperature, pressure, rel_humid, spec_humid, height, pbl_height,          &
    box_volume, i_mode_setup, nd_init, dp_init, mfrac_init,                    &
    h2so4_init, h2so4_prod, sec_org_init, sec_org_prod,                        &
    l_radaer, i_tune_bc, l_fix_nacl_density, l_fix_ukca_hygroscopicities,      &
    l_dust_mp_ageing
USE glomap_box_env_mod,    ONLY: set_box_env
USE glomap_box_state_mod,  ONLY: allocate_state, init_state
USE glomap_f2py_state,     ONLY: env, st, f2py_nbox, is_initialised,           &
                                 must_restart, init_i_mode_setup,              &
                                 init_l_radaer, init_i_tune_bc,                &
                                 init_l_fix_nacl_density,                      &
                                 init_l_fix_ukca_hygroscopicities,             &
                                 init_l_dust_mp_ageing
USE ereport_mod,           ONLY: ereport_shim_counts

IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: namelist_path
INTEGER,          INTENT(OUT) :: ierr
INTEGER :: fatal_before, fatal_after, warning, info

IF (must_restart) THEN
  ierr = 1
  RETURN
END IF

CALL ereport_shim_counts(fatal_before, warning, info)
CALL read_box_namelist(TRIM(namelist_path))

! A failed namelist open reports through ereport, which the shim makes
! non-fatal -- so without this check init would carry on into
! init_ukca_for_box and report success against a configuration it never read.
CALL ereport_shim_counts(fatal_after, warning, info)
IF (fatal_after > fatal_before) THEN
  ! Poison, for the same reason a setup change does. init_ukca_for_box may have
  ! half-built the mode tables before the ereport, so every later call would
  ! report success against state derived from a failed init.
  must_restart = .TRUE.
  ierr = 5
  RETURN
END IF

IF (is_initialised) THEN
  ! Compare EVERY variable init_ukca_for_box consumes, not just i_mode_setup.
  ! It is skipped on a matching re-init, so any of these that changed would be
  ! read from the new namelist and then ignored -- silently, with ierr = 0.
  IF ( (i_mode_setup /= init_i_mode_setup) .OR.                                &
       (l_radaer .NEQV. init_l_radaer) .OR.                                    &
       (i_tune_bc /= init_i_tune_bc) .OR.                                      &
       (l_fix_nacl_density .NEQV. init_l_fix_nacl_density) .OR.                &
       (l_fix_ukca_hygroscopicities .NEQV.                                     &
        init_l_fix_ukca_hygroscopicities) .OR.                                 &
       (l_dust_mp_ageing .NEQV. init_l_dust_mp_ageing) ) THEN
    ! Re-running init_ukca_for_box would leave stale nmas* indices pointing
    ! outside a re-sized bud_aer_mas; NOT re-running it would silently apply
    ! the old mode tables to the new namelist. Neither is safe, so refuse --
    ! and poison the process, because read_box_namelist above has already
    ! replaced every config scalar. What is left is the new switches paired
    ! with the old mode tables, a combination that never existed.
    must_restart = .TRUE.
    ierr = 1
    RETURN
  END IF
ELSE
  CALL init_ukca_for_box()
  is_initialised    = .TRUE.
  init_i_mode_setup = i_mode_setup
  init_l_radaer     = l_radaer
  init_i_tune_bc    = i_tune_bc
  init_l_fix_nacl_density          = l_fix_nacl_density
  init_l_fix_ukca_hygroscopicities = l_fix_ukca_hygroscopicities
  init_l_dust_mp_ageing            = l_dust_mp_ageing
END IF

CALL set_box_env(env, f2py_nbox, temperature, pressure, rel_humid, spec_humid, &
                 height, pbl_height, box_volume)
CALL allocate_state(st, f2py_nbox)
CALL init_state(st, env, nd_init, dp_init, mfrac_init,                         &
                h2so4_init, sec_org_init, h2so4_prod, sec_org_prod)

CALL ereport_shim_counts(fatal_after, warning, info)
IF (fatal_after > fatal_before) must_restart = .TRUE.
ierr = MERGE(5, 0, fatal_after > fatal_before)

END SUBROUTINE wrap_init

! ---------------------------------------------------------------------------
SUBROUTINE wrap_sizes(o_nbox, o_nmodes, o_ncp, o_nchemg, o_nadvg, o_nbudaer,   &
                      o_nsteps, o_setup, ierr)
! The sizes Python needs before it can allocate anything. All of them except
! nmodes are runtime module scalars: nbudaer alone takes seven distinct values
! across the seven supported setups.
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nadvg, nbudaer
USE glomap_box_config_mod,         ONLY: nsteps, i_mode_setup
USE glomap_f2py_state,             ONLY: f2py_nbox, is_initialised, must_restart

IMPLICIT NONE
INTEGER, INTENT(OUT) :: o_nbox, o_nmodes, o_ncp, o_nchemg, o_nadvg
INTEGER, INTENT(OUT) :: o_nbudaer, o_nsteps, o_setup, ierr

o_nbox = 0; o_nmodes = 0; o_ncp = 0; o_nchemg = 0
o_nadvg = 0; o_nbudaer = 0; o_nsteps = 0; o_setup = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF

o_nbox    = f2py_nbox
o_nmodes  = nmodes
o_ncp     = glomap_variables%ncp
o_nchemg  = nchemg
o_nadvg   = nadvg
o_nbudaer = nbudaer
o_nsteps  = nsteps
o_setup   = i_mode_setup
ierr      = 0

END SUBROUTINE wrap_sizes

! ---------------------------------------------------------------------------
SUBROUTINE wrap_step(ierr)
! One chemistry step: a transcription of the box driver's loop body, switch for
! switch. Any divergence here would make every gate-A comparison meaningless,
! which is why it is a transcription and not a re-derivation.
USE ukca_setup_indices,    ONLY: nchemg, nadvg, nbudaer, ichem
USE ukca_aero_step_mod,    ONLY: ukca_aero_step
USE glomap_box_config_mod, ONLY: dt_chem, nmts, nzts, cond_on, nucl_on,        &
    coag_on, bln_on, i_nuc_method, ibln, icoag, imerge, ifuchs, idcmfp,        &
    icondiam, intraoff, interoff, iactmethod, checkmd_nd, iextra_checks,       &
    act_dryr, verbose
USE glomap_box_state_mod,  ONLY: update_size
USE glomap_f2py_state,     ONLY: env, st, f2py_nbox, is_initialised,           &
                                 must_restart
USE ereport_mod,           ONLY: ereport_shim_counts

IMPLICIT NONE
INTEGER, INTENT(OUT) :: ierr
REAL    :: zeros(f2py_nbox)
INTEGER :: lday(f2py_nbox), jlabove(f2py_nbox), ilscat(f2py_nbox)
REAL    :: dtz
INTEGER :: fatal_before, fatal_after, warning, info

IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF

CALL ereport_shim_counts(fatal_before, warning, info)

zeros   = 0.0
lday    = 1
jlabove = 1
ilscat  = 1
dtz     = dt_chem / REAL(MAX(nmts,1) * MAX(nzts,1))

! Budgets and the merge counter accumulate over one chemistry step; upstream
! ukca_aero_ctl resets both per call, so do the same.
st%bud_aer_mas = 0.0
st%n_merge     = 0

CALL ukca_aero_step(                                                           &
  f2py_nbox, nchemg, nadvg, nbudaer,                                           &
  st%nd, st%mdt, st%md, st%mdwat, st%s0g, st%drydp, st%wetdp,                  &
  st%rhopar, st%dvol, st%wvol, env%sm,                                         &
  env%aird, env%airdm3, env%rhoa, env%mfpa, env%dvisc,                         &
  env%t, env%tsqrt, env%rh, env%rh_clr, env%s,                                 &
  env%pmid, env%pupper, env%plower,                                            &
  zeros, zeros, zeros,                    & ! zo3, zho2, zh2o2 (wet chem off)
  zeros, zeros, zeros,                    & ! ustr, znot, surtp (ddep off)
  zeros, zeros, zeros, zeros, zeros,      & ! crain, drain, crain_up, snow
  zeros, zeros, zeros, zeros, zeros,      & ! fconv, lowcloud, vfac, clf, autoconv
  dt_chem, dtz, nmts, nzts, lday, act_dryr, st%bud_aer_mas,                    &
  0, iextra_checks,                       & ! rainout_on
  0, 0, 0, 0, 0,                          & ! imscav, wetox, ddepaer, sedi, so2byo3
  1, 0, st%delso2, st%delso2_2,           & ! dryox_in_aer=1, wetox_in_aer
  cond_on, nucl_on, coag_on, bln_on, icoag,                                    &
  imerge, 0, 0, 0.0,                      & ! fine/coarse NO3 off, hno3_uptake
  ifuchs, idcmfp, icondiam, ibln, i_nuc_method,                                &
  iactmethod, 1, 1, ichem, .FALSE., .FALSE., & ! iddepaer, inucscav unused
  verbose, checkmd_nd, intraoff, interoff,                                     &
  st%s0g_dot, zeros, zeros, st%pvol, st%pvol_wat,                              &
  jlabove, ilscat, st%n_merge, env%height, env%htpbl)

! ukca_aero_step returns its own dry/wet sizes; refresh the diagnostic fields
! so the accessors report what the CSV output would have reported.
CALL update_size(st, env)

! ukca_aero_step has twenty reachable ereport sites, several of them inside the
! substep loop. Without this the shim would turn a fatal into a plausible
! trajectory.
! A fatal mid-step leaves st mutated part-way through; the next wrap_step would
! otherwise return 0 on it.
CALL ereport_shim_counts(fatal_after, warning, info)
IF (fatal_after > fatal_before) must_restart = .TRUE.
ierr = MERGE(5, 0, fatal_after > fatal_before)

END SUBROUTINE wrap_step

! ---------------------------------------------------------------------------
! Accessors. Shapes are passed in and checked rather than inferred.
! ---------------------------------------------------------------------------

SUBROUTINE wrap_get_2d(field, n1, n2, out, ierr)
USE ukca_mode_setup,   ONLY: nmodes
USE glomap_f2py_state, ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n1, n2
REAL(KIND=8),     INTENT(OUT) :: out(n1,n2)
INTEGER,          INTENT(OUT) :: ierr

out = 0.0_8
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('nd');       out = st%nd
CASE ('mdt');      out = st%mdt
CASE ('mdwat');    out = st%mdwat
CASE ('drydp');    out = st%drydp
CASE ('wetdp');    out = st%wetdp
CASE ('dvol');     out = st%dvol
CASE ('wvol');     out = st%wvol
CASE ('rhopar');   out = st%rhopar
CASE ('pvol_wat'); out = st%pvol_wat
CASE DEFAULT;      ierr = 3
END SELECT

END SUBROUTINE wrap_get_2d

SUBROUTINE wrap_set_2d(field, n1, n2, values, ierr)
! Driving the reference from Python is the point of gate A: it compares chosen
! inputs, not whatever state the trajectory happened to reach.
USE ukca_mode_setup,   ONLY: nmodes
USE glomap_f2py_state, ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n1, n2
REAL(KIND=8),     INTENT(IN)  :: values(n1,n2)
INTEGER,          INTENT(OUT) :: ierr
! See the header note on asymmetric signatures. f2py infers n1/n2 from
! `values`, so they always agree with each other -- the check below against the
! module's own nmodes is the only real one.

IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('nd');    st%nd    = values
CASE ('mdt');   st%mdt   = values
CASE ('mdwat'); st%mdwat = values
CASE ('drydp'); st%drydp = values
CASE ('wetdp'); st%wetdp = values
CASE DEFAULT;   ierr = 3
END SELECT

END SUBROUTINE wrap_set_2d

SUBROUTINE wrap_get_md(n1, n2, n3, out, ierr)
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n1, n2, n3
REAL(KIND=8), INTENT(OUT) :: out(n1,n2,n3)
INTEGER,      INTENT(OUT) :: ierr
out = 0.0_8
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nmodes .OR. n3 /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF
out  = st%md
ierr = 0
END SUBROUTINE wrap_get_md

SUBROUTINE wrap_set_md(n1, n2, n3, values, ierr)
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n1, n2, n3
REAL(KIND=8), INTENT(IN)  :: values(n1,n2,n3)
INTEGER,      INTENT(OUT) :: ierr
! f2py infers n1/n2/n3 from `values`; see the header note.
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nmodes .OR. n3 /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF
st%md = values
ierr  = 0
END SUBROUTINE wrap_set_md

SUBROUTINE wrap_get_s0g(n1, n2, out, ierr)
! Sized nadvg, not nchemg.
USE ukca_setup_indices, ONLY: nadvg
USE glomap_f2py_state,  ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n1, n2
REAL(KIND=8), INTENT(OUT) :: out(n1,n2)
INTEGER,      INTENT(OUT) :: ierr
out = 0.0_8
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nadvg) THEN
  ierr = 2
  RETURN
END IF
out  = st%s0g
ierr = 0
END SUBROUTINE wrap_get_s0g

SUBROUTINE wrap_get_budgets(n1, n2, out, ierr)
! bud_aer_mas is dimensioned (nbox, 0:nbudaer), so n2 must be nbudaer + 1.
! Slot 0 is a hole, not a null sink: every one of the ~684 writes is wrapped in
! IF (nmasxxx > 0), so a port that clamps unset indices to 0 changes semantics.
USE ukca_setup_indices, ONLY: nbudaer
USE glomap_f2py_state,  ONLY: st, f2py_nbox, is_initialised, must_restart
IMPLICIT NONE
INTEGER,      INTENT(IN)  :: n1, n2
REAL(KIND=8), INTENT(OUT) :: out(n1,n2)
INTEGER,      INTENT(OUT) :: ierr
out = 0.0_8
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= f2py_nbox .OR. n2 /= nbudaer + 1) THEN
  ierr = 2
  RETURN
END IF
out  = st%bud_aer_mas
ierr = 0
END SUBROUTINE wrap_get_budgets
