! *****************************COPYRIGHT*******************************
! (c) 2026. Standalone GLOMAP-mode box model driver.
! New code, BSD 3-Clause (see LICENCE).
! *****************************COPYRIGHT*******************************
!
! Description:
!   Main program for the standalone GLOMAP-mode aerosol box model.
!
!   Integrates a single, well-mixed air parcel forward in time using the
!   unmodified UKCA GLOMAP-mode microphysics (ukca_aero_step):
!
!     * nucleation   - binary homogeneous H2SO4-H2O (Vehkamaki/Kulmala),
!                      optional boundary layer nucleation
!     * condensation - H2SO4 and secondary organic vapour onto all modes
!     * coagulation  - intra- and inter-modal, soluble and insoluble
!     * ageing       - transfer of insoluble to soluble modes
!     * mode merging - remoding between adjacent size modes
!
!   Wet chemistry, cloud processing, dry deposition, sedimentation and
!   scavenging are compiled in but switched OFF: they need column or surface
!   information that a single box does not carry. Cloud processing is off by
!   virtue of iactmethod = 0 (ukca_aero_step gates it on iactmethod > 0), not
!   merely because the cloud fraction fields arrive as zero.
!
!   Usage:   glomap_box [namelist_file]
!   Default namelist: namelists/boundary_layer.nml
!
PROGRAM glomap_box

USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nchemg, nadvg, nbudaer, ichem
USE ukca_aero_step_mod,            ONLY: ukca_aero_step

USE glomap_box_config_mod,         ONLY: read_box_namelist, init_ukca_for_box,  &
    run_name, nsteps, dt_chem, nmts, nzts, output_every, output_file, verbose,  &
    temperature, pressure, rel_humid, spec_humid, height, pbl_height,           &
    box_volume, i_mode_setup, nd_init, dp_init, mfrac_init,                     &
    h2so4_init, h2so4_prod, sec_org_init, sec_org_prod,                         &
    cond_on, nucl_on, coag_on, bln_on, i_nuc_method, ibln, icoag, imerge,       &
    ifuchs, idcmfp, icondiam, intraoff, interoff, iactmethod, checkmd_nd,       &
    iextra_checks, act_dryr
USE glomap_box_env_mod,            ONLY: box_env_type, set_box_env
USE glomap_box_state_mod,          ONLY: box_state_type, allocate_state,        &
                                         init_state, update_size
USE glomap_box_output_mod,         ONLY: open_output, write_output,             &
                                         close_output, write_header_table,     &
                                         write_table_row

IMPLICIT NONE

INTEGER, PARAMETER :: nbox = 1

TYPE(box_env_type)   :: env
TYPE(box_state_type) :: st

CHARACTER(LEN=256) :: nml_file
INTEGER :: istep, iarg
REAL    :: time_s, dtz

! Fields required by ukca_aero_step for the processes the box model leaves
! switched off. They are allocated, zeroed and passed through unchanged.
REAL    :: zeros(nbox)
INTEGER :: lday(nbox), jlabove(nbox), ilscat(nbox)

! ---------------------------------------------------------------------------
! 1. Namelist
! ---------------------------------------------------------------------------
nml_file = 'namelists/boundary_layer.nml'
iarg = COMMAND_ARGUMENT_COUNT()
IF (iarg >= 1) CALL GET_COMMAND_ARGUMENT(1, nml_file)

CALL read_box_namelist(TRIM(nml_file))

! ---------------------------------------------------------------------------
! 2. UKCA initialisation (constants -> config -> modes -> indices)
! ---------------------------------------------------------------------------
CALL init_ukca_for_box()

WRITE(*,'(A)') '======================================================='
WRITE(*,'(A)') ' GLOMAP-mode standalone aerosol box model'
WRITE(*,'(A)') '======================================================='
WRITE(*,'(A,A)')    ' namelist       : ', TRIM(nml_file)
WRITE(*,'(A,A)')    ' run name       : ', TRIM(run_name)
WRITE(*,'(A,I0)')   ' i_mode_setup   : ', i_mode_setup
WRITE(*,'(A,I0,A,I0)') ' components     : ', glomap_variables%ncp,             &
                       '   active modes : ', COUNT(glomap_variables%mode)
WRITE(*,'(A,I0,A,I0,A,I0)') ' nchemg=', nchemg, ' nadvg=', nadvg,              &
                            ' nbudaer=', nbudaer
WRITE(*,'(A,F7.2,A,F9.1,A,F5.2)') ' T(K)=', temperature,                       &
                                  '  p(Pa)=', pressure, '  RH=', rel_humid
WRITE(*,'(A,I0,A,I0,A,I0)') ' switches: cond=', cond_on, ' nucl=', nucl_on,    &
                            ' coag=', coag_on
WRITE(*,'(A,I0,A,F8.1,A)') ' steps: ', nsteps, ' x ', dt_chem, ' s'
WRITE(*,'(A)') '======================================================='

! ---------------------------------------------------------------------------
! 3. Environment and initial state
! ---------------------------------------------------------------------------
CALL set_box_env(env, nbox, temperature, pressure, rel_humid, spec_humid,       &
                 height, pbl_height, box_volume)

CALL allocate_state(st, nbox)
CALL init_state(st, env, nd_init, dp_init, mfrac_init,                          &
                h2so4_init, sec_org_init, h2so4_prod, sec_org_prod)

zeros   = 0.0
lday    = 1
jlabove = 1
ilscat  = 1
dtz     = dt_chem / REAL(MAX(nmts,1) * MAX(nzts,1))

CALL open_output(TRIM(output_file))
CALL write_header_table()

time_s = 0.0
CALL write_output(time_s, st, env)
CALL write_table_row(time_s, st, env)

! ---------------------------------------------------------------------------
! 4. Time integration
! ---------------------------------------------------------------------------
DO istep = 1, nsteps

  ! Budget fields and the mode-merge counter accumulate over one chemistry
  ! step; upstream ukca_aero_ctl resets both per call, so do the same.
  st%bud_aer_mas = 0.0
  st%n_merge     = 0

  CALL ukca_aero_step(                                                         &
    nbox, nchemg, nadvg, nbudaer,                                              &
    st%nd, st%mdt, st%md, st%mdwat, st%s0g, st%drydp, st%wetdp,                 &
    st%rhopar, st%dvol, st%wvol, env%sm,                                       &
    env%aird, env%airdm3, env%rhoa, env%mfpa, env%dvisc,                        &
    env%t, env%tsqrt, env%rh, env%rh_clr, env%s,                                &
    env%pmid, env%pupper, env%plower,                                           &
    zeros, zeros, zeros,                    & ! zo3, zho2, zh2o2 (wet chem off)
    zeros, zeros, zeros,                    & ! ustr, znot, surtp (ddep off)
    zeros, zeros, zeros, zeros, zeros,      & ! crain, drain, crain_up, snow
    zeros, zeros, zeros, zeros, zeros,      & ! fconv, lowcloud, vfac, clf, autoconv
    dt_chem, dtz, nmts, nzts, lday, act_dryr, st%bud_aer_mas,                   &
    0, iextra_checks,                       & ! rainout_on
    0, 0, 0, 0, 0,                          & ! imscav, wetox, ddepaer, sedi, so2byo3
    1, 0, st%delso2, st%delso2_2,           & ! dryox_in_aer=1 (use s0g_dot), wetox_in_aer
    cond_on, nucl_on, coag_on, bln_on, icoag,                                   &
    imerge, 0, 0, 0.0,                      & ! fine/coarse NO3 off, hno3_uptake_coeff
    ifuchs, idcmfp, icondiam, ibln, i_nuc_method,                               &
    iactmethod, 1, 1, ichem, .FALSE., .FALSE., & ! iddepaer, inucscav unused (off)
    verbose, checkmd_nd, intraoff, interoff,                                    &
    st%s0g_dot, zeros, zeros, st%pvol, st%pvol_wat,                             &
    jlabove, ilscat, st%n_merge, env%height, env%htpbl)

  ! ukca_aero_step returns updated MD/ND and its own dry/wet sizes; refresh
  ! the diagnostic size and density fields so output is consistent.
  CALL update_size(st, env)

  time_s = REAL(istep) * dt_chem

  IF (MOD(istep, MAX(output_every,1)) == 0 .OR. istep == nsteps) THEN
    CALL write_output(time_s, st, env)
    CALL write_table_row(time_s, st, env)
  END IF

END DO

CALL close_output()

WRITE(*,'(A)') '======================================================='
WRITE(*,'(A,I0,A,F8.1,A)') ' completed ', nsteps, ' steps of ', dt_chem, ' s'
WRITE(*,'(A,A)') ' output written to: ', TRIM(output_file)
WRITE(*,'(A)') '======================================================='

END PROGRAM glomap_box
