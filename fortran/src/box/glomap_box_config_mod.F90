! *****************************COPYRIGHT*******************************
! (c) 2026. Standalone GLOMAP-mode box model driver.
!
! This file is NEW code written for the box model. It is released under
! the same BSD 3-Clause licence as the vendored UKCA sources (see LICENCE).
! *****************************COPYRIGHT*******************************
!
! Description:
!   Namelist-driven configuration for the standalone GLOMAP-mode box model.
!
!   Holds the box-model run settings and is responsible for driving the
!   UKCA initialisation sequence that the UM would normally perform:
!
!     1. init_config_constants()      - physical constants (avogadro, ...)
!     2. set ukca_config / glomap_config fields read by the microphysics
!     3. common_mode_setup_interface() - populates glomap_variables
!        (mode diameters, component molar masses/densities, mmid, x, ...)
!     4. ukca_indices_*()             - populates the gas-phase tracer and
!        aerosol budget index maps in ukca_setup_indices
!
!   Steps 1-4 MUST run, in that order, before any GLOMAP routine is called.
!   Skipping step 1 leaves avogadro/rho_so4 at rmdi and yields negative
!   particle masses; skipping step 4 leaves all gas indices at zero.
!
MODULE glomap_box_config_mod

USE ukca_mode_setup,               ONLY: nmodes, ncp_max
USE ukca_types_mod,                ONLY: integer_32

IMPLICIT NONE
PRIVATE

! ---------------------------------------------------------------------------
! Run control
! ---------------------------------------------------------------------------
CHARACTER(LEN=64), PUBLIC :: run_name    = 'glomap_box'
INTEGER,           PUBLIC :: nsteps      = 48        ! number of chemistry steps
REAL,              PUBLIC :: dt_chem     = 1800.0    ! chemistry timestep (s)
INTEGER,           PUBLIC :: nmts        = 1         ! microphysics substeps / dt_chem
INTEGER,           PUBLIC :: nzts        = 15        ! cond/nucl competition substeps
INTEGER,           PUBLIC :: output_every = 1        ! write output every N steps
CHARACTER(LEN=256),PUBLIC :: output_file = 'glomap_box.csv'
INTEGER,           PUBLIC :: verbose     = 0

! ---------------------------------------------------------------------------
! Environment (held fixed through the run)
! ---------------------------------------------------------------------------
REAL, PUBLIC :: temperature = 288.0     ! K
REAL, PUBLIC :: pressure    = 1.0e5     ! Pa
REAL, PUBLIC :: rel_humid   = 0.60      ! fraction, 0-1
REAL, PUBLIC :: spec_humid  = -1.0      ! kg/kg; < 0 => derive from rel_humid
REAL, PUBLIC :: height      = 500.0     ! height above surface (m)
REAL, PUBLIC :: pbl_height  = 1000.0    ! boundary layer depth (m)
REAL, PUBLIC :: box_volume  = 1.0       ! notional box volume (m3)

! ---------------------------------------------------------------------------
! Aerosol initial condition
! ---------------------------------------------------------------------------
! i_mode_setup selects the GLOMAP mode/component configuration. Values match
! ukca_config_specification_mod: 1=SU+SS 4-mode, 2=SUSSBCOC 5-mode,
! 3=SUSSBCOC 4-mode, 4=SUSSBCOCSO 5-mode, 5=SUSSBCOCSO 4-mode,
! 6=dust-only 2-mode, 8=SUSSBCOCDU 7-mode.
INTEGER, PUBLIC :: i_mode_setup = 1

! Initial number concentration per mode (particles cm-3). Index 1..nmodes:
!   1 nuc-sol  2 ait-sol  3 acc-sol  4 cor-sol
!   5 ait-ins  6 acc-ins  7 cor-ins  8 sup-ins
REAL, PUBLIC :: nd_init(nmodes) = 0.0

! Initial geometric-mean DRY diameter per mode (m).
! <= 0 => fall back to the mode mid-point diameter (ddpmid) from mode setup.
REAL, PUBLIC :: dp_init(nmodes) = -1.0

! Initial dry MASS fractions across components, per mode (nmodes x ncp_max).
! A row that sums to <= 0 falls back to mfrac_0 from the mode setup.
! Component order: 1=SU 2=BC 3=OC 4=NaCl 5=DU 6=SO 7=NO3 8=NaNO3 9=NH4 10=MP
REAL, PUBLIC :: mfrac_init(nmodes,ncp_max) = 0.0

! Mode setup options passed through to common_mode_setup_interface
LOGICAL, PUBLIC :: l_radaer                    = .FALSE.
INTEGER, PUBLIC :: i_tune_bc                   = 1
LOGICAL, PUBLIC :: l_fix_nacl_density          = .TRUE.
LOGICAL, PUBLIC :: l_fix_ukca_hygroscopicities = .TRUE.
LOGICAL, PUBLIC :: l_dust_mp_ageing            = .FALSE.

! ---------------------------------------------------------------------------
! Gas phase
! ---------------------------------------------------------------------------
! Initial concentrations (molecules cm-3) and sustained chemical production
! rates (molecules cm-3 s-1) of the condensable vapours.
REAL, PUBLIC :: h2so4_init   = 0.0
REAL, PUBLIC :: h2so4_prod   = 0.0
REAL, PUBLIC :: sec_org_init = 0.0
REAL, PUBLIC :: sec_org_prod = 0.0

! ---------------------------------------------------------------------------
! Process switches
! ---------------------------------------------------------------------------
INTEGER, PUBLIC :: cond_on       = 1   ! condensation of vapours onto particles
INTEGER, PUBLIC :: nucl_on       = 1   ! new particle formation
INTEGER, PUBLIC :: coag_on       = 1   ! coagulation
INTEGER, PUBLIC :: bln_on        = 0   ! boundary layer nucleation
INTEGER, PUBLIC :: i_nuc_method  = 2   ! 2 = BHN (+BLN in BL if bln_on)
INTEGER, PUBLIC :: ibln          = 1   ! 1=activation 2=kinetic 3=PNAS
INTEGER, PUBLIC :: icoag         = 1   ! 1=GLOMAP kernel 2=M7 3=UM 4=UM MFPP
INTEGER, PUBLIC :: imerge        = 1   ! 1=mid-points 2=edges 3=dynamic
INTEGER, PUBLIC :: ifuchs        = 1   ! 1=Fuchs(1964) 2=Fuchs-Sutugin(1971)
INTEGER, PUBLIC :: idcmfp        = 1   ! diffusion / mean-free-path variant
INTEGER, PUBLIC :: icondiam      = 1   ! 1=geometric mean 2=condensation diameter
INTEGER, PUBLIC :: intraoff      = 0   ! 1 => switch off intra-modal coagulation
INTEGER, PUBLIC :: interoff      = 0   ! 1 => switch off inter-modal coagulation
INTEGER, PUBLIC :: iactmethod    = 0   ! cloud processing / activation:
                                       ! 0=off, 1=fixed activation radius,
                                       ! 2=NSO3. Leave 0 unless cloud fields
                                       ! (lowcloud, vfac) are supplied.
INTEGER, PUBLIC :: checkmd_nd    = 0   ! run MD/ND consistency checks
INTEGER, PUBLIC :: iextra_checks = 0
REAL,    PUBLIC :: act_dryr      = 37.5e-9  ! activation dry radius (m)

! Set by read_group: was the group actually present in the file?
LOGICAL :: group_present = .FALSE.

PUBLIC :: read_box_namelist
PUBLIC :: init_ukca_for_box

CONTAINS

! ---------------------------------------------------------------------------
SUBROUTINE read_box_namelist(filename)
! Reads the box-model namelist groups. Any group may be omitted, in which
! case the defaults declared above apply.

USE ereport_mod, ONLY: ereport

IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN) :: filename

INTEGER :: unit_no, ios, errcode
CHARACTER(LEN=256) :: iomsg

NAMELIST /box_run/     run_name, nsteps, dt_chem, nmts, nzts,                  &
                       output_every, output_file, verbose
NAMELIST /box_env/     temperature, pressure, rel_humid, spec_humid,           &
                       height, pbl_height, box_volume
NAMELIST /box_aerosol/ i_mode_setup, nd_init, dp_init, mfrac_init,             &
                       l_radaer, i_tune_bc, l_fix_nacl_density,                &
                       l_fix_ukca_hygroscopicities, l_dust_mp_ageing
NAMELIST /box_gas/     h2so4_init, h2so4_prod, sec_org_init, sec_org_prod
NAMELIST /box_process/ cond_on, nucl_on, coag_on, bln_on, i_nuc_method,        &
                       ibln, icoag, imerge, ifuchs, idcmfp, icondiam,          &
                       intraoff, interoff, iactmethod, checkmd_nd,             &
                       iextra_checks, act_dryr

OPEN(NEWUNIT=unit_no, FILE=filename, STATUS='OLD', ACTION='READ', IOSTAT=ios)
IF (ios /= 0) THEN
  errcode = 1
  CALL ereport('READ_BOX_NAMELIST', errcode,                                   &
               'cannot open namelist file '//TRIM(filename))
END IF

! Each group is optional, but a group that IS present must parse completely.
! Both cases return IOSTAT /= 0, so "absent" and "present but malformed" are
! distinguished by scanning for the group header first. Without that check a
! single misspelled variable aborts the read part-way: names before it are
! applied, names after it silently keep their compiled-in defaults, and the
! model then runs a different case than the namelist asks for.
CALL read_group(unit_no, filename, 'box_run')
READ(unit_no, NML=box_run,     IOSTAT=ios, IOMSG=iomsg)
CALL check_group(filename, 'box_run', ios, iomsg)

CALL read_group(unit_no, filename, 'box_env')
READ(unit_no, NML=box_env,     IOSTAT=ios, IOMSG=iomsg)
CALL check_group(filename, 'box_env', ios, iomsg)

CALL read_group(unit_no, filename, 'box_aerosol')
READ(unit_no, NML=box_aerosol, IOSTAT=ios, IOMSG=iomsg)
CALL check_group(filename, 'box_aerosol', ios, iomsg)

CALL read_group(unit_no, filename, 'box_gas')
READ(unit_no, NML=box_gas,     IOSTAT=ios, IOMSG=iomsg)
CALL check_group(filename, 'box_gas', ios, iomsg)

CALL read_group(unit_no, filename, 'box_process')
READ(unit_no, NML=box_process, IOSTAT=ios, IOMSG=iomsg)
CALL check_group(filename, 'box_process', ios, iomsg)

CLOSE(unit_no)

CALL validate_config()

END SUBROUTINE read_box_namelist

! ---------------------------------------------------------------------------
SUBROUTINE read_group(unit_no, filename, group)
! Rewinds the unit and records whether `group` is present in the file, so a
! subsequent failed READ can be classified as "absent" or "malformed".

IMPLICIT NONE
INTEGER,          INTENT(IN) :: unit_no
CHARACTER(LEN=*), INTENT(IN) :: filename, group

CHARACTER(LEN=512) :: line, trimmed
INTEGER :: ios

group_present = .FALSE.
REWIND(unit_no)
DO
  READ(unit_no, '(A)', IOSTAT=ios) line
  IF (ios /= 0) EXIT
  trimmed = ADJUSTL(line)
  IF (trimmed(1:1) /= '&') CYCLE
  IF (lowercase(trimmed(2:LEN_TRIM(group)+1)) == lowercase(group)) THEN
    ! require the name to end here, so &box_run does not match &box_runx
    IF (LEN_TRIM(trimmed) == LEN_TRIM(group)+1 .OR.                            &
        trimmed(LEN_TRIM(group)+2:LEN_TRIM(group)+2) == ' ') THEN
      group_present = .TRUE.
      EXIT
    END IF
  END IF
END DO
REWIND(unit_no)

END SUBROUTINE read_group

! ---------------------------------------------------------------------------
SUBROUTINE check_group(filename, group, ios, iomsg)
! Absent group -> fine, defaults apply. Present but unreadable -> fatal.

USE ereport_mod, ONLY: ereport

IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN) :: filename, group, iomsg
INTEGER,          INTENT(IN) :: ios
INTEGER :: errcode

IF (ios == 0) RETURN
IF (.NOT. group_present) RETURN

errcode = 1
CALL ereport('READ_BOX_NAMELIST', errcode,                                     &
             'malformed namelist group &'//TRIM(group)//' in '//               &
             TRIM(filename)//' -- '//TRIM(iomsg)//                             &
             ' (check for a misspelled or mistyped variable name)')

END SUBROUTINE check_group

! ---------------------------------------------------------------------------
PURE FUNCTION lowercase(s) RESULT(out)
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN) :: s
CHARACTER(LEN=LEN(s)) :: out
INTEGER :: i, ic
out = s
DO i = 1, LEN(s)
  ic = IACHAR(out(i:i))
  IF (ic >= IACHAR('A') .AND. ic <= IACHAR('Z')) out(i:i) = ACHAR(ic + 32)
END DO
END FUNCTION lowercase

! ---------------------------------------------------------------------------
SUBROUTINE validate_config()
! Reject settings that would otherwise fail deep inside GLOMAP with a
! confusing message, or silently produce nonsense.

USE ereport_mod, ONLY: ereport

IMPLICIT NONE
INTEGER :: errcode
CHARACTER(LEN=128) :: msg

errcode = 1
IF (nsteps < 1) CALL ereport('VALIDATE_CONFIG', errcode, 'nsteps must be >= 1')
errcode = 1
IF (dt_chem <= 0.0) CALL ereport('VALIDATE_CONFIG', errcode,                   &
                                 'dt_chem must be > 0')
errcode = 1
IF (nmts < 1) CALL ereport('VALIDATE_CONFIG', errcode, 'nmts must be >= 1')
errcode = 1
IF (nzts < 1) CALL ereport('VALIDATE_CONFIG', errcode, 'nzts must be >= 1')
errcode = 1
IF (temperature <= 0.0) CALL ereport('VALIDATE_CONFIG', errcode,               &
                                     'temperature must be > 0 K')
errcode = 1
IF (pressure <= 0.0) CALL ereport('VALIDATE_CONFIG', errcode,                  &
                                  'pressure must be > 0 Pa')
errcode = 1
IF (rel_humid < 0.0 .OR. rel_humid > 1.0) THEN
  WRITE(msg,'(A,ES11.3)') 'rel_humid must be in [0,1], got ', rel_humid
  CALL ereport('VALIDATE_CONFIG', errcode, msg)
END IF
errcode = 1
IF (box_volume <= 0.0) CALL ereport('VALIDATE_CONFIG', errcode,                &
                                    'box_volume must be > 0 m3')
errcode = 1
IF (ANY(nd_init < 0.0)) CALL ereport('VALIDATE_CONFIG', errcode,               &
                                     'nd_init must be >= 0')
errcode = 1
IF (h2so4_init < 0.0 .OR. sec_org_init < 0.0 .OR.                              &
    h2so4_prod < 0.0 .OR. sec_org_prod < 0.0)                                  &
  CALL ereport('VALIDATE_CONFIG', errcode,                                     &
               'gas concentrations and production rates must be >= 0')

END SUBROUTINE validate_config

! ---------------------------------------------------------------------------
SUBROUTINE init_ukca_for_box()
! Performs the four-stage UKCA initialisation described in the module header.

USE ukca_config_constants_mod,     ONLY: init_config_constants
USE ukca_config_specification_mod, ONLY: glomap_variables, glomap_config,      &
                                         ukca_config
USE common_mode_setup_interface_mod, ONLY: common_mode_setup_interface

IMPLICIT NONE

! -- 1. physical constants ------------------------------------------------
CALL init_config_constants()

! -- 2. the config fields the vendored microphysics actually reads ---------
! (every other GLOMAP switch reaches ukca_aero_step as an explicit argument)
glomap_config%i_mode_setup             = i_mode_setup
glomap_config%l_fix_ukca_water_content = .TRUE.
glomap_config%l_fix_neg_pvol_wat       = .TRUE.
glomap_config%l_fix_ukca_impscav       = .TRUE.
glomap_config%solinsol_hygro_ratio     = 1.0
glomap_config%dry_depvel_acc_scaling   = 1.0
glomap_config%acc_cor_scav_scaling     = 1.0

ukca_config%l_ukca_chem      = .TRUE.
ukca_config%l_ukca_scale_ppe = .FALSE.
ukca_config%ntype            = 1

! -- 3. modes and components ----------------------------------------------
CALL common_mode_setup_interface(glomap_variables, i_mode_setup, l_radaer,     &
                                i_tune_bc, l_fix_nacl_density,                 &
                                l_fix_ukca_hygroscopicities, l_dust_mp_ageing)

! -- 4. gas tracer + aerosol budget index maps ----------------------------
CALL init_indices()

END SUBROUTINE init_ukca_for_box

! ---------------------------------------------------------------------------
SUBROUTINE init_indices()
! Pairs the gas-phase index set with the aerosol index set for the selected
! mode configuration. The pairings mirror those in the UM's ukca_init.

USE ereport_mod,          ONLY: ereport
USE ukca_setup_indices,   ONLY: ukca_indices_nochem,                           &
                                ukca_indices_sv1,                              &
                                ukca_indices_orgv1_soto3,                      &
                                ukca_indices_orgv1_soto6,                      &
                                ukca_indices_suss_4mode,                       &
                                ukca_indices_sussbcoc_5mode,                   &
                                ukca_indices_sussbcoc_4mode,                   &
                                ukca_indices_sussbcocso_5mode,                 &
                                ukca_indices_sussbcocso_4mode,                 &
                                ukca_indices_duonly_2mode,                     &
                                ukca_indices_sussbcocdu_7mode
USE ukca_config_specification_mod, ONLY: i_suss_4mode, i_sussbcoc_5mode,       &
                                i_sussbcoc_4mode, i_sussbcocso_5mode,          &
                                i_sussbcocso_4mode, i_du_2mode,                &
                                i_sussbcocdu_7mode

IMPLICIT NONE
INTEGER :: errcode
CHARACTER(LEN=64) :: msg

SELECT CASE (i_mode_setup)
CASE (i_suss_4mode)
  CALL ukca_indices_sv1
  CALL ukca_indices_suss_4mode
CASE (i_sussbcoc_5mode)
  CALL ukca_indices_orgv1_soto3
  CALL ukca_indices_sussbcoc_5mode
CASE (i_sussbcoc_4mode)
  CALL ukca_indices_orgv1_soto3
  CALL ukca_indices_sussbcoc_4mode
CASE (i_sussbcocso_5mode)
  CALL ukca_indices_orgv1_soto6
  CALL ukca_indices_sussbcocso_5mode
CASE (i_sussbcocso_4mode)
  CALL ukca_indices_orgv1_soto6
  CALL ukca_indices_sussbcocso_4mode
CASE (i_du_2mode)
  CALL ukca_indices_nochem
  CALL ukca_indices_duonly_2mode
CASE (i_sussbcocdu_7mode)
  CALL ukca_indices_orgv1_soto3
  CALL ukca_indices_sussbcocdu_7mode
CASE DEFAULT
  errcode = 2
  WRITE(msg,'(A,I0)') 'unsupported i_mode_setup = ', i_mode_setup
  CALL ereport('INIT_INDICES', errcode, msg)
END SELECT

END SUBROUTINE init_indices

END MODULE glomap_box_config_mod
