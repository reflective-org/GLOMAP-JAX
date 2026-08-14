! *****************************COPYRIGHT*******************************
! (c) 2026. Standalone GLOMAP-mode box model driver.
! New code, BSD 3-Clause (see LICENCE).
! *****************************COPYRIGHT*******************************
!
! Description:
!   Holds and initialises the GLOMAP-mode prognostic aerosol state for the
!   box model, and owns the gas-phase array in the units ukca_aero_step
!   expects.
!
!   Gas-phase units. ukca_aero_step recovers a volume mixing ratio as
!   s0g/sm and a number concentration as (s0g/sm)*aird, so
!
!       s0g(jv)     = (conc [molecules cm-3] / aird) * sm
!       s0g_dot(jv) =  prod [molecules cm-3 s-1] / aird     (vmr per second)
!
MODULE glomap_box_state_mod

USE ukca_mode_setup, ONLY: nmodes
USE ukca_types_mod,  ONLY: integer_32

IMPLICIT NONE
PRIVATE

TYPE, PUBLIC :: box_state_type
  INTEGER :: nbox
  INTEGER :: ncp

  ! -- prognostic aerosol ---------------------------------------------------
  REAL, ALLOCATABLE :: nd(:,:)        ! number concentration (cm-3)
  REAL, ALLOCATABLE :: md(:,:,:)      ! per-component mass (molecules ptcl-1)
  REAL, ALLOCATABLE :: mdt(:,:)       ! total mass (molecules ptcl-1)
  REAL, ALLOCATABLE :: mdwat(:,:)     ! aerosol water (molecules ptcl-1)

  ! -- derived size / density (recomputed each step by GLOMAP) --------------
  REAL, ALLOCATABLE :: drydp(:,:)     ! geometric mean dry diameter (m)
  REAL, ALLOCATABLE :: wetdp(:,:)     ! geometric mean wet diameter (m)
  REAL, ALLOCATABLE :: dvol(:,:)      ! geometric mean dry volume (m3)
  REAL, ALLOCATABLE :: wvol(:,:)      ! geometric mean wet volume (m3)
  REAL, ALLOCATABLE :: rhopar(:,:)    ! particle density incl. water (kg m-3)
  REAL, ALLOCATABLE :: pvol(:,:,:)    ! per-component partial volume
  REAL, ALLOCATABLE :: pvol_wat(:,:)  ! water partial volume

  ! -- gas phase ------------------------------------------------------------
  REAL, ALLOCATABLE :: s0g(:,:)       ! vmr * sm, per advected gas tracer
  REAL, ALLOCATABLE :: s0g_dot(:,:)   ! vmr per second, per chemistry tracer

  ! -- budget / bookkeeping -------------------------------------------------
  REAL, ALLOCATABLE :: bud_aer_mas(:,:)
  REAL, ALLOCATABLE :: delso2(:), delso2_2(:)
  INTEGER(KIND=integer_32), ALLOCATABLE :: n_merge(:,:)
END TYPE box_state_type

PUBLIC :: allocate_state
PUBLIC :: init_state
PUBLIC :: update_size

CONTAINS

! ---------------------------------------------------------------------------
SUBROUTINE allocate_state(st, nbox)

USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: nadvg, nchemg, nbudaer

IMPLICIT NONE
TYPE(box_state_type), INTENT(OUT) :: st
INTEGER, INTENT(IN) :: nbox

st%nbox = nbox
st%ncp  = glomap_variables%ncp

ALLOCATE(st%nd(nbox,nmodes), st%mdt(nbox,nmodes), st%mdwat(nbox,nmodes))
ALLOCATE(st%md(nbox,nmodes,st%ncp))
ALLOCATE(st%drydp(nbox,nmodes), st%wetdp(nbox,nmodes))
ALLOCATE(st%dvol(nbox,nmodes),  st%wvol(nbox,nmodes))
ALLOCATE(st%rhopar(nbox,nmodes))
ALLOCATE(st%pvol(nbox,nmodes,st%ncp), st%pvol_wat(nbox,nmodes))
ALLOCATE(st%s0g(nbox,nadvg), st%s0g_dot(nbox,nchemg))
ALLOCATE(st%bud_aer_mas(nbox,0:nbudaer))
ALLOCATE(st%delso2(nbox), st%delso2_2(nbox))
ALLOCATE(st%n_merge(nbox,nmodes))

st%nd = 0.0 ; st%md = 0.0 ; st%mdt = 0.0 ; st%mdwat = 0.0
st%drydp = 0.0 ; st%wetdp = 0.0 ; st%dvol = 0.0 ; st%wvol = 0.0
st%rhopar = 0.0 ; st%pvol = 0.0 ; st%pvol_wat = 0.0
st%s0g = 0.0 ; st%s0g_dot = 0.0 ; st%bud_aer_mas = 0.0
st%delso2 = 0.0 ; st%delso2_2 = 0.0 ; st%n_merge = 0

END SUBROUTINE allocate_state

! ---------------------------------------------------------------------------
SUBROUTINE init_state(st, env, nd_init, dp_init, mfrac_init,                   &
                      h2so4_init, sec_org_init, h2so4_prod, sec_org_prod)
! Builds a self-consistent initial aerosol population.
!
! Every ACTIVE mode is first seeded at its mid-point mass with the number
! concentration set to num_eps. GLOMAP requires MD/MDT to stay physical even
! for unpopulated modes: ukca_calc_drydiam aborts on dvol <= 0, which is what
! a literal zero-filled state produces.
!
! Modes with nd_init > num_eps are then given the requested number
! concentration and, if dp_init > 0, a per-component mass distribution
! consistent with that dry diameter:
!
!   dvol = (pi/6) * dp**3 * x            with x = exp(4.5 ln^2(sigma_g))
!   dvol = sum_cp  f_cp * M / rho_cp     =>  M = dvol / sum_cp (f_cp/rho_cp)
!   md_cp = f_cp * M * avogadro / mm_cp
!
! which is exactly the inverse of the dvol expression in ukca_calc_drydiam.

USE ukca_config_constants_mod,     ONLY: avogadro
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_constants,                ONLY: pi
USE ukca_setup_indices,            ONLY: mh2so4, msec_org
USE glomap_box_env_mod,            ONLY: box_env_type

IMPLICIT NONE
TYPE(box_state_type), INTENT(IN OUT) :: st
TYPE(box_env_type),   INTENT(IN)     :: env
REAL, INTENT(IN) :: nd_init(nmodes), dp_init(nmodes)
REAL, INTENT(IN) :: mfrac_init(:,:)
REAL, INTENT(IN) :: h2so4_init, sec_org_init, h2so4_prod, sec_org_prod

INTEGER :: imode, icp, ncp
REAL    :: frac(st%ncp), fsum, dvol_target, mass_tot, vol_per_kg

ncp = st%ncp

DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE

  ! -- 1. seed at mid-point mass so dvol > 0 even when the mode is empty ---
  DO icp = 1, ncp
    IF (glomap_variables%component(imode,icp)) THEN
      st%md(:,imode,icp) = glomap_variables%mmid(imode) *                      &
                           glomap_variables%mfrac_0(imode,icp)
    END IF
  END DO
  st%nd(:,imode) = glomap_variables%num_eps(imode)

  ! -- 2. populated modes: impose requested number and size ----------------
  IF (nd_init(imode) > glomap_variables%num_eps(imode)) THEN
    st%nd(:,imode) = nd_init(imode)

    IF (dp_init(imode) > 0.0) THEN
      ! composition: namelist row if it sums to > 0, else the setup default
      frac = mfrac_init(imode,1:ncp)
      fsum = SUM(frac)
      IF (fsum <= 0.0) THEN
        frac = glomap_variables%mfrac_0(imode,1:ncp)
        fsum = SUM(frac)
      END IF
      ! never place mass in a component the mode does not carry
      DO icp = 1, ncp
        IF (.NOT. glomap_variables%component(imode,icp)) frac(icp) = 0.0
      END DO
      fsum = SUM(frac)
      IF (fsum > 0.0) THEN
        frac = frac / fsum

        dvol_target = (pi / 6.0) * (dp_init(imode)**3) *                       &
                      glomap_variables%x(imode)

        vol_per_kg = 0.0
        DO icp = 1, ncp
          IF (frac(icp) > 0.0) THEN
            vol_per_kg = vol_per_kg + frac(icp) / glomap_variables%rhocomp(icp)
          END IF
        END DO
        mass_tot = dvol_target / vol_per_kg          ! kg per particle

        DO icp = 1, ncp
          st%md(:,imode,icp) = frac(icp) * mass_tot * avogadro /               &
                               glomap_variables%mm(icp)
        END DO
      END IF
    END IF
  END IF

  st%mdt(:,imode) = SUM(st%md(:,imode,:), DIM=2)
END DO

! -- 3. gas phase ---------------------------------------------------------
IF (mh2so4 > 0) THEN
  st%s0g(:,mh2so4)     = (h2so4_init / env%aird(:)) * env%sm(:)
  st%s0g_dot(:,mh2so4) =  h2so4_prod / env%aird(:)
END IF
IF (msec_org > 0) THEN
  st%s0g(:,msec_org)     = (sec_org_init / env%aird(:)) * env%sm(:)
  st%s0g_dot(:,msec_org) =  sec_org_prod / env%aird(:)
END IF

! -- 4. derived size and water content ------------------------------------
CALL update_size(st, env)

END SUBROUTINE init_state

! ---------------------------------------------------------------------------
SUBROUTINE update_size(st, env)
! Refreshes dry size, water content and wet size from the current MD/ND.

USE ukca_calc_drydiam_mod,         ONLY: ukca_calc_drydiam
USE ukca_volume_mode_mod,          ONLY: ukca_volume_mode
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_box_env_mod,            ONLY: box_env_type

IMPLICIT NONE
TYPE(box_state_type), INTENT(IN OUT) :: st
TYPE(box_env_type),   INTENT(IN)     :: env

CALL ukca_calc_drydiam(st%nbox, glomap_variables, st%nd, st%md, st%mdt,        &
                       st%drydp, st%dvol)

CALL ukca_volume_mode(glomap_variables, st%nbox, st%nd, st%md, st%mdt,         &
                      env%rh, st%dvol, st%drydp, env%t, env%pmid, env%s,       &
                      st%mdwat, st%wvol, st%wetdp, st%rhopar,                  &
                      st%pvol, st%pvol_wat)

END SUBROUTINE update_size

END MODULE glomap_box_state_mod
