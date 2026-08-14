! *****************************COPYRIGHT*******************************
! (c) 2026. Standalone GLOMAP-mode box model driver.
! New code, BSD 3-Clause (see LICENCE).
! *****************************COPYRIGHT*******************************
!
! Description:
!   Derives the air-property fields that ukca_aero_step expects from the
!   small set of thermodynamic variables the user supplies (T, p, RH).
!
!   All fields are rank-1 over nbox so the box model keeps the same shape
!   as the UM's segmented interface; nbox = 1 for a single-box run.
!
MODULE glomap_box_env_mod

IMPLICIT NONE
PRIVATE

TYPE, PUBLIC :: box_env_type
  INTEGER :: nbox
  REAL, ALLOCATABLE :: t(:)        ! temperature (K)
  REAL, ALLOCATABLE :: tsqrt(:)    ! sqrt(T)
  REAL, ALLOCATABLE :: pmid(:)     ! centre-level pressure (Pa)
  REAL, ALLOCATABLE :: pupper(:)   ! upper interface pressure (Pa)
  REAL, ALLOCATABLE :: plower(:)   ! lower interface pressure (Pa)
  REAL, ALLOCATABLE :: rh(:)       ! relative humidity (0-1)
  REAL, ALLOCATABLE :: rh_clr(:)   ! clear-sky relative humidity (0-1)
  REAL, ALLOCATABLE :: s(:)        ! specific humidity (kg/kg)
  REAL, ALLOCATABLE :: aird(:)     ! air number density (molecules cm-3)
  REAL, ALLOCATABLE :: airdm3(:)   ! air number density (molecules m-3)
  REAL, ALLOCATABLE :: rhoa(:)     ! air density (kg m-3)
  REAL, ALLOCATABLE :: mfpa(:)     ! mean free path of air (m)
  REAL, ALLOCATABLE :: dvisc(:)    ! dynamic viscosity (kg m-1 s-1)
  REAL, ALLOCATABLE :: sm(:)       ! mass of air in the box (kg)
  REAL, ALLOCATABLE :: height(:)   ! height above surface (m)
  REAL, ALLOCATABLE :: htpbl(:)    ! boundary layer depth (m)
END TYPE box_env_type

PUBLIC :: set_box_env

CONTAINS

! ---------------------------------------------------------------------------
SUBROUTINE set_box_env(env, nbox, t_in, p_in, rh_in, s_in, height_in,          &
                       pbl_in, volume_in)

USE ukca_config_constants_mod, ONLY: boltzmann, r
USE ukca_constants,            ONLY: pi

IMPLICIT NONE
TYPE(box_env_type), INTENT(OUT) :: env
INTEGER, INTENT(IN) :: nbox
REAL,    INTENT(IN) :: t_in, p_in, rh_in, s_in, height_in, pbl_in, volume_in

! Dynamic viscosity and mean free path are computed with exactly the
! expressions UKCA itself uses in ukca_aero_ctl (see ukca_aero_ctl.F90,
! "no conc of air" loop). Reproducing them rather than substituting an
! equivalent-looking fit matters: mfpa sets the Knudsen number in
! ukca_conden and in the coagulation kernel, and a generic lambda ~ T/p
! form diverges from UKCA's lambda ~ T^2/((T+120)*p) by ~9% at 200 hPa.
REAL, PARAMETER :: ma        = 4.78e-26    ! mass of an air molecule (kg)
REAL, PARAMETER :: dvisc_ref = 1.83e-5     ! kg m-1 s-1
REAL, PARAMETER :: t_suth    = 120.0       ! K, Sutherland constant
REAL, PARAMETER :: t_suth_num= 416.16      ! K, = t_ref_vis + t_suth
REAL, PARAMETER :: t_ref_vis = 296.16      ! K
REAL :: vba                                ! mean thermal speed of air (m/s)

env%nbox = nbox
ALLOCATE(env%t(nbox), env%tsqrt(nbox), env%pmid(nbox), env%pupper(nbox),       &
         env%plower(nbox), env%rh(nbox), env%rh_clr(nbox), env%s(nbox),        &
         env%aird(nbox), env%airdm3(nbox), env%rhoa(nbox), env%mfpa(nbox),     &
         env%dvisc(nbox), env%sm(nbox), env%height(nbox), env%htpbl(nbox))

env%t      = t_in
env%tsqrt  = SQRT(t_in)
env%pmid   = p_in
! Nominal +/-5% interface pressures: only used by processes (sedimentation,
! scavenging) that the box model leaves switched off by default.
env%pupper = 0.95 * p_in
env%plower = 1.05 * p_in

env%rh     = MIN(MAX(rh_in, 0.0), 1.0)
env%rh_clr = env%rh

IF (s_in >= 0.0) THEN
  env%s = s_in
ELSE
  env%s = spec_humid_from_rh(t_in, p_in, env%rh(1))
END IF

! Ideal gas relations
env%aird   = p_in / (boltzmann * 1.0e6 * t_in)   ! molecules cm-3
env%airdm3 = env%aird * 1.0e6
env%rhoa   = p_in / (r * t_in)

! Dynamic viscosity and mean free path, as in ukca_aero_ctl
env%dvisc = dvisc_ref * (t_suth_num / (t_in + t_suth)) *                       &
            (SQRT(t_in / t_ref_vis)**3)
vba       = SQRT(8.0 * boltzmann * t_in / (pi * ma))
env%mfpa  = 2.0 * env%dvisc / (env%rhoa * vba)

! Mass of air in the notional box. Only the ratio s0g/sm matters to the
! microphysics (it recovers a volume mixing ratio), so any consistent
! volume works; box_volume simply makes the choice explicit.
env%sm = env%rhoa * volume_in

env%height = height_in
env%htpbl  = pbl_in

END SUBROUTINE set_box_env

! ---------------------------------------------------------------------------
REAL FUNCTION spec_humid_from_rh(t, p, rh)
! Specific humidity from relative humidity using the Tetens saturation
! vapour pressure formula over liquid water.

USE ukca_config_constants_mod, ONLY: repsilon

IMPLICIT NONE
REAL, INTENT(IN) :: t, p, rh
REAL :: esat, e

esat = 610.94 * EXP(17.625 * (t - 273.15) / (t - 273.15 + 243.04))
e    = rh * esat
spec_humid_from_rh = repsilon * e / (p - (1.0 - repsilon) * e)

END FUNCTION spec_humid_from_rh

END MODULE glomap_box_env_mod
