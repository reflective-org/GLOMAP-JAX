! *****************************COPYRIGHT*******************************
! (c) 2026. Standalone GLOMAP-mode box model driver.
! New code, BSD 3-Clause (see LICENCE).
! *****************************COPYRIGHT*******************************
!
! Description:
!   CSV time-series output and a terminal progress table for the box model.
!   Only ACTIVE modes and components appear in the output, so the column
!   set follows the chosen i_mode_setup.
!
MODULE glomap_box_output_mod

USE ukca_mode_setup, ONLY: nmodes

IMPLICIT NONE
PRIVATE

INTEGER, SAVE :: csv_unit = -1

CHARACTER(LEN=7), PARAMETER :: mode_name(nmodes) =                             &
  [ 'nucsol ', 'aitsol ', 'accsol ', 'corsol ',                                &
    'aitins ', 'accins ', 'corins ', 'supins ' ]

PUBLIC :: open_output, write_output, close_output
PUBLIC :: write_header_table, write_table_row

CONTAINS

! ---------------------------------------------------------------------------
SUBROUTINE open_output(filename)

USE ukca_config_specification_mod, ONLY: glomap_variables

IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN) :: filename
INTEGER :: imode, icp
CHARACTER(LEN=8192) :: hdr

OPEN(NEWUNIT=csv_unit, FILE=filename, STATUS='REPLACE', ACTION='WRITE')

hdr = 'time_s,time_h'
DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  hdr = TRIM(hdr)//',N_'//TRIM(mode_name(imode))//'_cm3'
  hdr = TRIM(hdr)//',Ddry_'//TRIM(mode_name(imode))//'_nm'
  hdr = TRIM(hdr)//',Dwet_'//TRIM(mode_name(imode))//'_nm'
  hdr = TRIM(hdr)//',rhop_'//TRIM(mode_name(imode))//'_kgm3'
END DO
DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  DO icp = 1, glomap_variables%ncp
    IF (.NOT. glomap_variables%component(imode,icp)) CYCLE
    hdr = TRIM(hdr)//',M_'//TRIM(mode_name(imode))//'_'//                      &
          TRIM(glomap_variables%component_names(icp))//'_ugm3'
  END DO
END DO
hdr = TRIM(hdr)//',H2SO4_cm3,SEC_ORG_cm3'

WRITE(csv_unit,'(A)') TRIM(hdr)

END SUBROUTINE open_output

! ---------------------------------------------------------------------------
SUBROUTINE write_output(time_s, st, env)
! One CSV row. Component mass is converted from molecules per particle to a
! mass concentration:  ug m-3 = md * nd * (mm/avogadro) * 1e6 cm3/m3 * 1e9 ug/kg

USE ukca_config_constants_mod,     ONLY: avogadro
USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: mh2so4, msec_org
USE glomap_box_state_mod,          ONLY: box_state_type
USE glomap_box_env_mod,            ONLY: box_env_type

IMPLICIT NONE
REAL,                 INTENT(IN) :: time_s
TYPE(box_state_type), INTENT(IN) :: st
TYPE(box_env_type),   INTENT(IN) :: env

INTEGER :: imode, icp
CHARACTER(LEN=16384) :: row
CHARACTER(LEN=32) :: fld
REAL :: mass_ugm3, h2so4_c, secorg_c

WRITE(row,'(ES14.6,",",ES14.6)') time_s, time_s / 3600.0

DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  WRITE(fld,'(",",ES14.6)') st%nd(1,imode)          ; row = TRIM(row)//fld
  WRITE(fld,'(",",ES14.6)') st%drydp(1,imode)*1.0e9 ; row = TRIM(row)//fld
  WRITE(fld,'(",",ES14.6)') st%wetdp(1,imode)*1.0e9 ; row = TRIM(row)//fld
  WRITE(fld,'(",",ES14.6)') st%rhopar(1,imode)      ; row = TRIM(row)//fld
END DO

DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  DO icp = 1, glomap_variables%ncp
    IF (.NOT. glomap_variables%component(imode,icp)) CYCLE
    mass_ugm3 = st%md(1,imode,icp) * st%nd(1,imode) *                          &
                (glomap_variables%mm(icp) / avogadro) * 1.0e6 * 1.0e9
    WRITE(fld,'(",",ES14.6)') mass_ugm3 ; row = TRIM(row)//fld
  END DO
END DO

h2so4_c  = 0.0
secorg_c = 0.0
IF (mh2so4  > 0) h2so4_c  = st%s0g(1,mh2so4)  / env%sm(1) * env%aird(1)
IF (msec_org > 0) secorg_c = st%s0g(1,msec_org) / env%sm(1) * env%aird(1)
WRITE(fld,'(",",ES14.6)') h2so4_c  ; row = TRIM(row)//fld
WRITE(fld,'(",",ES14.6)') secorg_c ; row = TRIM(row)//fld

WRITE(csv_unit,'(A)') TRIM(row)

END SUBROUTINE write_output

! ---------------------------------------------------------------------------
SUBROUTINE close_output()
IMPLICIT NONE
IF (csv_unit > 0) CLOSE(csv_unit)
csv_unit = -1
END SUBROUTINE close_output

! ---------------------------------------------------------------------------
SUBROUTINE write_header_table()
! Terminal summary table header: number concentration of each active mode
! plus the two condensable vapour concentrations.

USE ukca_config_specification_mod, ONLY: glomap_variables

IMPLICIT NONE
INTEGER :: imode
CHARACTER(LEN=256) :: line

WRITE(line,'(A8)') '   t(h)'
DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  line = TRIM(line)//'   N_'//TRIM(mode_name(imode))
END DO
line = TRIM(line)//'    H2SO4    SEC_ORG'
WRITE(*,'(A)') REPEAT('-', LEN_TRIM(line)+2)
WRITE(*,'(A)') TRIM(line)
WRITE(*,'(A)') '        '//'   (cm-3) ...                          (molecules cm-3)'
WRITE(*,'(A)') REPEAT('-', LEN_TRIM(line)+2)

END SUBROUTINE write_header_table

! ---------------------------------------------------------------------------
SUBROUTINE write_table_row(time_s, st, env)

USE ukca_config_specification_mod, ONLY: glomap_variables
USE ukca_setup_indices,            ONLY: mh2so4, msec_org
USE glomap_box_state_mod,          ONLY: box_state_type
USE glomap_box_env_mod,            ONLY: box_env_type

IMPLICIT NONE
REAL,                 INTENT(IN) :: time_s
TYPE(box_state_type), INTENT(IN) :: st
TYPE(box_env_type),   INTENT(IN) :: env
INTEGER :: imode
REAL    :: h2so4_c, secorg_c

WRITE(*,'(F8.2)',ADVANCE='NO') time_s / 3600.0
DO imode = 1, nmodes
  IF (.NOT. glomap_variables%mode(imode)) CYCLE
  WRITE(*,'(ES11.3)',ADVANCE='NO') st%nd(1,imode)
END DO
h2so4_c  = 0.0
secorg_c = 0.0
IF (mh2so4  > 0) h2so4_c  = st%s0g(1,mh2so4)  / env%sm(1) * env%aird(1)
IF (msec_org > 0) secorg_c = st%s0g(1,msec_org) / env%sm(1) * env%aird(1)
WRITE(*,'(2ES11.3)') h2so4_c, secorg_c

END SUBROUTINE write_table_row

END MODULE glomap_box_output_mod
