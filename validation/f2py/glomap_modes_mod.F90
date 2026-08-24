! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see fortran/LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Accessors for glomap_variables_type -- the mode and component tables
!   (phase C, task 24).
!
!   These are what `common_mode_setup_interface` builds from `i_mode_setup` and
!   the five density/hygroscopicity switches, and they are the input to every
!   process routine. Port them wrong and everything downstream is wrong in a way
!   no trajectory tolerance would attribute correctly, so phase C's acceptance
!   is **byte equality**, not `allclose`.
!
!   Typed dispatch rather than 26 subroutines, mirroring wrap_get_2d: the field
!   set is a table, and a table is better read by name than by generating one
!   entry point per row. Shapes are still explicit and checked -- see the header
!   of glomap_f2py_mod.F90 for why.
!
!   LOGICALs come back as INTEGER 0/1. The Fortran declares them with several
!   kinds across modules and f2py's mapping of LOGICAL is not worth relying on;
!   MERGE(1, 0, x) is kind-agnostic and unambiguous.
!
!   component_names is returned as one concatenated string rather than a
!   character array: f2py's handling of CHARACTER(LEN=n), DIMENSION(:) is
!   fragile, and splitting a fixed-width string in Python is not.
!
! ---------------------------------------------------------------------------
SUBROUTINE wrap_mode_real(field, n, out, ierr)
! Per-mode REAL tables, dimension(nmodes).
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
REAL(KIND=8),     INTENT(OUT) :: out(n)
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
IF (n /= nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('fracbcem'); out = glomap_variables%fracbcem
CASE ('fracocem'); out = glomap_variables%fracocem
CASE ('ddplim0');  out = glomap_variables%ddplim0
CASE ('ddpmid');   out = glomap_variables%ddpmid
CASE ('ddplim1');  out = glomap_variables%ddplim1
CASE ('mmid');     out = glomap_variables%mmid
CASE ('mlo');      out = glomap_variables%mlo
CASE ('mhi');      out = glomap_variables%mhi
CASE ('num_eps');  out = glomap_variables%num_eps
CASE ('sigmag');   out = glomap_variables%sigmag
CASE ('x');        out = glomap_variables%x
CASE DEFAULT;      ierr = 3
END SELECT

END SUBROUTINE wrap_mode_real

! ---------------------------------------------------------------------------
SUBROUTINE wrap_mode_int(field, n, out, ierr)
! Per-mode INTEGER and LOGICAL tables, dimension(nmodes).
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
INTEGER,          INTENT(OUT) :: out(n)
INTEGER,          INTENT(OUT) :: ierr

out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n /= nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('mode_choice'); out = glomap_variables%mode_choice
CASE ('modesol');     out = glomap_variables%modesol
CASE ('mode');        out = MERGE(1, 0, glomap_variables%mode)
CASE DEFAULT;         ierr = 3
END SELECT

END SUBROUTINE wrap_mode_int

! ---------------------------------------------------------------------------
SUBROUTINE wrap_cp_real(field, n, out, ierr)
! Per-component REAL tables, dimension(ncp). ncp is 6 in every supported setup,
! but it is a runtime scalar and is checked as one.
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
REAL(KIND=8),     INTENT(OUT) :: out(n)
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
IF (n /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('mm');      out = glomap_variables%mm
CASE ('rhocomp'); out = glomap_variables%rhocomp
CASE ('no_ions'); out = glomap_variables%no_ions
CASE DEFAULT;     ierr = 3
END SELECT

END SUBROUTINE wrap_cp_real

! ---------------------------------------------------------------------------
SUBROUTINE wrap_cp_int(field, n, out, ierr)
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n
INTEGER,          INTENT(OUT) :: out(n)
INTEGER,          INTENT(OUT) :: ierr

out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('component_choice'); out = glomap_variables%component_choice
CASE ('soluble_choice');   out = glomap_variables%soluble_choice
CASE ('soluble');          out = MERGE(1, 0, glomap_variables%soluble)
CASE DEFAULT;              ierr = 3
END SELECT

END SUBROUTINE wrap_cp_int

! ---------------------------------------------------------------------------
SUBROUTINE wrap_mode_cp_real(field, n1, n2, out, ierr)
! (nmodes, ncp) tables.
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
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
IF (n1 /= nmodes .OR. n2 /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('mfrac_0'); out = glomap_variables%mfrac_0
CASE DEFAULT;     ierr = 3
END SELECT

END SUBROUTINE wrap_mode_cp_real

! ---------------------------------------------------------------------------
SUBROUTINE wrap_mode_cp_int(field, n1, n2, out, ierr)
USE ukca_mode_setup,               ONLY: nmodes
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: field
INTEGER,          INTENT(IN)  :: n1, n2
INTEGER,          INTENT(OUT) :: out(n1,n2)
INTEGER,          INTENT(OUT) :: ierr

out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n1 /= nmodes .OR. n2 /= glomap_variables%ncp) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
SELECT CASE (TRIM(field))
CASE ('component_mode'); out = glomap_variables%component_mode
CASE ('component');      out = MERGE(1, 0, glomap_variables%component)
CASE DEFAULT;            ierr = 3
END SELECT

END SUBROUTINE wrap_mode_cp_int

! ---------------------------------------------------------------------------
SUBROUTINE wrap_component_names(n, names, ierr)
! One concatenated fixed-width string, split in Python. See the header.
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER,            INTENT(IN)  :: n
CHARACTER(LEN=140), INTENT(OUT) :: names
INTEGER,            INTENT(OUT) :: ierr
INTEGER :: icp

names = ''
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
! 140 = 7 chars x 20 components, comfortably above ncp_max.
IF (n /= glomap_variables%ncp .OR. 7*n > 140) THEN
  ierr = 2
  RETURN
END IF

DO icp = 1, n
  names(7*(icp-1)+1 : 7*icp) = glomap_variables%component_names(icp)
END DO
ierr = 0

END SUBROUTINE wrap_component_names

! ---------------------------------------------------------------------------
SUBROUTINE wrap_topmode(out, ierr)
USE ukca_config_specification_mod, ONLY: glomap_variables
USE glomap_f2py_state,             ONLY: is_initialised, must_restart
IMPLICIT NONE
INTEGER, INTENT(OUT) :: out, ierr
out = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
out  = glomap_variables%topmode
ierr = 0
END SUBROUTINE wrap_topmode
