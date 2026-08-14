! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   A drop-in replacement for `ereport_mod`, linked into the gate-A extension
!   module ONLY.
!
!   THE PROBLEM. The real `ereport` handles a fatal error with `STOP 1`. In a
!   compiled executable that is correct and desirable -- it is what
!   `fortran/patches/0002` exists to ensure, and the reference build keeps it.
!   Inside a Python extension it terminates the interpreter, with no traceback
!   and no chance to report which of twenty reachable call sites fired. A test
!   that drives a routine towards an error path therefore takes the whole test
!   session with it.
!
!   THE SUBSTITUTION. This module presents the same name, the same public
!   subroutine and the same signature, so every already-compiled caller links
!   against it unchanged. Instead of stopping it records the call and returns.
!   The caller then continues with whatever it would have done next -- which is
!   NOT what the reference does, and is the point: it lets Python see the error
!   rather than inherit the exit.
!
!   THIS IS A DELIBERATE DIVERGENCE, and a narrow one. It applies only to the
!   f2py extension; `validation/build_reference.sh` never sees it, so no golden
!   and no committed number is affected. Documented in `docs/harness.md`.
!
!   READ THE FLAG. Because a caller continues past a fatal error, anything it
!   computes afterwards is meaningless. Gate-A drivers must check
!   `wrap_ereport_count()` after every call and discard the result if it is
!   non-zero. Treating a recorded fatal as a warning would be worse than the
!   `STOP` it replaces, because it looks like a number.
!
!   Warnings (`error_status < 0`) and info (`== 0`) are recorded too, and
!   counted separately, since the real routine already returns for those and
!   they are useful signal rather than a failure.
!
MODULE ereport_mod

IMPLICIT NONE
PRIVATE

INTEGER, PARAMETER :: msg_len = 256

INTEGER, SAVE :: n_fatal   = 0
INTEGER, SAVE :: n_warning = 0
INTEGER, SAVE :: n_info    = 0
INTEGER, SAVE :: last_status = 0
CHARACTER(LEN=msg_len), SAVE :: last_routine = ''
CHARACTER(LEN=msg_len), SAVE :: last_message = ''

PUBLIC :: ereport
PUBLIC :: ereport_shim_counts, ereport_shim_last, ereport_shim_reset

CONTAINS

SUBROUTINE ereport(routine_name, error_status, message)
! Same signature as ukca/ereport_mod.F90, including INTENT(IN OUT) on
! error_status and the reset to 0 on exit -- callers rely on both.
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)     :: routine_name
INTEGER,          INTENT(IN OUT) :: error_status
CHARACTER(LEN=*), INTENT(IN)     :: message

IF (error_status > 0) THEN
  n_fatal = n_fatal + 1
  WRITE(*,'(A,A,A,A)') 'UKCA ERROR (shim, not fatal) in ',                     &
                       TRIM(routine_name), ': ', TRIM(message)
ELSE IF (error_status < 0) THEN
  n_warning = n_warning + 1
  WRITE(*,'(A,A,A,A)') 'UKCA WARNING in ',                                     &
                       TRIM(routine_name), ': ', TRIM(message)
ELSE
  n_info = n_info + 1
  WRITE(*,'(A,A,A,A)') 'UKCA INFO in ',                                        &
                       TRIM(routine_name), ': ', TRIM(message)
END IF

last_status  = error_status
last_routine = routine_name
last_message = message

! The real routine resets error_status before returning; matching that keeps
! callers that re-test it behaving identically.
error_status = 0

END SUBROUTINE ereport

SUBROUTINE ereport_shim_counts(fatal, warning, info)
IMPLICIT NONE
INTEGER, INTENT(OUT) :: fatal, warning, info
fatal   = n_fatal
warning = n_warning
info    = n_info
END SUBROUTINE ereport_shim_counts

SUBROUTINE ereport_shim_last(status, routine, message)
IMPLICIT NONE
INTEGER,                INTENT(OUT) :: status
CHARACTER(LEN=msg_len), INTENT(OUT) :: routine, message
status  = last_status
routine = last_routine
message = last_message
END SUBROUTINE ereport_shim_last

SUBROUTINE ereport_shim_reset()
IMPLICIT NONE
n_fatal      = 0
n_warning    = 0
n_info       = 0
last_status  = 0
last_routine = ''
last_message = ''
END SUBROUTINE ereport_shim_reset

END MODULE ereport_mod
