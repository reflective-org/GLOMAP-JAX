! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Accessor for `coag_mode` -- the mode coagulation table (phase C, task 33).
!
!   `coag_mode(nmodes,nmodes)` is declared at ukca_mode_setup.F90:174 as an
!   INTEGER PARAMETER, and is read at exactly one place,
!   ukca_coagwithnucl.F90:534-535, where it says which mode receives the mass
!   leaving IMODE when IMODE coagulates with JMODE.
!
!   NOTE the deliberate absence of the `is_initialised` / `must_restart` guard
!   that every accessor in glomap_modes_mod.F90 carries. That is the point of
!   this file rather than a shortcut. The mode tables are built per process by
!   common_mode_setup_interface and are meaningless before init; `coag_mode` is
!   a compile-time constant and is not. Leaving the guard out lets
!   validation/capture_coag_mode.py read the table BEFORE any init at all and
!   then again after each i_mode_setup, in separate processes, and assert the
!   twelve results are identical -- which is the evidence for the claim that
!   this one table is setup-independent. With the guard in place that
!   comparison could not be made, and the claim would rest on grep.
!
!   Byte equality is the acceptance criterion, and these are INTEGERs, so
!   unlike every other golden in this repo this one IS bit-reproducible across
!   compilers and platforms: no floating point, no libm, no FMA contraction.
!   That does not exempt it from ADR-005's "generated once, committed, never in
!   CI" rule -- it just means a drift here would be a real defect rather than a
!   platform difference.
!
! ---------------------------------------------------------------------------
SUBROUTINE wrap_coag_nmodes(out)
! nmodes, readable without an init. A PARAMETER, like coag_mode itself, so the
! capture does not have to hard-code the extent it then checks.
USE ukca_mode_setup, ONLY: nmodes
IMPLICIT NONE
INTEGER, INTENT(OUT) :: out

out = nmodes

END SUBROUTINE wrap_coag_nmodes

! ---------------------------------------------------------------------------
SUBROUTINE wrap_coag_mode(n, out, ierr)
! The whole table. out(i,j) is coag_mode(i,j): the mode that receives mass
! leaving mode i when i coagulates with j.
!
! ierr: 0 ok, 2 wrong extent. There is no ierr=1/4 here because there is no
! init to be missing -- see the header.
USE ukca_mode_setup, ONLY: nmodes, coag_mode
IMPLICIT NONE
INTEGER, INTENT(IN)  :: n
INTEGER, INTENT(OUT) :: out(n,n)
INTEGER, INTENT(OUT) :: ierr

out = 0
IF (n /= nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
out = coag_mode

END SUBROUTINE wrap_coag_mode

! ---------------------------------------------------------------------------
SUBROUTINE wrap_coag_dest(imode, jmode, out, ierr)
! One entry, read the way ukca_coagwithnucl.F90:534 reads it -- with the two
! subscripts in the consumer's order rather than as a whole-array copy.
!
! Present because the whole-array path above cannot detect a transposed
! transcription: coag_mode is symmetric, so out and TRANSPOSE(out) are the same
! bytes. This gives the capture an independent, index-by-index read of the same
! constant. It still cannot break the symmetry -- nothing can -- but it does
! pin that the Python side and the Fortran side agree on which subscript is
! which for the asymmetric *shape* checks around it.
USE ukca_mode_setup, ONLY: nmodes, coag_mode
IMPLICIT NONE
INTEGER, INTENT(IN)  :: imode
INTEGER, INTENT(IN)  :: jmode
INTEGER, INTENT(OUT) :: out
INTEGER, INTENT(OUT) :: ierr

out = 0
IF (imode < 1 .OR. imode > nmodes .OR. jmode < 1 .OR. jmode > nmodes) THEN
  ierr = 2
  RETURN
END IF

ierr = 0
out = coag_mode(imode,jmode)

END SUBROUTINE wrap_coag_dest
