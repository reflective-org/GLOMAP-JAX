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
!   THIS FILE covers the numerics primitives (task 21, feeding the compat layer
!   at task 34). They come first because they are consumed by remode,
!   volume_mode, the coagulation kernels and binapara alike, and because three
!   of them are known hazards where gfortran and XLA need not agree:
!
!     * ERF feeds `ukca_remode`'s FRAC_N, which is cut at exactly 0.5 --
!       i.e. at erf(x) = 0. A discrepancy there is a merge/no-merge flip, not
!       a rounding difference, so the grid below is dense around zero.
!
!     * cubrt_v is literally `x ** (1.0/3.0)`, NOT a cube root function. The
!       two are not the same computation and need not give the same bits, and
!       the constant 1.0/3.0 itself changes value under -fdefault-real-8.
!       Both forms are exposed so the port can be checked against the one the
!       Fortran actually performs.
!
!     * Fortran NINT rounds half AWAY FROM ZERO; numpy and jnp.round round half
!       to EVEN. `ukca_vapour.F90:226` computes `(NINT(wts/5))*5` with wts
!       clamped to [41, 99], so wts = 42.5, 47.5, ... land exactly on ties.
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
