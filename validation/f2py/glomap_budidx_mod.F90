! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Accessors for the 283 nmas* aerosol mass-budget slot indices (task 32).
!
!   `ukca_setup_indices` declares 283 INTEGER scalars, each the second index
!   into `bud_aer_mas(nbox, 0:nbudaer)` for one (process, component, mode)
!   flux. Each `ukca_indices_*` routine assigns a different subset and
!   `nbudaer` itself takes seven distinct values across the seven supported
!   setups, so the map is a per-setup table -- and reading it out is the only
!   way to port it without retyping 283 numbers seven times.
!
!   FOUR ENTRY POINTS, NOT 283. One accessor per name would be 283 f2py
!   signatures, 283 call sites in the capture, and 283 chances to pair a name
!   with the wrong value. Instead the names come back as one concatenated
!   fixed-width blob and the values as one INTEGER array in the SAME order, so
!   the pairing is positional: it cannot be wrong for one name and right for
!   the other 282. `wrap_bud_index` looks a single name up by string and
!   exists so the capture can *prove* the two orders agree rather than assume
!   it.
!
!   GENERATED, NOT TYPED. The table below was extracted from the
!   `INTEGER :: nmas*` declarations of
!   `fortran/src/ukca/ukca_setup_indices.F90`, in declaration order.
!   `tests/test_budget_indices.py` re-parses both files and asserts they still
!   agree, so a vendored-tree update cannot leave this file quietly short a
!   name -- which would present as a shifted map, not as a missing one.
!
!   NO MODULE, and the table lives in exactly one subroutine. f2py's
!   f90mod_rules cracks every module in a file it is handed and tries to
!   expose its data; that is what forced glomap_f2py_state_mod.F90 out of the
!   f2py source list, and a CHARACTER(LEN=n) PARAMETER array is the same class
!   of thing. A local PARAMETER in `wrap_bud_names` is invisible to f2py, and
!   `wrap_bud_index` calls that routine rather than holding a second copy.
!
!   Sizes are explicit and checked, as in glomap_modes_mod.F90: an accessor
!   that quietly returns a short array is worse than one that refuses.
!
! ---------------------------------------------------------------------------
SUBROUTINE wrap_bud_count(o_nnames, o_namelen, ierr)
! The two sizes Python needs BEFORE it can size the output buffers, so unlike
! every other accessor here this one is not gated on initialisation: both are
! properties of the vendored source, not of the setup.
IMPLICIT NONE
INTEGER, INTENT(OUT) :: o_nnames, o_namelen, ierr

o_nnames  = 283
o_namelen = 20
ierr      = 0

END SUBROUTINE wrap_bud_count

! ---------------------------------------------------------------------------
SUBROUTINE wrap_bud_names(n, ln, blob, ierr)
! The 283 names as one concatenated fixed-width string, split in Python.
! Same reasoning as wrap_component_names: f2py's handling of
! CHARACTER(LEN=n), DIMENSION(:) is fragile, and slicing a fixed-width string
! in Python is not. Not gated on initialisation -- the names are a property of
! the source. Fixed width 20; the longest declared name is 17 characters.
IMPLICIT NONE
INTEGER,              INTENT(IN)  :: n, ln
CHARACTER(LEN=5660), INTENT(OUT) :: blob
INTEGER,              INTENT(OUT) :: ierr
INTEGER :: i

CHARACTER(LEN=20), PARAMETER :: bud_names(283) = [                              &
  'nmasprimsuaitsol    ', 'nmasprimsuaccsol    ', 'nmasprimsucorsol    ',     &
  'nmasprimssaccsol    ', 'nmasprimsscorsol    ', 'nmasprimbcaitsol    ',     &
  'nmasprimbcaitins    ', 'nmasprimocaitsol    ', 'nmasprimocaitins    ',     &
  'nmasddepsunucsol    ', 'nmasddepsuaitsol    ', 'nmasddepsuaccsol    ',     &
  'nmasddepsucorsol    ', 'nmasddepssaccsol    ', 'nmasddepsscorsol    ',     &
  'nmasddepbcaitsol    ', 'nmasddepbcaccsol    ', 'nmasddepbccorsol    ',     &
  'nmasddepbcaitins    ', 'nmasddepocnucsol    ', 'nmasddepocaitsol    ',     &
  'nmasddepocaccsol    ', 'nmasddepoccorsol    ', 'nmasddepocaitins    ',     &
  'nmasddepsonucsol    ', 'nmasddepsoaitsol    ', 'nmasddepsoaccsol    ',     &
  'nmasddepsocorsol    ', 'nmasnuscsunucsol    ', 'nmasnuscsuaitsol    ',     &
  'nmasnuscsuaccsol    ', 'nmasnuscsucorsol    ', 'nmasnuscssaccsol    ',     &
  'nmasnuscsscorsol    ', 'nmasnuscbcaitsol    ', 'nmasnuscbcaccsol    ',     &
  'nmasnuscbccorsol    ', 'nmasnuscbcaitins    ', 'nmasnuscocnucsol    ',     &
  'nmasnuscocaitsol    ', 'nmasnuscocaccsol    ', 'nmasnuscoccorsol    ',     &
  'nmasnuscocaitins    ', 'nmasnuscsonucsol    ', 'nmasnuscsoaitsol    ',     &
  'nmasnuscsoaccsol    ', 'nmasnuscsocorsol    ', 'nmasimscsunucsol    ',     &
  'nmasimscsuaitsol    ', 'nmasimscsuaccsol    ', 'nmasimscsucorsol    ',     &
  'nmasimscssaccsol    ', 'nmasimscsscorsol    ', 'nmasimscbcaitsol    ',     &
  'nmasimscbcaccsol    ', 'nmasimscbccorsol    ', 'nmasimscbcaitins    ',     &
  'nmasimscocnucsol    ', 'nmasimscocaitsol    ', 'nmasimscocaccsol    ',     &
  'nmasimscoccorsol    ', 'nmasimscocaitins    ', 'nmasimscsonucsol    ',     &
  'nmasimscsoaitsol    ', 'nmasimscsoaccsol    ', 'nmasimscsocorsol    ',     &
  'nmasclprsuaitsol1   ', 'nmasclprsuaccsol1   ', 'nmasclprsucorsol1   ',     &
  'nmasclprsuaitsol2   ', 'nmasclprsuaccsol2   ', 'nmasclprsucorsol2   ',     &
  'nmascondsunucsol    ', 'nmascondsuaitsol    ', 'nmascondsuaccsol    ',     &
  'nmascondsucorsol    ', 'nmascondsuaitins    ', 'nmasnuclsunucsol    ',     &
  'nmascondocnucsol    ', 'nmascondocaitsol    ', 'nmascondocaccsol    ',     &
  'nmascondoccorsol    ', 'nmascondocaitins    ', 'nmascondocinucsol   ',     &
  'nmascondociaitsol   ', 'nmascondociaccsol   ', 'nmascondocicorsol   ',     &
  'nmascondociaitins   ', 'nmascondociaccins   ', 'nmascondocicorins   ',     &
  'nmascondsonucsol    ', 'nmascondsoaitsol    ', 'nmascondsoaccsol    ',     &
  'nmascondsocorsol    ', 'nmascondsoaitins    ', 'nmascoagsuintr12    ',     &
  'nmascoagsuintr13    ', 'nmascoagsuintr14    ', 'nmascoagsuintr15    ',     &
  'nmascoagocintr12    ', 'nmascoagocintr13    ', 'nmascoagocintr14    ',     &
  'nmascoagocintr15    ', 'nmascoagsointr12    ', 'nmascoagsointr13    ',     &
  'nmascoagsointr14    ', 'nmascoagsointr15    ', 'nmascoagsuintr23    ',     &
  'nmascoagbcintr23    ', 'nmascoagocintr23    ', 'nmascoagsointr23    ',     &
  'nmascoagsuintr24    ', 'nmascoagbcintr24    ', 'nmascoagocintr24    ',     &
  'nmascoagsointr24    ', 'nmascoagsuintr34    ', 'nmascoagbcintr34    ',     &
  'nmascoagocintr34    ', 'nmascoagssintr34    ', 'nmascoagsointr34    ',     &
  'nmascoagbcintr53    ', 'nmascoagocintr53    ', 'nmascoagbcintr54    ',     &
  'nmascoagocintr54    ', 'nmasagedsuintr52    ', 'nmasagedbcintr52    ',     &
  'nmasagedocintr52    ', 'nmasagedsointr52    ', 'nmasmergsuintr12    ',     &
  'nmasmergocintr12    ', 'nmasmergsointr12    ', 'nmasmergsuintr23    ',     &
  'nmasmergbcintr23    ', 'nmasmergocintr23    ', 'nmasmergsointr23    ',     &
  'nmasmergsuintr34    ', 'nmasmergssintr34    ', 'nmasmergbcintr34    ',     &
  'nmasmergocintr34    ', 'nmasmergsointr34    ', 'nmasprocsuintr23    ',     &
  'nmasprocbcintr23    ', 'nmasprococintr23    ', 'nmasprocsointr23    ',     &
  'nmasprimduaccsol    ', 'nmasprimducorsol    ', 'nmasprimduaccins    ',     &
  'nmasprimducorins    ', 'nmasprimdusupins    ', 'nmasddepduaccsol    ',     &
  'nmasddepducorsol    ', 'nmasddepduaccins    ', 'nmasddepducorins    ',     &
  'nmasddepdusupins    ', 'nmasnuscduaccsol    ', 'nmasnuscducorsol    ',     &
  'nmasnuscduaccins    ', 'nmasnuscducorins    ', 'nmasnuscdusupins    ',     &
  'nmasimscduaccsol    ', 'nmasimscducorsol    ', 'nmasimscduaccins    ',     &
  'nmasimscducorins    ', 'nmasimscdusupins    ', 'nmascondsuaccins    ',     &
  'nmascondsucorins    ', 'nmascondsusupins    ', 'nmascondocaccins    ',     &
  'nmascondoccorins    ', 'nmascondocsupins    ', 'nmascondsoaccins    ',     &
  'nmascondsocorins    ', 'nmascondsosupins    ', 'nmascoagsuintr16    ',     &
  'nmascoagsuintr17    ', 'nmascoagsuintr18    ', 'nmascoagocintr16    ',     &
  'nmascoagocintr17    ', 'nmascoagocintr18    ', 'nmascoagsointr16    ',     &
  'nmascoagsointr17    ', 'nmascoagsointr18    ', 'nmascoagduintr34    ',     &
  'nmascoagduintr64    ', 'nmasagedsuintr63    ', 'nmasagedduintr63    ',     &
  'nmasagedocintr63    ', 'nmasagedsointr63    ', 'nmasagedsuintr74    ',     &
  'nmasagedduintr74    ', 'nmasagedocintr74    ', 'nmasagedsointr74    ',     &
  'nmasagedsuintr84    ', 'nmasagedduintr84    ', 'nmasagedocintr84    ',     &
  'nmasagedsointr84    ', 'nmasmergduintr34    ', 'nmasprimntnucsol    ',     &
  'nmasprimntaitsol    ', 'nmasprimntaccsol    ', 'nmasprimntcorsol    ',     &
  'nmasprimnhnucsol    ', 'nmasprimnhaitsol    ', 'nmasprimnhaccsol    ',     &
  'nmasprimnhcorsol    ', 'nmascondnnaccsol    ', 'nmascondnncorsol    ',     &
  'nmasddepntaitsol    ', 'nmasddepntaccsol    ', 'nmasddepntcorsol    ',     &
  'nmasddepnhaitsol    ', 'nmasddepnhaccsol    ', 'nmasddepnhcorsol    ',     &
  'nmasddepnnaccsol    ', 'nmasddepnncorsol    ', 'nmasnuscntaitsol    ',     &
  'nmasnuscntaccsol    ', 'nmasnuscntcorsol    ', 'nmasnuscnhaitsol    ',     &
  'nmasnuscnhaccsol    ', 'nmasnuscnhcorsol    ', 'nmasnuscnnaccsol    ',     &
  'nmasnuscnncorsol    ', 'nmasimscntaitsol    ', 'nmasimscntaccsol    ',     &
  'nmasimscntcorsol    ', 'nmasimscnhaitsol    ', 'nmasimscnhaccsol    ',     &
  'nmasimscnhcorsol    ', 'nmasimscnnaccsol    ', 'nmasimscnncorsol    ',     &
  'nmascoagntintr23    ', 'nmascoagnhintr23    ', 'nmascoagntintr24    ',     &
  'nmascoagnhintr24    ', 'nmascoagntintr34    ', 'nmascoagnhintr34    ',     &
  'nmascoagnnintr34    ', 'nmasmergntintr23    ', 'nmasmergnhintr23    ',     &
  'nmasmergntintr34    ', 'nmasmergnhintr34    ', 'nmasmergnnintr34    ',     &
  'nmasprocntintr23    ', 'nmasprocnhintr23    ', 'nmasprimmpaitins    ',     &
  'nmasprimmpaccins    ', 'nmasprimmpcorins    ', 'nmasprimmpsupins    ',     &
  'nmasddepmpaitins    ', 'nmasddepmpaccins    ', 'nmasddepmpcorins    ',     &
  'nmasddepmpaitsol    ', 'nmasddepmpaccsol    ', 'nmasddepmpcorsol    ',     &
  'nmasddepmpsupins    ', 'nmasnuscmpaitins    ', 'nmasnuscmpaccins    ',     &
  'nmasnuscmpcorins    ', 'nmasnuscmpaitsol    ', 'nmasnuscmpaccsol    ',     &
  'nmasnuscmpcorsol    ', 'nmasnuscmpsupins    ', 'nmasimscmpaitins    ',     &
  'nmasimscmpaccins    ', 'nmasimscmpcorins    ', 'nmasimscmpaitsol    ',     &
  'nmasimscmpaccsol    ', 'nmasimscmpcorsol    ', 'nmasimscmpsupins    ',     &
  'nmasprocmpintr23    ', 'nmascoagmpintr23    ', 'nmascoagmpintr24    ',     &
  'nmascoagmpintr34    ', 'nmascoagmpintr53    ', 'nmascoagmpintr54    ',     &
  'nmascoagmpintr64    ', 'nmasagedmpintr52    ', 'nmasagedmpintr63    ',     &
  'nmasagedmpintr74    ', 'nmasagedmpintr84    ', 'nmasmergmpintr23    ',     &
  'nmasmergmpintr34    ' ]

blob = ''
IF (n /= 283 .OR. ln /= 20) THEN
  ierr = 2
  RETURN
END IF

DO i = 1, n
  blob(ln*(i-1)+1 : ln*i) = bud_names(i)
END DO
ierr = 0

END SUBROUTINE wrap_bud_names

! ---------------------------------------------------------------------------
SUBROUTINE wrap_bud_values(n, out, o_nbudaer, ierr)
! The 283 slot indices for the setup this process initialised, in the same
! order as wrap_bud_names, plus nbudaer so the caller can bound-check them
! without a second call.
!
! Values are 1-based Fortran slot numbers into bud_aer_mas(nbox, 0:nbudaer).
! ZERO MEANS "this flux is not carried in this setup" -- it does not mean slot
! 0. Every one of the 344 writes in the vendored tree is guarded by
! `IF (nmasxxx > 0)`, so slot 0 is allocated and never written.
!
! Note that each ukca_indices_* routine assigns only 245 of the 283: the 38
! nmas*mp* names are assigned by ukca_indices_sussbcocdump_8mode alone, and in
! every supported setup they are read (34 of them appear in a live
! `IF (nmasxxx > 0)` guard) without ever having been assigned. Module scalars
! have static storage, so gfortran gives them a .bss zero and the guard is
! false -- but that is the compiler's answer, not the standard's. The capture
! records what this build actually returns rather than assuming it.
USE ukca_setup_indices,  ONLY: nbudaer
USE glomap_f2py_state,   ONLY: is_initialised, must_restart
USE ukca_setup_indices, ONLY:                                                 &
    nmasprimsuaitsol, nmasprimsuaccsol, nmasprimsucorsol,                     &
    nmasprimssaccsol, nmasprimsscorsol, nmasprimbcaitsol,                     &
    nmasprimbcaitins, nmasprimocaitsol, nmasprimocaitins,                     &
    nmasddepsunucsol, nmasddepsuaitsol, nmasddepsuaccsol,                     &
    nmasddepsucorsol, nmasddepssaccsol, nmasddepsscorsol,                     &
    nmasddepbcaitsol, nmasddepbcaccsol, nmasddepbccorsol,                     &
    nmasddepbcaitins, nmasddepocnucsol, nmasddepocaitsol,                     &
    nmasddepocaccsol, nmasddepoccorsol, nmasddepocaitins,                     &
    nmasddepsonucsol, nmasddepsoaitsol, nmasddepsoaccsol,                     &
    nmasddepsocorsol, nmasnuscsunucsol, nmasnuscsuaitsol,                     &
    nmasnuscsuaccsol, nmasnuscsucorsol, nmasnuscssaccsol,                     &
    nmasnuscsscorsol, nmasnuscbcaitsol, nmasnuscbcaccsol,                     &
    nmasnuscbccorsol, nmasnuscbcaitins, nmasnuscocnucsol,                     &
    nmasnuscocaitsol, nmasnuscocaccsol, nmasnuscoccorsol,                     &
    nmasnuscocaitins, nmasnuscsonucsol, nmasnuscsoaitsol,                     &
    nmasnuscsoaccsol, nmasnuscsocorsol, nmasimscsunucsol,                     &
    nmasimscsuaitsol, nmasimscsuaccsol, nmasimscsucorsol,                     &
    nmasimscssaccsol, nmasimscsscorsol, nmasimscbcaitsol,                     &
    nmasimscbcaccsol, nmasimscbccorsol, nmasimscbcaitins,                     &
    nmasimscocnucsol, nmasimscocaitsol, nmasimscocaccsol,                     &
    nmasimscoccorsol, nmasimscocaitins, nmasimscsonucsol,                     &
    nmasimscsoaitsol, nmasimscsoaccsol, nmasimscsocorsol,                     &
    nmasclprsuaitsol1, nmasclprsuaccsol1, nmasclprsucorsol1,                  &
    nmasclprsuaitsol2, nmasclprsuaccsol2, nmasclprsucorsol2,                  &
    nmascondsunucsol, nmascondsuaitsol, nmascondsuaccsol,                     &
    nmascondsucorsol, nmascondsuaitins, nmasnuclsunucsol,                     &
    nmascondocnucsol, nmascondocaitsol, nmascondocaccsol,                     &
    nmascondoccorsol, nmascondocaitins, nmascondocinucsol,                    &
    nmascondociaitsol, nmascondociaccsol, nmascondocicorsol,                  &
    nmascondociaitins, nmascondociaccins, nmascondocicorins,                  &
    nmascondsonucsol, nmascondsoaitsol, nmascondsoaccsol,                     &
    nmascondsocorsol, nmascondsoaitins, nmascoagsuintr12,                     &
    nmascoagsuintr13, nmascoagsuintr14, nmascoagsuintr15,                     &
    nmascoagocintr12, nmascoagocintr13, nmascoagocintr14,                     &
    nmascoagocintr15, nmascoagsointr12, nmascoagsointr13,                     &
    nmascoagsointr14, nmascoagsointr15, nmascoagsuintr23,                     &
    nmascoagbcintr23, nmascoagocintr23, nmascoagsointr23,                     &
    nmascoagsuintr24, nmascoagbcintr24, nmascoagocintr24,                     &
    nmascoagsointr24, nmascoagsuintr34, nmascoagbcintr34,                     &
    nmascoagocintr34, nmascoagssintr34, nmascoagsointr34,                     &
    nmascoagbcintr53, nmascoagocintr53, nmascoagbcintr54,                     &
    nmascoagocintr54, nmasagedsuintr52, nmasagedbcintr52,                     &
    nmasagedocintr52, nmasagedsointr52, nmasmergsuintr12,                     &
    nmasmergocintr12, nmasmergsointr12, nmasmergsuintr23,                     &
    nmasmergbcintr23, nmasmergocintr23, nmasmergsointr23,                     &
    nmasmergsuintr34, nmasmergssintr34, nmasmergbcintr34,                     &
    nmasmergocintr34, nmasmergsointr34, nmasprocsuintr23,                     &
    nmasprocbcintr23, nmasprococintr23, nmasprocsointr23,                     &
    nmasprimduaccsol, nmasprimducorsol, nmasprimduaccins,                     &
    nmasprimducorins, nmasprimdusupins, nmasddepduaccsol,                     &
    nmasddepducorsol, nmasddepduaccins, nmasddepducorins,                     &
    nmasddepdusupins, nmasnuscduaccsol, nmasnuscducorsol,                     &
    nmasnuscduaccins, nmasnuscducorins, nmasnuscdusupins,                     &
    nmasimscduaccsol, nmasimscducorsol, nmasimscduaccins,                     &
    nmasimscducorins, nmasimscdusupins, nmascondsuaccins,                     &
    nmascondsucorins, nmascondsusupins, nmascondocaccins,                     &
    nmascondoccorins, nmascondocsupins, nmascondsoaccins,                     &
    nmascondsocorins, nmascondsosupins, nmascoagsuintr16,                     &
    nmascoagsuintr17, nmascoagsuintr18, nmascoagocintr16,                     &
    nmascoagocintr17, nmascoagocintr18, nmascoagsointr16,                     &
    nmascoagsointr17, nmascoagsointr18, nmascoagduintr34,                     &
    nmascoagduintr64, nmasagedsuintr63, nmasagedduintr63,                     &
    nmasagedocintr63, nmasagedsointr63, nmasagedsuintr74,                     &
    nmasagedduintr74, nmasagedocintr74, nmasagedsointr74,                     &
    nmasagedsuintr84, nmasagedduintr84, nmasagedocintr84,                     &
    nmasagedsointr84, nmasmergduintr34, nmasprimntnucsol,                     &
    nmasprimntaitsol, nmasprimntaccsol, nmasprimntcorsol,                     &
    nmasprimnhnucsol, nmasprimnhaitsol, nmasprimnhaccsol,                     &
    nmasprimnhcorsol, nmascondnnaccsol, nmascondnncorsol,                     &
    nmasddepntaitsol, nmasddepntaccsol, nmasddepntcorsol,                     &
    nmasddepnhaitsol, nmasddepnhaccsol, nmasddepnhcorsol,                     &
    nmasddepnnaccsol, nmasddepnncorsol, nmasnuscntaitsol,                     &
    nmasnuscntaccsol, nmasnuscntcorsol, nmasnuscnhaitsol,                     &
    nmasnuscnhaccsol, nmasnuscnhcorsol, nmasnuscnnaccsol,                     &
    nmasnuscnncorsol, nmasimscntaitsol, nmasimscntaccsol,                     &
    nmasimscntcorsol, nmasimscnhaitsol, nmasimscnhaccsol,                     &
    nmasimscnhcorsol, nmasimscnnaccsol, nmasimscnncorsol,                     &
    nmascoagntintr23, nmascoagnhintr23, nmascoagntintr24,                     &
    nmascoagnhintr24, nmascoagntintr34, nmascoagnhintr34,                     &
    nmascoagnnintr34, nmasmergntintr23, nmasmergnhintr23,                     &
    nmasmergntintr34, nmasmergnhintr34, nmasmergnnintr34,                     &
    nmasprocntintr23, nmasprocnhintr23, nmasprimmpaitins,                     &
    nmasprimmpaccins, nmasprimmpcorins, nmasprimmpsupins,                     &
    nmasddepmpaitins, nmasddepmpaccins, nmasddepmpcorins,                     &
    nmasddepmpaitsol, nmasddepmpaccsol, nmasddepmpcorsol,                     &
    nmasddepmpsupins, nmasnuscmpaitins, nmasnuscmpaccins,                     &
    nmasnuscmpcorins, nmasnuscmpaitsol, nmasnuscmpaccsol,                     &
    nmasnuscmpcorsol, nmasnuscmpsupins, nmasimscmpaitins,                     &
    nmasimscmpaccins, nmasimscmpcorins, nmasimscmpaitsol,                     &
    nmasimscmpaccsol, nmasimscmpcorsol, nmasimscmpsupins,                     &
    nmasprocmpintr23, nmascoagmpintr23, nmascoagmpintr24,                     &
    nmascoagmpintr34, nmascoagmpintr53, nmascoagmpintr54,                     &
    nmascoagmpintr64, nmasagedmpintr52, nmasagedmpintr63,                     &
    nmasagedmpintr74, nmasagedmpintr84, nmasmergmpintr23,                     &
    nmasmergmpintr34
IMPLICIT NONE
INTEGER, INTENT(IN)  :: n
INTEGER, INTENT(OUT) :: out(n)
INTEGER, INTENT(OUT) :: o_nbudaer
INTEGER, INTENT(OUT) :: ierr

out = 0
o_nbudaer = 0
IF (must_restart) THEN
  ierr = 1
  RETURN
END IF
IF (.NOT. is_initialised) THEN
  ierr = 4
  RETURN
END IF
IF (n /= 283) THEN
  ierr = 2
  RETURN
END IF

out = [                                                                        &
  nmasprimsuaitsol, nmasprimsuaccsol, nmasprimsucorsol, nmasprimssaccsol,     &
  nmasprimsscorsol, nmasprimbcaitsol, nmasprimbcaitins, nmasprimocaitsol,     &
  nmasprimocaitins, nmasddepsunucsol, nmasddepsuaitsol, nmasddepsuaccsol,     &
  nmasddepsucorsol, nmasddepssaccsol, nmasddepsscorsol, nmasddepbcaitsol,     &
  nmasddepbcaccsol, nmasddepbccorsol, nmasddepbcaitins, nmasddepocnucsol,     &
  nmasddepocaitsol, nmasddepocaccsol, nmasddepoccorsol, nmasddepocaitins,     &
  nmasddepsonucsol, nmasddepsoaitsol, nmasddepsoaccsol, nmasddepsocorsol,     &
  nmasnuscsunucsol, nmasnuscsuaitsol, nmasnuscsuaccsol, nmasnuscsucorsol,     &
  nmasnuscssaccsol, nmasnuscsscorsol, nmasnuscbcaitsol, nmasnuscbcaccsol,     &
  nmasnuscbccorsol, nmasnuscbcaitins, nmasnuscocnucsol, nmasnuscocaitsol,     &
  nmasnuscocaccsol, nmasnuscoccorsol, nmasnuscocaitins, nmasnuscsonucsol,     &
  nmasnuscsoaitsol, nmasnuscsoaccsol, nmasnuscsocorsol, nmasimscsunucsol,     &
  nmasimscsuaitsol, nmasimscsuaccsol, nmasimscsucorsol, nmasimscssaccsol,     &
  nmasimscsscorsol, nmasimscbcaitsol, nmasimscbcaccsol, nmasimscbccorsol,     &
  nmasimscbcaitins, nmasimscocnucsol, nmasimscocaitsol, nmasimscocaccsol,     &
  nmasimscoccorsol, nmasimscocaitins, nmasimscsonucsol, nmasimscsoaitsol,     &
  nmasimscsoaccsol, nmasimscsocorsol, nmasclprsuaitsol1, nmasclprsuaccsol1,   &
  nmasclprsucorsol1, nmasclprsuaitsol2, nmasclprsuaccsol2, nmasclprsucorsol2, &
  nmascondsunucsol, nmascondsuaitsol, nmascondsuaccsol, nmascondsucorsol,     &
  nmascondsuaitins, nmasnuclsunucsol, nmascondocnucsol, nmascondocaitsol,     &
  nmascondocaccsol, nmascondoccorsol, nmascondocaitins, nmascondocinucsol,    &
  nmascondociaitsol, nmascondociaccsol, nmascondocicorsol, nmascondociaitins, &
  nmascondociaccins, nmascondocicorins, nmascondsonucsol, nmascondsoaitsol,   &
  nmascondsoaccsol, nmascondsocorsol, nmascondsoaitins, nmascoagsuintr12,     &
  nmascoagsuintr13, nmascoagsuintr14, nmascoagsuintr15, nmascoagocintr12,     &
  nmascoagocintr13, nmascoagocintr14, nmascoagocintr15, nmascoagsointr12,     &
  nmascoagsointr13, nmascoagsointr14, nmascoagsointr15, nmascoagsuintr23,     &
  nmascoagbcintr23, nmascoagocintr23, nmascoagsointr23, nmascoagsuintr24,     &
  nmascoagbcintr24, nmascoagocintr24, nmascoagsointr24, nmascoagsuintr34,     &
  nmascoagbcintr34, nmascoagocintr34, nmascoagssintr34, nmascoagsointr34,     &
  nmascoagbcintr53, nmascoagocintr53, nmascoagbcintr54, nmascoagocintr54,     &
  nmasagedsuintr52, nmasagedbcintr52, nmasagedocintr52, nmasagedsointr52,     &
  nmasmergsuintr12, nmasmergocintr12, nmasmergsointr12, nmasmergsuintr23,     &
  nmasmergbcintr23, nmasmergocintr23, nmasmergsointr23, nmasmergsuintr34,     &
  nmasmergssintr34, nmasmergbcintr34, nmasmergocintr34, nmasmergsointr34,     &
  nmasprocsuintr23, nmasprocbcintr23, nmasprococintr23, nmasprocsointr23,     &
  nmasprimduaccsol, nmasprimducorsol, nmasprimduaccins, nmasprimducorins,     &
  nmasprimdusupins, nmasddepduaccsol, nmasddepducorsol, nmasddepduaccins,     &
  nmasddepducorins, nmasddepdusupins, nmasnuscduaccsol, nmasnuscducorsol,     &
  nmasnuscduaccins, nmasnuscducorins, nmasnuscdusupins, nmasimscduaccsol,     &
  nmasimscducorsol, nmasimscduaccins, nmasimscducorins, nmasimscdusupins,     &
  nmascondsuaccins, nmascondsucorins, nmascondsusupins, nmascondocaccins,     &
  nmascondoccorins, nmascondocsupins, nmascondsoaccins, nmascondsocorins,     &
  nmascondsosupins, nmascoagsuintr16, nmascoagsuintr17, nmascoagsuintr18,     &
  nmascoagocintr16, nmascoagocintr17, nmascoagocintr18, nmascoagsointr16,     &
  nmascoagsointr17, nmascoagsointr18, nmascoagduintr34, nmascoagduintr64,     &
  nmasagedsuintr63, nmasagedduintr63, nmasagedocintr63, nmasagedsointr63,     &
  nmasagedsuintr74, nmasagedduintr74, nmasagedocintr74, nmasagedsointr74,     &
  nmasagedsuintr84, nmasagedduintr84, nmasagedocintr84, nmasagedsointr84,     &
  nmasmergduintr34, nmasprimntnucsol, nmasprimntaitsol, nmasprimntaccsol,     &
  nmasprimntcorsol, nmasprimnhnucsol, nmasprimnhaitsol, nmasprimnhaccsol,     &
  nmasprimnhcorsol, nmascondnnaccsol, nmascondnncorsol, nmasddepntaitsol,     &
  nmasddepntaccsol, nmasddepntcorsol, nmasddepnhaitsol, nmasddepnhaccsol,     &
  nmasddepnhcorsol, nmasddepnnaccsol, nmasddepnncorsol, nmasnuscntaitsol,     &
  nmasnuscntaccsol, nmasnuscntcorsol, nmasnuscnhaitsol, nmasnuscnhaccsol,     &
  nmasnuscnhcorsol, nmasnuscnnaccsol, nmasnuscnncorsol, nmasimscntaitsol,     &
  nmasimscntaccsol, nmasimscntcorsol, nmasimscnhaitsol, nmasimscnhaccsol,     &
  nmasimscnhcorsol, nmasimscnnaccsol, nmasimscnncorsol, nmascoagntintr23,     &
  nmascoagnhintr23, nmascoagntintr24, nmascoagnhintr24, nmascoagntintr34,     &
  nmascoagnhintr34, nmascoagnnintr34, nmasmergntintr23, nmasmergnhintr23,     &
  nmasmergntintr34, nmasmergnhintr34, nmasmergnnintr34, nmasprocntintr23,     &
  nmasprocnhintr23, nmasprimmpaitins, nmasprimmpaccins, nmasprimmpcorins,     &
  nmasprimmpsupins, nmasddepmpaitins, nmasddepmpaccins, nmasddepmpcorins,     &
  nmasddepmpaitsol, nmasddepmpaccsol, nmasddepmpcorsol, nmasddepmpsupins,     &
  nmasnuscmpaitins, nmasnuscmpaccins, nmasnuscmpcorins, nmasnuscmpaitsol,     &
  nmasnuscmpaccsol, nmasnuscmpcorsol, nmasnuscmpsupins, nmasimscmpaitins,     &
  nmasimscmpaccins, nmasimscmpcorins, nmasimscmpaitsol, nmasimscmpaccsol,     &
  nmasimscmpcorsol, nmasimscmpsupins, nmasprocmpintr23, nmascoagmpintr23,     &
  nmascoagmpintr24, nmascoagmpintr34, nmascoagmpintr53, nmascoagmpintr54,     &
  nmascoagmpintr64, nmasagedmpintr52, nmasagedmpintr63, nmasagedmpintr74,     &
  nmasagedmpintr84, nmasmergmpintr23, nmasmergmpintr34 ]
o_nbudaer = nbudaer
ierr = 0

END SUBROUTINE wrap_bud_values

! ---------------------------------------------------------------------------
SUBROUTINE wrap_bud_index(name, o_value, o_pos, ierr)
! One name looked up by string, returning its slot index and its 1-based
! position in the blob. ierr = 3 for a name that is not in the table, so a
! typo is an error rather than a zero that reads as "not carried here".
IMPLICIT NONE
CHARACTER(LEN=*), INTENT(IN)  :: name
INTEGER,          INTENT(OUT) :: o_value, o_pos, ierr
INTEGER, PARAMETER :: nnames = 283
INTEGER, PARAMETER :: namelen = 20
CHARACTER(LEN=nnames*namelen) :: blob
INTEGER :: i, values(nnames), nbudaer_out, err

o_value = 0
o_pos   = 0

CALL wrap_bud_names(nnames, namelen, blob, err)
IF (err /= 0) THEN
  ierr = err
  RETURN
END IF

DO i = 1, nnames
  IF (TRIM(blob(namelen*(i-1)+1 : namelen*i)) == TRIM(name)) THEN
    o_pos = i
    EXIT
  END IF
END DO
IF (o_pos == 0) THEN
  ierr = 3
  RETURN
END IF

! Gating (must_restart, is_initialised) lives in wrap_bud_values; reporting a
! position for an uninitialised process would be harmless but reporting a
! value would not, so the value comes from the gated routine.
CALL wrap_bud_values(nnames, values, nbudaer_out, err)
IF (err /= 0) THEN
  o_pos = 0
  ierr  = err
  RETURN
END IF
o_value = values(o_pos)
ierr = 0

END SUBROUTINE wrap_bud_index
