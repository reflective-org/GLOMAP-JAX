! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see LICENCE). Not part of UKCA.
! *****************************COPYRIGHT*******************************
!
! Description:
!   Box state for the gate-A binding, held apart from the wrapper.
!
!   This split exists because of a hard f2py limitation, not a design
!   preference. f2py's f90mod_rules tries to expose EVERY module-level variable
!   in a file it cracks, and a derived type has no C mapping, so
!
!       TYPE(box_env_type), SAVE :: env
!
!   aborts the build with `KeyError: 'void'` -- even though the variable never
!   crosses the boundary and nothing asked for it to be exposed. Declaring the
!   state in a module f2py is never handed, and reaching it by USE from the
!   free subroutines it does crack, sidesteps the problem completely.
!
!   Nothing but declarations lives here.
!
MODULE glomap_f2py_state

USE glomap_box_env_mod,   ONLY: box_env_type
USE glomap_box_state_mod, ONLY: box_state_type

IMPLICIT NONE
PUBLIC

! Matches the box driver. Multi-box is order 2 (task 2.6) and would relax the
! accessors' shape checks rather than rewrite them.
INTEGER, PARAMETER :: f2py_nbox = 1

TYPE(box_env_type),   SAVE :: env
TYPE(box_state_type), SAVE :: st

! ukca_mode_setup allocates under `IF (.NOT. ALLOCATED)` and never deallocates,
! and the 283 nmas* budget indices have no initialiser. A second init with a
! different setup therefore leaves stale indices, and since nbudaer also
! changes (8 vs 138 across the seven setups) a stale index can be out of
! bounds. These two let wrap_init refuse rather than corrupt memory; running
! several setups needs one process each, which is task 20b.
LOGICAL, SAVE :: is_initialised    = .FALSE.
INTEGER, SAVE :: initialised_setup = -1

! Set when a re-init is refused. read_box_namelist has to run before the setup
! is even knowable, so by the time wrap_init can refuse it has already
! overwritten every config scalar -- leaving the switches from the new namelist
! paired with the mode setup of the old one. Silently continuing from there
! would produce plausible numbers from a configuration that never existed, so
! every entry point refuses until the process is restarted.
LOGICAL, SAVE :: must_restart = .FALSE.

END MODULE glomap_f2py_state
