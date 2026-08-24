! *****************************COPYRIGHT*******************************
! (c) 2026. Validation instrumentation for the GLOMAP-JAX port.
! New code, BSD 3-Clause (see fortran/LICENCE). Not part of UKCA.
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

! Every namelist variable init_ukca_for_box consumes, recorded at the init that
! actually ran. Keying only on i_mode_setup was not enough: the phase B review
! showed that changing l_fix_nacl_density between two wrap_init calls in one
! process was silently ignored -- common_mode_setup_interface was never
! re-called, glomap_variables kept the first namelist's densities, and drydp
! came out 2.3e-5 wrong with ierr = 0. Against a gate advertised at ~1e-14.
INTEGER, SAVE :: init_i_mode_setup = -1
LOGICAL, SAVE :: init_l_radaer     = .FALSE.
INTEGER, SAVE :: init_i_tune_bc    = -1
LOGICAL, SAVE :: init_l_fix_nacl_density = .FALSE.
LOGICAL, SAVE :: init_l_fix_ukca_hygroscopicities = .FALSE.
LOGICAL, SAVE :: init_l_dust_mp_ageing = .FALSE.

! Set when a re-init is refused. read_box_namelist has to run before the setup
! is even knowable, so by the time wrap_init can refuse it has already
! overwritten every config scalar -- leaving the switches from the new namelist
! paired with the mode setup of the old one. Silently continuing from there
! would produce plausible numbers from a configuration that never existed, so
! every entry point refuses until the process is restarted.
LOGICAL, SAVE :: must_restart = .FALSE.

! Set by wrap_set_fix_water_content. ukca_water_content_v is the one science
! routine that needs no box init at all -- it reads only ncation/nanion (both
! PARAMETERs), its own DATA tables, and glomap_config's flag -- and it is the
! one routine that MUST be reachable before init, because :235 patches its
! SAVEd `y` table in place and never restores it (issue #22).
!
! init_ukca_for_box hardcodes the flag .TRUE. at glomap_box_config_mod.F90:322
! and then init_state runs volume_mode, which runs water_content_v, which fires
! the latch. So after wrap_init the unpatched table is gone for the life of the
! process and no setter can bring it back. The only way to sweep the unfixed
! arm is a process that sets the flag and calls the leaf WITHOUT init.
!
! Nothing gives the flag a value before then -- glomap_config_type declares no
! default and init_ukca_configuration, which would set .FALSE., has no caller.
! So the leaf refuses unless the flag has been set explicitly or init has run.
LOGICAL, SAVE :: water_flag_set = .FALSE.

END MODULE glomap_f2py_state
