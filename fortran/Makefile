# ---------------------------------------------------------------------------
# GLOMAP-mode standalone aerosol box model
#
#   make            build bin/glomap_box
#   make run        build and run the boundary-layer case
#   make test       build and run the regression tests
#   make check      report any external (unstubbed) module dependencies
#   make clean      remove build products
#
# Dependency ordering is generated from the sources by
# tools/gen_build_order.py, so adding a file needs no Makefile edit.
# ---------------------------------------------------------------------------

# `FC ?=` will not work here: make defines FC=f77 as a built-in default, so
# the variable is already set. Only override when the value came from make.
ifeq ($(origin FC),default)
  FC := gfortran
endif

BUILD   ?= build
BIN     ?= bin
SRC     ?= src
UKCA_ROOT ?= ../ukca
TARGET  := $(BIN)/glomap_box

# -fno-range-check is required: the vendored UKCA sources use integer
# constants (missing-data indicators) that overflow the default checks.
FCFLAGS ?= -O2 -ffree-line-length-none -fno-range-check -J$(BUILD) -I$(BUILD)

# Deliberately `:=` not `?=`: the box model needs no link flags, and `?=`
# would inherit a stray LDFLAGS from the user's shell profile. A command-line
# override (make LDFLAGS=...) still takes precedence over this.
LDFLAGS :=

# Stricter build used by `make debug`.
DEBUGFLAGS := -O0 -g -ffree-line-length-none -fno-range-check \
              -fcheck=bounds,pointer -ffpe-trap=invalid,zero,overflow \
              -finit-real=snan -J build-debug -I build-debug

COMPILE = $(FC) $(FCFLAGS) -c $< -o $@

DEPFILE := $(BUILD)/depend.mk

.PHONY: all run test check clean debug debugtest distclean verify-vendor

all: $(TARGET)

$(BUILD) $(BIN):
	@mkdir -p $@

# Regenerate dependencies whenever any source file changes.
$(DEPFILE): $(shell find $(SRC) -name '*.F90') | $(BUILD)
	@python3 tools/gen_build_order.py --make $(BUILD) $(SRC) > $@

-include $(DEPFILE)

$(TARGET): $(OBJS) | $(BIN)
	$(FC) $(FCFLAGS) $(OBJS) -o $@ $(LDFLAGS)
	@echo "built $@"

# Debug gets its own object and binary directories. Sharing them with the
# release build means `make && make debug` finds every object already up to
# date and quietly leaves you with the -O2, no-checking binary -- exactly when
# you are trying to chase a bounds or NaN problem.
debug:
	@$(MAKE) BUILD=build-debug BIN=bin-debug FCFLAGS="$(DEBUGFLAGS)" all
	@echo "debug binary: bin-debug/glomap_box"

# Run every shipped case under the debug build's checking.
debugtest: debug
	@mkdir -p out
	@rc=0; for n in namelists/*.nml tests/namelists/*.nml; do \
	  case "$$n" in *.nml.in) continue;; esac; \
	  ./bin-debug/glomap_box "$$n" > /dev/null 2>&1 \
	    && printf '  ok   %s\n' "$$n" \
	    || { printf '  FAIL %s\n' "$$n"; rc=1; }; \
	done; \
	[ $$rc -eq 0 ] && echo "all cases clean under $(DEBUGFLAGS)" || exit 1

run: $(TARGET)
	@mkdir -p out
	./$(TARGET) namelists/boundary_layer.nml

test: $(TARGET)
	@./tests/run_tests.sh

check:
	@python3 tools/gen_build_order.py --check $(SRC)

# Confirm src/ukca/ is upstream UKCA plus exactly patches/, nothing more.
verify-vendor:
	@UKCA_ROOT=$(UKCA_ROOT) ./tools/verify_vendor.sh

clean:
	rm -rf $(BUILD) $(BIN) build-debug bin-debug

distclean: clean
	rm -rf out
