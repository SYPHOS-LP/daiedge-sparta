# ============================================================================
# Encoder Build System (Vitis HLS)
#
# Live projects (each is a workspace/<PROJECT>/ with its own hls_config.cfg):
#   encoder_layer      - Y = MLP-block(MHA-block(X)), synthetic-weight self-checking TB
#   encoder_e2e        - same kernel, real compiled model (sim/e2e), checked by sim/
#
# Targets:
#   make config   - Generate the project's HLS config (.cfg)
#   make sim      - Run C simulation
#   make synth    - Run HLS synthesis
#   make full     - config -> sim -> synth
#   make clean    - Remove the workspace + generated configs
#
# Native regression (g++ against the Vitis HLS headers; no license or device needed):
#   make native-units         - edge cases the golden data cannot reach
#   make native-e2e IMAGES=2  - real-model TB on the first IMAGES images in $(E2E_DIR)
#   make golden-check         - $(E2E_DIR) outputs vs $(GOLDEN_DIR), bit-exact
#   make regress IMAGES=2     - all three
#
# Usage:
#   make sim  PROJECT=encoder_layer
#   make synth PROJECT=encoder_e2e
#   make sim-encoder-layer            (shorthand)
# ============================================================================

.PHONY: config sim synth full clean help all \
        native-units native-e2e golden-check regress \
        $(foreach p,encoder-layer encoder-e2e,config-$(p) sim-$(p) synth-$(p) full-$(p))

# ============================================================================
# Configuration
# ============================================================================
PYTHON            := python3
BUILD_CONFIG      := config/build_config.yaml
GEN_HLS_CONFIG    := scripts/gen_hls_config.py
GEN_LAYER_CONFIG  := scripts/gen_layer_config.py

WORKSPACE_DIR := workspace

# Project selection (see the list above).
PROJECT     ?= encoder_layer
PROJECT_DIR := $(WORKSPACE_DIR)/$(PROJECT)
HLS_CONFIG  := $(PROJECT_DIR)/hls_config.cfg

# Per-component config headers: config/yml/<name>.yaml -> config/inc/<name>_cfg.h
# (encoder_types.yaml -> encoder_types.h is the one naming exception).
CFG_YML_DIR    := config/yml
CFG_INC_DIR    := config/inc
CFG_COMPONENTS := rmsnorm mha encoder_layer
CFG_HEADERS    := $(addprefix $(CFG_INC_DIR)/,$(addsuffix _cfg.h,$(CFG_COMPONENTS))) \
                   $(CFG_INC_DIR)/encoder_types.h

# Vitis tool launchers (.bat wrappers on Windows).
ifeq ($(OS),Windows_NT)
VITIS_RUN := vitis-run.bat
VPP       := v++.bat
else
VITIS_RUN := vitis-run
VPP       := v++
endif

all: help

# ============================================================================
# Config: regenerate the per-component config headers from config/yml/*.yaml,
# then generate the project's HLS config from build_config.yaml
# ============================================================================
config: $(CFG_HEADERS) $(HLS_CONFIG)

$(CFG_INC_DIR)/%_cfg.h: $(CFG_YML_DIR)/%.yaml $(GEN_LAYER_CONFIG)
	@echo "Generating $@..."
	@$(PYTHON) $(GEN_LAYER_CONFIG) $< $@

$(CFG_INC_DIR)/encoder_types.h: $(CFG_YML_DIR)/encoder_types.yaml $(GEN_LAYER_CONFIG)
	@echo "Generating $@..."
	@$(PYTHON) $(GEN_LAYER_CONFIG) $< $@

$(HLS_CONFIG): $(BUILD_CONFIG) $(GEN_HLS_CONFIG)
	@echo "Generating HLS config for $(PROJECT)..."
	@$(PYTHON) $(GEN_HLS_CONFIG) $(BUILD_CONFIG) $(HLS_CONFIG)

# ============================================================================
# Sim: run HLS C simulation
# ============================================================================
sim: config
	@echo "Running C simulation: $(PROJECT)"
	$(VITIS_RUN) --mode hls --csim --config $(HLS_CONFIG) --work_dir $(PROJECT_DIR)

# ============================================================================
# Synth: run HLS synthesis
# ============================================================================
synth: config
	@echo "Running HLS synthesis: $(PROJECT)"
	$(VPP) -c --mode hls --config $(HLS_CONFIG) --work_dir $(PROJECT_DIR)

# ============================================================================
# Full flow: config -> sim -> synth
# ============================================================================
full: sim synth

# ============================================================================
# Native regression: the kernel sources built with g++ and the Vitis HLS headers.
# Every kernel change must keep golden-check bit-exact (see docs/plans/).
# ============================================================================
VITIS_INCLUDE ?= $(firstword $(wildcard $(XILINX_HLS)/include $(XILINX_VITIS)/include /opt/Xilinx/*/Vitis/include))
E2E_DIR       ?= $(CURDIR)/sim/e2e
# The reference outputs. They are not always inside E2E_DIR: a payload emitted for a different
# engine configuration is still checked against the ORIGINAL design's golden, which is the
# point of the check -- the kernel must stay bit-exact across those changes.
GOLDEN_DIR    ?= $(E2E_DIR)/golden
IMAGES        ?= 2
NATIVE_DIR    := $(WORKSPACE_DIR)/native
BUILD_DEFINES := $(shell $(PYTHON) -c "import yaml; print(' '.join('-D' + d for d in (yaml.safe_load(open('$(BUILD_CONFIG)')).get('defines') or [])))")
NATIVE_CXXFLAGS ?= -std=c++17 -O2 -w
NATIVE_INCLUDES := -I$(VITIS_INCLUDE) -Iconfig -I$(CFG_INC_DIR) -Iinc/layers -Iinc/helpers -Iinc/top -Itb
KERNEL_SRCS     := $(wildcard src/top/*.cpp src/layers/*.cpp)
KERNEL_HDRS     := $(wildcard inc/*/*.h) $(CFG_HEADERS)

$(NATIVE_DIR)/%: tb/%.cpp $(KERNEL_SRCS) $(KERNEL_HDRS) $(wildcard tb/*.h) $(BUILD_CONFIG)
	@test -f "$(VITIS_INCLUDE)/ap_int.h" || { echo "ap_int.h not found; set VITIS_INCLUDE=<Vitis>/include"; exit 1; }
	@mkdir -p $(NATIVE_DIR)
	g++ $(NATIVE_CXXFLAGS) $(BUILD_DEFINES) $(NATIVE_INCLUDES) $(KERNEL_SRCS) $< -o $@ -lpthread

native-units: $(NATIVE_DIR)/encoder_units_tb
	$<

native-e2e: $(NATIVE_DIR)/encoder_e2e_tb
	E2E_DIR=$(E2E_DIR)/ E2E_IMAGES=$(IMAGES) $<

golden-check:
	$(PYTHON) sim/perm_e2e.py compare --dir $(E2E_DIR) --golden $(GOLDEN_DIR)

regress: native-units native-e2e golden-check

# ============================================================================
# Per-project shorthands (make sim-encoder-layer, etc.)
# ============================================================================
config-encoder-layer:      ; @$(MAKE) config PROJECT=encoder_layer
sim-encoder-layer:         ; @$(MAKE) sim    PROJECT=encoder_layer
synth-encoder-layer:       ; @$(MAKE) synth  PROJECT=encoder_layer
full-encoder-layer:        ; @$(MAKE) full   PROJECT=encoder_layer

config-encoder-e2e:        ; @$(MAKE) config PROJECT=encoder_e2e
sim-encoder-e2e:           ; @$(MAKE) sim    PROJECT=encoder_e2e
synth-encoder-e2e:         ; @$(MAKE) synth  PROJECT=encoder_e2e
full-encoder-e2e:          ; @$(MAKE) full   PROJECT=encoder_e2e

# ============================================================================
# Clean
# ============================================================================
clean:
	@echo "Cleaning build artifacts..."
	-rm -rf $(WORKSPACE_DIR) .Xil 2>/dev/null || true
	-find . -maxdepth 1 -name "*.log" -delete 2>/dev/null || true
	@echo "Clean complete"

# ============================================================================
# Help
# ============================================================================
help:
	@echo "Encoder Build System (Vitis HLS)"
	@echo ""
	@echo "Projects: encoder_layer (synthetic weights) encoder_e2e (real model)"
	@echo ""
	@echo "  make config PROJECT=<p>   - generate the HLS config"
	@echo "  make sim    PROJECT=<p>   - run C simulation"
	@echo "  make synth  PROJECT=<p>   - run HLS synthesis"
	@echo "  make full   PROJECT=<p>   - config -> sim -> synth"
	@echo "  make clean                - remove workspace + generated configs"
	@echo ""
	@echo "Native regression (no Vitis license/device needed):"
	@echo "  make native-units         - edge cases the golden data cannot reach"
	@echo "  make native-e2e IMAGES=2  - real-model TB on the first IMAGES images (E2E_DIR=$(E2E_DIR))"
	@echo "  make golden-check         - outputs vs GOLDEN_DIR, bit-exact (default E2E_DIR/golden)"
	@echo "  make regress IMAGES=2     - all three"
	@echo ""
	@echo "Shorthands: make sim-encoder-layer, make synth-encoder-layer, ..."
	@echo "encoder_e2e needs sim/e2e artifacts + E2E_DIR set - see sim/README.md"
	@echo "Default PROJECT = encoder_layer"
	@echo ""
	@echo "Build settings (device/clock): config/build_config.yaml"
