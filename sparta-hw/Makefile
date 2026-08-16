# ============================================================================
# Encoder Build System (Vitis HLS)
#
# Live projects (each is a workspace/<PROJECT>/ with its own hls_config.cfg):
#   rmsnorm            - RMSNorm layer
#   encoder_mlp_block  - Y = X + MLP(RMSNorm(X))       (pre-norm residual)
#   encoder_mha_block  - Y = z + MHA(RMSNorm(z))       (pre-norm residual)
#   encoder_layer      - Y = MLP-block(MHA-block(X))   (full pre-norm layer)
#
# Targets:
#   make config   - Generate the project's HLS config (.cfg)
#   make sim      - Run C simulation
#   make synth    - Run HLS synthesis
#   make full     - config -> sim -> synth
#   make clean    - Remove the workspace + generated configs
#
# Usage:
#   make sim  PROJECT=encoder_layer
#   make synth PROJECT=encoder_mha_block
#   make sim-encoder-layer            (shorthand)
# ============================================================================

.PHONY: config sim synth full clean help all \
        $(foreach p,rmsnorm encoder-mlp-block encoder-mha-block encoder-layer,\
                 config-$(p) sim-$(p) synth-$(p) full-$(p))

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
CFG_COMPONENTS := rmsnorm mlp mha encoder_mha_block encoder_mlp_block encoder_layer
CFG_HEADERS    := $(addprefix $(CFG_INC_DIR)/,$(addsuffix _cfg.h,$(CFG_COMPONENTS))) \
                   $(CFG_INC_DIR)/encoder_types.h

VITIS_RUN := vitis-run.bat

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
	v++.bat -c --mode hls --config $(HLS_CONFIG) --work_dir $(PROJECT_DIR)

# ============================================================================
# Full flow: config -> sim -> synth
# ============================================================================
full: sim synth

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
	@echo "Shorthands: make sim-encoder-layer, make synth-encoder-layer, ..."
	@echo "encoder_e2e needs sim/e2e artifacts + E2E_DIR set - see sim/README.md"
	@echo "Default PROJECT = encoder_layer"
	@echo ""
	@echo "Build settings (device/clock): config/build_config.yaml"
