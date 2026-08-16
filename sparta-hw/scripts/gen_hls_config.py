#!/usr/bin/env python3
# ============================================================================
# HLS Config Generator for the sparse encoder kernels
#
# Generates hls_config.cfg for Vitis HLS. The two arguments play different
# roles:
#   <input.yaml>  only supplies target device/clock (config/build_config.yaml).
#                 Per-component types/bounds come from config/yml/*.yaml via
#                 the separate gen_layer_config.py -> config/inc/*_cfg.h step,
#                 run first (see `make config`).
#   <output.cfg>  its PARENT DIRECTORY NAME selects which component gets built
#                 (workspace/<name>/hls_config.cfg -> project_name = <name>),
#                 not anything read from <input.yaml>.
#
# Usage:
#   python3 scripts/gen_hls_config.py config/build_config.yaml workspace/encoder_layer/hls_config.cfg
# ============================================================================

import yaml
import sys
from pathlib import Path

def parse_config(yaml_path):
    """Load and parse YAML configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def generate_hls_config(config, output_path):
    """Generate HLS config file from configuration."""
    
    hls_cfg = config.get('hls', {})

    clock_period = hls_cfg.get('target_clock_period_ns', 10)
    target_device = hls_cfg.get('target_device', 'xck26-sfvc784-2LV-c')

    # Project-wide preprocessor defines (e.g. POT_SHIFT_MAC), applied to the
    # syn + csim + tb cflags alike.
    extra_defines = config.get('defines', []) or []
    defines_flags = "".join(f" -D{d}" for d in extra_defines)

    # Get absolute paths
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent.absolute()

    # Determine project variant from output path
    # workspace/encoder_layer/... -> encoder_layer (top: encoder_layer_top)
    project_name = output_dir.name
    
    syn_top = {
        "encoder_layer":      "encoder_layer_top",
        "encoder_e2e":        "encoder_layer_top",   # same kernel; file-driven e2e TB (real weights/images)
    }.get(project_name)
    if syn_top is None:
        raise SystemExit(f"unknown project '{project_name}'")
    
    # Calculate paths relative to the config file location (workspace)
    # This assumes: workspace/<project_name>/hls_config.cfg
    # And root is: ../../ (back to sparta-hw)
    sparta_root = output_dir.parent.parent

    def p(path):
        """Stringify a path with forward slashes (Vitis HLS config files want '/' even on Windows)."""
        return str(path).replace("\\", "/")

    config_dir = sparta_root / "config"
    config_inc_dir = config_dir / "inc"   # generated per-component config headers
    src_dir = sparta_root / "src"
    inc_dir = sparta_root / "inc"
    tb_dir = sparta_root / "tb"

    # Source/header directories (restructured: src/{layers,top}, inc/{layers,helpers,top}).
    layers_src_dir  = src_dir / "layers"
    layers_inc_dir  = inc_dir / "layers"
    top_src_dir     = src_dir / "top"
    top_inc_dir     = inc_dir / "top"
    helpers_inc_dir = inc_dir / "helpers"

    config_dir_str      = p(config_dir)
    config_inc_dir_str  = p(config_inc_dir)
    layers_inc_dir_str  = p(layers_inc_dir)
    top_inc_dir_str     = p(top_inc_dir)
    helpers_inc_dir_str = p(helpers_inc_dir)

    # Reusable per-module file groups (layer cores live in src/layers + inc/layers;
    # shared helper headers -- recip/rsqrt LUTs -- in inc/helpers; generated configs
    # in config/inc).  Composed below per project so each module's file list is
    # written once instead of repeated in every project that pulls it in.
    rmsnorm_files = [
        p(layers_src_dir / "rmsnorm.cpp"),
        p(layers_inc_dir / "rmsnorm.h"),
        p(config_inc_dir / "rmsnorm_cfg.h"),
        p(helpers_inc_dir / "rmsnorm_rsqrt.h"),
    ]
    mha_files = [
        p(layers_src_dir / "mha.cpp"),
        p(layers_inc_dir / "mha.h"),
        p(helpers_inc_dir / "mha_recip.h"),
        p(config_inc_dir / "mha_cfg.h"),
    ]
    mlp_files = [
        p(layers_src_dir / "mlp.cpp"),
        p(layers_inc_dir / "mlp.h"),
        p(config_inc_dir / "mlp_cfg.h"),
    ]
    residual_files = [
        p(layers_src_dir / "residual.cpp"),
        p(layers_inc_dir / "residual.h"),
    ]

    # Select testbench and source files based on project variant.  The block/
    # full-layer glue (src/top + inc/top) is listed per project; the layer-core
    # modules it wraps are pulled in from the groups above.
    if project_name in ("encoder_layer", "encoder_e2e"):
        # encoder_e2e = the same encoder_layer sources, driven by the file-based e2e TB.
        tb_filename = ("encoder_e2e_tb.cpp" if project_name == "encoder_e2e"
                       else "encoder_layer_tb.cpp")
        src_files = [
            p(top_src_dir / "encoder_layer.cpp"),
            p(top_inc_dir / "encoder_layer.h"),
            p(config_inc_dir / "encoder_layer_cfg.h"),
            p(top_src_dir / "encoder_mha_block.cpp"),
            p(top_inc_dir / "encoder_mha_block.h"),
            p(config_inc_dir / "encoder_mha_block_cfg.h"),
            p(top_src_dir / "encoder_mlp_block.cpp"),
            p(top_inc_dir / "encoder_mlp_block.h"),
            p(config_inc_dir / "encoder_mlp_block_cfg.h"),
            *residual_files,
            *mha_files,
            *mlp_files,
            *rmsnorm_files,
        ]
    else:
        raise SystemExit(f"unknown project '{project_name}' "
                         "(live: encoder_layer, encoder_e2e)")

    # Every live component pulls the shared types header (T_Activation / T_Scale)
    # transitively via its *_cfg.h and the shared quantize.h (saturate/requant
    # helpers); list both as syn.files so Vitis tracks them as explicit dependencies
    # (they live on the -I path already: config/inc and inc/layers respectively).
    src_files.append(p(config_inc_dir / "encoder_types.h"))
    src_files.append(p(helpers_inc_dir / "quantize.h"))

    tb_file_str = p(tb_dir / tb_filename)

    # Build syn.file entries (one per source file)
    syn_files_str = "\n".join([f"syn.file={f}" for f in src_files])
    
    # ========================================================================
    # Generate config file content
    # ========================================================================
    config_desc = {
        "encoder_layer":     "Encoder Layer (Y = MLP-block(MHA-block(X)), full pre-norm layer)",
        "encoder_e2e":       "Encoder E2E (full encoder_layer kernel, file-driven real weights/images)",
    }[project_name]

    # All live components share the same include set: the layer/helper/top headers
    # and the generated configs.  (config_dir is on the base -I below.)
    extra_includes = (f" -I{layers_inc_dir_str} -I{helpers_inc_dir_str}"
                      f" -I{top_inc_dir_str} -I{config_inc_dir_str}")

    config_content = f"""# ============================================================================
# HLS Configuration File for {config_desc}
#
# Generated from: config/build_config.yaml
# Generated by: scripts/gen_hls_config.py
# ============================================================================
part={target_device}


[hls]
# HLS flow target (vitis for unified flow, vivado for legacy)
flow_target=vitis

# Package output format (xo = Xilinx Object format)
package.output.format=xo
package.output.syn=false

# Top-level function for synthesis
syn.top={syn_top}

# Clock period in nanoseconds
clock={clock_period}ns

# Clock uncertainty (percentage)
clock_uncertainty=12%

# Testbench file (absolute path)
tb.file={tb_file_str}

# Source files for synthesis
{syn_files_str}

# Compiler flags - include paths for synthesis
syn.cflags=-I{config_dir_str}{extra_includes}{defines_flags}

# Compiler flags - include paths for C simulation
syn.csimflags=-I{config_dir_str}{extra_includes}{defines_flags}

# Compiler flags for testbench
tb.cflags=-I{config_dir_str}{extra_includes}{defines_flags}
"""
    
    # Ensure output directory exists
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Write config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    return config_content

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python3 gen_hls_config.py <input.yaml> <output_hls_config.cfg>")
        sys.exit(1)
    
    yaml_path = sys.argv[1]
    config_path = sys.argv[2]
    
    try:
        config = parse_config(yaml_path)
        generate_hls_config(config, config_path)
        
        hls_cfg = config.get('hls', {})
        print(f"Generated {config_path}")
        print(f"  - Target device: {hls_cfg.get('target_device', 'xck26-sfvc784-2LV-c')}")
        print(f"  - Clock period: {hls_cfg.get('target_clock_period_ns', 10)}ns")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
