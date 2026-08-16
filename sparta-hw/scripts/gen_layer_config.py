#!/usr/bin/env python3
# ============================================================================
# Generic per-component config generator
#
# Converts a self-describing component config YAML into a C++ header of typedefs
# and #defines for HLS.  ONE generator serves every component: the YAML says
# WHAT to emit (types + defines); the generator only knows HOW.
#
#   config/yml/<name>.yaml  ->  config/inc/<name>.h
#
# Design rules (per project rework):
#   * Each component is INDEPENDENT — its header pulls in only the generic HLS
#     type headers (ap_int.h, ap_fixed.h), never another component's config.
#     The ONE sanctioned exception is the shared encoder_types.h (T_Activation /
#     T_Residual): a component that reuses those global types lists it under the
#     optional `includes:` key so it does not re-mint the int8/residual typedefs.
#   * The include guard is DERIVED from the file name: <name>.yaml -> <NAME>_CFG_H.
#   * No header.guard field in the YAML.
#
# YAML schema:
#   includes:                   # OPTIONAL — extra headers to #include after the
#     - encoder_types.h         # ap_* headers (e.g. the shared encoder_types.h).
#   defines:                    # emitted FIRST (so typedefs may reference them)
#     - { name: RMS_MAX_D,        value: 768 }
#     - { name: RMS_EPS,          value: 1e-6f }
#     - { name: RMS_RSQRT_LOG,    value: 2 }
#     - { name: RMS_LUT_SIZE,     value: "(1 << RMS_LUT_ADDR_BITS)" }  # expression
#   types:                      # emitted AFTER defines, in order
#     - { name: T_Foo,  kind: ap_int,    width: 8 }
#     - { name: T_Bar,  kind: ap_uint,   width: 16 }
#     - { name: T_Baz,  kind: ap_fixed,  total: 32, int: 16 }
#     - { name: T_Qux,  kind: ap_ufixed, total: 56, int: 26 }
#     - { name: T_Wide, kind: ap_uint,   width_expr: "FOO_BITS * FOO_PACK" }  # width from defines above
#   Optional per-entry "comment:" adds an inline trailing comment.
#
# Emit ORDER is defines-then-types, so an ap_int/ap_uint typedef may set its width
# via width_expr referencing any #define above it.
#
# Usage:
#   python3 scripts/gen_layer_config.py config/yml/rmsnorm.yaml config/inc/rmsnorm_cfg.h
# ============================================================================

import sys
from pathlib import Path

import yaml


def guard_from_name(out_path):
    """Derive the include guard from the output file stem: rmsnorm_cfg.h -> RMSNORM_CFG_H."""
    return Path(out_path).stem.upper() + "_H" if not Path(out_path).stem.upper().endswith("_H") \
        else Path(out_path).stem.upper()


def emit_typedef(t):
    """Render one typedef entry to a C++ typedef line."""
    name = t["name"]
    kind = t["kind"]
    comment = t.get("comment")
    if kind in ("ap_int", "ap_uint"):
        # Width is either a literal (width:) or an expression (width_expr:) that
        # may reference #defines emitted earlier in this header.  Exactly one.
        has_w = "width" in t
        has_we = "width_expr" in t
        if has_w == has_we:
            raise ValueError(f"type '{name}': set exactly one of width / width_expr")
        w = t["width"] if has_w else t["width_expr"]
        decl = f"typedef {kind}<{w}> {name};"
    elif kind in ("ap_fixed", "ap_ufixed"):
        decl = f"typedef {kind}<{t['total']}, {t['int']}> {name};"
    else:
        raise ValueError(f"type '{name}': unknown kind '{kind}' "
                         f"(expected ap_int/ap_uint/ap_fixed/ap_ufixed)")
    return decl + (f"   // {comment}" if comment else "")


def emit_define(d):
    """Render one define entry to a C++ #define line."""
    name = d["name"]
    value = d["value"]
    comment = d.get("comment")
    line = f"#define {name} {value}"
    return line + (f"   // {comment}" if comment else "")


def generate_header(config, src_yaml, guard):
    """Build the full header text from a parsed component config."""
    types = config.get("types", []) or []
    defines = config.get("defines", []) or []
    includes = config.get("includes", []) or []

    lines = []
    lines.append("// AUTO-GENERATED FILE - DO NOT EDIT MANUALLY")
    lines.append(f"// Generated from {src_yaml} by scripts/gen_layer_config.py")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    # Independent component header: only the generic HLS type headers, plus any
    # shared-type headers the component opted into via `includes:` (e.g. the global
    # encoder_types.h).
    lines.append('#include "ap_int.h"')
    lines.append('#include "ap_fixed.h"')
    for inc in includes:
        lines.append(f'#include "{inc}"')
    lines.append("")

    # Defines are emitted BEFORE types so a typedef's width_expr can reference them.
    if defines:
        lines.append("// ---- Defines --------------------------------------------------------------")
        for d in defines:
            lines.append(emit_define(d))
        lines.append("")

    if types:
        lines.append("// ---- Types ----------------------------------------------------------------")
        for t in types:
            lines.append(emit_typedef(t))
        lines.append("")

    lines.append(f"#endif // {guard}")
    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gen_layer_config.py <config/yml/<name>.yaml> <config/inc/<name>.h>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    out_path = sys.argv[2]

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}

        guard = guard_from_name(out_path)
        header = generate_header(config, yaml_path, guard)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(header)

        n_types = len(config.get("types", []) or [])
        n_defs = len(config.get("defines", []) or [])
        print(f"Generated {out_path} from {yaml_path}")
        print(f"  - guard: {guard}")
        print(f"  - types: {n_types}, defines: {n_defs}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Config schema error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
