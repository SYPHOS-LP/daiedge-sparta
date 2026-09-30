#!/usr/bin/env bash
# ============================================================================
# package_app.sh — turn a bitstream into an app directory xmutil can load.
#
# xmutil does not load a raw .bit.  It wants a directory under
# /lib/firmware/xilinx/<app>/ holding three files:
#
#   <app>.bit.bin   the bitstream in bootgen's format
#   <app>.dtbo      a device-tree overlay naming that file and setting the
#                   fabric clock
#   shell.json      declares the shell type; a fixed constant for XRT_FLAT
#
# The overlay below is a reproduction of the one already running on the board
# (recovered with `dtc -I dtb -O dts` from an existing app), with the firmware
# name and the clock rate substituted.  The reset and clock indices are the
# KR260's and are not guesses: 116..119 are the four PL resets and 71 is the
# PL0 fabric clock, matching the decompiled original exactly.
#
# THE CLOCK RATE MUST MATCH THE BUILD.  The overlay is what actually sets the
# fabric frequency when the app loads, so a rate here that differs from the one
# the design was implemented and timed for runs the design at a speed it was
# never closed for.
#
# Usage
#   scripts/package_app.sh <bitstream.bit> <app-name> [mhz]
#
# Then, on the board:
#   sudo cp -r deploy/<app-name> /lib/firmware/xilinx/
#   sudo xmutil unloadapp && sudo xmutil loadapp <app-name>
# ============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <bitstream.bit> <app-name> [mhz]" >&2
    exit 1
fi

BIT="$1"
APP="$2"
MHZ="${3:-200}"
RATE=$(( MHZ * 1000000 ))

for tool in bootgen dtc; do
    command -v "$tool" >/dev/null || {
        echo "ERROR: $tool not on PATH; source the Vitis settings script" >&2
        exit 1
    }
done
[ -f "$BIT" ] || { echo "ERROR: no bitstream at $BIT" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/deploy/$APP"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

rm -rf "$OUT"
mkdir -p "$OUT"

# ---- 1. bitstream -> .bit.bin ---------------------------------------------
cat > "$WORK/boot.bif" <<BIF
all:
{
    [destination_device = pl] $(readlink -f "$BIT")
}
BIF
echo "=== bootgen: $(basename "$BIT") -> $APP.bit.bin ==="
bootgen -image "$WORK/boot.bif" -arch zynqmp -w -o "$WORK/$APP.bit.bin" >"$WORK/bootgen.log" 2>&1 || {
    echo "ERROR: bootgen failed:" >&2; cat "$WORK/bootgen.log" >&2; exit 1
}
cp "$WORK/$APP.bit.bin" "$OUT/$APP.bit.bin"

# ---- 2. device-tree overlay ------------------------------------------------
cat > "$WORK/$APP.dts" <<DTS
/dts-v1/;
/plugin/;

&fpga_full {
    #address-cells = <2>;
    #size-cells = <2>;
    firmware-name = "$APP.bit.bin";
    resets = <&zynqmp_reset 116>, <&zynqmp_reset 117>,
             <&zynqmp_reset 118>, <&zynqmp_reset 119>;
};

&amba {
    clocking0: clocking0 {
        #clock-cells = <0>;
        compatible = "xlnx,fclk";
        clock-output-names = "fabric_clk";
        clocks = <&zynqmp_clk 71>;
        assigned-clocks = <&zynqmp_clk 71>;
        assigned-clock-rates = <$RATE>;
    };
};
DTS
echo "=== dtc: overlay at ${MHZ} MHz ==="
dtc -q -I dts -O dtb -o "$OUT/$APP.dtbo" "$WORK/$APP.dts"

# ---- 3. shell.json --------------------------------------------------------
cat > "$OUT/shell.json" <<'JSON'
{
    "shell_type" : "XRT_FLAT",
    "num_slots": "1"
}
JSON

echo
echo "app ready: $OUT"
ls -la "$OUT"
cat <<EOF

On the board:
  sudo cp -r $APP /lib/firmware/xilinx/
  sudo xmutil unloadapp
  sudo xmutil loadapp $APP
EOF
