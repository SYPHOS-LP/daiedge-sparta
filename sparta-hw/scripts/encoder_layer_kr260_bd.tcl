# ============================================================================
# encoder_layer_kr260_bd.tcl — Vivado block design for encoder_layer_top on the
# Kria KR260.
#
# Builds a project containing the Zynq UltraScale+ PS, the packaged
# encoder_layer_top HLS IP, and the AXI wiring between them; validates it and
# generates the HDL wrapper. Does NOT run synthesis, implementation, or
# bitstream generation.
#
# Prerequisites
#   - KR260 board files installed (xilinx.com:kr260_som:part0:1.1)
#   - encoder_layer_top packaged as IP-XACT in ip/encoder_layer/, produced by
#     the HLS flow with package.output.format=ip_catalog
#
# Usage
#   vivado -mode batch -source scripts/encoder_layer_kr260_bd.tcl
#   vivado -mode batch -source scripts/encoder_layer_kr260_bd.tcl -tclargs 200
#
#   arg 1: PL0 fabric clock in MHz (default 125)
#
# Vivado returns exit code 0 even when a sourced script errors, so check the
# log for the final CHECKPOINT line rather than the exit status.
# ============================================================================

# ---- Build parameters ------------------------------------------------------
# The clock is part of the project name so that builds at different
# frequencies land in separate directories instead of overwriting each other's
# implementation results.
set pl0_mhz [expr {$argc >= 1 ? [lindex $argv 0] : 125}]

set repo_root  [file normalize [file dirname [info script]]/..]
set proj_dir   "C:/vp"
set proj_name  "enc_kr260_${pl0_mhz}mhz"
set bd_name    "enc_system"
set cell_name  "enc_top_0"
set ip_repo    "$repo_root/ip/encoder_layer"
set board_part "xilinx.com:kr260_som:part0:1.1"

puts "=== PL0 = ${pl0_mhz}MHz, project = $proj_dir/$proj_name ==="

file mkdir $proj_dir
create_project $proj_name $proj_dir/$proj_name -part xck26-sfvc784-2LV-c -force
set_property board_part $board_part [current_project]
set_property ip_repo_paths $ip_repo [current_fileset]
update_ip_catalog -rebuild

create_bd_design $bd_name

# ---- Zynq UltraScale+ PS ---------------------------------------------------
# The board preset configures DDR, MIO and clocking for the KR260 SOM.
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} [get_bd_cells $ps]

# PS port configuration:
#   S_AXI_GP0/GP1  = HPC0/HPC1, coherent slave ports for the kernel's masters
#   S_AXI_GP2/3/4  = HP0/HP1/HP2, non-coherent slave ports
#   S_AXI_GP5      = HP3, disabled. An enabled slave port exposes its own aclk
#                    pin; with no master attached nothing drives that clock and
#                    validate_bd_design rejects the design. The 11-bundle
#                    interface only needs five physical ports.
#   M_AXI_GP0/GP1  = HPM0/HPM1_FPD, disabled.
#   M_AXI_GP2      = HPM0_LPD, the AXI-Lite control master (see below).
#
# PL0 drives the kernel's ap_clk. The board preset defaults it to 100MHz, so
# it is set explicitly to match the frequency the design was synthesized for.
set_property -dict [list \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {0} \
    CONFIG.PSU__USE__M_AXI_GP0 {0} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl0_mhz \
] $ps

# ---- encoder_layer_top HLS IP ----------------------------------------------
set enc [create_bd_cell -type ip -vlnv xilinx.com:hls:encoder_layer_top:1.0 $cell_name]

# ---- AXI-Lite control ------------------------------------------------------
# Control goes through M_AXI_HPM0_LPD (GP2), which is natively 32-bit and so
# matches the kernel's 32-bit AXI4-Lite control port directly. The FPD ports
# (GP0/GP1) are 128-bit and would cause Vivado to insert a width converter in
# front of the control interface.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config { Master "/zynq_ultra_ps_e_0/M_AXI_HPM0_LPD" Clk "Auto" } \
    [get_bd_intf_pins $cell_name/s_axi_control]

# ---- Data path: 11 m_axi masters -> 5 PS slave ports ------------------------
# Each entry is:  <PS port> {<primary master> {<additional masters>}}
# Bundles are grouped to spread traffic: small load-once buffers together,
# weights split across ports, and the Q'/K' scratch alongside the weight
# stream that produces it.
set groups {
    S_AXI_HPC0_FPD {sc_mem      {in_mem}}
    S_AXI_HPC1_FPD {y_mem       {h_mem}}
    S_AXI_HP0_FPD  {w_mha_mem   {qv_mem qc_mem}}
    S_AXI_HP1_FPD  {w1_mem      {kv_mem kc_mem}}
    S_AXI_HP2_FPD  {w2_mem      {}}
}

# Pass 1: connect the first master on each port with connection automation,
# which creates the interconnect and its clock/reset wiring.
foreach {port_suffix spec} $groups {
    set primary [lindex $spec 0]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
        -config [list Master "/$cell_name/m_axi_$primary" Clk "Auto"] \
        [get_bd_intf_pins zynq_ultra_ps_e_0/$port_suffix]
}

validate_bd_design
save_bd_design
puts "CHECKPOINT: primary masters wired, validated OK"

# Pass 2: attach the remaining masters to the interconnect that pass 1 created
# on each port, by widening it and connecting into the new slave ports.
#
# Automation is not reused here: invoking it again for a second master on an
# already-connected port does not extend the existing interconnect reliably.
# Instead the interconnect is located by walking back from the PS pin, its
# NUM_SI is increased, and each extra master is connected directly, taking
# clock and reset from the interconnect's own S00 pins so no clock name is
# assumed.
proc find_primary_interconnect {port_suffix} {
    set ps_pin [get_bd_intf_pins zynq_ultra_ps_e_0/$port_suffix]
    set net [get_bd_intf_nets -quiet -of_objects $ps_pin]
    foreach p [get_bd_intf_pins -quiet -of_objects $net] {
        set cell [get_bd_cells -quiet -of_objects $p]
        if {$cell ne "" && [get_property VLNV $cell] eq "xilinx.com:ip:axi_interconnect:2.1"} {
            return $cell
        }
    }
    error "No axi_interconnect found feeding $port_suffix"
}

foreach {port_suffix spec} $groups {
    set extras [lindex $spec 1]
    if {[llength $extras] == 0} { continue }

    set primary_ic [find_primary_interconnect $port_suffix]
    set n [llength $extras]
    set new_nsi [expr {1 + $n}]
    puts "\n=== $primary_ic ($port_suffix): adding $n extra bundle(s), NUM_SI 1 -> $new_nsi ==="

    set aclk_pin  [get_bd_pins -quiet ${primary_ic}/S00_ACLK]
    set arstn_pin [get_bd_pins -quiet ${primary_ic}/S00_ARESETN]
    set aclk_src  [get_bd_pins -quiet -of_objects [get_bd_nets -quiet -of_objects $aclk_pin]  -filter {DIR == O}]
    set arstn_src [get_bd_pins -quiet -of_objects [get_bd_nets -quiet -of_objects $arstn_pin] -filter {DIR == O}]
    puts "    clk source: $aclk_src   rstn source: $arstn_src"

    set_property CONFIG.NUM_SI $new_nsi [get_bd_cells $primary_ic]

    set idx 1
    foreach bundle $extras {
        set sidx [format "S%02d" $idx]
        set kernel_pin   [get_bd_intf_pins $cell_name/m_axi_$bundle]
        set primary_snew [get_bd_intf_pins ${primary_ic}/${sidx}_AXI]
        puts "    connecting $cell_name/m_axi_$bundle -> ${primary_ic}/${sidx}_AXI"
        connect_bd_intf_net $kernel_pin $primary_snew
        connect_bd_net $aclk_src  [get_bd_pins ${primary_ic}/${sidx}_ACLK]
        connect_bd_net $arstn_src [get_bd_pins ${primary_ic}/${sidx}_ARESETN]
        incr idx
    }
}

puts "\n=== Assigning addresses ==="
assign_bd_address

validate_bd_design
save_bd_design
puts "CHECKPOINT: all bundles wired, validated OK"

# ---- Connectivity verification ---------------------------------------------
# validate_bd_design accepts a master whose interconnect output is left
# dangling, which produces a design that builds but never completes on
# hardware. This walks forward from every kernel master through the
# interconnect chain and requires it to terminate at a PS pin.
puts "\n=== Verifying every bundle reaches a physical PS port ==="
set all_bundles {sc_mem in_mem w_mha_mem w1_mem w2_mem h_mem y_mem qv_mem qc_mem kv_mem kc_mem}
set bad {}
foreach b $all_bundles {
    set kernel_pin [get_bd_intf_pins -quiet $cell_name/m_axi_$b]
    if {$kernel_pin eq ""} {
        lappend bad "$b: no such pin on $cell_name"
        continue
    }
    set net [get_bd_intf_nets -quiet -of_objects $kernel_pin]
    if {$net eq ""} {
        lappend bad "$b: m_axi_$b has no net"
        continue
    }
    set cur $kernel_pin
    set reached_ps 0
    for {set hop 0} {$hop < 10} {incr hop} {
        set n [get_bd_intf_nets -quiet -of_objects $cur]
        if {$n eq ""} { break }
        set other {}
        foreach p [get_bd_intf_pins -quiet -of_objects $n] { if {$p ne $cur} { lappend other $p } }
        if {[llength $other] == 0} { break }
        set nxt [lindex $other 0]
        set cell [get_bd_cells -quiet -of_objects $nxt]
        if {$cell eq ""} { break }
        # The PS is a regular cell, so reaching any of its pins resolves to
        # /zynq_ultra_ps_e_0 -- that is the success condition.
        if {[string match "*zynq_ultra_ps_e*" $cell]} {
            set reached_ps 1
            break
        }
        if {[get_property VLNV $cell] ne "xilinx.com:ip:axi_interconnect:2.1"} { break }
        set cur [get_bd_intf_pins -quiet ${cell}/M00_AXI]
    }
    if {!$reached_ps} {
        lappend bad "$b: does not terminate at a PS pin"
    }
}
if {[llength $bad] > 0} {
    puts "FAIL -- bundles with a connectivity problem:"
    foreach b $bad { puts "  $b" }
    error "Bundle connectivity verification FAILED"
}
puts "PASS -- all [llength $all_bundles] bundles reach a physical PS port."

# ---- Wrapper ---------------------------------------------------------------
make_wrapper -files [get_files "$proj_dir/$proj_name/${proj_name}.srcs/sources_1/bd/${bd_name}/${bd_name}.bd"] -top
add_files -norecurse "$proj_dir/$proj_name/${proj_name}.gen/sources_1/bd/${bd_name}/hdl/${bd_name}_wrapper.v"
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "CHECKPOINT: block design built, wired, verified, validated, and wrapped OK"
