# ============================================================================
# build_bitstream.tcl — synthesis, implementation and bitstream for the block
# design that encoder_layer_kr260_bd.tcl produced.
#
# That script deliberately stops after wrapping the design, so this is the
# second half: it opens the project, runs both stages, reports timing, and
# copies the bitstream somewhere predictable.
#
# Usage
#   vivado -mode batch -source scripts/build_bitstream.tcl \
#       -tclargs <mhz> [dbg] [jobs] [threads]
#
# MEMORY: this design needs ~5 GB to synthesise at jobs=1.  Measured, not
# guessed: Vivado's own log reports `peak = 4748.938` MB at Timing
# Optimization, and a cgroup kill caught it at anon-rss 3.4 GB.
#
# So just run it plainly:
#
#   nice -n 10 vivado -mode batch -nojournal -nolog \
#     -source scripts/build_bitstream.tcl -tclargs 200 0 1
#
# DO NOT wrap it in a systemd scope with MemoryMax.  That was tried at 9 GB and
# 12 GB and killed the run both times, because cgroup v2 charges PAGE CACHE to
# the cgroup as well as the process: Vivado's file I/O fills it, MemorySwapMax=0
# makes dirty pages unreclaimable until written back, and the cgroup hits its
# ceiling while the process is still at 4 GB.  The limit does not protect the
# run, it ends it.
#
# What DOES need care is the job count, which is the one real memory failure
# seen: jobs=12 on a 15 GB host exhausted memory and the system OOM killer took
# the DESKTOP (an editor usually scores higher than Vivado), losing the run.
# jobs=2 then made the machine thrash against a 2 GB swapfile.  At ~5 GB per
# run, jobs=1 is right for this host and jobs=2 needs ~12 GB free.
#
# `jobs` bounds how many RUNS go in parallel; `threads` bounds the threads
# WITHIN one run, which -jobs does not control.  Lower it too if a serial run
# still struggles.  0 leaves Vivado's default.
#
# Reads SPARTA_VIVADO_DIR the same way the block-design script does, so both
# halves agree on where the project lives.
#
# TIMING IS THE POINT.  The HLS estimate is not a signoff number: it knows
# nothing about placement or routing.  This script prints the worst negative
# slack and FAILS if it is negative, because a design that does not close
# timing will produce wrong answers intermittently rather than not working --
# much harder to diagnose from the host side.
# ============================================================================
set pl0_mhz [expr {$argc >= 1 ? [lindex $argv 0] : 200}]
set dbg     [expr {$argc >= 2 ? [lindex $argv 1] : 0}]
set jobs    [expr {$argc >= 3 ? [lindex $argv 2] : 2}]
set threads [expr {$argc >= 4 ? [lindex $argv 3] : 0}]

set repo_root [file normalize [file dirname [info script]]/..]
if {[info exists ::env(SPARTA_VIVADO_DIR)]} {
    set proj_dir $::env(SPARTA_VIVADO_DIR)
} else {
    set proj_dir "$repo_root/workspace/vivado"
}
set proj_name [expr {$dbg ? "enc_kr260_${pl0_mhz}mhz_dbg" : "enc_kr260_${pl0_mhz}mhz"}]
set xpr "$proj_dir/$proj_name/$proj_name.xpr"

if {![file exists $xpr]} {
    error "no project at $xpr -- run scripts/encoder_layer_kr260_bd.tcl first"
}
if {$threads > 0} {
    # Caps threads inside each run; -jobs only caps how many runs are concurrent.
    set_param general.maxThreads $threads
    puts "=== general.maxThreads capped at $threads ==="
}
puts "=== opening $xpr (jobs = $jobs) ==="
open_project $xpr

# ---- recover from an interrupted run --------------------------------------
# A process killed mid-synthesis leaves a __synthesis_is_running__ marker in its
# run directory, and Vivado then refuses to relaunch that run because it
# believes it is still going.  Resetting every run that is not complete clears
# those markers while KEEPING the checkpoints of the runs that did finish, so a
# retry only redoes what actually failed.
foreach r [get_runs -quiet *synth_1] {
    if {[get_property PROGRESS $r] ne "100%"} {
        puts "  resetting incomplete run: $r"
        reset_run $r
    }
}

# ---- synthesis -------------------------------------------------------------
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    puts "\n=== synthesis ==="
    reset_run synth_1
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        error "synthesis failed -- see $proj_dir/$proj_name/${proj_name}.runs/synth_1/runme.log"
    }
}
puts "CHECKPOINT: synthesis complete"

# ---- implementation + bitstream -------------------------------------------
puts "\n=== implementation and bitstream ==="
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "implementation failed -- see $proj_dir/$proj_name/${proj_name}.runs/impl_1/runme.log"
}
puts "CHECKPOINT: implementation and bitstream complete"

# ---- timing ----------------------------------------------------------------
open_run impl_1
set wns [get_property SLACK [get_timing_paths -delay_type max]]
set whs [get_property SLACK [get_timing_paths -delay_type min]]
set period [expr {1000.0 / $pl0_mhz}]
puts "\n=== timing at ${pl0_mhz} MHz (period ${period} ns) ==="
puts "  worst negative slack (setup): $wns ns"
puts "  worst hold slack            : $whs ns"
if {$wns < 0} {
    puts "  achievable frequency        : [format %.1f [expr {1000.0 / ($period - $wns)}]] MHz"
}

# ---- collect the bitstream -------------------------------------------------
set bit [glob -nocomplain "$proj_dir/$proj_name/${proj_name}.runs/impl_1/*_wrapper.bit"]
if {[llength $bit] == 0} {
    error "no bitstream written in impl_1"
}
set bit [lindex $bit 0]
file mkdir "$repo_root/deploy"
set out "$repo_root/deploy/[file tail $bit]"
file copy -force $bit $out
puts "\nbitstream: $out"

if {$wns < 0} {
    error "TIMING NOT MET at ${pl0_mhz} MHz (WNS $wns ns) -- do not deploy this bitstream"
}
puts "CHECKPOINT: bitstream built and timing met at ${pl0_mhz} MHz"
