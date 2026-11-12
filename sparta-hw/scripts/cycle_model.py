#!/usr/bin/env python3
"""Cycles per layer from the REAL weights and the EXACT per-loop latencies.

The budget in docs/plans/04-tiled-mlp.md was built from HLS's latency estimate, which uses the
LOOP_TRIPCOUNT pragmas -- guesses set from average density.  The board measured 2.17 M cycles per
layer against that budget's 734 k, so the budget is wrong by 3x and it is worth knowing where.

This multiplies two things that are each solid:

  * the per-loop initiation interval and pipeline DEPTH from csynth, which are exact;
  * the actual nonzero counts per (weight, output row, column block) from the model file, which is
    what drives every data-dependent trip count.

Latencies used, all read from the csynth reports of the current build:

  MAC_STEP          II=1, depth 5   -> trip + 4,  trip = max_t(nnz[row][t]) * MHA_BANK_ROW_WORDS
  tiled_rows_compute                -> MAC_STEP + 8 of fixed per-row work
  REQUANT_ROW       II=1            -> 10
  SLICE_COUNT       II=1, depth 77  -> 172 (fixed: 96 count words)
  BANK_WORD         II=1, depth 75  -> trip + 74, ONCE PER COLUMN BLOCK because BANK_TILE is not
                                       pipelined -- 12 x 74 = 888 cycles of memory latency a stage
  TAIL_K                            -> 50 per row (K stages only)
  TAIL_BYTES                        -> 7 per row  (Q and V stages)
  TAIL_Q                            -> 3204 per Q stage
  rmsnorm / residual / tile load    -> 10850 / 5381 / 5377
  head_core                         -> data dependent; avg 3114, worst 7527 (both reported)

Usage
  python3 scripts/cycle_model.py -i <quant_values.txt> [--layers 0]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_assign import WEIGHTS, WMODULE, contiguous, cost  # noqa: E402

# --- geometry (config/inc/mha_cfg.h) ---------------------------------------------------
TILES, ROW_WORDS, TILE_PACK, ROWS_PER_STAGE = 12, 7, 4, 64

# --- exact per-loop numbers from csynth -----------------------------------------------
MAC_DEPTH_EXTRA   = 4      # MAC_STEP latency = trip + 4
ROW_FIXED         = 8      # tiled_rows_compute, outside MAC_STEP
REQUANT_ROW       = 10
SLICE_COUNT       = 172
BANK_WORD_LATENCY = 74     # paid PER COLUMN BLOCK: BANK_TILE is not pipelined
SLICE_OVERHEAD    = 38     # load_weight_slice, outside its two loops
TAIL_K, TAIL_BYTES, TAIL_Q = 50, 7, 3204
RMSNORM, RESIDUAL, TILE_LOAD = 10850, 5381, 5377
HEAD_CORE_AVG, HEAD_CORE_WORST = 3114, 7527

# which stages exist, and how many of each (108 total)
STAGES = [("w_q", 12), ("w_k", 12), ("w_v", 12), ("out_proj", 12),
          ("linear_1", 48), ("linear_2", 12)]


def stage_cycles(counts, row0, tag):
    """Cycles for one stage: load this row block's weights, then walk its 64 output rows.

    counts[row][tile] = nonzeros of that output row in that column block.
    """
    blk = counts[row0:row0 + ROWS_PER_STAGE]

    # ---- load_weight_slice ----
    # Entries per column block for this row block, in packed words.  BANK_TILE is a plain loop
    # over the 12 blocks with a pipelined inner loop, so each block pays the read latency again.
    per_block_entries = blk.sum(axis=0)
    words = np.ceil(per_block_entries / TILE_PACK).astype(np.int64)
    load = SLICE_COUNT + int((words + BANK_WORD_LATENCY).sum()) + SLICE_OVERHEAD

    # ---- the 64 output rows ----
    # A row costs the BUSIEST column block: the twelve advance in lockstep.
    busiest = blk.max(axis=1)
    engine = int((busiest * ROW_WORDS + MAC_DEPTH_EXTRA + ROW_FIXED).sum())
    requant = REQUANT_ROW * ROWS_PER_STAGE

    tail = 0
    if tag == "w_k":
        tail = TAIL_K * ROWS_PER_STAGE
    elif tag in ("w_q", "w_v"):
        tail = TAIL_BYTES * ROWS_PER_STAGE
        if tag == "w_q":
            tail += TAIL_Q

    return {"weight load": load, "engine": engine, "requantize": requant, "tails": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--measured", type=float, default=2.17,
                    help="board-measured M cycles per layer, for comparison")
    args = ap.parse_args()

    import compile_model as C
    model = C.compile_model(args.input)
    layer_ids = model.layers() if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    totals, per_layer = {}, []
    for L in layer_ids:
        acc = {}
        for tag, n_stages in STAGES:
            wi = np.asarray(model.linears[f"encoder.encoder_layer_{L}.{WMODULE[tag]}"].weight_int)
            nz = wi != 0
            per_block = nz.shape[1] // TILES
            counts = np.zeros((nz.shape[0], TILES), dtype=np.int64)
            for t in range(TILES):
                counts[:, t] = nz[:, t * per_block:(t + 1) * per_block].sum(axis=1)
            for s in range(n_stages):
                for k, v in stage_cycles(counts, s * ROWS_PER_STAGE, tag).items():
                    acc[k] = acc.get(k, 0) + v
        acc["rmsnorm x2"] = 2 * RMSNORM
        acc["residual x2"] = 2 * RESIDUAL
        acc["tile loads x3"] = 3 * TILE_LOAD
        acc["attention x12"] = 12 * HEAD_CORE_AVG
        per_layer.append(sum(acc.values()))
        for k, v in acc.items():
            totals[k] = totals.get(k, 0) + v
        print(f"  layer {L:2d}: {sum(acc.values()):>10,} cycles")

    n = len(layer_ids)
    mean = {k: v / n for k, v in totals.items()}
    total = sum(mean.values())
    print(f"\n{'component':<20}{'cycles/layer':>14}{'share':>9}")
    for k, v in sorted(mean.items(), key=lambda kv: -kv[1]):
        print(f"{k:<20}{v:>14,.0f}{100 * v / total:>8.1f}%")
    print(f"{'TOTAL':<20}{total:>14,.0f}")

    alt = total - 12 * HEAD_CORE_AVG + 12 * HEAD_CORE_WORST
    print(f"\nwith attention at its worst case instead of average: {alt:,.0f}")
    print(f"board measured: {args.measured:.2f} M  ->  model accounts for "
          f"{100 * total / (args.measured * 1e6):.0f}% "
          f"({100 * alt / (args.measured * 1e6):.0f}% at worst-case attention)")

    # The two defects this exposes, quantified.
    idle = 108 * TILES * BANK_WORD_LATENCY
    print(f"\nof which, recoverable without changing the datapath:")
    print(f"  BANK_TILE not pipelined: {idle:,} cycles/layer "
          f"({100 * idle / total:.1f}%) -- 12 blocks x 74 cycles of memory latency x 108 stages")


if __name__ == "__main__":
    main()
