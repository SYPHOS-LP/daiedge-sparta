#!/usr/bin/env python3
"""What each engine schedule costs, and whether the column assignment still matters under it.

Three schedules, all on the real weights, all in nonzero-steps (one step = MHA_BANK_ROW_WORDS
cycles, 7 at N = 50):

  lockstep   what the engine does now.  The 12 blocks all work on ONE output row and the row ends
             when the slowest finishes:      sum over rows of (max over blocks)

  latin      12 output rows in flight, blocks assigned to rows on a rotating diagonal, all workers
             stepping to the next block together:
                                            sum over phases of (max over the diagonal)

  free       each block streams through the stage's rows at its own pace.  Nothing is shared
             between blocks -- separate weight memories, separate activation tiles -- so there is no
             contention; the stage ends when the slowest BLOCK is done:
                                            max over blocks of (sum over rows)
             The price is accumulators for the rows in flight, which this also reports as the drift
             between the leading and trailing block.

The point of the comparison: lockstep and latin are hurt by per-ROW structure, which a column
assignment can fix.  free depends only on each block's TOTAL, which no assignment changes -- the
total number of nonzeros is what it is.  So the question is whether the gather still earns anything
once the schedule is free-running.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lane_assign import WEIGHTS, WMODULE, greedy, contiguous   # noqa: E402

ROWS_PER_STAGE = 64      # MHA_FEATURE_HEAD: the rows one loaded weight block holds


def balance_totals(nz_list, n_cols, T):
    """Equalise each group's TOTAL nonzeros, ignoring the per-row spread.

    This is all a free-running schedule cares about, and it is a far easier problem than the per-row
    objective: longest-processing-time first, i.e. place the densest column in the emptiest group
    that still has room.
    """
    per_col = sum(nz.sum(axis=0) for nz in nz_list).astype(np.int64)
    size = n_cols // T
    groups = np.full(n_cols, -1, dtype=np.int32)
    load = np.zeros(T, dtype=np.int64)
    room = np.full(T, size, dtype=np.int64)
    for c in np.argsort(-per_col):
        t = int(np.lexsort((load, np.where(room > 0, 0, 1)))[0])
        groups[c] = t
        load[t] += per_col[c]
        room[t] -= 1
    return groups


def counts_of(nz, groups, T):
    c = np.zeros((nz.shape[0], T), dtype=np.int64)
    for t in range(T):
        c[:, t] = nz[:, groups == t].sum(axis=1)
    return c


def lockstep(c):
    return int(c.max(axis=1).sum())


def latin(c, T):
    """12 rows in flight on a rotating diagonal, workers stepping together."""
    total = 0
    for s in range(0, c.shape[0], T):
        blk = c[s:s + T]                      # up to T rows
        n = blk.shape[0]
        for p in range(T):
            # worker w holds row w and works block (w + p) % T
            total += int(max(blk[w, (w + p) % T] for w in range(n)))
    return total


def free(c, T, rows_per_stage=ROWS_PER_STAGE):
    """Each block streams its own rows; the stage ends with the slowest block."""
    total, drift = 0, 0
    for s in range(0, c.shape[0], rows_per_stage):
        blk = c[s:s + rows_per_stage]
        per_block = blk.sum(axis=0)           # cycles each block needs for this stage
        total += int(per_block.max())
        # How far apart the blocks get: at the moment the fastest finishes, how many rows behind is
        # the slowest?  Bounds the accumulator ring.
        cum = np.cumsum(blk, axis=0)          # [rows, T]
        done_fast = per_block.min()
        behind = (cum <= done_fast).sum(axis=0)
        drift = max(drift, int(behind.max() - behind.min()))
    return total, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    import compile_model as C
    model = C.compile_model(args.input)
    layer_ids = model.layers() if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    T = args.tiles

    # [schedule][assignment] -> steps
    KINDS = ("contig", "gather", "totals")
    acc = {s: {a: 0 for a in KINDS} for s in ("lockstep", "latin", "free")}
    ideal = 0
    drift_max = {a: 0 for a in KINDS}
    block_skew = {a: [] for a in KINDS}

    for L in layer_ids:
        nz = {w: (np.asarray(model.linears[f"encoder.encoder_layer_{L}.{WMODULE[w]}"].weight_int) != 0)
              for w in WEIGHTS}
        share = [["w_q", "w_k", "w_v"], ["out_proj"], ["linear_1"], ["linear_2"]]
        for grp in share:
            n_cols = nz[grp[0]].shape[1]
            mats = [nz[w] for w in grp]
            assign = {"contig": contiguous(n_cols, T),
                      "gather": greedy(mats, n_cols, T),
                      "totals": balance_totals(mats, n_cols, T)}
            for w in grp:
                ideal += int(nz[w].sum()) / T
                for a, g in assign.items():
                    c = counts_of(nz[w], g, T)
                    acc["lockstep"][a] += lockstep(c)
                    acc["latin"][a] += latin(c, T)
                    f, d = free(c, T)
                    acc["free"][a] += f
                    drift_max[a] = max(drift_max[a], d)
                    # how uneven each block's TOTAL is, per stage
                    for s in range(0, c.shape[0], ROWS_PER_STAGE):
                        pb = c[s:s + ROWS_PER_STAGE].sum(axis=0)
                        block_skew[a].append(pb.max() / pb.mean())
        print(f"  layer {L} done")

    print(f"\n{'schedule':<12}" + "".join(f"{k:>14}" for k in KINDS) +
          "".join(f"{k:>9}" for k in KINDS))
    for s in ("lockstep", "latin", "free"):
        print(f"{s:<12}" + "".join(f"{acc[s][k]:>14d}" for k in KINDS) +
              "".join(f"{acc[s][k]/ideal:>8.2f}x" for k in KINDS))
    print(f"{'ideal':<12}{ideal:>14.0f}")

    print(f"\nper-stage block skew (max block total / mean), which is what 'free' pays:")
    for a in KINDS:
        v = np.array(block_skew[a])
        print(f"  {a:<8} mean {v.mean():.3f}   p95 {np.percentile(v,95):.3f}   max {v.max():.3f}")
    print("\nrows of drift between fastest and slowest block (sets the accumulator ring): "
          + ", ".join(f"{a} {drift_max[a]}" for a in KINDS))

    cyc = lambda steps: steps * 7 / len(layer_ids)
    print(f"\ncycles per layer (7 per step):")
    for s in ("lockstep", "latin", "free"):
        print(f"  {s:<10}" + "".join(f"  {k} {cyc(acc[s][k]):>10,.0f}" for k in KINDS))


if __name__ == "__main__":
    main()
