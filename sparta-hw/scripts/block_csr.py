#!/usr/bin/env python3
"""Block-CSR layout and Latin-square schedule model for tiled SpMM (Plan 02, step 2).

For a weight W (R x C) and a feature-major activation A (C x N), split W into an n x n grid of
blocks W_ij (row block i = output rows, column block j = input features).  Worker (i, j) reads only
A_j and writes only Y_i.  A round runs n workers whose (i, j) pairs share no i and no j, so they never
collide; n rounds cover all blocks (a Latin square).  Inside a block the nonzeros are row-major, so the
row being accumulated lives in a register: read Y[r] on entering the row, write it back on leaving.

This module provides, working from a compiled payload (compile_model.py output: manifest.json +
model.bin, POT weight bytes):

  * balancing permutations that respect the encoder's permutation-invariance constraints
    (see compile_model.apply_permutations): one stream order pi for all layers, a hidden order sigma
    per layer, and for attention a head order shared by the Q/K and V/O sides plus within-head orders;
  * the block-CSR layout: per-block row counts + bit-packed {POT value nibble, block-local column};
  * a cycle model comparing, at an equal number of parallel lanes L:
      today      - the current loop: 4 row lanes, 16-token tiles, inner-loop re-entry per nonzero
      row lanes  - L lanes over rows sorted by nnz, whole N-token row per step (needs L copies of A)
      blocks     - L = n Latin-square workers, whole N-token row per step (one partitioned copy of A)
  * an exactness check: pack -> unpack -> execute the schedule on random int8 activations and compare
    with the dense product.

The cycle model is a first-order count, not a timing result: 1 cycle per nonzero step, `--row-switch`
cycles to enter a row (accumulator read), `--round-switch` cycles per round (loop exit/entry, table load).

Usage (numpy only):
    python scripts/block_csr.py analyze --payload sim/e2e --n 4 8 12
    python scripts/block_csr.py verify  --payload sim/e2e --n 8 --layer 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

WEIGHTS = ["w_q", "w_k", "w_v", "out_proj", "linear_1", "linear_2"]
N_HEADS = 12
N_TOKENS = 50
TODAY_LANES, TODAY_TILE = 4, 16


# ----------------------------------------------------------------------------
# payload access
# ----------------------------------------------------------------------------
@dataclass
class Coo:
    """A weight as row-major sorted coordinates with raw POT bytes (bit7 sign, bits0..6 exponent)."""
    rows: np.ndarray
    cols: np.ndarray
    vals: np.ndarray
    shape: tuple

    @staticmethod
    def build(rows, cols, vals, shape):
        order = np.lexsort((cols, rows))
        return Coo(rows[order], cols[order], vals[order], tuple(shape))

    def row_nnz(self):
        return np.bincount(self.rows, minlength=self.shape[0])

    def col_nnz(self):
        return np.bincount(self.cols, minlength=self.shape[1])

    def permuted(self, row_perm, col_perm):
        """perm[old] = new."""
        return Coo.build(row_perm[self.rows], col_perm[self.cols], self.vals, self.shape)

    def weights(self):
        mag = np.left_shift(1, (self.vals & 0x7F).astype(np.int64))
        return np.where(self.vals & 0x80, -mag, mag)

    def dense_product(self, act):
        """Exact Y = W . A for A (C x N) -> (R x N) int64."""
        w = np.zeros(self.shape, dtype=np.int64)
        w[self.rows, self.cols] = self.weights()
        return w @ act.astype(np.int64)


class Payload:
    def __init__(self, directory):
        self.man = json.load(open(os.path.join(directory, "manifest.json")))
        if self.man.get("format") != "pot":
            raise SystemExit("block_csr expects a POT-weight payload (compile_model pot=True)")
        self.blob = np.memmap(os.path.join(directory, self.man["bin"]), dtype=np.uint8, mode="r")

    def _arr(self, d):
        return np.frombuffer(self.blob[d["offset"]:d["offset"] + d["bytes"]].tobytes(), dtype=d["dtype"])

    def layers(self):
        return sorted(int(k) for k in self.man["layers"])

    def weight(self, layer, tag):
        w = self.man["layers"][str(layer)]["weights"][tag]
        ptr = self._arr(w["rowptr"]).astype(np.int64)
        idx = self._arr(w["col"]).astype(np.int64)
        vals = self._arr(w["values"])
        shape = w["shape"]
        if w["layout"] == "csc":   # (values, row index, column pointer)
            cols = np.repeat(np.arange(shape[1]), np.diff(ptr))
            return Coo.build(idx, cols, vals, shape)
        rows = np.repeat(np.arange(shape[0]), np.diff(ptr))
        return Coo.build(rows, idx, vals, shape)


# ----------------------------------------------------------------------------
# balancing permutations
# ----------------------------------------------------------------------------
def balanced_order(loads, n):
    """perm[old] = new: n equal-size bins filled heaviest-item-first into the lightest open bin."""
    m = len(loads)
    cap = m // n
    fill, total = np.zeros(n, dtype=int), np.zeros(n)
    members = [[] for _ in range(n)]
    for item in np.argsort(-np.asarray(loads, dtype=float), kind="stable"):
        open_bins = np.flatnonzero(fill < cap)
        b = open_bins[np.argmin(total[open_bins])]
        members[b].append(item)
        fill[b] += 1
        total[b] += loads[item]
    perm = np.empty(m, dtype=int)
    for b, items in enumerate(members):
        perm[np.asarray(items)] = b * cap + np.arange(len(items))
    return perm


def _norm(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum()


def head_orders(qk_loads, vo_loads, n, rng, iters=6000):
    """Head order theta (shared by both sides) + within-head orders for the Q/K and V/O sides.

    Local search over moves allowed by the invariance: swap two whole heads (both sides), or swap two
    features within a head on one side.  Objective: sum of squared row-block loads on both sides.
    Returns (qk_perm, vo_perm), each perm[old] = new over d_model features."""
    d_head = len(qk_loads) // N_HEADS
    head_of = np.arange(len(qk_loads)) // d_head
    band = np.arange(N_HEADS)                                   # band[head] = position of that head
    local = [np.tile(np.arange(d_head), N_HEADS), np.tile(np.arange(d_head), N_HEADS)]
    loads = [np.asarray(qk_loads, float), np.asarray(vo_loads, float)]
    block = len(qk_loads) // n

    def perm_of(side):
        return band[head_of] * d_head + local[side]

    def cost():
        return sum(float((np.bincount(perm_of(s) // block, weights=loads[s], minlength=n) ** 2).sum())
                   for s in (0, 1))

    best = cost()
    for _ in range(iters):
        if rng.random() < 0.2:
            h1, h2 = rng.choice(N_HEADS, 2, replace=False)
            band[h1], band[h2] = band[h2], band[h1]
            new = cost()
            if new < best:
                best = new
            else:
                band[h1], band[h2] = band[h2], band[h1]
        else:
            side = int(rng.integers(2))
            h = int(rng.integers(N_HEADS))
            a, b = h * d_head + rng.choice(d_head, 2, replace=False)
            local[side][a], local[side][b] = local[side][b], local[side][a]
            new = cost()
            if new < best:
                best = new
            else:
                local[side][a], local[side][b] = local[side][b], local[side][a]
    return perm_of(0), perm_of(1)


def feasible_permutations(payload, n, seed=0):
    """Balancing permutations for n x n blocks that keep the encoder bit-exact.

    Returns {layer: {tag: (row_perm, col_perm)}}."""
    rng = np.random.default_rng(seed)
    layers = payload.layers()
    w = {L: {t: payload.weight(L, t) for t in WEIGHTS} for L in layers}

    stream = sum(_norm(w[L][t].col_nnz()) for L in layers for t in ("w_q", "w_k", "w_v", "linear_1"))
    stream = stream + sum(_norm(w[L][t].row_nnz()) for L in layers for t in ("out_proj", "linear_2"))
    pi = balanced_order(stream, n)

    out = {}
    for L in layers:
        W = w[L]
        sigma = balanced_order(_norm(W["linear_1"].row_nnz()) + _norm(W["linear_2"].col_nnz()), n)
        qk, vo = head_orders(_norm(W["w_q"].row_nnz()) + _norm(W["w_k"].row_nnz()),
                             _norm(W["w_v"].row_nnz()) + _norm(W["out_proj"].col_nnz()), n, rng)
        out[L] = {"w_q": (qk, pi), "w_k": (qk, pi), "w_v": (vo, pi),
                  "out_proj": (pi, vo), "linear_1": (sigma, pi), "linear_2": (pi, sigma)}
    return out


def unconstrained_permutations(coo, n):
    """Per-matrix row + column balancing, ignoring invariance constraints (an upper bound)."""
    return balanced_order(coo.row_nnz(), n), balanced_order(coo.col_nnz(), n)


# ----------------------------------------------------------------------------
# block-CSR layout (reference format)
# ----------------------------------------------------------------------------
def bits_for(count):
    return max(1, math.ceil(math.log2(count))) if count > 1 else 1


@dataclass
class BlockCsr:
    """n x n blocks of a row-major weight.

    row_counts[i][j] : (R/n,) nonzeros per local row of block (i, j)
    words[i][j]      : uint64 bit stream, fields {val nibble (4 b) | local column (col_bits)}
    nnz[i, j]        : nonzeros per block
    """
    n: int
    shape: tuple
    col_bits: int
    row_counts: list
    words: list
    nnz: np.ndarray

    def field_bits(self):
        return 4 + self.col_bits

    def storage_bits(self):
        count_bits = bits_for(max(int(rc.max()) for row in self.row_counts for rc in row) + 1)
        return int(self.nnz.sum()) * self.field_bits() + self.n * self.shape[0] * count_bits


def _pack(fields, width):
    per_word = 64 // width
    words = np.zeros(max(1, math.ceil(len(fields) / per_word)), dtype=np.uint64)
    for k in range(per_word):
        chunk = fields[k::per_word].astype(np.uint64)
        words[:len(chunk)] |= chunk << np.uint64(k * width)
    return words


def _unpack(words, count, width):
    per_word = 64 // width
    mask = np.uint64((1 << width) - 1)
    lanes = [(words >> np.uint64(k * width)) & mask for k in range(per_word)]
    return np.stack(lanes, axis=1).reshape(-1)[:count].astype(np.int64)


def to_block_csr(coo, n):
    R, C = coo.shape
    rb, cb = R // n, C // n
    col_bits = bits_for(cb)
    nibble = ((coo.vals >> 4) & 0x8 | (coo.vals & 0x7)).astype(np.int64)   # {sign(bit3), exp(bits0..2)}
    if np.any((coo.vals & 0x7F) > 7):
        raise ValueError("POT exponent does not fit the 3-bit on-chip nibble")
    bi, bj = coo.rows // rb, coo.cols // cb
    row_counts = [[None] * n for _ in range(n)]
    words = [[None] * n for _ in range(n)]
    nnz = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            sel = (bi == i) & (bj == j)                   # row-major order preserved by the mask
            nnz[i, j] = int(sel.sum())
            row_counts[i][j] = np.bincount(coo.rows[sel] - i * rb, minlength=rb)
            fields = nibble[sel] | ((coo.cols[sel] - j * cb) << 4)
            words[i][j] = _pack(fields, 4 + col_bits)
    return BlockCsr(n, (R, C), col_bits, row_counts, words, nnz)


# ----------------------------------------------------------------------------
# schedule + cycle model
# ----------------------------------------------------------------------------
def latin_square(cost, rng, trials=300):
    """n rounds of n non-colliding blocks: round k runs (i, col_label[(row_label[i] + k) % n]).

    Random search over row/column block relabelings minimizing sum over rounds of the slowest block.
    Returns (rounds, total) with rounds[k] = list of (i, j)."""
    n = cost.shape[0]
    best_total, best = None, None
    for t in range(trials):
        rl = np.arange(n) if t == 0 else rng.permutation(n)
        cl = np.arange(n) if t == 0 else rng.permutation(n)
        rounds = [[(i, int(cl[(rl[i] + k) % n])) for i in range(n)] for k in range(n)]
        total = sum(max(cost[i, j] for i, j in r) for r in rounds)
        if best_total is None or total < best_total:
            best_total, best = total, rounds
    return best, best_total


def lane_cycles(row_nnz, lanes, row_switch):
    """`lanes` rows advance in lockstep over rows sorted by nnz; each group waits for its longest row."""
    nz = np.sort(row_nnz[row_nnz > 0])[::-1]
    nz = np.concatenate([nz, np.zeros((-len(nz)) % lanes, dtype=nz.dtype)])
    groups = nz.reshape(-1, lanes)
    return int(groups.max(1).sum()) + len(groups) * row_switch


def block_cycles(blocks, rng, row_switch, round_switch, lanes_per_worker=1):
    """Latin-square rounds; each worker runs `lanes_per_worker` row lanes over its own block
    (dual-port memories give 2 per A_j / Y_i without copies)."""
    cost = np.array([[lane_cycles(blocks.row_counts[i][j], lanes_per_worker, row_switch)
                      for j in range(blocks.n)] for i in range(blocks.n)])
    _, total = latin_square(cost, rng)
    return total + blocks.n * round_switch


def row_lane_cycles(coo, lanes, row_switch):
    return lane_cycles(coo.row_nnz(), lanes, row_switch)


def today_cycles(coo, reentry):
    nz = coo.row_nnz()
    groups = nz[:len(nz) // TODAY_LANES * TODAY_LANES].reshape(-1, TODAY_LANES)
    tiles = math.ceil(N_TOKENS / TODAY_TILE)
    return int(groups.max(1).sum()) * (tiles + reentry)


def today_storage_bits(coo):
    R, C = coo.shape
    return len(coo.rows) * (4 + bits_for(C)) + (R + 1) * 32


# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def analyze(args):
    payload = Payload(args.payload)
    layers = payload.layers()
    rng = np.random.default_rng(args.seed)
    base = {L: {t: payload.weight(L, t) for t in WEIGHTS} for L in layers}
    nnz_total = {t: sum(len(base[L][t].rows) for L in layers) for t in WEIGHTS}

    print("Cycle model (12-layer sums, millions of MAC steps; whole N-token row per step except 'today')")
    print(f"assumptions: row switch = {args.row_switch} cycle(s), round switch = {args.round_switch}, "
          f"today inner-loop re-entry = {args.reentry} per nonzero, today tiles = "
          f"{math.ceil(N_TOKENS / TODAY_TILE)} x {TODAY_TILE} tokens")
    today = {t: sum(today_cycles(base[L][t], args.reentry) for L in layers) for t in WEIGHTS[:5]}
    print("\ntoday (4 lanes, current loop; linear_2 uses the H-driven scatter instead):  " +
          "  ".join(f"{t}={today[t]/1e6:.2f}" for t in WEIGHTS[:5]))

    k = args.lanes_per_worker
    for n in args.n:
        feas = feasible_permutations(payload, n, seed=args.seed)
        lanes = n * k
        print(f"\n== L = {lanes} lanes: row lanes {lanes} | blocks {n} x {n} with {k} lane(s) per worker ==")
        print(f"{'weight':9s} {'ideal':>6s} {'rowlanes':>8s} {'blk-id':>7s} {'blk-feas':>8s} {'blk-ub':>7s} "
              f"| overhead vs ideal: rowlanes blk-id blk-feas blk-ub | storage vs today (feas)")
        totals = np.zeros(5)
        for t in WEIGHTS:
            ideal = nnz_total[t] / lanes
            rl = bi = bf = bu = 0
            st_today = st_blk = 0
            for L in layers:
                coo = base[L][t]
                rl += row_lane_cycles(coo, lanes, args.row_switch)
                bi += block_cycles(to_block_csr(coo, n), rng, args.row_switch, args.round_switch, k)
                blk = to_block_csr(coo.permuted(*feas[L][t]), n)
                bf += block_cycles(blk, rng, args.row_switch, args.round_switch, k)
                bu += block_cycles(to_block_csr(coo.permuted(*unconstrained_permutations(coo, n)), n),
                                   rng, args.row_switch, args.round_switch, k)
                st_today += today_storage_bits(coo)
                st_blk += blk.storage_bits()
            totals += np.array([ideal, rl, bi, bf, bu])
            print(f"{t:9s} {ideal/1e6:6.2f} {rl/1e6:8.2f} {bi/1e6:7.2f} {bf/1e6:8.2f} {bu/1e6:7.2f} "
                  f"|                    {rl/ideal:5.2f}  {bi/ideal:5.2f}   {bf/ideal:5.2f}   {bu/ideal:5.2f} "
                  f"| {st_blk/st_today:.2f}x")
        ideal, rl, bi, bf, bu = totals
        print(f"{'all':9s} {ideal/1e6:6.2f} {rl/1e6:8.2f} {bi/1e6:7.2f} {bf/1e6:8.2f} {bu/1e6:7.2f} "
              f"|                    {rl/ideal:5.2f}  {bi/ideal:5.2f}   {bf/ideal:5.2f}   {bu/ideal:5.2f}")
        print(f"activation copies needed: row lanes {lanes}, blocks 1 (partitioned into {n}, "
              f"{k} reader(s) per block)")
    return 0


def execute_schedule(blocks, act, rng):
    """Run the Latin-square rounds from the packed layout; returns Y (R x N) int64."""
    n = blocks.n
    R, C = blocks.shape
    rb, cb = R // n, C // n
    width = blocks.field_bits()
    y = np.zeros((R, act.shape[1]), dtype=np.int64)
    rounds, _ = latin_square(blocks.nnz, rng, trials=10)
    seen = np.zeros((n, n), dtype=int)
    for r in rounds:
        assert len({i for i, _ in r}) == n and len({j for _, j in r}) == n, "round collides"
        for i, j in r:                                   # independent workers in hardware
            seen[i, j] += 1
            fields = _unpack(blocks.words[i][j], int(blocks.nnz[i, j]), width)
            exp, neg, col = fields & 0x7, (fields >> 3) & 1, (fields >> 4) + j * cb
            w = np.where(neg == 1, -(1 << exp), 1 << exp)
            a_j = act                                     # worker reads only rows [j*cb, (j+1)*cb)
            pos = 0
            for local_row, count in enumerate(blocks.row_counts[i][j]):
                if count == 0:
                    continue
                row = i * rb + local_row
                sl = slice(pos, pos + count)
                assert np.all((col[sl] >= j * cb) & (col[sl] < (j + 1) * cb))
                acc = y[row].copy()                       # enter row: read accumulator
                acc += (w[sl, None] * a_j[col[sl]]).sum(0)
                y[row] = acc                              # leave row: write back
                pos += count
    assert np.all(seen == 1), "every block must run exactly once"
    return y


def verify(args):
    payload = Payload(args.payload)
    rng = np.random.default_rng(args.seed)
    feas = feasible_permutations(payload, args.n, seed=args.seed)
    ok = True
    for t in WEIGHTS:
        coo = payload.weight(args.layer, t).permuted(*feas[args.layer][t])
        blocks = to_block_csr(coo, args.n)
        act = rng.integers(-128, 128, size=(coo.shape[1], N_TOKENS))
        y = execute_schedule(blocks, act, rng)
        exact = np.array_equal(y, coo.dense_product(act))
        ok &= exact
        print(f"layer {args.layer} {t:9s} n={args.n}: blocks nnz min/max {blocks.nnz.min()}/{blocks.nnz.max()}, "
              f"col bits {blocks.col_bits}, {'bit-exact' if exact else 'MISMATCH'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--payload", required=True)
    a.add_argument("--n", type=int, nargs="+", default=[4, 8, 12])
    a.add_argument("--row-switch", type=int, default=1)
    a.add_argument("--round-switch", type=int, default=3)
    a.add_argument("--reentry", type=int, default=2)
    a.add_argument("--lanes-per-worker", type=int, default=1,
                   help="row lanes inside each block worker (2 = one per port of a dual-port memory)")
    a.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--payload", required=True)
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--layer", type=int, default=0)
    v.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return analyze(args) if args.cmd == "analyze" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
