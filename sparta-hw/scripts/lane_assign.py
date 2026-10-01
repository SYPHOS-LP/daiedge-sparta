#!/usr/bin/env python3
"""How much lane idling a per-matrix column assignment would remove.

The engine splits a weight's columns into MHA_TILES groups; a row costs as long as the group with
the most nonzeros in it, so the cost is  sum over rows of (max over groups), while the work is
sum over rows of (mean over groups).  Today a group is a CONTIGUOUS range of columns, which nothing
requires -- load_activation_tiles is a copy and can gather any set of columns into a group.

This measures what a deliberately chosen assignment would buy, on the real weights, before any
hardware is written.  Groups are kept EQUAL SIZE so the activation tiles keep their current depth.

Which matrices may have their own assignment is set by how often the tiles are loaded:

  qkv       w_q, w_k, w_v share one load of the normalized input -> ONE shared assignment
  out_proj  its own load (the attention result)                  -> its own
  linear_1  its own load (the second normalized input)           -> its own
  linear_2  reads H, which the linear_1 tail writes into the tiles, so where each hidden feature
            lands is equally free                                -> its own

The "split_qkv" variant asks what separating w_q/w_k/w_v would add, at the price of two more tile
loads per layer.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WEIGHTS = ["w_q", "w_k", "w_v", "out_proj", "linear_1", "linear_2"]
WMODULE = {"w_q": "mha.w_q", "w_k": "mha.w_k", "w_v": "mha.w_v",
           "out_proj": "mha.out_proj", "linear_1": "mlp.linear_1", "linear_2": "mlp.linear_2"}


def cost(nz, groups, n_groups):
    """sum over rows of (max over groups of that row's nonzeros in the group).

    nz      boolean [rows, cols]
    groups  int [cols] -> group index
    """
    counts = np.zeros((nz.shape[0], n_groups), dtype=np.int32)
    for t in range(n_groups):
        counts[:, t] = nz[:, groups == t].sum(axis=1)
    return int(counts.max(axis=1).sum())


def contiguous(n_cols, n_groups):
    return np.repeat(np.arange(n_groups), n_cols // n_groups)


def greedy(nz_list, n_cols, n_groups, seed_order=True):
    """Assign columns to equal-size groups, minimising sum over rows of the row maximum.

    nz_list is a list of boolean [rows, cols] matrices that must SHARE one assignment (w_q/w_k/w_v),
    so the objective is the sum of their costs.

    Placing column c in group t raises count[r][t] by one for every row r where c is nonzero, and
    that raises the row's maximum only where count[r][t] was already the maximum.  So the exact
    increase in the objective is the number of such rows, which is cheap to evaluate for all groups
    at once.
    """
    size = n_cols // n_groups
    counts = [np.zeros((nz.shape[0], n_groups), dtype=np.int32) for nz in nz_list]
    curmax = [np.zeros(nz.shape[0], dtype=np.int32) for nz in nz_list]
    load = np.zeros(n_groups, dtype=np.int64)
    groups = np.full(n_cols, -1, dtype=np.int32)

    # Densest columns first: they constrain the result most, and placing them while every group is
    # still empty is what lets the sparse ones fill the gaps afterwards.
    per_col = sum(nz.sum(axis=0) for nz in nz_list)
    order = np.argsort(-per_col) if seed_order else np.arange(n_cols)

    for c in order:
        delta = np.zeros(n_groups, dtype=np.int64)
        rows = []
        for m, nz in enumerate(nz_list):
            s = np.flatnonzero(nz[:, c])
            rows.append(s)
            if s.size:
                delta += (counts[m][s] == curmax[m][s][:, None]).sum(axis=0)
        delta = delta.astype(np.float64)
        delta[load >= size] = np.inf
        # Tie-break towards the emptiest group, which keeps later columns free to choose.
        t = int(np.lexsort((load, delta))[0])
        groups[c] = t
        load[t] += 1
        for m, s in enumerate(rows):
            if s.size:
                counts[m][s, t] += 1
                # Fancy indexing returns a copy, so the running maximum has to be assigned back.
                curmax[m][s] = counts[m][s].max(axis=1)
    assert (groups >= 0).all() and (load == size).all()
    return groups


def refine(nz_list, groups, n_groups, rounds=12, partners=4, seed=0):
    """Improve an assignment by swapping pairs of columns between groups.

    Groups must stay equal size, so a column cannot simply move; it has to trade places with one
    from another group.  A swap only changes the rows where either column is nonzero, so its effect
    on the objective is evaluated on those rows alone.  First improvement wins, which is much faster
    than scanning every partner and converges to the same kind of result.
    """
    rng = np.random.default_rng(seed)
    n_cols = groups.size
    S = [[np.flatnonzero(nz[:, c]) for c in range(n_cols)] for nz in nz_list]
    counts, curmax = [], []
    for nz in nz_list:
        cnt = np.zeros((nz.shape[0], n_groups), dtype=np.int32)
        for t in range(n_groups):
            cnt[:, t] = nz[:, groups == t].sum(axis=1)
        counts.append(cnt)
        curmax.append(cnt.max(axis=1))

    for _ in range(rounds):
        taken = 0
        for c in rng.permutation(n_cols):
            a = int(groups[c])
            for b in rng.permutation(n_groups)[:partners]:
                b = int(b)
                if b == a:
                    continue
                pool = np.flatnonzero(groups == b)
                d = int(pool[rng.integers(pool.size)])
                delta, staged = 0, []
                for m in range(len(nz_list)):
                    rows = np.union1d(S[m][c], S[m][d])
                    if rows.size == 0:
                        staged.append(None)
                        continue
                    sub = counts[m][rows].copy()
                    mc, md = np.isin(rows, S[m][c]), np.isin(rows, S[m][d])
                    sub[mc, a] -= 1; sub[mc, b] += 1
                    sub[md, b] -= 1; sub[md, a] += 1
                    nm = sub.max(axis=1)
                    delta += int(nm.sum()) - int(curmax[m][rows].sum())
                    staged.append((rows, sub, nm))
                if delta < 0:
                    groups[c], groups[d] = b, a
                    for m, st in enumerate(staged):
                        if st is None:
                            continue
                        rows, sub, nm = st
                        counts[m][rows] = sub
                        curmax[m][rows] = nm
                    taken += 1
                    break
        if taken == 0:
            break
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--random", type=int, default=0,
                    help="instead of the greedy, use N seeded random equal-size assignments "
                         "and report the best; tests whether the search is doing any work")
    ap.add_argument("--refine", type=int, default=0,
                    help="rounds of pairwise swap improvement after the greedy pass")
    ap.add_argument("--split-qkv", action="store_true",
                    help="give w_q/w_k/w_v their own assignments (costs two extra tile loads)")
    args = ap.parse_args()

    import compile_model as C
    t0 = time.time()
    model = C.compile_model(args.input)
    layer_ids = model.layers() if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"loaded {len(layer_ids)} layers in {time.time()-t0:.1f}s, {args.tiles} groups\n")

    T = args.tiles
    tot = {w: [0, 0, 0, 0] for w in WEIGHTS}  # [contiguous, greedy, nnz, floor]
    per_layer = []

    for L in layer_ids:
        nz = {}
        for w in WEIGHTS:
            wi = model.linears[f"encoder.encoder_layer_{L}.{WMODULE[w]}"].weight_int
            nz[w] = (np.asarray(wi) != 0)
        # groups that must share one assignment
        share = [["w_q"], ["w_k"], ["w_v"]] if args.split_qkv else [["w_q", "w_k", "w_v"]]
        share += [["out_proj"], ["linear_1"], ["linear_2"]]

        row = {}
        for grp in share:
            n_cols = nz[grp[0]].shape[1]
            g_base = contiguous(n_cols, T)
            if args.random:
                # A random equal partition already removes whatever structure made the CONTIGUOUS
                # ranges uneven.  If it matches the greedy, the search is not earning its keep.
                mats, best = [nz[w] for w in grp], None
                for k in range(args.random):
                    g = np.random.default_rng(1000 * L + k).permutation(
                        np.repeat(np.arange(T), n_cols // T)).astype(np.int32)
                    c = sum(cost(m, g, T) for m in mats)
                    if best is None or c < best[0]:
                        best = (c, g)
                g_opt = best[1]
            else:
                g_opt = greedy([nz[w] for w in grp], n_cols, T)
            if args.refine:
                g_opt = refine([nz[w] for w in grp], g_opt, T, rounds=args.refine, seed=L)
            for w in grp:
                base, opt = cost(nz[w], g_base, T), cost(nz[w], g_opt, T)
                ideal = int(nz[w].sum()) / T
                # No assignment can beat this: a row with n nonzeros spread over T groups still
                # leaves one group holding ceil(n / T) of them.
                floor = int(np.ceil(nz[w].sum(axis=1) / T).sum())
                tot[w][0] += base; tot[w][1] += opt; tot[w][2] += int(nz[w].sum())
                tot[w][3] += floor
                row[w] = (base, opt, ideal)
        per_layer.append((L, row))
        b = sum(v[0] for v in row.values()); o = sum(v[1] for v in row.values())
        i = sum(v[2] for v in row.values())
        print(f"  layer {L:2d}   contiguous {b/i:5.2f}x   assigned {o/i:5.2f}x")

    print(f"\n{'weight':<10}{'contiguous':>12}{'assigned':>12}{'floor':>12}"
          f"{'was':>8}{'now':>8}{'best':>8}")
    gb = go = gi = gf = 0
    for w in WEIGHTS:
        base, opt, nnz, floor = tot[w]
        ideal = nnz / T
        gb += base; go += opt; gi += ideal; gf += floor
        print(f"{w:<10}{base:>12d}{opt:>12d}{floor:>12d}"
              f"{base/ideal:>7.2f}x{opt/ideal:>7.2f}x{floor/ideal:>7.2f}x")
    print(f"{'ALL':<10}{gb:>12d}{go:>12d}{gf:>12d}"
          f"{gb/gi:>7.2f}x{go/gi:>7.2f}x{gf/gi:>7.2f}x")
    print(f"\nheadroom left to the floor: {go-gf} nonzero-steps "
          f"= {(go-gf)*7/len(layer_ids):,.0f} cycles per layer")

    # A nonzero costs MHA_BANK_ROW_WORDS cycles (7 at N=50, 8 tokens per word).
    saved = (gb - go) * 7 / len(layer_ids)
    print(f"\nsaved: {gb-go} nonzero-steps over {len(layer_ids)} layers "
          f"= {saved:,.0f} cycles per layer at 7 cycles each")


if __name__ == "__main__":
    main()
