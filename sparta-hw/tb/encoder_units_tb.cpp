/**********************************************************************************
 * Unit testbench — the cases the golden data cannot reach
 *
 * `make golden-check` compares 32 real images bit-exact and is the authority on whether the kernel
 * is correct.  What it cannot do is reach states the trained weights never produce: an output row
 * with no nonzeros, a token count that is not a multiple of the packing, saturation at both ends,
 * or an accumulator large enough to wrap the requantize cast.  Those are what this file covers.
 *
 * Everything here is exact-integer, so every check is an equality against a reference computed in
 * the obvious way.  Where a check pins down behaviour that is arguably wrong but is what the golden
 * data encodes (see docs/plans/00-open-issues.md), it says so, so that changing it is a deliberate
 * act and not a surprise.
 *
 *   make native-units
 *********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "spmm_tiled.h"
#include "quantize.h"
#include "residual.h"

static int g_failures = 0;
static int g_checks   = 0;

static void check(bool ok, const char *what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        std::printf("  FAIL  %s\n", what);
    }
}

static void check_eq(long got, long want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        std::printf("  FAIL  %s: got %ld, want %ld\n", what, got, want);
    }
}

/* ============================================================================
 * Building a packed weight payload, the format compile_model.py emits
 *
 * counts blob: MHA_COUNT_HDR_WORDS transfers of int32 block offsets (MHA_COUNT_PACK/4 per word),
 *              then one byte per (column block, output row) at byte t * rows + r.
 * bank blob:   MHA_TILE_PACK entries per word, each MHA_TILE_ENTRY_BITS wide, holding the value in
 *              the low MHA_ENTRY_VAL_BITS and the block-local column above it.  A block's entries
 *              run in output-row order and start at a word boundary.
 * ==========================================================================*/
struct Payload {
    std::vector<T_MhaTileWord>  bank;
    std::vector<T_MhaCountWord> counts;
};

/* A weight the shift datapath can represent: +/- a power of two, up to 2^7. */
static int pot_code(int w) {
    const int mag = w < 0 ? -w : w;
    int e = 0;
    while ((1 << e) < mag) e++;
    if ((1 << e) != mag || e > 7) {
        std::printf("  FAIL  weight %d is not a power of two in range\n", w);
        g_failures++;
        return 0;
    }
    return (w < 0 ? 8 : 0) | e;
}

/* dense[row][col], col over the full feature axis (MHA_TILES blocks of MHA_TILE_FEATURES). */
static Payload pack_weights(const std::vector<std::vector<int>> &dense) {
    const int rows = (int) dense.size();
    const int cols = (int) dense[0].size();
    const int per_block = cols / MHA_TILES;

    Payload p;
    p.counts.assign(MHA_COUNT_HDR_WORDS + MHA_TILES * rows / MHA_COUNT_PACK, 0);

    for (int t = 0; t < MHA_TILES; t++) {
        /* Each block starts on a word boundary, so the first slice's lane cursor is zero. */
        const int word0 = (int) p.bank.size();
        const int hw = t / (MHA_COUNT_PACK / 4), hp = t % (MHA_COUNT_PACK / 4);
        p.counts[hw].range(hp * 32 + 31, hp * 32) = (ap_uint<32>) word0;

        int entry = 0;
        for (int r = 0; r < rows; r++) {
            int n = 0;
            for (int c = t * per_block; c < (t + 1) * per_block; c++) {
                if (dense[r][c] == 0) continue;
                if ((entry % MHA_TILE_PACK) == 0) p.bank.push_back(0);
                const int lo = (entry % MHA_TILE_PACK) * MHA_TILE_ENTRY_BITS;
                T_MhaTileWord &w = p.bank.back();
                w.range(lo + MHA_ENTRY_VAL_BITS - 1, lo) = (ap_uint<MHA_ENTRY_VAL_BITS>) pot_code(dense[r][c]);
                w.range(lo + MHA_ENTRY_COL_LSB + 7, lo + MHA_ENTRY_COL_LSB) =
                    (ap_uint<8>) (c - t * per_block);
                entry++;
                n++;
            }
            const int byte = t * rows + r;
            p.counts[MHA_COUNT_HDR_WORDS + byte / MHA_COUNT_PACK]
                .range((byte % MHA_COUNT_PACK) * 8 + 7, (byte % MHA_COUNT_PACK) * 8) = (ap_uint<8>) n;
            if (n > 255) { std::printf("  FAIL  row %d block %d has %d nonzeros (byte count)\n", r, t, n); g_failures++; }
        }
        while ((p.bank.size() % 1) != 0) p.bank.push_back(0);   /* blocks are word-aligned already */
    }
    return p;
}

static void fill_tiles(const std::vector<std::vector<int>> &act, int tokens) {
    /* act[feature][token], written into the tile that owns the feature. */
    const int per_block = (int) act.size() / MHA_TILES;
    std::memset((void *) g_act_tile, 0, sizeof(g_act_tile));
    for (int f = 0; f < (int) act.size(); f++) {
        const int t = f / per_block, local = f % per_block;
        for (int w = 0; w < MHA_BANK_ROW_WORDS; w++) {
            T_MhaBankWord word = 0;
            for (int p = 0; p < MHA_BANK_PACK; p++) {
                const int tok = w * MHA_BANK_PACK + p;
                const int v = tok < tokens ? act[f][tok] : 0;
                word.range(p * 8 + 7, p * 8) = (ap_uint<8>) (ap_int<8>) v;
            }
            g_act_tile[t][local * MHA_BANK_ROW_WORDS + w] = word;
        }
    }
}

/* Run the engine over every row and return out[row][token], draining as the kernel's tail does. */
static std::vector<std::vector<long>> run_engine(Payload &p, int rows, int cols, int tokens) {
    /* The slice the loader hands to the engine; the kernel declares the same three inside its
     * stage loop.  Static, because a bank is 60 kB and this runs on the host stack. */
    static T_MhaSliceBank   bank;
    static T_MhaSliceCount  counts;
    static T_MhaSliceCursor cursor0;

    load_weight_bases(p.counts.data(), SPMM_W_K);
    load_weight_slice(p.bank.data(), p.counts.data(), rows, SPMM_W_K, 0, rows,
                      bank, counts, cursor0);

    std::vector<std::vector<long>> out(rows, std::vector<long>(tokens, 0));
    for (int r = 0; r < rows; r++) {
        tiled_rows_compute(r, tokens, 0, bank, counts, cursor0);
        /* Every token below tokens_dim must be drained, or the next row inherits the leftovers --
         * which is exactly the contract this testbench is here to police. */
        for (int tok = 0; tok < tokens; tok++) {
            out[r][tok] = (long) (int) tiled_lane_sum(tok);
        }
    }
    return out;
}

static void expect_engine(const char *what,
                          const std::vector<std::vector<int>> &dense,
                          const std::vector<std::vector<int>> &act,
                          int tokens) {
    const int rows = (int) dense.size(), cols = (int) dense[0].size();
    Payload p = pack_weights(dense);
    fill_tiles(act, tokens);
    const std::vector<std::vector<long>> got = run_engine(p, rows, cols, tokens);

    for (int r = 0; r < rows; r++) {
        for (int tok = 0; tok < tokens; tok++) {
            long want = 0;
            for (int c = 0; c < cols; c++) want += (long) dense[r][c] * act[c][tok];
            if (got[r][tok] != want) {
                std::printf("  FAIL  %s: row %d token %d got %ld want %ld\n",
                            what, r, tok, got[r][tok], want);
                g_failures++;
                g_checks++;
                return;
            }
        }
    }
    g_checks++;
}

/* ============================================================================
 * The engine
 * ==========================================================================*/
static void test_engine() {
    const int COLS = MHA_FEATURE_W_MAX;         /* 768: MHA_TILES blocks of MHA_TILE_FEATURES */
    const int ROWS = MHA_COUNT_PACK;            /* one count transfer's worth of output rows */
    std::printf("engine\n");

    std::vector<std::vector<int>> zero_w(ROWS, std::vector<int>(COLS, 0));
    std::vector<std::vector<int>> act(COLS, std::vector<int>(MHA_TOKEN_W_MAX, 0));
    for (int f = 0; f < COLS; f++)
        for (int t = 0; t < MHA_TOKEN_W_MAX; t++)
            act[f][t] = ((f * 7 + t * 3) % 255) - 127;      /* spans the int8 range */

    /* A weight row with no nonzeros anywhere: the accumulators must come back zero, which is only
     * true if the previous row drained them completely. */
    expect_engine("all-zero weights", zero_w, act, MHA_TOKEN_W_MAX);

    /* Nonzeros in one block only: the other eleven contribute nothing and must not be left dirty. */
    {
        std::vector<std::vector<int>> w = zero_w;
        for (int r = 0; r < ROWS; r++) w[r][5 * MHA_TILE_FEATURES + r] = (r % 2) ? -4 : 8;
        expect_engine("one block populated", w, act, MHA_TOKEN_W_MAX);
    }

    /* Token counts that are not a multiple of MHA_BANK_PACK, including one and one over a word. */
    {
        std::vector<std::vector<int>> w = zero_w;
        for (int r = 0; r < ROWS; r++)
            for (int c = r; c < COLS; c += 37) w[r][c] = ((c / 37) % 2) ? -2 : 16;
        const int cases[] = {1, 7, 8, 9, 15, 17, MHA_TOKEN_W_MAX};
        for (int i = 0; i < (int) (sizeof(cases) / sizeof(cases[0])); i++) {
            char name[64];
            std::snprintf(name, sizeof(name), "tokens_dim = %d", cases[i]);
            expect_engine(name, w, act, cases[i]);
        }
    }

    /* A heavily loaded block: entries spanning many packed words, and a lane far busier than the
     * others, which is the case the cursor arithmetic gets wrong most easily. */
    {
        std::vector<std::vector<int>> w = zero_w;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < MHA_TILE_FEATURES; c++) w[r][c] = ((r + c) % 2) ? -1 : 1;
        expect_engine("block 0 fully dense", w, act, MHA_TOKEN_W_MAX);
    }

    /* Accumulators past the int16 range, where the requantize cast wraps (see below). */
    {
        std::vector<std::vector<int>> w = zero_w;
        std::vector<std::vector<int>> big(COLS, std::vector<int>(MHA_TOKEN_W_MAX, 127));
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) w[r][c] = 128;
        expect_engine("large accumulators", w, big, MHA_TOKEN_W_MAX);
    }
}

/* ============================================================================
 * Requantize and saturation
 * ==========================================================================*/
static void test_quantize() {
    std::printf("quantize\n");
    const T_Scale one = (T_Scale) 1.0f;

    check_eq((int) saturate_to_int8((T_Scale) 126.6f), 127, "round up to the top of the range");
    check_eq((int) saturate_to_int8((T_Scale) 200.0f), 127, "clamp above");
    check_eq((int) saturate_to_int8((T_Scale) -200.0f), -128, "clamp below");
    check_eq((int) saturate_to_int8((T_Scale) 0.5f), 1, "round half away from zero, positive");
    check_eq((int) saturate_to_int8((T_Scale) -0.5f), -1, "round half away from zero, negative");
    check_eq((int) saturate_to_int8((T_Scale) 0.4f), 0, "round toward zero below half");

    check_eq((int) requant_to_int8((T_MhaAcc) 100, one), 100, "requant passes small values");
    check_eq((int) requant_to_int8((T_MhaAcc) -100, one), -100, "requant passes small negatives");

    /* KNOWN, and encoded in the golden data: requant_to_int8 casts to T_Scale, which has 16 integer
     * bits, so an accumulator past +/-32768 WRAPS rather than saturating -- a large positive
     * accumulator can come back negative.  See docs/plans/00-open-issues.md section 1.  This check
     * pins the current behaviour so that fixing it is a deliberate change with its own accuracy
     * run, not something that silently moves the reference. */
    const int wrapped = (int) requant_to_int8((T_MhaAcc) 40000, one);
    check(wrapped < 0, "large positive accumulator still wraps negative (known, see plan 00)");
}

/* ============================================================================
 * The residual add
 * ==========================================================================*/
static void test_residual() {
    std::printf("residual\n");
    const int WORDS = 4;
    std::vector<T_MhaBankWord> a(WORDS, 0), b(WORDS, 0), out(WORDS, 0);

    auto put = [](std::vector<T_MhaBankWord> &v, int i, int val) {
        v[i / MHA_BANK_PACK].range((i % MHA_BANK_PACK) * 8 + 7, (i % MHA_BANK_PACK) * 8) =
            (ap_uint<8>) (ap_int<8>) val;
    };
    auto get = [](const std::vector<T_MhaBankWord> &v, int i) {
        return (int) (ap_int<8>) (ap_uint<8>) v[i / MHA_BANK_PACK]
            .range((i % MHA_BANK_PACK) * 8 + 7, (i % MHA_BANK_PACK) * 8);
    };

    put(a, 0, 127);  put(b, 0, 127);      /* must clamp, not wrap */
    put(a, 1, -128); put(b, 1, -128);
    put(a, 2, 100);  put(b, 2, -100);     /* cancels exactly */
    put(a, 3, 0);    put(b, 3, 0);
    residual_add(a.data(), b.data(), WORDS, (T_Scale) 1.0f, (T_Scale) 1.0f, out.data());

    check_eq(get(out, 0), 127, "residual saturates high rather than wrapping");
    check_eq(get(out, 1), -128, "residual saturates low rather than wrapping");
    check_eq(get(out, 2), 0, "residual cancels to zero");
    check_eq(get(out, 3), 0, "residual of zeros is zero");

    /* Each product is reduced to T_Scale BEFORE the addition, and residual.cpp carries a comment
     * saying it must not be folded into one expression.  What that actually guards against is an
     * OVERFLOW, not a rounding difference: an int8 operand times a T_Scale has at most T_Scale's
     * 16 fractional bits, so the narrowing loses nothing, but T_Scale holds only 16 integer bits
     * and the product wraps past +/-32768.  The folded form keeps the multiply's full width until
     * the end and saturates instead.
     *
     * A scale of 512 makes that reachable: 100 * 512 = 51 200 wraps to -14 336, so the two-step
     * form saturates low where the folded form would saturate high. */
    std::vector<T_MhaBankWord> c(1, 0), d(1, 0), o(1, 0);
    put(c, 0, 100); put(d, 0, 0);
    residual_add(c.data(), d.data(), 1, (T_Scale) 512.0f, (T_Scale) 1.0f, o.data());
    check_eq(get(o, 0), -128, "a product past T_Scale's integer range wraps before the add");

    /* At the scales the model actually uses -- order 1, so |operand * scale| stays well inside
     * 32768 -- the two forms agree exactly.  This is the check that would catch someone widening
     * T_Scale or introducing a large scale without revisiting the golden data. */
    for (int v = -128; v < 128; v += 7) {
        std::vector<T_MhaBankWord> e(1, 0), f(1, 0), g(1, 0);
        put(e, 0, v); put(f, 0, -v / 2);
        residual_add(e.data(), f.data(), 1, (T_Scale) 0.75f, (T_Scale) 1.25f, g.data());
        const T_Scale folded = (T_Scale)(int) v * (T_Scale) 0.75f
                             + (T_Scale)(int) (-v / 2) * (T_Scale) 1.25f;
        check_eq(get(g, 0), (int) saturate_to_int8(folded), "two-step matches folded at model scales");
    }
}

int main() {
    std::printf("Unit testbench: edge cases the golden data cannot reach\n"
                "%d lanes, %d tokens per word, %d features\n\n",
                MHA_TILES, MHA_BANK_PACK, MHA_FEATURE_W_MAX);
    test_quantize();
    test_residual();
    test_engine();
    std::printf("\n%d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
