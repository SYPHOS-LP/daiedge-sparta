// ============================================================================
// Encoder E2E testbench (file-driven, real POT4 weights, many images).
//
// Runs the REAL shipping encoder_layer_top through all 12 HW layers back-to-back
// on real per-layer POT4 weights + a real SW-embedded image, in ONE csim process.
// POT4 model = fused gammas (none read) + int8 weights + ap_fixed<32,16> scales.
//
//   model.bin       : per layer, MHA weights as packed bank words + counts, MLP as CSR/CSC
//                     + 15 ap_fixed scales.  NO gammas.  (scripts/compile_model.py
//                     --scales fixed: pot=True, pot_scales=False, fused=True)
//   model_index.bin : per-layer (nnz, n_rows) so we stream model.bin without JSON.
//   inputs.bin      : per-image layer-0 input (D x N int8, feature-major) from SW embed.
//   -> hw_out.bin   : per-image encoder output (D x N int8) after all 12 HW layers.
//   -> hw_layers.bin: image-0 per-layer outputs (12 x D x N) for divergence localization.
//
// E2E_IMAGES=<k> runs only the first k images (quick regression).
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include "encoder_layer.h"

static std::string resolve_dir(int argc, char** argv) {
    if (const char* e = std::getenv("E2E_DIR")) { std::string s = e; if (!s.empty() && s.back() != '/' && s.back() != '\\') s += '/'; return s; }
    if (argc > 1) { std::string s = argv[1]; if (!s.empty() && s.back() != '/' && s.back() != '\\') s += '/'; return s; }
    return "../../sim/e2e/";
}

static void die(const char* m) { std::fprintf(stderr, "E2E TB ERROR: %s\n", m); std::exit(1); }
template <typename T> static void read_raw(FILE* f, T* d, size_t c, const char* w) {
    if (std::fread(d, sizeof(T), c, f) != c) { std::fprintf(stderr, "short read: %s\n", w); die("fread"); }
}
static int32_t read_i32(FILE* f) { int32_t v; read_raw(f, &v, 1, "i32"); return v; }

// The kernel reads weight values/columns in wide words and reads the last word whole, so
// the buffers must be readable (zero-padded) up to a multiple of this many elements.
static const int WIDE_READ_PAD = 32;

struct WMeta { int nnz; int n_rows; };
struct LayerMeta { WMeta w[6]; };   // w_q,w_k,w_v,out_proj,linear_1,linear_2

struct LayerW {
    // All six weights are pre-packed bank words ({column, value} per slot) plus the nonzero counts
    // per (column block, row).
    std::vector<T_MhaTileWord>  bank[6];
    std::vector<T_MhaCountWord> counts[6];
    std::vector<T_Scale>       scales;   // 15
};

int main(int argc, char** argv) {
    const std::string g_dir = resolve_dir(argc, argv);
    std::printf("E2E TB: data dir = %s\n", g_dir.c_str());

    // ---- model_index.bin ----
    FILE* fi = std::fopen((g_dir + "model_index.bin").c_str(), "rb");
    if (!fi) die("cannot open model_index.bin");
    if ((uint32_t)read_i32(fi) != 0x45324D49u) die("bad model_index magic");
    const int n_layers = read_i32(fi);
    const int D        = read_i32(fi);
    const int d_h      = read_i32(fi);
    (void)read_i32(fi);   // n_heads
    {   /* Weight entry layout: 0 = power-of-two values (shift datapath), 1 = plain integers
         * (DSP multiply).  compile_model.py --weights writes it; it must match this build. */
        const int fmt = read_i32(fi);
        if (fmt != MHA_ENTRY_FORMAT) {
            std::fprintf(stderr, "model_index entry_format=%d but this kernel is built for %d "
                                 "(POT_SHIFT_MAC %s); re-run compile_model.py --weights auto\n",
                         fmt, MHA_ENTRY_FORMAT,
#ifdef POT_SHIFT_MAC
                         "set"
#else
                         "not set"
#endif
                         );
            die("weight entry format mismatch");
        }
    }
    std::vector<LayerMeta> lm(n_layers);
    for (int L = 0; L < n_layers; L++)
        for (int w = 0; w < 6; w++) { lm[L].w[w].nnz = read_i32(fi); lm[L].w[w].n_rows = read_i32(fi); }
    std::fclose(fi);

    // ---- inputs.bin header ----
    FILE* fin = std::fopen((g_dir + "inputs.bin").c_str(), "rb");
    if (!fin) die("cannot open inputs.bin");
    if ((uint32_t)read_i32(fin) != 0x45324E49u) die("bad inputs magic");
    const int n_in_file = read_i32(fin);
    int n_images = n_in_file;
    if (const char* e = std::getenv("E2E_IMAGES")) {
        const int k = std::atoi(e);
        if (k > 0 && k < n_images) n_images = k;
    }
    const int in_D = read_i32(fin);
    const int N    = read_i32(fin);
    if (in_D != D) die("inputs D != model D");
    if (D != ENC_FEATURE_W_MAX) die("model D != ENC_FEATURE_W_MAX (rebuild config)");
    if (N > ENC_TOKEN_W_MAX) die("N > ENC_TOKEN_W_MAX (raise ENC_TOKEN_W_MAX)");
    const int XY = D * N;

    // ---- model.bin: per layer, 6 weights (bank u64, counts i32, s_row i32), then 15 scales ----
    FILE* fm = std::fopen((g_dir + "model.bin").c_str(), "rb");
    if (!fm) die("cannot open model.bin");
    std::vector<LayerW> LW(n_layers);
    for (int L = 0; L < n_layers; L++) {
        for (int w = 0; w < 6; w++) {
            const int nnz = lm[L].w[w].nnz, nr = lm[L].w[w].n_rows;
            if (nnz % MHA_TILE_PACK) die("weight nnz is not a multiple of MHA_TILE_PACK");
            const int words = nnz / MHA_TILE_PACK;
            LW[L].bank[w].assign(words + MHA_TILE_PACK, (T_MhaTileWord) 0);
            std::vector<uint64_t> t(words); read_raw(fm, t.data(), words, "w.bank");
            for (int i = 0; i < words; i++) LW[L].bank[w][i] = (T_MhaTileWord) t[i];
            /* The counts blob is bytes: MHA_TILES int32 block offsets padded to a whole transfer,
             * then one byte per (column block, row) count.  Read it as transfers. */
            const int count_bytes = MHA_COUNT_HDR_WORDS * MHA_COUNT_PACK
                                  + (nr + 1) - MHA_TILES;   /* model_index counts the entries */
            const int count_words = (count_bytes + MHA_COUNT_PACK - 1) / MHA_COUNT_PACK;
            std::vector<uint8_t> cb(count_words * MHA_COUNT_PACK, 0);
            read_raw(fm, cb.data(), count_bytes, "w.counts");
            LW[L].counts[w].assign(count_words, (T_MhaCountWord) 0);
            for (int i = 0; i < count_bytes; i++) {
                LW[L].counts[w][i / MHA_COUNT_PACK].range((i % MHA_COUNT_PACK) * 8 + 7,
                                                          (i % MHA_COUNT_PACK) * 8) = cb[i];
            }
            (void)read_i32(fm);   // s_row (per-row scale, ignored; scalar scales used)
        }
        // 15 scales: int32 raw ap_fixed<32,16> bit pattern -> reinterpret
        LW[L].scales.resize(SCALE_IDX_MAX);
        { std::vector<int32_t> t(SCALE_IDX_MAX); read_raw(fm, t.data(), SCALE_IDX_MAX, "scales");
          for (int i = 0; i < SCALE_IDX_MAX; i++) {
              T_Scale s; s.range() = (ap_int<32>)t[i]; LW[L].scales[i] = s; } }
    }
    std::fclose(fm);

    // ---- output file ----
    FILE* fo = std::fopen((g_dir + "hw_out.bin").c_str(), "wb");
    if (!fo) die("cannot open hw_out.bin");
    { int32_t hdr[4] = { (int32_t)0x45324F55, n_images, D, N }; std::fwrite(hdr, 4, 4, fo); }

    std::vector<T_Activation> cur(XY), hidden(XY), nxt(XY);

    std::printf("E2E TB: %d of %d images, %d layers, D=%d N=%d d_h=%d\n",
                n_images, n_in_file, n_layers, D, N, d_h);

    for (int img = 0; img < n_images; img++) {
        { std::vector<int8_t> t(XY); read_raw(fin, t.data(), XY, "img input");
          for (int i = 0; i < XY; i++) cur[i] = (T_Activation)t[i]; }

        /* The kernel keeps the activation on-chip between layers, so only the first layer reads it
         * from memory and only the last writes it back.  Image 0 forces the write on every layer so
         * that hw_layers.bin still holds each layer's output for the reference comparison; the other
         * images run the production flags, which is what the board will do. */
        FILE* fdbg = (img == 0) ? std::fopen((g_dir + "hw_layers.bin").c_str(), "wb") : nullptr;

        for (int L = 0; L < n_layers; L++) {
            LayerW& w = LW[L];
            encoder_layer_top(
                cur.data(), D, N, d_h,
                w.bank[0].data(), w.counts[0].data(),   // w_q
                w.bank[1].data(), w.counts[1].data(),   // w_k
                w.bank[2].data(), w.counts[2].data(),   // w_v
                w.bank[3].data(), w.counts[3].data(),   // out_proj
                w.bank[4].data(), w.counts[4].data(),   // linear_1
                w.bank[5].data(), w.counts[5].data(),   // linear_2
                w.scales.data(),
                nxt.data(),
                (L == 0),                            /* first_layer: read the input from memory */
                (fdbg != nullptr) || (L == n_layers - 1));   /* last_layer: write the result out */
            if (fdbg) {
                std::vector<int8_t> ty(XY);
                for (int i = 0; i < XY; i++) { ty[i] = (int8_t)nxt[i]; }
                std::fwrite(ty.data(), 1, XY, fdbg);
            }
            cur.swap(nxt);
        }
        if (fdbg) { std::fflush(fdbg); std::fclose(fdbg); }

        { std::vector<int8_t> t(XY); for (int i = 0; i < XY; i++) t[i] = (int8_t)cur[i];
          std::fwrite(t.data(), 1, XY, fo); std::fflush(fo); }
        std::fprintf(stderr, "  image %d/%d done\n", img, n_images); std::fflush(stderr);
    }

    std::fclose(fin); std::fclose(fo);
    std::printf("E2E TB: wrote hw_out.bin (%d images)\n", n_images);
    return 0;
}
