// ============================================================================
// Encoder E2E testbench (file-driven, real POT4 weights, many images).
//
// Runs the REAL shipping encoder_layer_top through all 12 HW layers back-to-back
// on real per-layer POT4 weights + a real SW-embedded image, in ONE csim process.
// POT4 model = fused gammas (none read) + int8 weights + ap_fixed<32,16> scales.
//
//   model.bin       : per-layer CSR weights (int8 val, u16 col, i32 rowptr, i32 s_row)
//                     + 15 ap_fixed scales.  NO gammas.  (scripts/compile_model.emit_model
//                     pot=False fused=True)
//   model_index.bin : per-layer (nnz, n_rows) so we stream model.bin without JSON.
//   inputs.bin      : per-image layer-0 input (D x N int8, feature-major) from SW embed.
//   -> hw_out.bin   : per-image encoder output (D x N int8) after all 12 HW layers.
//   -> hw_layers.bin: image-0 per-layer {hidden, output} boundaries for divergence localization.
//
// File formats mirror the emit script scratchpad/emit_e2e_pot4.py.
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

struct WMeta { int nnz; int n_rows; };
struct LayerMeta { WMeta w[6]; };   // w_q,w_k,w_v,out_proj,linear_1,linear_2

struct LayerW {
    std::vector<T_Activation> val[6];
    std::vector<uint16_t>     col[6];
    std::vector<int>          ptr[6];
    std::vector<T_Scale>      scales;   // 15
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
    std::vector<LayerMeta> lm(n_layers);
    for (int L = 0; L < n_layers; L++)
        for (int w = 0; w < 6; w++) { lm[L].w[w].nnz = read_i32(fi); lm[L].w[w].n_rows = read_i32(fi); }
    std::fclose(fi);

    // ---- inputs.bin header ----
    FILE* fin = std::fopen((g_dir + "inputs.bin").c_str(), "rb");
    if (!fin) die("cannot open inputs.bin");
    if ((uint32_t)read_i32(fin) != 0x45324E49u) die("bad inputs magic");
    const int n_images = read_i32(fin);
    const int in_D = read_i32(fin);
    const int N    = read_i32(fin);
    if (in_D != D) die("inputs D != model D");
    if (D != ENC_FEATURE_W_MAX) die("model D != ENC_FEATURE_W_MAX (rebuild config)");
    if (N > ENC_TOKEN_W_MAX) die("N > ENC_TOKEN_W_MAX (raise ENC_TOKEN_W_MAX)");
    const int XY = D * N;

    // ---- model.bin: per layer, 6 weights (val i8, col u16, ptr i32, s_row i32) + 15 scales ----
    FILE* fm = std::fopen((g_dir + "model.bin").c_str(), "rb");
    if (!fm) die("cannot open model.bin");
    std::vector<LayerW> LW(n_layers);
    for (int L = 0; L < n_layers; L++) {
        for (int w = 0; w < 6; w++) {
            const int nnz = lm[L].w[w].nnz, nr = lm[L].w[w].n_rows;
            LW[L].val[w].resize(nnz); LW[L].col[w].resize(nnz); LW[L].ptr[w].resize(nr + 1);
            { std::vector<int8_t> t(nnz); read_raw(fm, t.data(), nnz, "w.val");
              for (int i = 0; i < nnz; i++) LW[L].val[w][i] = (T_Activation)t[i]; }
            read_raw(fm, LW[L].col[w].data(), nnz, "w.col");
            { std::vector<int32_t> t(nr + 1); read_raw(fm, t.data(), nr + 1, "w.ptr");
              for (int i = 0; i <= nr; i++) LW[L].ptr[w][i] = (int)t[i]; }
            (void)read_i32(fm);   // s_row (per-row scale, ignored; scalar scales used)
        }
        // 15 scales: int32 raw ap_fixed<32,16> bit pattern -> reinterpret
        LW[L].scales.resize(SCALE_IDX_MAX);
        { std::vector<int32_t> t(SCALE_IDX_MAX); read_raw(fm, t.data(), SCALE_IDX_MAX, "scales");
          for (int i = 0; i < SCALE_IDX_MAX; i++) {
              T_Scale s; s.range() = (ap_int<32>)t[i]; LW[L].scales[i] = s; } }
    }
    std::fclose(fm);

    // widened column buffers (T_MhaIndex / T_MlpIndex)
    std::vector<std::vector<T_MhaIndex>> mha_col(n_layers * 4);
    std::vector<std::vector<T_MlpIndex>> mlp_col(n_layers * 2);
    for (int L = 0; L < n_layers; L++) {
        for (int w = 0; w < 4; w++) { auto& d = mha_col[L*4+w]; d.resize(LW[L].col[w].size());
            for (size_t i = 0; i < d.size(); i++) d[i] = (T_MhaIndex)LW[L].col[w][i]; }
        for (int w = 0; w < 2; w++) { auto& d = mlp_col[L*2+w]; d.resize(LW[L].col[4+w].size());
            for (size_t i = 0; i < d.size(); i++) d[i] = (T_MlpIndex)LW[L].col[4+w][i]; }
    }

    // ---- output file ----
    FILE* fo = std::fopen((g_dir + "hw_out.bin").c_str(), "wb");
    if (!fo) die("cannot open hw_out.bin");
    { int32_t hdr[4] = { (int32_t)0x45324F55, n_images, D, N }; std::fwrite(hdr, 4, 4, fo); }

    std::vector<T_Activation> cur(XY), hidden(XY), nxt(XY);

    // DDR-resident activation scratch, reused across all layers/images.
    // Q'/K': full grid D*N.  H: ONE region (d_h*N) exposed via two pointer pairs (h0/h1) so the
    // two stage-3 W2 PEs read it on independent AXI bundles -> both alias the SAME buffer here.
    std::vector<T_Activation>   qpv(MHA_MAX_QK_NNZ), kpv(MHA_MAX_QK_NNZ);
    std::vector<T_MhaHeadIndex> qpc(MHA_MAX_QK_NNZ), kpc(MHA_MAX_QK_NNZ);
    std::vector<int>            qpp(MHA_SCR_QP_PTR_CNT), kpp(MHA_SCR_KP_PTR_CNT);
    std::printf("E2E TB: %d images, %d layers, D=%d N=%d d_h=%d\n", n_images, n_layers, D, N, d_h);

    for (int img = 0; img < n_images; img++) {
        { std::vector<int8_t> t(XY); read_raw(fin, t.data(), XY, "img input");
          for (int i = 0; i < XY; i++) cur[i] = (T_Activation)t[i]; }

        FILE* fdbg = (img == 0) ? std::fopen((g_dir + "hw_layers.bin").c_str(), "wb") : nullptr;

        for (int L = 0; L < n_layers; L++) {
            LayerW& w = LW[L];
            encoder_layer_top(
                cur.data(), D, N, d_h,
                w.val[0].data(), mha_col[L*4+0].data(), w.ptr[0].data(),   // w_q
                w.val[1].data(), mha_col[L*4+1].data(), w.ptr[1].data(),   // w_k
                w.val[2].data(), mha_col[L*4+2].data(), w.ptr[2].data(),   // w_v
                w.val[3].data(), mha_col[L*4+3].data(), w.ptr[3].data(),   // out_proj
                w.val[4].data(), mlp_col[L*2+0].data(), w.ptr[4].data(),   // linear_1
                w.val[5].data(), mlp_col[L*2+1].data(), w.ptr[5].data(),   // linear_2
                w.scales.data(),
                qpv.data(), qpc.data(),
                kpv.data(), kpc.data(),
                hidden.data(), nxt.data());
            if (fdbg) {
                std::vector<int8_t> th(XY), ty(XY);
                for (int i = 0; i < XY; i++) { th[i] = (int8_t)hidden[i]; ty[i] = (int8_t)nxt[i]; }
                std::fwrite(th.data(), 1, XY, fdbg);
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
