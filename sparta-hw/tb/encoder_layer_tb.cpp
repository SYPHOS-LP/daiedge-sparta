// ============================================================================
// Encoder Layer Testbench
//
// Tests encoder_layer(): Y = MLP-block(MHA-block(X)), the full pre-norm encoder
//   H = X + MHA(RMSNorm(X))      (int8)
//   Y = H + MLP(RMSNorm(H))      (int8)
// feature-major int8 (D x N, element (f,t) at f*N+t), sparse weights, biases
// ignored.  The reference recomputes BOTH sub-blocks in double precision, applying
// the POT4 power-of-two scales via ref_requant so the requants bit-match the DUT, and
// crucially REQUANTIZES H to int8 between the blocks (H is the DDR boundary tensor)
// exactly as the hardware does.  Mirrors the two block testbenches chained.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "encoder_layer.h"
#include "tb_pot_scales.h"   // POT4 power-of-two scales + ref_requant

#define TEST_VERBOSE 1
#define TOL_STEPS    4.0
#define ABS_TOL      (TOL_STEPS * pot_scale(S_MLP_BRANCH))

static inline int sat8(long v){ if(v>127)return 127; if(v<-128)return -128; return (int)v; }

// ---------------------------------------------------------------------------
// MHA sub-block reference: feature-major z_in (D x N) -> dense int8 H (D x N)
// (this is what the hardware writes to the H DDR buffer).
// ---------------------------------------------------------------------------
static std::vector<int> reference_mha_block_i8(
    const std::vector<T_Activation>& z_in, int D, int N, int n_heads,
    const std::vector<T_Activation>& wqv, const std::vector<T_MhaIndex>& wqc, const std::vector<int>& wqp,
    const std::vector<T_Activation>& wkv, const std::vector<T_MhaIndex>& wkc, const std::vector<int>& wkp,
    const std::vector<T_Activation>& wvv, const std::vector<T_MhaIndex>& wvc, const std::vector<int>& wvp,
    const std::vector<T_Activation>& wov, const std::vector<T_MhaIndex>& woc, const std::vector<int>& wop)
{
    const int dh = D / n_heads;
    auto Z = [&](int f,int t){ return (double)z_in[f*N+t]; };

    std::vector<int> z(D*N,0);
    for (int t=0;t<N;t++){
        double ss=0; for(int f=0;f<D;f++){ double r=Z(f,t); ss+=r*r; }
        double inv=1.0/std::sqrt(ss/(double)D+(double)RMS_EPS);
        for(int f=0;f<D;f++){
            double nr=Z(f,t)*inv*pot_scale(S_MHA_RMS_INV);   /* gamma fused into the following linear */
            z[f*N+t]=sat8(std::lround(nr));
        }
    }
    auto project=[&](const std::vector<T_Activation>&wv,const std::vector<T_MhaIndex>&wc,
                     const std::vector<int>&wp,int shift,bool relu)->std::vector<int>{
        std::vector<int> out(D*N,0);
        for(int i=0;i<D;i++) for(int t=0;t<N;t++){
            long acc=0; for(int k=wp[i];k<wp[i+1];k++) acc += (long)wv[k]*z[(int)wc[k]*N+t];
            if(relu && acc<=0){ out[i*N+t]=0; continue; }
            out[i*N+t]=ref_requant(acc,shift);
            if(relu && out[i*N+t]<0) out[i*N+t]=0;
        }
        return out;
    };
    std::vector<int> Qp=project(wqv,wqc,wqp,S_MHA_Q,true);
    std::vector<int> Kp=project(wkv,wkc,wkp,S_MHA_K,true);
    std::vector<int> V =project(wvv,wvc,wvp,S_MHA_V,false);

    std::vector<int> Zhat(D*N,0);
    for(int hh=0;hh<n_heads;hh++){
        int fb=hh*dh;
        std::vector<long> A(dh*dh,0), sK(dh,0);
        for(int a=0;a<dh;a++) for(int t=0;t<N;t++){
            int kv=Kp[(fb+a)*N+t]; if(!kv) continue; sK[a]+=kv;
            for(int b=0;b<dh;b++) A[a*dh+b]+=(long)kv*V[(fb+b)*N+t];
        }
        for(int t=0;t<N;t++){
            std::vector<long> nrow(dh,0); long d=0;
            for(int a=0;a<dh;a++){ int qv=Qp[(fb+a)*N+t]; if(!qv) continue;
                d+=(long)qv*sK[a];
                for(int b=0;b<dh;b++) nrow[b]+=(long)qv*A[a*dh+b];
            }
            double recip = (d<=1L) ? 1.0 : (1.0/(double)d);
            for(int b=0;b<dh;b++){
                double ratio=(double)nrow[b]*recip*pot_scale(S_MHA_DIV);
                Zhat[(fb+b)*N+t]=sat8(std::lround(ratio));
            }
        }
    }
    std::vector<int> O(D*N,0);
    for(int i=0;i<D;i++) for(int t=0;t<N;t++){
        long acc=0; for(int k=wop[i];k<wop[i+1];k++) acc+=(long)wov[k]*Zhat[(int)woc[k]*N+t];
        O[i*N+t]=ref_requant(acc,S_MHA_O);
    }
    // H = z_in + O with the residual folded as power-of-two scales, quantized to int8
    // (the boundary tensor the hardware stores in DDR).
    std::vector<int> H(D*N,0);
    for(int i=0;i<D;i++) for(int t=0;t<N;t++){
        double a = (double)(int)z_in[i*N+t]*pot_scale(S_MHA_RES) + (double)O[i*N+t]*pot_scale(S_MHA_BRANCH);
        H[i*N+t]=sat8(std::lround(a));
    }
    return H;
}

// ---------------------------------------------------------------------------
// MLP sub-block reference: int8 H (D x N) -> dense real-valued Y (D x N).
// ---------------------------------------------------------------------------
static std::vector<double> reference_mlp_block(
    const std::vector<int>& H, int D, int N,
    const std::vector<T_Activation>& w1v, const std::vector<T_MlpIndex>& w1c, const std::vector<int>& w1p, int d_h,
    const std::vector<T_Activation>& w2v, const std::vector<T_MlpIndex>& w2c, const std::vector<int>& w2p)
{
    auto X = [&](int f,int t){ return (double)H[f*N+t]; };
    std::vector<int> n(D*N,0);
    for (int t=0;t<N;t++){
        double ss=0; for(int f=0;f<D;f++){ double r=X(f,t); ss+=r*r; }
        double inv=1.0/std::sqrt(ss/(double)D+(double)RMS_EPS);
        for(int f=0;f<D;f++){
            double nr=X(f,t)*inv*pot_scale(S_MLP_RMS_INV);   /* gamma fused into the following linear */
            n[f*N+t]=sat8(std::lround(nr));
        }
    }
    // Hh = ReLU(W1 . n), requantized to int8 by the fc1 power-of-two scale.
    std::vector<int> Hh(d_h*N,0);
    for(int i=0;i<d_h;i++) for(int t=0;t<N;t++){
        long acc=0; for(int k=w1p[i];k<w1p[i+1];k++) acc += (long)w1v[k]*n[(int)w1c[k]*N+t];
        if(acc<=0){ Hh[i*N+t]=0; continue; }
        int v=ref_requant(acc,S_MLP_FC1);
        Hh[i*N+t]=(v<0)?0:v;
    }
    // M = W2 . Hh, requantized to int8 by the fc2 power-of-two scale.
    std::vector<int> M(D*N,0);
    for(int i=0;i<D;i++) for(int t=0;t<N;t++){
        long acc=0; for(int k=w2p[i];k<w2p[i+1];k++) acc += (long)w2v[k]*Hh[(int)w2c[k]*N+t];
        M[i*N+t]=ref_requant(acc,S_MLP_FC2);
    }
    // Y = H + M, residual folded as power-of-two scales (real units).
    std::vector<double> y(D*N,0.0);
    for(int i=0;i<D;i++) for(int t=0;t<N;t++)
        y[i*N+t]=X(i,t)*pot_scale(S_MLP_RES) + (double)M[i*N+t]*pot_scale(S_MLP_BRANCH);
    return y;
}

struct TestResult{ bool passed; const char* name; };

static TestResult run_case(
    const char* name, int D, int N, int n_heads, int d_h,
    std::vector<T_Activation> x,
    std::vector<T_Activation> wqv, std::vector<T_MhaIndex> wqc, std::vector<int> wqp,
    std::vector<T_Activation> wkv, std::vector<T_MhaIndex> wkc, std::vector<int> wkp,
    std::vector<T_Activation> wvv, std::vector<T_MhaIndex> wvc, std::vector<int> wvp,
    std::vector<T_Activation> wov, std::vector<T_MhaIndex> woc, std::vector<int> wop,
    std::vector<T_Activation> w1v, std::vector<T_MlpIndex> w1c, std::vector<int> w1p,
    std::vector<T_Activation> w2v, std::vector<T_MlpIndex> w2c, std::vector<int> w2p)
{
    TestResult tr; tr.name=name; tr.passed=true;
    if(TEST_VERBOSE) std::cout<<"\n=== "<<name<<" (D="<<D<<" N="<<N<<" heads="<<n_heads<<" d_h="<<d_h<<") ===\n";

    auto H_ref = reference_mha_block_i8(x,D,N,n_heads,
                    wqv,wqc,wqp,wkv,wkc,wkp,wvv,wvc,wvp,wov,woc,wop);
    auto ref   = reference_mlp_block(H_ref,D,N,w1v,w1c,w1p,d_h,w2v,w2c,w2p);

    const int WPTR=MAX_ROWS+1, XY=ENC_FEATURE_W_MAX*ENC_TOKEN_W_MAX;
    auto padW=[&](std::vector<T_Activation>&v,std::vector<T_MhaIndex>&c,std::vector<int>&p){
        v.resize(MAX_NNZ,(T_Activation)0); c.resize(MAX_NNZ,(T_MhaIndex)0); p.resize(WPTR,0);
    };
    auto padWm=[&](std::vector<T_Activation>&v,std::vector<T_MlpIndex>&c,std::vector<int>&p){
        v.resize(MAX_NNZ,(T_Activation)0); c.resize(MAX_NNZ,(T_MlpIndex)0); p.resize(WPTR,0);
    };
    padW(wqv,wqc,wqp); padW(wkv,wkc,wkp); padW(wvv,wvc,wvp); padW(wov,woc,wop);
    padWm(w1v,w1c,w1p); padWm(w2v,w2c,w2p);
    x.resize(XY,(T_Activation)0);
    std::vector<T_Activation> h(XY,(T_Activation)0);
    std::vector<T_Activation> y(XY,(T_Activation)0);

    // Fill the per-layer scale array (SCALE_*_IDX order = the host/hardware contract).
    // The host would fold these from the layer's QONNX scales; here = layer-0 refs.
    std::vector<T_Scale> scales(SCALE_IDX_MAX, (T_Scale)0);
    scales[SCALE_Q_IDX]                   = (T_Scale)pot_scale(S_MHA_Q);
    scales[SCALE_K_IDX]                   = (T_Scale)pot_scale(S_MHA_K);
    scales[SCALE_V_IDX]                   = (T_Scale)pot_scale(S_MHA_V);
    scales[SCALE_LINEAR_OUT_IDX]          = (T_Scale)pot_scale(S_MHA_O);
    scales[SCALE_DIV_OUT_IDX]             = (T_Scale)pot_scale(S_MHA_DIV);
    scales[SCALE_ATT_RMSNORM_OUT_INV_IDX] = (T_Scale)pot_scale(S_MHA_RMS_INV);
    scales[SCALE_ATT_RESIDUAL_IDX]        = (T_Scale)pot_scale(S_MHA_RES);
    scales[SCALE_ATT_BRANCH_RATIO_IDX]    = (T_Scale)pot_scale(S_MHA_BRANCH);
    scales[SCALE_FC1_IDX]                 = (T_Scale)pot_scale(S_MLP_FC1);
    scales[SCALE_FC2_IDX]                 = (T_Scale)pot_scale(S_MLP_FC2);
    scales[SCALE_FF_RMSNORM_OUT_INV_IDX]  = (T_Scale)pot_scale(S_MLP_RMS_INV);
    scales[SCALE_FF_RESIDUAL_IDX]         = (T_Scale)pot_scale(S_MLP_RES);
    scales[SCALE_FF_BRANCH_RATIO_IDX]     = (T_Scale)pot_scale(S_MLP_BRANCH);

    /* On-chip Q'/K' CSR scratch: the kernel builds the ReLU-sparsified Q'/K' into
     * these, so the caller owns the storage. Sized to the fully-dense worst case. */
    std::vector<T_Activation>   qpv(MHA_MAX_QK_NNZ), kpv(MHA_MAX_QK_NNZ);
    std::vector<T_MhaHeadIndex> qpc(MHA_MAX_QK_NNZ), kpc(MHA_MAX_QK_NNZ);

    encoder_layer_top(
        x.data(),
        D, N, d_h,
        wqv.data(),wqc.data(),wqp.data(),
        wkv.data(),wkc.data(),wkp.data(),
        wvv.data(),wvc.data(),wvp.data(),
        wov.data(),woc.data(),wop.data(),
        w1v.data(),w1c.data(),w1p.data(),
        w2v.data(),w2c.data(),w2p.data(),
        scales.data(),
        qpv.data(), qpc.data(),
        kpv.data(), kpc.data(),
        h.data(), y.data());

    double maxerr=0; int off=0,worse=0;
    for(int i=0;i<D;i++) for(int t=0;t<N;t++){
        double got=(double)y[i*N+t];   /* layer output already in real units per the residual fold */
        double e=std::fabs(got-ref[i*N+t]); if(e>maxerr)maxerr=e;
        if(e>ABS_TOL){ (e<=2.0*ABS_TOL?off:worse)++; }
    }
    if(TEST_VERBOSE){
        std::cout<<"    max abs err (real): "<<std::setprecision(5)<<maxerr<<"  tol="<<ABS_TOL<<"\n";
        std::cout<<"    elems > tol       : "<<(off+worse)<<"  (badly off: "<<worse<<")\n";
    }
    int total=D*N;
    tr.passed=(worse==0)&&(off*50<=total);
    std::cout<<"    "<<(tr.passed?"PASS":"FAIL")<<"\n";
    return tr;
}

// Identity projections on both blocks (real D=768, 12 heads, d_h=D so MLP W1/W2
// are square identities) — exercises the full chain with a known reference.
static TestResult test_identity(){
    const int D=ENC_FEATURE_W_MAX, N=8, H=MHA_N_HEADS, d_h=D;
    std::vector<T_Activation> x(D*N);
    for(int f=0;f<D;f++) for(int t=0;t<N;t++) x[f*N+t]=(T_Activation)((((f-300)*(t+1))%50));
    auto identM=[&](std::vector<T_Activation>&v,std::vector<T_MhaIndex>&c,std::vector<int>&p){
        v.assign(D,(T_Activation)1); c.resize(D); p.resize(D+1);
        for(int i=0;i<D;i++){c[i]=(T_MhaIndex)i;p[i]=i;} p[D]=D;
    };
    auto identL=[&](std::vector<T_Activation>&v,std::vector<T_MlpIndex>&c,std::vector<int>&p){
        v.assign(D,(T_Activation)1); c.resize(D); p.resize(D+1);
        for(int i=0;i<D;i++){c[i]=(T_MlpIndex)i;p[i]=i;} p[D]=D;
    };
    std::vector<T_Activation> qv,kv,vv,ov; std::vector<T_MhaIndex> qc,kc,vc,oc; std::vector<int> qp,kp,vp,op;
    identM(qv,qc,qp); identM(kv,kc,kp); identM(vv,vc,vp); identM(ov,oc,op);
    std::vector<T_Activation> w1,w2; std::vector<T_MlpIndex> w1c,w2c; std::vector<int> w1p,w2p;
    identL(w1,w1c,w1p); identL(w2,w2c,w2p);
    return run_case("identity MHA+MLP (real D=768, 12 heads, d_h=768)",D,N,H,d_h,x,
                    qv,qc,qp,kv,kc,kp,vv,vc,vp,ov,oc,op,w1,w1c,w1p,w2,w2c,w2p);
}

int main(){
    std::cout<<"Encoder layer testbench — Y = MLP-block(MHA-block(X))\n"
             <<"rsqrt variant "<<RMS_RSQRT_VARIANT<<", recip LUT addr bits "<<MHA_RECIP_LUT_ADDR_BITS<<"\n";
    std::vector<TestResult> r;
    r.push_back(test_identity());
    int passed=0; for(auto&t:r) passed+=t.passed?1:0;
    std::cout<<"\n================ "<<passed<<"/"<<r.size()<<" cases passed ================\n";
    return (passed==(int)r.size())?0:1;
}
