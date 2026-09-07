// fMLLR statistics accumulation, kernel 1 of 2.
//
// One invocation per frame: form the posteriors over the gaussians of that frame's
// aligned pdf (the same dot products the scoring kernel does, against the same
// resident packed rows `[gconst, mean*invvar, -0.5*invvar]`), scale them by the
// frame weight, and reduce them to the two model-dim vectors Kaldi's
// `SingleFrameStats` needs:
//
//     a[i] = sum_g p_g * invvar_g[i] * mean_g[i]   (packed column 1 + i)
//     b[i] = sum_g p_g * invvar_g[i]               (-2 * packed column 1 + dim + i)
//     count = sum_g p_g
//
// written to `a_out[t*dim+i]`, `b_out[t*dim+i]` and `cnt_out[t]`. Frames with weight
// zero (silence under MFA's `silence_weight = 0`) write zeros and are skipped, which
// makes them contribute nothing to either kernel-2 product.
//
// The gaussian loop runs three times rather than buffering per-gaussian scores: a
// pdf can have hundreds of gaussians, far more than fits in registers, and the
// packed rows it re-reads stay hot in cache.

struct Params {
    frames: u32,
    dim: u32,      // model dim
    dim1: u32,     // dim + 1
    width4: u32,   // padded packed row width / 4
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<uniform> par: Params;
@group(0) @binding(1) var<storage, read> feats: array<vec4<f32>>;  // [frames, width4] rows [1,x,x*x]
@group(0) @binding(2) var<storage, read> packed: array<vec4<f32>>; // resident model rows
@group(0) @binding(3) var<storage, read> seg_start: array<u32>;
@group(0) @binding(4) var<storage, read> seg_end: array<u32>;
@group(0) @binding(5) var<storage, read> pdf_of: array<u32>;       // [frames]
@group(0) @binding(6) var<storage, read> weight: array<f32>;       // [frames]
@group(0) @binding(7) var<storage, read_write> a_out: array<f32>;  // [frames, dim]
@group(0) @binding(8) var<storage, read_write> b_out: array<f32>;  // [frames, dim]
@group(0) @binding(9) var<storage, read_write> cnt_out: array<f32>;// [frames]

fn score(g: u32, f_off: u32, w4: u32) -> f32 {
    var acc: f32 = 0.0;
    let b_off = g * w4;
    for (var kk: u32 = 0u; kk < w4; kk = kk + 1u) {
        acc = acc + dot(feats[f_off + kk], packed[b_off + kk]);
    }
    return acc;
}

@compute @workgroup_size(64, 1, 1)
fn fmllr_ab(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= par.frames) {
        return;
    }
    let dim = par.dim;
    let w4 = par.width4;
    let ab_base = t * dim;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        a_out[ab_base + i] = 0.0;
        b_out[ab_base + i] = 0.0;
    }
    cnt_out[t] = 0.0;

    let w = weight[t];
    let pdf = pdf_of[t];
    let lo = seg_start[pdf];
    let hi = seg_end[pdf];
    if (w == 0.0 || hi <= lo) {
        return;
    }
    let f_off = t * w4;

    var mx: f32 = -3.4028235e38;
    for (var g: u32 = lo; g < hi; g = g + 1u) {
        mx = max(mx, score(g, f_off, w4));
    }
    var sum: f32 = 0.0;
    for (var g: u32 = lo; g < hi; g = g + 1u) {
        sum = sum + exp(score(g, f_off, w4) - mx);
    }
    let inv = w / sum;

    var count: f32 = 0.0;
    for (var g: u32 = lo; g < hi; g = g + 1u) {
        let p = exp(score(g, f_off, w4) - mx) * inv;
        count = count + p;
        let b_off = g * w4;
        for (var i: u32 = 0u; i < dim; i = i + 1u) {
            let mo = 1u + i;
            let ho = 1u + dim + i;
            let mi = packed[b_off + mo / 4u][mo % 4u];
            let hv = packed[b_off + ho / 4u][ho % 4u];
            a_out[ab_base + i] = a_out[ab_base + i] + p * mi;
            b_out[ab_base + i] = b_out[ab_base + i] + p * (-2.0 * hv);
        }
    }
    cnt_out[t] = count;
}
