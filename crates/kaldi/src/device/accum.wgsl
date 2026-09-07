// GPU GMM statistics accumulation, two kernels sharing the resident packed model.
//
// Kernel 1 (`posteriors`): one invocation per frame. Reads the frame's raw
// feature row `x` and its aligned pdf, evaluates every gaussian row of that pdf
// (gconst + mean*invvar . x - 0.5*invvar . x^2, the same math as the scoring
// kernel with the squares formed in-kernel), log-sum-exps, and
// writes the normalised posteriors, scaled by the frame weight, into
// `post[post_off[frame] + c]`. The frame's *unweighted* log-likelihood goes to
// `loglike[frame]`.
//
// Kernel 2 (`reduce`): one workgroup per (active gaussian, 64-dimension slice).
// The CPU has counting sorted the frames by pdf, so a gaussian's contributing
// frames are the contiguous range `bucket[bstart[p] .. bend[p]]` of its pdf `p`.
// Lane `d` of the workgroup owns output dimension `d` and walks that range in
// registers, so every lane writes its own output slot exactly once: no
// barriers, no workgroup memory, no atomics, no cross-workgroup races.

struct Params {
    frames: u32,
    xpad: u32,       // raw feature row stride in floats (dim rounded up to 4)
    dim: u32,
    num_active: u32, // number of active gaussians (rows of `out`)
    out_width: u32,  // 1 + 2*dim
    pwidth: u32,     // packed gaussian row stride in floats (1 + 2*dim rounded up to 4)
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> feats: array<f32>;          // [frames, xpad] raw x
@group(0) @binding(2) var<storage, read> packed: array<f32>;         // [total_gauss, pwidth]: gconst, mean*invvar, -0.5*invvar
@group(0) @binding(3) var<storage, read> seg_start: array<u32>;      // [num_pdfs]
@group(0) @binding(4) var<storage, read> seg_end: array<u32>;        // [num_pdfs]
@group(0) @binding(5) var<storage, read> frame_pdf: array<u32>;      // [frames]
@group(0) @binding(6) var<storage, read> frame_weight: array<f32>;   // [frames]
@group(0) @binding(7) var<storage, read> post_off: array<u32>;       // [frames]
@group(0) @binding(8) var<storage, read_write> post: array<f32>;     // [sum ngauss(frame)]
@group(0) @binding(9) var<storage, read_write> loglike: array<f32>;  // [frames]
// Frame indices grouped by pdf, and each active gaussian's descriptor.
@group(0) @binding(10) var<storage, read> bucket: array<u32>;        // [frames]
@group(0) @binding(11) var<storage, read> g_bstart: array<u32>;      // [num_active]
@group(0) @binding(12) var<storage, read> g_bend: array<u32>;        // [num_active]
@group(0) @binding(13) var<storage, read> g_comp: array<u32>;        // [num_active] index within pdf
@group(0) @binding(14) var<storage, read_write> out: array<f32>;     // [num_active, out_width]

const NEG_INF: f32 = -3.4028235e38;
const RED_THREADS: u32 = 64u;

@compute @workgroup_size(64, 1, 1)
fn posteriors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let frame = gid.x;
    if (frame >= params.frames) {
        return;
    }
    let dim = params.dim;
    let x_off = frame * params.xpad;
    let pdf = frame_pdf[frame];
    let lo = seg_start[pdf];
    let hi = seg_end[pdf];
    let base = post_off[frame];

    // Pass 1: component log-likelihoods gconst + sum(m*iv * x) + sum(-0.5*iv * x^2),
    // tracking the max.
    var mx = NEG_INF;
    for (var g: u32 = lo; g < hi; g = g + 1u) {
        let b_off = g * params.pwidth;
        var acc = packed[b_off];
        for (var d: u32 = 0u; d < dim; d = d + 1u) {
            let x = feats[x_off + d];
            acc = acc + packed[b_off + 1u + d] * x + packed[b_off + 1u + dim + d] * x * x;
        }
        post[base + (g - lo)] = acc;
        mx = max(mx, acc);
    }
    if (hi <= lo) {
        loglike[frame] = NEG_INF;
        return;
    }
    // Pass 2: exponentiate and normalise.
    var sum = 0.0;
    for (var c: u32 = 0u; c < hi - lo; c = c + 1u) {
        let e = exp(post[base + c] - mx);
        post[base + c] = e;
        sum = sum + e;
    }
    loglike[frame] = mx + log(sum);
    let scale = frame_weight[frame] / sum;
    for (var c: u32 = 0u; c < hi - lo; c = c + 1u) {
        post[base + c] = post[base + c] * scale;
    }
}

// The reduce kernel assigns each of the 64 lanes one output *dimension* of one
// gaussian and walks the gaussian's frames. Lane `d` accumulates `p*x[d]` and
// `p*x[d]^2`; lane 0 additionally accumulates the occupancy. Every lane writes
// its own output slot, so there are no barriers and no workgroup memory at all.
// Gaussians whose dim exceeds 64 are dispatched as several workgroups in y,
// each covering a 64-wide slice of dimensions.
@compute @workgroup_size(64, 1, 1)
fn reduce(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let a = wg.x;
    if (a >= params.num_active) {
        return;
    }
    let dim = params.dim;
    let d = wg.y * RED_THREADS + lid.x;
    let want_occ = (d == 0u);
    if (d >= dim && !want_occ) {
        return;
    }
    let lo = g_bstart[a];
    let hi = g_bend[a];
    let comp = g_comp[a];
    let xpad = params.xpad;
    let obase = a * params.out_width;

    var occ = 0.0;
    var sx = 0.0;
    var sx2 = 0.0;
    for (var i: u32 = lo; i < hi; i = i + 1u) {
        let f = bucket[i];
        let p = post[post_off[f] + comp];
        occ = occ + p;
        if (d < dim) {
            let x = feats[f * xpad + d];
            sx = sx + p * x;
            sx2 = sx2 + p * x * x;
        }
    }
    if (want_occ) {
        out[obase] = occ;
    }
    if (d < dim) {
        out[obase + 1u + d] = sx;
        out[obase + 1u + dim + d] = sx2;
    }
}
