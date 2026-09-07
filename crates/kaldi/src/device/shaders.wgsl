// Fused diagonal-GMM scoring kernel, one job (utterance) per dispatch.
//
// The whole packed model is resident: `packed[g]` rows `[gconst, mean*invvar,
// -0.5*invvar]` zero-padded to a multiple of 4 floats and viewed as vec4s, and
// `seg_start/seg_end[pdf]` give each pdf's gaussian row range. A job scores its
// frames `[1, x, x*x]` against only the pdfs it lists in `sel`, writing
// `scores[out_off + frame * nsel + s]`. Batches of jobs share one submit through
// dynamic uniform offsets into a job table.
//
// Each invocation owns FRAMES_PER_THREAD frames of one selected pdf, so every
// gaussian row it loads feeds four dot products; the 32 frames of a workgroup share
// a tile of A in workgroup memory, and threads with the same pdf read the same
// gaussian row, which the GPU broadcasts.

struct Job {
    frames: u32,
    width4: u32,        // padded width / 4, <= MAX_WIDTH4
    nsel: u32,          // number of selected pdfs
    sel_off: u32,       // offset into `sel`
    feat_off: u32,      // offset into `feats`, in vec4 units
    out_off: u32,       // offset into `scores`, in floats
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> feats: array<vec4<f32>>;   // all jobs' rows
@group(0) @binding(2) var<storage, read> packed: array<vec4<f32>>;  // [total_gauss, width4]
@group(0) @binding(3) var<storage, read> seg_start: array<u32>;     // [num_pdfs]
@group(0) @binding(4) var<storage, read> seg_end: array<u32>;       // [num_pdfs]
@group(0) @binding(5) var<storage, read> sel: array<u32>;           // all jobs' pdf lists
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;  // all jobs' outputs

const FRAMES_PER_WG: u32 = 32u;
const FRAMES_PER_THREAD: u32 = 4u;
const FRAME_THREADS: u32 = 8u;   // FRAMES_PER_WG / FRAMES_PER_THREAD
const PDFS_PER_WG: u32 = 16u;
const THREADS: u32 = 128u;       // FRAME_THREADS * PDFS_PER_WG
const MAX_WIDTH4: u32 = 24u;     // 96 floats
const NEG_INF: f32 = -3.4028235e38;

var<workgroup> tile_a: array<vec4<f32>, 768>; // FRAMES_PER_WG * MAX_WIDTH4

@compute @workgroup_size(8, 16, 1)
fn score_pdfs(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let frames = job.frames;
    let w4 = job.width4;
    let nsel = job.nsel;
    let frame_base = wg.x * FRAMES_PER_WG;

    // Cooperative load of this workgroup's 32 feature rows (as vec4s).
    let tid = lid.y * FRAME_THREADS + lid.x;
    let tile_len = FRAMES_PER_WG * w4;
    for (var idx: u32 = tid; idx < tile_len; idx = idx + THREADS) {
        let f = idx / w4;
        let kk = idx - f * w4;
        let frame = frame_base + f;
        if (frame < frames) {
            tile_a[idx] = feats[job.feat_off + frame * w4 + kk];
        } else {
            tile_a[idx] = vec4<f32>(0.0);
        }
    }
    workgroupBarrier();

    let s = wg.y * PDFS_PER_WG + lid.y;
    // Adjacent lanes read adjacent frame rows. With contiguous four-frame blocks,
    // a 40-dimensional model gives a shared-memory lane stride of 336 floats:
    // only two of 32 starting banks are used, causing four-way bank conflicts.
    // Interleaving each lane's four frames keeps the arithmetic per frame intact.
    let f0 = lid.x;
    if (frame_base + f0 >= frames || s >= nsel) {
        return;
    }

    let pdf = sel[job.sel_off + s];
    let lo = seg_start[pdf];
    let hi = seg_end[pdf];
    let a0 = f0 * w4;
    let a1 = a0 + FRAME_THREADS * w4;
    let a2 = a1 + FRAME_THREADS * w4;
    let a3 = a2 + FRAME_THREADS * w4;

    var mx = vec4<f32>(NEG_INF);
    var sum = vec4<f32>(0.0);
    // Four gaussian rows per pass over the tile: each tile load feeds 16 dots
    // instead of 4, which is what bounds this loop (workgroup-memory bandwidth).
    var g: u32 = lo;
    for (; g + 4u <= hi; g = g + 4u) {
        let b0 = g * w4;
        let b1 = b0 + w4;
        let b2 = b1 + w4;
        let b3 = b2 + w4;
        var acc0 = vec4<f32>(0.0);
        var acc1 = vec4<f32>(0.0);
        var acc2 = vec4<f32>(0.0);
        var acc3 = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < w4; kk = kk + 1u) {
            let x0 = tile_a[a0 + kk];
            let x1 = tile_a[a1 + kk];
            let x2 = tile_a[a2 + kk];
            let x3 = tile_a[a3 + kk];
            let r0 = packed[b0 + kk];
            let r1 = packed[b1 + kk];
            let r2 = packed[b2 + kk];
            let r3 = packed[b3 + kk];
            acc0 = acc0 + vec4<f32>(dot(x0, r0), dot(x1, r0), dot(x2, r0), dot(x3, r0));
            acc1 = acc1 + vec4<f32>(dot(x0, r1), dot(x1, r1), dot(x2, r1), dot(x3, r1));
            acc2 = acc2 + vec4<f32>(dot(x0, r2), dot(x1, r2), dot(x2, r2), dot(x3, r2));
            acc3 = acc3 + vec4<f32>(dot(x0, r3), dot(x1, r3), dot(x2, r3), dot(x3, r3));
        }
        // Same running log-sum-exp as the scalar path, one row at a time so the
        // rounding matches the CPU reference.
        var acc = acc0;
        var bigger = acc > mx;
        var new_mx = max(mx, acc);
        sum = select(sum + exp(acc - mx), sum * exp(mx - acc) + vec4<f32>(1.0), bigger);
        mx = new_mx;
        acc = acc1;
        bigger = acc > mx;
        new_mx = max(mx, acc);
        sum = select(sum + exp(acc - mx), sum * exp(mx - acc) + vec4<f32>(1.0), bigger);
        mx = new_mx;
        acc = acc2;
        bigger = acc > mx;
        new_mx = max(mx, acc);
        sum = select(sum + exp(acc - mx), sum * exp(mx - acc) + vec4<f32>(1.0), bigger);
        mx = new_mx;
        acc = acc3;
        bigger = acc > mx;
        new_mx = max(mx, acc);
        sum = select(sum + exp(acc - mx), sum * exp(mx - acc) + vec4<f32>(1.0), bigger);
        mx = new_mx;
    }
    for (; g < hi; g = g + 1u) {
        let b_off = g * w4;
        var acc = vec4<f32>(0.0);
        for (var kk: u32 = 0u; kk < w4; kk = kk + 1u) {
            let b = packed[b_off + kk];
            acc.x = acc.x + dot(tile_a[a0 + kk], b);
            acc.y = acc.y + dot(tile_a[a1 + kk], b);
            acc.z = acc.z + dot(tile_a[a2 + kk], b);
            acc.w = acc.w + dot(tile_a[a3 + kk], b);
        }
        let bigger = acc > mx;
        let new_mx = max(mx, acc);
        sum = select(sum + exp(acc - mx), sum * exp(mx - acc) + vec4<f32>(1.0), bigger);
        mx = new_mx;
    }

    var result = vec4<f32>(NEG_INF);
    if (hi > lo) {
        result = mx + log(sum);
    }
    let frame = frame_base + f0;
    let base = job.out_off + frame * nsel + s;
    scores[base] = result.x;
    if (frame + FRAME_THREADS < frames) { scores[base + FRAME_THREADS * nsel] = result.y; }
    if (frame + 2u * FRAME_THREADS < frames) { scores[base + 2u * FRAME_THREADS * nsel] = result.z; }
    if (frame + 3u * FRAME_THREADS < frames) { scores[base + 3u * FRAME_THREADS * nsel] = result.w; }
}
