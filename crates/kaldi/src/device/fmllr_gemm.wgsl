// fMLLR statistics accumulation, kernel 2 of 2: the two products over the batch.
//
// With `xplus[t] = [x_t; 1]` (dim+1 wide) and the per-frame `a`, `b` from kernel 1,
//
//     K[i][j]    = sum_t a[t][i] * xplus[t][j]                (dim x dim1)
//     G[i][r][c] = sum_t b[t][i] * xplus[t][r] * xplus[t][c]  (dim symmetric dim1^2)
//
// which is exactly Kaldi's committed single-frame stats summed over the batch, with
// the `beta` count coming from `cnt_out` (reduced on the host).
//
// Dispatch grid: (ceil(dim1/TILE), ceil(dim1/TILE), dim + 1). Layer `z < dim`
// computes the `G[z]` tile; layer `z == dim` computes the K tile (its x index is the
// model-dim row `i`, its y index the column `j`). Each workgroup walks the frames in
// tiles of TILE_T, staging that tile's two xplus column slices and the one `b`
// column it needs in workgroup memory, so each thread keeps a single f32
// accumulator and the global traffic per workgroup is O(T * (2*TILE + 1)) instead of
// O(T * TILE^2).
//
// Only `r <= c` is written; the host mirrors into the symmetric lower triangle.

struct Params {
    frames: u32,
    dim: u32,
    dim1: u32,
    xw: u32,       // xplus row stride in floats (dim1 padded to 4)
    _p0: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
};

@group(0) @binding(0) var<uniform> par: Params;
@group(0) @binding(1) var<storage, read> xplus: array<f32>;       // [frames, xw]
@group(0) @binding(2) var<storage, read> a_in: array<f32>;        // [frames, dim]
@group(0) @binding(3) var<storage, read> b_in: array<f32>;        // [frames, dim]
@group(0) @binding(4) var<storage, read_write> k_out: array<f32>; // [dim, dim1]
@group(0) @binding(5) var<storage, read_write> g_out: array<f32>; // [dim, dim1, dim1]

const TILE: u32 = 8u;
const TILE_T: u32 = 64u;
const THREADS: u32 = 64u; // TILE * TILE

var<workgroup> sh_r: array<f32, 512>; // TILE_T * TILE
var<workgroup> sh_c: array<f32, 512>; // TILE_T * TILE
var<workgroup> sh_w: array<f32, 64>;  // TILE_T, the a/b column for this layer

@compute @workgroup_size(8, 8, 1)
fn fmllr_gemm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let dim = par.dim;
    let dim1 = par.dim1;
    let frames = par.frames;
    let xw = par.xw;
    let tid = lid.y * TILE + lid.x;
    let is_k = wg.z == dim;

    // Row index: the K layer indexes `a`'s model-dim rows, a G layer indexes xplus.
    let r = wg.x * TILE + lid.x;
    let c = wg.y * TILE + lid.y;
    let r_lim = select(dim1, dim, is_k);
    // For a G layer only the upper triangle is stored.
    let live = r < r_lim && c < dim1 && (is_k || r <= c);
    let col = select(wg.z, 0u, is_k); // which model-dim column of a/b this layer uses

    var acc: f32 = 0.0;
    var t0: u32 = 0u;
    loop {
        if (t0 >= frames) { break; }
        let n = min(TILE_T, frames - t0);
        // Stage: sh_r[t][lid.x] = xplus[t][r_tile + lid.x] (K layer: a[t][i]),
        //        sh_c[t][lid.y] = xplus[t][c_tile + lid.y],
        //        sh_w[t]        = b[t][col]  (unused on the K layer).
        for (var s: u32 = tid; s < n * TILE; s = s + THREADS) {
            let tt = s / TILE;
            let k = s - tt * TILE;
            let t = t0 + tt;
            let rr = wg.x * TILE + k;
            let cc = wg.y * TILE + k;
            if (is_k) {
                sh_r[s] = select(0.0, a_in[t * dim + rr], rr < dim);
            } else {
                sh_r[s] = select(0.0, xplus[t * xw + rr], rr < dim1);
            }
            sh_c[s] = select(0.0, xplus[t * xw + cc], cc < dim1);
        }
        if (!is_k) {
            for (var t: u32 = tid; t < n; t = t + THREADS) {
                sh_w[t] = b_in[(t0 + t) * dim + col];
            }
        }
        workgroupBarrier();
        if (live) {
            if (is_k) {
                for (var tt: u32 = 0u; tt < n; tt = tt + 1u) {
                    acc = acc + sh_r[tt * TILE + lid.x] * sh_c[tt * TILE + lid.y];
                }
            } else {
                for (var tt: u32 = 0u; tt < n; tt = tt + 1u) {
                    acc = acc + sh_w[tt] * sh_r[tt * TILE + lid.x] * sh_c[tt * TILE + lid.y];
                }
            }
        }
        workgroupBarrier();
        t0 = t0 + TILE_T;
    }

    if (!live) {
        return;
    }
    if (is_k) {
        k_out[r * dim1 + c] = acc;
    } else {
        g_out[(wg.z * dim1 + r) * dim1 + c] = acc;
    }
}
