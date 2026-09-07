//! Port of `plans/kaldi/src/tree/cluster-utils.{h,cc}`: the clustering
//! algorithms used by tree building —
//! `ClusterBottomUp`, `ClusterBottomUpCompartmentalized`, `RefineClusters` and
//! `ClusterKMeans`. `TreeCluster` lives in [`super::tree_cluster`].
//!
//! Determinism notes (parity-relevant):
//!
//! * Kaldi uses `std::priority_queue`, a binary heap whose sift-up/sift-down
//!   ordering decides ties between equal-distance merges. [`super::heap::Heap`]
//!   reimplements libstdc++'s `push_heap`/`pop_heap` element-for-element so
//!   that equal keys pop in exactly the same order.
//! * `ClusterKMeans` seeds its initial assignment from `Rand()`. Kaldi's is a
//!   global libc `rand()`; we use a `Xoshiro256PlusPlus` seeded with 0, created
//!   fresh per `cluster_kmeans` call, so runs are reproducible.
//!   CONTRACT-DEVIATION: bit-exact parity with a Kaldi run is impossible here
//!   because Kaldi's stream depends on process-global `rand()` state; the
//!   algorithm, the coprime-skip construction, and the number of tries match.
//! * `RefineClusters::InitPoint` uses `std::nth_element`, whose ordering among
//!   equal keys is unspecified. We sort by `(distance, cluster index)`, which
//!   is deterministic and agrees with `nth_element` whenever distances are
//!   distinct. CONTRACT-DEVIATION: documented tie-break, no Kaldi equivalent.

use super::clusterable::{
    GaussClusterable, add_to_clusters, sum_clusterable, sum_clusterable_normalizer,
    sum_clusterable_objf,
};
use super::heap::{Heap, OrdF64};
use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Kaldi `RefineClustersOptions`.
#[derive(Clone, Copy, Debug)]
pub struct RefineClustersOptions {
    pub num_iters: i32,
    pub top_n: i32,
}

impl Default for RefineClustersOptions {
    fn default() -> Self {
        Self {
            num_iters: 100,
            top_n: 5,
        }
    }
}

impl RefineClustersOptions {
    pub fn new(num_iters: i32, top_n: i32) -> Self {
        Self { num_iters, top_n }
    }
}

/// Kaldi `ClusterKMeansOptions`.
#[derive(Clone, Copy, Debug)]
pub struct ClusterKMeansOptions {
    pub refine_cfg: RefineClustersOptions,
    pub num_iters: i32,
    pub num_tries: i32,
}

impl Default for ClusterKMeansOptions {
    fn default() -> Self {
        Self {
            refine_cfg: RefineClustersOptions::default(),
            num_iters: 20,
            num_tries: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Bottom-up clustering (cluster-utils.cc:193)
// ---------------------------------------------------------------------------

/// Kaldi `ClusterBottomUp`. Merges the closest pair repeatedly while the merge
/// cost is `<= max_merge_thresh` and the number of clusters exceeds `min_clust`.
/// Returns `(total objf change (<= 0), clusters, assignments)`.
pub fn cluster_bottom_up(
    points: &[GaussClusterable],
    max_merge_thresh: f64,
    min_clust: usize,
) -> (f64, Vec<GaussClusterable>, Vec<usize>) {
    let npoints = points.len();
    let mut clusters: Vec<Option<GaussClusterable>> =
        points.iter().map(|p| Some(p.clone())).collect();
    let mut assignments: Vec<usize> = (0..npoints).collect();
    let mut dist_vec = vec![0.0f64; npoints * npoints.saturating_sub(1) / 2];
    let idx = |i: usize, j: usize| (i * (i - 1)) / 2 + j;

    // Min-heap on (dist, i, j): std::greater<pair<float,pair<i,j>>>.
    type Elem = std::cmp::Reverse<(OrdF64, u32, u32)>;
    let mut queue: Heap<Elem> = Heap::new();
    let mut nclusters = npoints;

    // SetInitialDistances
    for i in 0..npoints {
        for j in 0..i {
            let d = clusters[i]
                .as_ref()
                .unwrap()
                .distance(clusters[j].as_ref().unwrap());
            dist_vec[idx(i, j)] = d;
            if d <= max_merge_thresh {
                queue.push(std::cmp::Reverse((OrdF64(d), i as u32, j as u32)));
            }
        }
    }

    let mut ans = 0.0f64;
    while nclusters > min_clust && !queue.is_empty() {
        let std::cmp::Reverse((OrdF64(dist), i, j)) = queue.pop().unwrap();
        let (i, j) = (i as usize, j as usize);
        // CanMerge
        if clusters[i].is_none() || clusters[j].is_none() {
            continue;
        }
        let cached = dist_vec[idx(i, j)];
        if (cached - dist).abs() > 1.0e-05 * dist.abs() {
            continue;
        }
        // MergeClusters
        let cj = clusters[j].take().unwrap();
        clusters[i].as_mut().unwrap().add(&cj);
        assignments[j] = i;
        ans -= dist_vec[idx(i, j)];
        nclusters -= 1;
        for k in 0..npoints {
            if k != i && clusters[k].is_some() {
                let (a, b) = if k < i { (i, k) } else { (k, i) };
                let d = clusters[a]
                    .as_ref()
                    .unwrap()
                    .distance(clusters[b].as_ref().unwrap());
                dist_vec[idx(a, b)] = d;
                if d < max_merge_thresh {
                    queue.push(std::cmp::Reverse((OrdF64(d), a as u32, b as u32)));
                }
                // ReconstructQueue when the queue grows too large.
                if queue.len() >= npoints * npoints {
                    queue.clear();
                    for ii in 0..npoints {
                        if clusters[ii].is_none() {
                            continue;
                        }
                        for jj in 0..ii {
                            if clusters[jj].is_none() {
                                continue;
                            }
                            let dd = dist_vec[idx(ii, jj)];
                            if dd <= max_merge_thresh {
                                queue.push(std::cmp::Reverse((OrdF64(dd), ii as u32, jj as u32)));
                            }
                        }
                    }
                }
            }
        }
    }

    // Renumber
    let mut mapping = vec![usize::MAX; npoints];
    let mut new_clusters = Vec::with_capacity(nclusters);
    for i in 0..npoints {
        if let Some(c) = clusters[i].take() {
            mapping[i] = new_clusters.len();
            new_clusters.push(c);
        }
    }
    let mut new_assignments = vec![0usize; npoints];
    for i in 0..npoints {
        let mut ii = i;
        while assignments[ii] != ii {
            ii = assignments[ii];
        }
        new_assignments[i] = mapping[ii];
    }
    (ans, new_clusters, new_assignments)
}

/// Kaldi `ClusterBottomUpCompartmentalized`: bottom-up clustering where merges
/// happen only within a compartment. Returns `(objf change, assignments)`.
pub fn cluster_bottom_up_compartmentalized(
    points: &[Vec<GaussClusterable>],
    max_merge_thresh: f64,
    min_clust: usize,
) -> (f64, Vec<Vec<usize>>) {
    let ncomp = points.len();
    let npoints: Vec<usize> = points.iter().map(|p| p.len()).collect();
    let mut nclusters: usize = npoints.iter().sum();
    let mut clusters: Vec<Vec<Option<GaussClusterable>>> = points
        .iter()
        .map(|p| p.iter().map(|c| Some(c.clone())).collect())
        .collect();
    let mut assignments: Vec<Vec<usize>> = npoints.iter().map(|n| (0..*n).collect()).collect();
    let mut dist_vec: Vec<Vec<f64>> = npoints
        .iter()
        .map(|n| vec![0.0f64; n * n.saturating_sub(1) / 2])
        .collect();
    let idx = |i: usize, j: usize| (i * (i - 1)) / 2 + j;

    // CompBotClustElem compares on dist only; std::greater -> min-heap.
    // We keep (dist, comp, i, j) in the key so that the ordering is total and
    // reproducible; Kaldi's own tie order is heap-implementation defined.
    type Elem = std::cmp::Reverse<(OrdF64, u32, u32, u32)>;
    let mut queue: Heap<Elem> = Heap::new();

    let set_distance = |queue: &mut Heap<Elem>,
                        dist_vec: &mut [Vec<f64>],
                        clusters: &[Vec<Option<GaussClusterable>>],
                        comp: usize,
                        i: usize,
                        j: usize| {
        let d = clusters[comp][i]
            .as_ref()
            .unwrap()
            .distance(clusters[comp][j].as_ref().unwrap());
        dist_vec[comp][idx(i, j)] = d;
        if d < max_merge_thresh {
            queue.push(std::cmp::Reverse((
                OrdF64(d),
                comp as u32,
                i as u32,
                j as u32,
            )));
        }
    };

    for comp in 0..ncomp {
        for i in 0..npoints[comp] {
            for j in 0..i {
                set_distance(&mut queue, &mut dist_vec, &clusters, comp, i, j);
            }
        }
    }

    let mut total_obj_change = 0.0f64;
    while nclusters > min_clust && !queue.is_empty() {
        let std::cmp::Reverse((OrdF64(dist), comp, i, j)) = queue.pop().unwrap();
        let (comp, i, j) = (comp as usize, i as usize, j as usize);
        if clusters[comp][i].is_none() || clusters[comp][j].is_none() {
            continue;
        }
        let cached = dist_vec[comp][idx(i, j)];
        if (cached - dist).abs() > 1.0e-05 * dist.abs() {
            continue;
        }
        let cj = clusters[comp][j].take().unwrap();
        clusters[comp][i].as_mut().unwrap().add(&cj);
        assignments[comp][j] = i;
        total_obj_change += -dist_vec[comp][idx(i, j)];
        nclusters -= 1;
        for k in 0..npoints[comp] {
            if k != i && clusters[comp][k].is_some() {
                let (a, b) = if k < i { (i, k) } else { (k, i) };
                set_distance(&mut queue, &mut dist_vec, &clusters, comp, a, b);
            }
        }
        if queue.len() >= nclusters * nclusters {
            queue.clear();
            for c in 0..ncomp {
                for ii in 0..npoints[c] {
                    if clusters[c][ii].is_none() {
                        continue;
                    }
                    for jj in 0..ii {
                        if clusters[c][jj].is_none() {
                            continue;
                        }
                        set_distance(&mut queue, &mut dist_vec, &clusters, c, ii, jj);
                    }
                }
            }
        }
    }

    // Renumber each compartment.
    let mut out_assignments = Vec::with_capacity(ncomp);
    for comp in 0..ncomp {
        let mut mapping = vec![usize::MAX; npoints[comp]];
        let mut next = 0usize;
        for i in 0..npoints[comp] {
            if clusters[comp][i].is_some() {
                mapping[i] = next;
                next += 1;
            }
        }
        let mut new_assignments = vec![0usize; npoints[comp]];
        for i in 0..npoints[comp] {
            let mut ii = i;
            while assignments[comp][ii] != ii {
                ii = assignments[comp][ii];
            }
            new_assignments[i] = mapping[ii];
        }
        out_assignments.push(new_assignments);
    }
    (total_obj_change, out_assignments)
}

// ---------------------------------------------------------------------------
// RefineClusters (cluster-utils.cc:686)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct PointInfo {
    clust: i32,
    time: i32,
    objf: f64,
}

/// Kaldi `RefineClusters`: hill-climb the assignment of points to clusters,
/// each point only considering its `top_n` nearest clusters. Mutates
/// `clusters` and `assignments`; returns the objf improvement (>= 0).
pub fn refine_clusters(
    points: &[GaussClusterable],
    clusters: &mut [GaussClusterable],
    assignments: &mut [usize],
    cfg: RefineClustersOptions,
) -> f64 {
    if cfg.num_iters <= 0 {
        return 0.0;
    }
    let num_points = points.len();
    let num_clust = clusters.len();
    let mut top_n = cfg.top_n;
    if top_n > num_clust as i32 {
        top_n = num_clust as i32;
    }
    if top_n <= 1 {
        return 0.0;
    }
    let top_n = top_n as usize;

    let mut t: i32 = 0;
    let mut my_clust_index = vec![0usize; num_points];
    let mut clust_time = vec![0i32; num_clust];
    let mut clust_objf: Vec<f64> = clusters.iter().map(|c| c.objf()).collect();
    let mut info = vec![
        PointInfo {
            clust: 0,
            time: 0,
            objf: 0.0
        };
        num_points * top_n
    ];

    // InitPoints
    for point in 0..num_points {
        let my_clust = assignments[point];
        let mut distances: Vec<(f64, usize)> = Vec::with_capacity(num_clust.saturating_sub(1));
        for clust in 0..num_clust {
            if clust != my_clust {
                let other_clust_objf = clust_objf[clust];
                let other_clust_plus_me_objf = clusters[clust].objf_plus(&points[point]);
                distances.push((other_clust_objf - other_clust_plus_me_objf, clust));
            }
        }
        // Kaldi: nth_element(begin, begin+(top_n-2), end). See module note.
        distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        for index in 0..(top_n - 1) {
            let (distance, clust) = distances[index];
            let other_clust_objf = clust_objf[clust];
            info[point * top_n + index] = PointInfo {
                clust: clust as i32,
                time: 0,
                objf: -(distance - other_clust_objf),
            };
        }
        info[point * top_n + (top_n - 1)] = PointInfo {
            clust: my_clust as i32,
            time: 0,
            objf: clusters[my_clust].objf_minus(&points[point]),
        };
        my_clust_index[point] = top_n - 1;
    }

    let mut ans = 0.0f64;

    // Iterate
    for _ in 0..cfg.num_iters {
        let cur_t = t;
        for point in 0..num_points {
            // ProcessPoint
            let self_index = my_clust_index[point];
            let self_clust = info[point * top_n + self_index].clust as usize;

            // UpdateInfo(point, self_index)
            update_info(
                &mut info,
                top_n,
                point,
                self_index,
                &my_clust_index,
                clusters,
                points,
                &clust_time,
                t,
            );

            let own_clust_objf = clust_objf[self_clust];
            let own_clust_minus_me_objf = info[point * top_n + self_index].objf;

            for index in 0..top_n {
                if index == self_index {
                    continue;
                }
                update_info(
                    &mut info,
                    top_n,
                    point,
                    index,
                    &my_clust_index,
                    clusters,
                    points,
                    &clust_time,
                    t,
                );
                let other = info[point * top_n + index];
                let other_clust_objf = clust_objf[other.clust as usize];
                let impr = other.objf + own_clust_minus_me_objf - other_clust_objf - own_clust_objf;
                if impr > 0.0 {
                    ans += impr;
                    // MovePoint
                    t += 1;
                    let old_index = my_clust_index[point];
                    my_clust_index[point] = index;
                    let old_clust = info[point * top_n + old_index].clust as usize;
                    let new_clust = other.clust as usize;
                    assignments[point] = new_clust;
                    clusters[old_clust].sub(&points[point]);
                    clusters[new_clust].add(&points[point]);
                    clust_objf[old_clust] = clusters[old_clust].objf();
                    clust_time[old_clust] = t;
                    clust_objf[new_clust] = clusters[new_clust].objf();
                    clust_time[new_clust] = t;
                    break;
                }
            }
        }
        if t == cur_t {
            break; // converged
        }
    }
    ans
}

#[allow(clippy::too_many_arguments)]
fn update_info(
    info: &mut [PointInfo],
    top_n: usize,
    point: usize,
    idx: usize,
    my_clust_index: &[usize],
    clusters: &[GaussClusterable],
    points: &[GaussClusterable],
    clust_time: &[i32],
    t: i32,
) {
    let pinfo = info[point * top_n + idx];
    if pinfo.time < clust_time[pinfo.clust as usize] {
        let mut tmp = clusters[pinfo.clust as usize].clone();
        if idx == my_clust_index[point] {
            tmp.sub(&points[point]);
        } else {
            tmp.add(&points[point]);
        }
        info[point * top_n + idx].time = t;
        info[point * top_n + idx].objf = tmp.objf();
    }
}

// ---------------------------------------------------------------------------
// ClusterKMeans (cluster-utils.cc:918)
// ---------------------------------------------------------------------------

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Kaldi `ClusterKMeansOnce`. Returns `(objf improvement, clusters, assignments)`.
fn cluster_kmeans_once(
    points: &[GaussClusterable],
    num_clust: usize,
    cfg: &ClusterKMeansOptions,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, Vec<GaussClusterable>, Vec<usize>) {
    let num_points = points.len();
    assert!(num_points != 0);
    assert!(num_clust <= num_points);
    let mut clusters_out: Vec<Option<GaussClusterable>> = vec![None; num_clust];
    let mut assignments_out = vec![0usize; num_points];

    // Pseudo-random initial assignment via a skip coprime to num_points.
    let skip: usize = if num_points == 1 {
        1
    } else {
        let mut skip = 1 + (rng.random::<u32>() as usize % (num_points - 1));
        while gcd(skip as i64, num_points as i64) != 1 {
            if skip == num_points - 1 {
                skip = 0;
            }
            skip += 1;
        }
        skip
    };
    {
        let (mut i, mut j) = (0usize, 0usize);
        for _ in 0..num_points {
            match &mut clusters_out[j] {
                None => clusters_out[j] = Some(points[i].clone()),
                Some(c) => c.add(&points[i]),
            }
            assignments_out[i] = j;
            i = (i + skip) % num_points;
            j = (j + 1) % num_clust;
        }
    }

    let mut ans = {
        let all_stats = sum_clusterable(&clusters_out).expect("non-empty");
        sum_clusterable_objf(&clusters_out) - all_stats.objf()
    };

    let mut clusters: Vec<GaussClusterable> = clusters_out
        .into_iter()
        .map(|c| c.expect("filled"))
        .collect();

    for _ in 0..cfg.num_iters {
        let impr = refine_clusters(points, &mut clusters, &mut assignments_out, cfg.refine_cfg);
        ans += impr;
        if impr == 0.0 {
            break;
        }
    }
    (ans, clusters, assignments_out)
}

/// Kaldi `ClusterKMeans`. Returns `(objf improvement, clusters, assignments)`.
///
/// Seeding: a `Xoshiro256PlusPlus` seeded with 0, created per call (Kaldi uses
/// the global libc `rand()`). See the module-level note.
pub fn cluster_kmeans(
    points: &[GaussClusterable],
    num_clust: usize,
    cfg: &ClusterKMeansOptions,
) -> (f64, Vec<GaussClusterable>, Vec<usize>) {
    if points.is_empty() {
        return (0.0, Vec::new(), Vec::new());
    }
    assert!(cfg.num_tries >= 1 && cfg.num_iters >= 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    if cfg.num_tries == 1 {
        return cluster_kmeans_once(points, num_clust, cfg, &mut rng);
    }
    let mut best: Option<(f64, Vec<GaussClusterable>, Vec<usize>)> = None;
    for i in 0..cfg.num_tries {
        let (ans, clusters, assignments) = cluster_kmeans_once(points, num_clust, cfg, &mut rng);
        let better = match &best {
            None => true,
            Some((b, _, _)) => i == 0 || ans > *b,
        };
        if better {
            best = Some((ans, clusters, assignments));
        }
    }
    best.expect("at least one try")
}

/// Kaldi `SumClusterableNormalizer` over a plain slice.
pub fn sum_normalizer(v: &[GaussClusterable]) -> f64 {
    let opts: Vec<Option<GaussClusterable>> = v.iter().map(|c| Some(c.clone())).collect();
    sum_clusterable_normalizer(&opts)
}

/// Helper: run `add_to_clusters` where every input stat is present.
pub fn add_all_to_clusters(
    stats: &[GaussClusterable],
    assignments: &[usize],
) -> Vec<Option<GaussClusterable>> {
    let stats: Vec<Option<GaussClusterable>> = stats.iter().map(|c| Some(c.clone())).collect();
    let mut clusters = Vec::new();
    add_to_clusters(&stats, assignments, &mut clusters);
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, var_floor: f64) -> GaussClusterable {
        let mut c = GaussClusterable::new(1, var_floor);
        c.add_stats(&[x], 1.0);
        c.add_stats(&[x + 0.1], 1.0);
        c
    }

    #[test]
    fn bottom_up_merges_close_points() {
        let pts = vec![
            point(0.0, 0.01),
            point(0.05, 0.01),
            point(10.0, 0.01),
            point(10.05, 0.01),
        ];
        let (change, clusters, assignments) = cluster_bottom_up(&pts, 1e10, 2);
        assert_eq!(clusters.len(), 2);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
        assert!(change <= 1e-9);
    }

    #[test]
    fn bottom_up_threshold_zero_merges_nothing() {
        let pts = vec![point(0.0, 0.01), point(10.0, 0.01)];
        let (_, clusters, _) = cluster_bottom_up(&pts, 0.0, 0);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn kmeans_separates_two_groups() {
        let pts = vec![
            point(0.0, 0.01),
            point(0.2, 0.01),
            point(20.0, 0.01),
            point(20.2, 0.01),
        ];
        let cfg = ClusterKMeansOptions::default();
        let (impr, clusters, assignments) = cluster_kmeans(&pts, 2, &cfg);
        assert_eq!(clusters.len(), 2);
        assert!(impr > 0.0);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn kmeans_is_deterministic() {
        let pts: Vec<GaussClusterable> = (0..8).map(|i| point(i as f32 * 3.0, 0.01)).collect();
        let cfg = ClusterKMeansOptions::default();
        let a = cluster_kmeans(&pts, 3, &cfg).2;
        let b = cluster_kmeans(&pts, 3, &cfg).2;
        assert_eq!(a, b);
    }

    #[test]
    fn compartmentalized_keeps_compartments_separate() {
        let comp = vec![
            vec![point(0.0, 0.01), point(0.05, 0.01)],
            vec![point(10.0, 0.01), point(10.05, 0.01)],
        ];
        let (_, assignments) = cluster_bottom_up_compartmentalized(&comp, f64::INFINITY, 3);
        // 4 points, min_clust 3 -> exactly one merge, inside one compartment.
        let merged: usize = assignments
            .iter()
            .map(|a| a.len() - (a.iter().max().unwrap() + 1))
            .sum();
        assert_eq!(merged, 1);
    }
}
