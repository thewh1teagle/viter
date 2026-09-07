//! Port of `TreeClusterer` / `TreeCluster` from
//! `plans/kaldi/src/tree/cluster-utils.cc:1032`.
//!
//! Top-down binary clustering: every leaf keeps its best k-means split, and the
//! highest-scoring leaf is split next. `AutomaticallyObtainQuestions` uses this
//! to turn phone stats into a binary tree of phone sets.

use super::cluster::{ClusterKMeansOptions, cluster_kmeans};
use super::clusterable::{GaussClusterable, sum_clusterable};
use super::heap::{Heap, OrdF64};

/// Kaldi `TreeClusterOptions`.
#[derive(Clone, Copy, Debug)]
pub struct TreeClusterOptions {
    pub kmeans_cfg: ClusterKMeansOptions,
    pub branch_factor: usize,
    pub thresh: f64,
}

impl Default for TreeClusterOptions {
    fn default() -> Self {
        Self {
            kmeans_cfg: ClusterKMeansOptions::default(),
            branch_factor: 2,
            thresh: 0.0,
        }
    }
}

struct TreeNode {
    is_leaf: bool,
    index: usize,
    parent: Option<usize>, // index into `nodes`
    node_total: Option<GaussClusterable>,
    points: Vec<GaussClusterable>,
    point_indices: Vec<usize>,
    best_split: f64,
    clusters: Vec<GaussClusterable>,
    assignments: Vec<usize>,
    children: Vec<usize>, // indices into `nodes`
}

/// Kaldi `TreeCluster`: top-down binary clustering. Returns
/// `(objf improvement, clusters, assignments, clust_assignments, num_leaves)`.
///
/// `clusters` holds the leaf clusters first (indices `0..num_leaves`), then the
/// non-leaf nodes in reverse topological order (root last).
/// `clust_assignments[i]` is the parent index of node `i`; the root is its own
/// parent.
pub fn tree_cluster(
    points: &[GaussClusterable],
    max_clust: usize,
    cfg: &TreeClusterOptions,
) -> (f64, Vec<GaussClusterable>, Vec<usize>, Vec<usize>, usize) {
    if points.is_empty() {
        return (0.0, Vec::new(), Vec::new(), Vec::new(), 0);
    }
    assert!(cfg.branch_factor > 1);

    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut leaf_nodes: Vec<usize> = Vec::new();
    let mut nonleaf_nodes: Vec<usize> = Vec::new();
    // Max-heap on (impr, node id); Kaldi's is priority_queue<pair<float,Node*>>
    // whose secondary key is a pointer. We use the node id, which is assigned
    // in the same creation order as Kaldi's allocations.
    let mut queue: Heap<(OrdF64, usize)> = Heap::new();

    // Init: top node.
    let top = TreeNode {
        is_leaf: true,
        index: 0,
        parent: None,
        node_total: sum_clusterable(&points.iter().map(|p| Some(p.clone())).collect::<Vec<_>>()),
        points: points.to_vec(),
        point_indices: (0..points.len()).collect(),
        best_split: 0.0,
        clusters: Vec::new(),
        assignments: Vec::new(),
        children: Vec::new(),
    };
    nodes.push(top);
    leaf_nodes.push(0);
    find_best_split(&mut nodes, 0, cfg, &mut queue);

    let mut ans = 0.0f64;
    while leaf_nodes.len() < max_clust && !queue.is_empty() {
        let (OrdF64(impr), node_id) = queue.pop().unwrap();
        ans += impr;
        // DoSplit
        let (node_points, node_indices, node_assignments, node_clusters, node_index) = {
            let n = &mut nodes[node_id];
            (
                std::mem::take(&mut n.points),
                std::mem::take(&mut n.point_indices),
                std::mem::take(&mut n.assignments),
                std::mem::take(&mut n.clusters),
                n.index,
            )
        };
        let mut child_ids = Vec::with_capacity(cfg.branch_factor);
        for (i, cluster) in node_clusters.into_iter().enumerate() {
            let child_id = nodes.len();
            let child_index = if i == 0 {
                leaf_nodes[node_index] = child_id;
                node_index
            } else {
                let new_index = leaf_nodes.len();
                leaf_nodes.push(child_id);
                new_index
            };
            nodes.push(TreeNode {
                is_leaf: true,
                index: child_index,
                parent: Some(node_id),
                node_total: Some(cluster),
                points: Vec::new(),
                point_indices: Vec::new(),
                best_split: 0.0,
                clusters: Vec::new(),
                assignments: Vec::new(),
                children: Vec::new(),
            });
            child_ids.push(child_id);
        }
        for (i, p) in node_points.into_iter().enumerate() {
            let child = child_ids[node_assignments[i]];
            nodes[child].points.push(p);
            nodes[child].point_indices.push(node_indices[i]);
        }
        {
            let n = &mut nodes[node_id];
            n.is_leaf = false;
            n.index = nonleaf_nodes.len();
            n.children = child_ids.clone();
        }
        nonleaf_nodes.push(node_id);
        for child in child_ids {
            find_best_split(&mut nodes, child, cfg, &mut queue);
        }
    }

    // CreateOutput
    let num_leaves = leaf_nodes.len();
    let total_nodes = num_leaves + nonleaf_nodes.len();
    let nonleaf_output_index = |index: usize| total_nodes - 1 - index;

    let mut assignments_out = vec![usize::MAX; points.len()];
    for (leaf, &node_id) in leaf_nodes.iter().enumerate() {
        for &pi in &nodes[node_id].point_indices {
            assignments_out[pi] = leaf;
        }
    }

    let mut clust_assignments = vec![0usize; total_nodes];
    for (leaf, &node_id) in leaf_nodes.iter().enumerate() {
        let parent_index = match nodes[node_id].parent {
            None => 0,
            Some(p) => {
                if nodes[p].is_leaf {
                    nodes[p].index
                } else {
                    nonleaf_output_index(nodes[p].index)
                }
            }
        };
        clust_assignments[leaf] = parent_index;
    }
    for &node_id in nonleaf_nodes.iter() {
        let index = nonleaf_output_index(nodes[node_id].index);
        let parent_index = match nodes[node_id].parent {
            None => index,
            Some(p) => nonleaf_output_index(nodes[p].index),
        };
        clust_assignments[index] = parent_index;
    }

    let mut clusters_out: Vec<Option<GaussClusterable>> = vec![None; total_nodes];
    for (leaf, &node_id) in leaf_nodes.iter().enumerate() {
        clusters_out[leaf] = nodes[node_id].node_total.take();
    }
    for &node_id in nonleaf_nodes.iter() {
        let index = nonleaf_output_index(nodes[node_id].index);
        clusters_out[index] = nodes[node_id].node_total.take();
    }
    let clusters_out: Vec<GaussClusterable> = clusters_out
        .into_iter()
        .map(|c| c.expect("every node has a total"))
        .collect();

    (
        ans,
        clusters_out,
        assignments_out,
        clust_assignments,
        num_leaves,
    )
}

fn find_best_split(
    nodes: &mut [TreeNode],
    node_id: usize,
    cfg: &TreeClusterOptions,
    queue: &mut Heap<(OrdF64, usize)>,
) {
    if nodes[node_id].points.len() <= 1 {
        nodes[node_id].best_split = 0.0;
        return;
    }
    let points = std::mem::take(&mut nodes[node_id].points);
    let (impr, clusters, assignments) = cluster_kmeans(&points, cfg.branch_factor, &cfg.kmeans_cfg);
    let n = &mut nodes[node_id];
    n.points = points;
    n.clusters = clusters;
    n.assignments = assignments;
    n.best_split = impr;
    if impr > cfg.thresh {
        queue.push((OrdF64(impr), node_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32) -> GaussClusterable {
        let mut c = GaussClusterable::new(1, 0.01);
        c.add_stats(&[x], 1.0);
        c.add_stats(&[x + 0.1], 1.0);
        c
    }

    #[test]
    fn builds_a_binary_tree() {
        let pts = vec![point(0.0), point(0.2), point(20.0), point(20.2)];
        let cfg = TreeClusterOptions::default();
        let (_, clusters, assignments, clust_assignments, num_leaves) = tree_cluster(&pts, 4, &cfg);
        assert!(num_leaves >= 2);
        assert_eq!(clusters.len(), clust_assignments.len());
        assert_eq!(assignments.len(), 4);
        let root = clust_assignments.len() - 1;
        assert_eq!(clust_assignments[root], root);
        for (i, &p) in clust_assignments.iter().enumerate() {
            assert!(p > i || i == root, "parent {p} must come after node {i}");
        }
        // Every point is assigned to a leaf.
        assert!(assignments.iter().all(|&a| a < num_leaves));
    }

    #[test]
    fn max_clust_caps_the_leaf_count() {
        let pts: Vec<GaussClusterable> = (0..8).map(|i| point(i as f32 * 5.0)).collect();
        let cfg = TreeClusterOptions::default();
        let (_, _, _, _, num_leaves) = tree_cluster(&pts, 3, &cfg);
        assert!(num_leaves <= 3);
    }

    #[test]
    fn single_point_is_one_leaf() {
        let pts = vec![point(1.0)];
        let cfg = TreeClusterOptions::default();
        let (impr, clusters, assignments, clust_assignments, num_leaves) =
            tree_cluster(&pts, 4, &cfg);
        assert_eq!(num_leaves, 1);
        assert_eq!(impr, 0.0);
        assert_eq!(assignments, vec![0]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clust_assignments, vec![0]);
    }

    #[test]
    fn empty_input() {
        let cfg = TreeClusterOptions::default();
        let (_, clusters, assignments, clust_assignments, num_leaves) = tree_cluster(&[], 4, &cfg);
        assert_eq!(num_leaves, 0);
        assert!(clusters.is_empty() && assignments.is_empty() && clust_assignments.is_empty());
    }
}
