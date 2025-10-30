//! Earth Mover's Distance (EMD) computation using tree embeddings.
//!
//! Implements dynamic EMD from paper Corollary 1.3. Given two distributions (weighted point sets),
//! computes approximate EMD using the tree embedding structure.
//!
//! ## Algorithm
//!
//! EMD in Euclidean space is expensive (O(n³) Hungarian algorithm).
//! EMD on a tree is cheap (O(n) dynamic programming).
//! Our approach: Embed into tree, compute EMD on tree → O(log n)-approximation.
//!
//! ## Complexity
//!
//! - EMD computation: O(n) on the tree
//! - Approximation: O(log n) factor due to tree distortion
//! - Updates: Õ(n^ε + d) when distributions change

use crate::dynamic::DynamicTreeEmbedding;
use crate::metric::{EuclideanPoint, Point};
use crate::tree_embedding::PointId;
use std::collections::HashMap;
use ordered_float::OrderedFloat;

/// Represents a weighted distribution as a set of points with weights.
#[derive(Clone, Debug)]
pub struct Distribution {
    pub points: Vec<(PointId, f64)>,  // (point_id, weight)
}

impl Distribution {
    pub fn new(points: Vec<(PointId, f64)>) -> Self {
        Self { points }
    }

    pub fn total_weight(&self) -> f64 {
        self.points.iter().map(|(_, w)| w).sum()
    }

    pub fn normalize(&mut self) {
        let total = self.total_weight();
        if total > 0.0 {
            for (_, w) in &mut self.points {
                *w /= total;
            }
        }
    }
}

impl DynamicTreeEmbedding {
    /// Compute approximate Earth Mover's Distance between two distributions.
    ///
    /// Uses the tree embedding to compute EMD efficiently. Returns O(log n)-approximate EMD
    /// due to the tree's distortion.
    ///
    /// Both distributions should have equal total weight (typically normalized to 1.0).
    ///
    /// Complexity: O(n + m) where n = points in distributions, m = tree levels.
    /// Much better than O(n³) Hungarian algorithm for exact EMD!
    pub fn emd_distance(&self, dist_a: &Distribution, dist_b: &Distribution) -> f64 {
        // EMD on tree: sum of flow costs on tree edges
        // Flow on edge = excess/deficit propagated up the tree
        
        // Build weight map for all points
        let mut weights = HashMap::new();
        
        for (point_id, weight) in &dist_a.points {
            *weights.entry(*point_id).or_insert(0.0) += weight;
        }
        
        for (point_id, weight) in &dist_b.points {
            *weights.entry(*point_id).or_insert(0.0) -= weight;
        }

        // Compute excess at each tree node by aggregating from leaves
        // For simplicity, use a greedy approach: pair up excesses at each level
        let mut total_cost = 0.0;

        for level in (0..self.num_levels()).rev() {
            // Group points by their label at this level (same subtree)
            let mut subtree_excess = HashMap::new();

            for (point_id, excess) in &weights {
                if let Some(labels) = self.get_labels(*point_id) {
                    if level < labels.len() {
                        let subtree_label = OrderedFloat(labels[level]);
                        *subtree_excess.entry(subtree_label).or_insert(0.0) += excess;
                    }
                }
            }

            // Flow cost at this level: sum of absolute flows × edge length
            let w_i = self.aspect_ratio() * 2.0_f64.powi(-(level as i32));
            let edge_length = 2.0 * w_i;

            for (_label, excess) in subtree_excess {
                total_cost += excess.abs() * edge_length;
            }
        }

        total_cost
    }

    /// Compute approximate EMD between two equal-sized point sets.
    ///
    /// Convenience method that creates uniform distributions and computes EMD.
    /// Each point has weight 1/n in its distribution.
    pub fn emd_between_sets(&self, set_a: &[PointId], set_b: &[PointId]) -> f64 {
        let weight_a = 1.0 / set_a.len() as f64;
        let weight_b = 1.0 / set_b.len() as f64;

        let dist_a = Distribution::new(set_a.iter().map(|&id| (id, weight_a)).collect());
        let dist_b = Distribution::new(set_b.iter().map(|&id| (id, weight_b)).collect());

        self.emd_distance(&dist_a, &dist_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::EuclideanPoint;

    #[test]
    fn test_emd_basic() {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);

        // Two clusters
        let cluster_a: Vec<_> = (0..5)
            .map(|i| store.insert(EuclideanPoint::new(vec![i as f64, 0.0])))
            .collect();

        let cluster_b: Vec<_> = (0..5)
            .map(|i| store.insert(EuclideanPoint::new(vec![50.0 + i as f64, 50.0])))
            .collect();

        let emd = store.emd_between_sets(&cluster_a, &cluster_b);

        println!("EMD between clusters: {:.2}", emd);
        assert!(emd > 0.0, "EMD should be positive for different clusters");
    }

    #[test]
    fn test_emd_identical_distributions() {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);

        let points: Vec<_> = (0..10)
            .map(|i| store.insert(EuclideanPoint::new(vec![i as f64, i as f64])))
            .collect();

        let emd = store.emd_between_sets(&points[0..5], &points[0..5]);

        println!("EMD of identical distributions: {:.2}", emd);
        assert!(emd.abs() < 1e-6, "EMD should be zero for identical distributions");
    }

    #[test]
    fn test_emd_with_weights() {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);

        let id1 = store.insert(EuclideanPoint::new(vec![0.0, 0.0]));
        let id2 = store.insert(EuclideanPoint::new(vec![10.0, 10.0]));

        let dist_a = Distribution::new(vec![(id1, 1.0)]);
        let dist_b = Distribution::new(vec![(id2, 1.0)]);

        let emd = store.emd_distance(&dist_a, &dist_b);

        println!("EMD between two points: {:.2}", emd);
        assert!(emd > 0.0);

        // EMD should be approximately the tree distance
        if let Some(tree_dist) = store.tree_distance(id1, id2) {
            println!("Tree distance: {:.2}", tree_dist);
            println!("Ratio: {:.2}", emd / tree_dist);
        }
    }

    #[test]
    fn test_emd_approximation_quality() {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);

        // Create two distributions
        let set_a: Vec<_> = (0..10)
            .map(|i| store.insert(EuclideanPoint::new(vec![i as f64 * 2.0, 0.0])))
            .collect();

        let set_b: Vec<_> = (0..10)
            .map(|i| store.insert(EuclideanPoint::new(vec![i as f64 * 2.0 + 1.0, 0.0])))
            .collect();

        let emd = store.emd_between_sets(&set_a, &set_b);

        println!("EMD between slightly shifted distributions: {:.2}", emd);
        assert!(emd > 0.0);

        // The EMD should reflect the shift
        // Approximate lower bound: average distance between corresponding points
        let avg_exact_dist = set_a
            .iter()
            .zip(set_b.iter())
            .filter_map(|(&id_a, &id_b)| {
                let p_a = store.get_point(id_a)?;
                let p_b = store.get_point(id_b)?;
                Some(p_a.distance(p_b))
            })
            .sum::<f64>()
            / set_a.len() as f64;

        println!("Average exact distance: {:.2}", avg_exact_dist);
        println!("EMD / avg_dist ratio: {:.2}", emd / avg_exact_dist);

        // EMD should be within O(log n) factor of true EMD
        assert!(
            emd > avg_exact_dist * 0.1,
            "EMD seems too small relative to exact distances"
        );
    }
}
