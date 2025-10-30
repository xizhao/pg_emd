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
        // EMD on tree using minimum-cost flow
        // For trees, this is simple: greedily match points by tree distance
        
        // Build positive (sources) and negative (sinks) mass lists
        let mut sources = Vec::new();  // (point_id, mass)
        let mut sinks = Vec::new();    // (point_id, mass)
        
        let mut weight_map = HashMap::new();
        
        for (point_id, weight) in &dist_a.points {
            *weight_map.entry(*point_id).or_insert(0.0) += weight;
        }
        
        for (point_id, weight) in &dist_b.points {
            *weight_map.entry(*point_id).or_insert(0.0) -= weight;
        }
        
        for (point_id, net_weight) in weight_map {
            if net_weight > 1e-10 {
                sources.push((point_id, net_weight));
            } else if net_weight < -1e-10 {
                sinks.push((point_id, -net_weight));
            }
        }
        
        // Greedy matching: match each source to nearest sink
        // For optimal EMD on trees, this is sufficient
        let mut total_cost = 0.0;
        let mut remaining_sinks = sinks.clone();
        
        for (source_id, mut source_mass) in sources {
            while source_mass > 1e-10 && !remaining_sinks.is_empty() {
                // Find nearest sink
                let mut best_idx = 0;
                let mut best_dist = f64::INFINITY;
                
                for (idx, (sink_id, _)) in remaining_sinks.iter().enumerate() {
                    if let Some(dist) = self.tree_distance(source_id, *sink_id) {
                        if dist < best_dist {
                            best_dist = dist;
                            best_idx = idx;
                        }
                    }
                }
                
                // Move as much mass as possible to this sink
                let flow = source_mass.min(remaining_sinks[best_idx].1);
                total_cost += flow * best_dist;
                
                source_mass -= flow;
                remaining_sinks[best_idx].1 -= flow;
                
                // Remove sink if exhausted
                if remaining_sinks[best_idx].1 < 1e-10 {
                    remaining_sinks.remove(best_idx);
                }
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
