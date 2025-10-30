//! Static tree embedding implementation (Algorithm 1 from paper).
//!
//! This module implements the core tree embedding algorithm that creates
//! a probabilistic embedding into a tree metric with O(Γ log Γ · log n) distortion.
//!
//! ## Algorithm 1 from Paper
//!
//! Input: Point set P ⊆ V, parameter Γ ≥ 1, metric hashes {φᵢ}
//! Output: Labels {ℓₚ⁽ⁱ⁾} for each p ∈ P, i ∈ [m]
//!
//! 1. Sample β ∈ [1/4, 1/2] uniformly at random
//! 2. Let π be a uniform random map from P to [0,1]
//! 3. For i = 1 to m:
//!    4. rᵢ ← β/Γ · wᵢ
//!    5. For each p ∈ P: ℓₚ⁽ⁱ⁾ ← πₘᵢₙ(B̃ᵢᴾ(p, rᵢ))
//! 6. Return labels
//!
//! ## Critical Implementation Notes
//!
//! ### Random Map π (Line 2)
//! - Must be truly random, NOT sorted!
//! - Common bug: sorting random values destroys randomness
//! - Makes labels equal to min point ID instead of random minimum
//!
//! ### B̃ Computation (Line 5)
//! - B̃ᵢᴾ(p, r) = buksPᵢ(B(p, r))
//! - This is the union of ALL points whose bucket intersects B(p, r)
//! - NOT just points inside the ball!
//! - Points outside ball can be included if their bucket intersects
//! - Essential for distortion analysis (representative sets)

use crate::hashing::GridHash;
use crate::metric::{EuclideanPoint, Point};
use rand::Rng;

pub type PointId = usize;
pub type Label = f64;
pub type LevelLabels = Vec<Label>;

pub struct TreeEmbedding {
    gamma: f64,
    num_levels: usize,
    aspect_ratio: f64,
}

impl TreeEmbedding {
    pub fn new(gamma: f64, aspect_ratio: f64) -> Self {
        let num_levels = (aspect_ratio.log2().ceil() as usize) + 1;
        Self {
            gamma,
            num_levels,
            aspect_ratio,
        }
    }

    pub fn compute_embedding(&self, points: &[EuclideanPoint]) -> EmbeddingResult {
        if points.is_empty() {
            return EmbeddingResult::new(Vec::new(), 0);
        }

        let dimension = points[0].dimension();
        let n = points.len();
        let m = self.num_levels;

        let beta: f64 = rand::thread_rng().gen_range(0.25..0.5);
        let pi = self.generate_random_map(n);

        let mut labels = vec![Vec::new(); n];
        let hashes: Vec<GridHash> = (0..m)
            .map(|i| {
                let w_i = self.aspect_ratio * 2.0_f64.powi(-(i as i32));
                let tau_i = w_i / 2.0;
                let cell_size = tau_i / (dimension as f64).sqrt();
                GridHash::new(cell_size, dimension)
            })
            .collect();

        for i in 0..m {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(i as i32));
            let r_i = beta / self.gamma * w_i;

            for (p_idx, p) in points.iter().enumerate() {
                let b_tilde_p = self.compute_b_tilde(points, p, r_i, &hashes[i]);
                
                let label = self.pi_min(&pi, &b_tilde_p);
                labels[p_idx].push(label);
            }
        }

        EmbeddingResult::new(labels, m)
    }

    /// Generate random map π: P → [0,1].
    ///
    /// CRITICAL: Do NOT sort these values!
    /// - Sorting would make π(p₀) < π(p₁) < ... < π(pₙ) always
    /// - Labels would become min point ID, not random
    /// - This was a major bug in initial implementation
    fn generate_random_map(&self, n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..n).map(|_| rng.gen()).collect()
    }

    /// Compute B̃ᵢᴾ(center, radius) = union of points in buckets intersecting B(center, radius).
    ///
    /// This is NOT the same as points inside the ball!
    ///
    /// From the paper (Section 3):
    /// - B̃ᵢᴾ(p, r) := buksPᵢ(B(p, r))
    /// - Where buksPᵢ(S) := ⋃_{x ∈ S} bukᵢ(x) ∩ P
    ///
    /// Implementation:
    /// - For each point q ∈ P, test if bucket(q) intersects B(center, radius)
    /// - If yes, include q in B̃
    /// - O(n·d) time by scanning all points
    fn compute_b_tilde(
        &self,
        points: &[EuclideanPoint],
        center: &EuclideanPoint,
        radius: f64,
        hash: &GridHash,
    ) -> Vec<PointId> {
        points
            .iter()
            .enumerate()
            .filter(|(_, q)| hash.bucket_intersects_ball(q, center, radius))
            .map(|(idx, _)| idx)
            .collect()
    }

    fn pi_min(&self, pi: &[f64], point_ids: &[PointId]) -> Label {
        point_ids
            .iter()
            .map(|&id| pi[id])
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    pub fn tree_distance(&self, labels1: &[Label], labels2: &[Label]) -> f64 {
        let lv = self.lowest_common_ancestor_level(labels1, labels2);
        
        // From paper equation (3): dist_T(p,q) = sum_{i=lv}^m of 2*w_i
        let mut dist = 0.0;
        for i in lv..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(i as i32));
            dist += 2.0 * w_i;
        }
        dist
    }

    fn lowest_common_ancestor_level(&self, labels1: &[Label], labels2: &[Label]) -> usize {
        for i in 0..labels1.len().min(labels2.len()) {
            if (labels1[i] - labels2[i]).abs() > 1e-10 {
                return i;
            }
        }
        labels1.len().min(labels2.len())
    }
}

pub struct EmbeddingResult {
    labels: Vec<LevelLabels>,
    num_levels: usize,
}

impl EmbeddingResult {
    pub fn new(labels: Vec<LevelLabels>, num_levels: usize) -> Self {
        Self { labels, num_levels }
    }

    pub fn get_labels(&self, point_id: PointId) -> &[Label] {
        &self.labels[point_id]
    }

    pub fn num_points(&self) -> usize {
        self.labels.len()
    }

    pub fn num_levels(&self) -> usize {
        self.num_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_embedding_basic() {
        let points = vec![
            EuclideanPoint::new(vec![0.0, 0.0]),
            EuclideanPoint::new(vec![1.0, 0.0]),
            EuclideanPoint::new(vec![10.0, 10.0]),
        ];

        let embedding = TreeEmbedding::new(2.0, 20.0);
        let result = embedding.compute_embedding(&points);

        assert_eq!(result.num_points(), 3);
        assert!(result.num_levels() > 0);
    }

    #[test]
    fn test_tree_distance() {
        let points = vec![
            EuclideanPoint::new(vec![0.0, 0.0]),
            EuclideanPoint::new(vec![1.0, 0.0]),
        ];

        let embedding = TreeEmbedding::new(2.0, 10.0);
        let result = embedding.compute_embedding(&points);

        let dist = embedding.tree_distance(
            result.get_labels(0),
            result.get_labels(1),
        );

        assert!(dist > 0.0);
    }

    #[test]
    fn test_dominating_property() {
        let points = vec![
            EuclideanPoint::new(vec![0.0, 0.0]),
            EuclideanPoint::new(vec![3.0, 4.0]),
        ];

        let embedding = TreeEmbedding::new(2.0, 20.0);
        let result = embedding.compute_embedding(&points);

        let euclidean_dist = points[0].distance(&points[1]);
        let tree_dist = embedding.tree_distance(
            result.get_labels(0),
            result.get_labels(1),
        );

        assert!(tree_dist >= euclidean_dist);
    }
}
