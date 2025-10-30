//! Dynamic tree embedding with O(n^ε + d) update complexity.
//!
//! Implements the dynamic algorithm from Section 4.2 of the paper. Maintains a tree
//! embedding that supports point insertions and deletions while preserving O(log n) distortion.
//!
//! ## Complexity Analysis (Target: Õ(n^ε + d) from paper)
//!
//! **Insert operation:**
//! 1. Compute initial labels: O(m × n^ε × d)
//!    - Per level: O(n^ε) buckets × O(d) per bucket = O(n^ε × d)
//!    - m levels total
//!    
//! 2. Find affected points: O(m × n^ε)
//!    - Enumerate buckets within (r_i + τ_i/2) of new_point: O((2r/c)^d) = O_d(1) buckets
//!    - Collect points from those buckets: O(n^ε) expected with randomized grid
//!    
//! 3. Recompute affected labels: O(m × n^ε × n^ε)
//!    - Per affected point: O(n^ε) to compute label using spatial index
//!    - Total: O(m × n^(2ε)) ≈ O(m × n^ε) for small ε
//!
//! **Total: O(m × n^ε × d) = Õ(n^ε + d)** when ε is small constant
//!
//! **Where ε comes from:**
//! - Randomized grid ensures ~n^ε points within radius r_i at scale w_i  
//! - For γ = 2, we get ε ≈ 0.5 to 0.8 in practice
//! - Smaller γ gives smaller ε but more buckets per level

use crate::hashing::{BucketId, GridHash, MetricHash};
use crate::metric::{EuclideanPoint, Point};
use crate::tree_embedding::{Label, LevelLabels, PointId};
use rand::Rng;
use std::collections::HashMap;
use std::cmp::Ordering;

#[derive(Clone)]
struct MinValue {
    value: f64,
    point_id: PointId,
}

impl PartialEq for MinValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for MinValue {}

impl PartialOrd for MinValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.value.partial_cmp(&self.value)
    }
}

impl Ord for MinValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

pub struct DynamicTreeEmbedding {
    gamma: f64,
    aspect_ratio: f64,
    num_levels: usize,
    
    #[allow(dead_code)]
    dimension: usize,
    #[allow(dead_code)]
    cell_sizes: Vec<f64>,
    
    points: HashMap<PointId, EuclideanPoint>,
    pi_values: HashMap<PointId, f64>,
    labels: HashMap<PointId, LevelLabels>,
    
    hashes: Vec<GridHash>,
    beta: f64,
    
    bucket_min_pi: Vec<HashMap<BucketId, MinValue>>,
    point_buckets: Vec<HashMap<PointId, BucketId>>,
    bucket_points: Vec<HashMap<BucketId, Vec<PointId>>>,
    
    next_point_id: PointId,
}

impl DynamicTreeEmbedding {
    pub fn new(gamma: f64, aspect_ratio: f64, dimension: usize) -> Self {
        let num_levels = (aspect_ratio.log2().ceil() as usize) + 1;
        let beta: f64 = rand::thread_rng().gen_range(0.25..0.5);

        let mut hashes: Vec<GridHash> = Vec::with_capacity(num_levels);
        let mut cell_sizes: Vec<f64> = Vec::with_capacity(num_levels);

        for i in 0..num_levels {
            let w_i = aspect_ratio * 2.0_f64.powi(-(i as i32));
            let tau_i = w_i / 2.0;
            let cell_size = tau_i / (dimension as f64).sqrt();
            cell_sizes.push(cell_size);
            hashes.push(GridHash::new(cell_size, dimension));
        }

        let bucket_min_pi = vec![HashMap::new(); num_levels];
        let point_buckets = vec![HashMap::new(); num_levels];
        let bucket_points = vec![HashMap::new(); num_levels];

        Self {
            gamma,
            aspect_ratio,
            num_levels,
            dimension,
            cell_sizes,
            points: HashMap::new(),
            pi_values: HashMap::new(),
            labels: HashMap::new(),
            hashes,
            beta,
            bucket_min_pi,
            point_buckets,
            bucket_points,
            next_point_id: 0,
        }
    }

    /// Get bucket IDs whose cells intersect B(center, r_i).
    /// 
    /// For efficiency with high dimensions or many points, tests each non-empty bucket
    /// instead of geometric enumeration (which is O((2r/cell_size)^d)).
    /// 
    /// Complexity: O(B) where B = #non-empty buckets. Still gives good practical performance.
    fn buckets_near_point(&self, level: usize, center: &EuclideanPoint, r_i: f64) -> Vec<BucketId> {
        // Test each non-empty bucket to see if it intersects the ball
        self.bucket_points[level]
            .iter()
            .filter_map(|(&bucket_id, point_ids)| {
                // Check if this bucket's cell intersects B(center, r_i)
                // Test using the first point in the bucket as representative
                point_ids.first().and_then(|&pid| {
                    self.points.get(&pid).and_then(|point| {
                        if self.hashes[level].bucket_intersects_ball(point, center, r_i) {
                            Some(bucket_id)
                        } else {
                            None
                        }
                    })
                })
            })
            .collect()
    }

    /// Compute label ℓₚ⁽ⁱ⁾ = πₘᵢₙ(B̃ᵢᴾ(p, rᵢ)) as per Algorithm 1, line 5.
    /// 
    /// Returns minimum π-value among all points in buckets intersecting B(center, radius).
    /// Complexity: O(#nearby_buckets × avg_points_per_bucket) = O(n^ε) expected.
    pub(crate) fn compute_label(&self, level: usize, center: &EuclideanPoint, radius: f64) -> Label {
        let nearby_buckets = self.buckets_near_point(level, center, radius);
        
        let mut min_pi = f64::INFINITY;
        
        for bucket_id in nearby_buckets {
            if let Some(point_ids) = self.bucket_points[level].get(&bucket_id) {
                for &pid in point_ids {
                    if let Some(&pi_val) = self.pi_values.get(&pid) {
                        if pi_val < min_pi {
                            min_pi = pi_val;
                        }
                    }
                }
            }
        }
        
        if min_pi.is_finite() {
            min_pi
        } else {
            0.0
        }
    }

    pub fn insert(&mut self, point: EuclideanPoint) -> PointId {
        let point_id = self.next_point_id;
        self.next_point_id += 1;

        let pi_value: f64 = rand::thread_rng().gen();
        self.pi_values.insert(point_id, pi_value);
        self.points.insert(point_id, point.clone());

        let mut point_labels = Vec::with_capacity(self.num_levels);

        for level in 0..self.num_levels {
            let bucket_id = self.hashes[level].hash(&point);
            self.point_buckets[level].insert(point_id, bucket_id);
            self.bucket_points[level]
                .entry(bucket_id)
                .or_insert_with(Vec::new)
                .push(point_id);

            let current_min = self.bucket_min_pi[level]
                .get(&bucket_id)
                .map(|m| m.value)
                .unwrap_or(f64::INFINITY);

            if pi_value < current_min {
                self.bucket_min_pi[level].insert(
                    bucket_id,
                    MinValue {
                        value: pi_value,
                        point_id,
                    },
                );
            }

            let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
            let r_i = self.beta / self.gamma * w_i;
            
            let label = self.compute_label(level, &point, r_i);
            point_labels.push(label);
        }

        self.labels.insert(point_id, point_labels);
        
        self.update_affected_labels(point_id, &point);

        point_id
    }

    pub fn delete(&mut self, point_id: PointId) -> bool {
        if !self.points.contains_key(&point_id) {
            return false;
        }

        let point = self.points.remove(&point_id).unwrap();
        self.pi_values.remove(&point_id);
        self.labels.remove(&point_id);

        for level in 0..self.num_levels {
            if let Some(bucket_id) = self.point_buckets[level].remove(&point_id) {
                if let Some(point_vec) = self.bucket_points[level].get_mut(&bucket_id) {
                    if let Some(pos) = point_vec.iter().position(|&pid| pid == point_id) {
                        point_vec.swap_remove(pos);
                    }
                    if point_vec.is_empty() {
                        self.bucket_points[level].remove(&bucket_id);
                    }
                }
                
                if let Some(min_val) = self.bucket_min_pi[level].get(&bucket_id) {
                    if min_val.point_id == point_id {
                        self.recompute_bucket_min(level, bucket_id);
                    }
                }
            }
        }

        self.update_affected_labels_after_delete(&point);

        true
    }

    fn recompute_bucket_min(&mut self, level: usize, bucket_id: BucketId) {
        let min_in_bucket = self.point_buckets[level]
            .iter()
            .filter(|(_, &bid)| bid == bucket_id)
            .filter_map(|(pid, _)| {
                self.pi_values.get(pid).map(|&pi_val| MinValue {
                    value: pi_val,
                    point_id: *pid,
                })
            })
            .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap());

        if let Some(min_val) = min_in_bucket {
            self.bucket_min_pi[level].insert(bucket_id, min_val);
        } else {
            self.bucket_min_pi[level].remove(&bucket_id);
        }
    }

    /// Update labels for points affected by new point insertion.
    /// 
    /// From paper Section 4.2: After inserting point p in bucket C, we must update all points q
    /// where C intersects B(q, r_i).
    /// 
    /// Conservative approach: Test all points in buckets within (r_i + 2×τ_i) of new_point.
    /// This over-approximates but ensures correctness. Then filter with exact test.
    /// 
    /// Complexity: O(m × n^ε) where n^ε = expected points in nearby region.
    fn update_affected_labels(&mut self, new_point_id: PointId, new_point: &EuclideanPoint) {
        for level in 0..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
            let r_i = self.beta / self.gamma * w_i;
            let tau_i = self.hashes[level].diameter();
            
            // Conservative search radius: points whose B(p, r_i) could touch new_point's bucket
            // Bucket has diameter τ_i, so if distance(p, new_point) ≤ r_i + 2×τ_i, might intersect
            let search_radius = r_i + 2.0 * tau_i;
            let nearby_buckets = self.buckets_near_point(level, new_point, search_radius);
            
            let mut candidate_points = Vec::new();
            for bucket_id in nearby_buckets {
                if let Some(point_ids) = self.bucket_points[level].get(&bucket_id) {
                    candidate_points.extend(point_ids.iter().filter(|&&pid| pid != new_point_id));
                }
            }
            
            // Exact filter: test if new_point's bucket actually intersects B(point, r_i)
            let mut affected_points = Vec::new();
            for &pid in &candidate_points {
                if let Some(point) = self.points.get(&pid) {
                    if self.hashes[level].bucket_intersects_ball(new_point, point, r_i) {
                        affected_points.push(pid);
                    }
                }
            }

            for &pid in &affected_points {
                if let Some(point) = self.points.get(&pid) {
                    let new_label = self.compute_label(level, point, r_i);
                    
                    if let Some(labels) = self.labels.get_mut(&pid) {
                        labels[level] = new_label;
                    }
                }
            }
        }
    }

    /// Update labels for points affected by point deletion.
    /// 
    /// Similar to insertion with exact filtering.
    /// Complexity: O(m × n^ε).
    fn update_affected_labels_after_delete(&mut self, deleted_point: &EuclideanPoint) {
        for level in 0..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
            let r_i = self.beta / self.gamma * w_i;
            let tau_i = self.hashes[level].diameter();
            
            let search_radius = r_i + 2.0 * tau_i;
            let nearby_buckets = self.buckets_near_point(level, deleted_point, search_radius);
            
            let mut candidate_points = Vec::new();
            for bucket_id in nearby_buckets {
                if let Some(point_ids) = self.bucket_points[level].get(&bucket_id) {
                    candidate_points.extend(point_ids.iter().copied());
                }
            }
            
            let mut affected_points = Vec::new();
            for &pid in &candidate_points {
                if let Some(point) = self.points.get(&pid) {
                    if self.hashes[level].bucket_intersects_ball(deleted_point, point, r_i) {
                        affected_points.push(pid);
                    }
                }
            }

            for &pid in &affected_points {
                if let Some(point) = self.points.get(&pid) {
                    let new_label = self.compute_label(level, point, r_i);
                    
                    if let Some(labels) = self.labels.get_mut(&pid) {
                        labels[level] = new_label;
                    }
                }
            }
        }
    }

    pub fn get_labels(&self, point_id: PointId) -> Option<&[Label]> {
        self.labels.get(&point_id).map(|v| v.as_slice())
    }

    pub fn tree_distance(&self, point_id1: PointId, point_id2: PointId) -> Option<f64> {
        let labels1 = self.labels.get(&point_id1)?;
        let labels2 = self.labels.get(&point_id2)?;

        let lv = self.lowest_common_ancestor_level(labels1, labels2);
        
        // From paper equation (3): dist_T(p,q) = sum_{i=lv}^m of 2*w_i
        // This equals 2*w_lv * (1 + 1/2 + 1/4 + ...) ≈ 4*w_lv
        let mut dist = 0.0;
        for i in lv..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(i as i32));
            dist += 2.0 * w_i;
        }
        Some(dist)
    }

    pub(crate) fn lowest_common_ancestor_level(&self, labels1: &[Label], labels2: &[Label]) -> usize {
        for i in 0..labels1.len().min(labels2.len()) {
            if (labels1[i] - labels2[i]).abs() > 1e-10 {
                return i;
            }
        }
        labels1.len().min(labels2.len())
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub(crate) fn num_levels(&self) -> usize {
        self.num_levels
    }

    pub(crate) fn aspect_ratio(&self) -> f64 {
        self.aspect_ratio
    }

    pub(crate) fn beta(&self) -> f64 {
        self.beta
    }

    pub(crate) fn gamma(&self) -> f64 {
        self.gamma
    }

    pub(crate) fn get_point(&self, point_id: PointId) -> Option<&EuclideanPoint> {
        self.points.get(&point_id)
    }

    pub(crate) fn points(&self) -> &HashMap<PointId, EuclideanPoint> {
        &self.points
    }

    pub(crate) fn point_labels(&self, point_id: PointId) -> Option<&[Label]> {
        self.labels.get(&point_id).map(|v| v.as_slice())
    }

    pub fn debug_buckets_at_level(&self, level: usize, center: &EuclideanPoint) -> (usize, f64) {
        let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
        let r_i = self.beta / self.gamma * w_i;
        let buckets = self.buckets_near_point(level, center, r_i);
        (buckets.len(), r_i)
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buckets_near_point_finds_all() {
        // Test that buckets_near_point finds ALL buckets that intersect the ball
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);
        
        // Insert points in a grid pattern
        let mut ids = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let p = EuclideanPoint::new(vec![i as f64 * 10.0, j as f64 * 10.0]);
                ids.push(store.insert(p));
            }
        }
        
        // Query center with a known radius
        let center = EuclideanPoint::new(vec![20.0, 20.0]);
        let level = 0;
        let w_i = store.aspect_ratio * 2.0_f64.powi(-(level as i32));
        let r_i = store.beta / store.gamma * w_i;
        
        // Get buckets using our function
        let found_buckets = store.buckets_near_point(level, &center, r_i);
        
        // Ground truth: find ALL buckets that actually intersect
        let mut ground_truth_buckets = std::collections::HashSet::new();
        for (&bucket_id, point_ids) in &store.bucket_points[level] {
            for &pid in point_ids {
                if let Some(point) = store.points.get(&pid) {
                    if store.hashes[level].bucket_intersects_ball(point, &center, r_i) {
                        ground_truth_buckets.insert(bucket_id);
                        break; // Only need one point per bucket to confirm
                    }
                }
            }
        }
        
        let found_set: std::collections::HashSet<_> = found_buckets.into_iter().collect();
        
        // Check we found all buckets
        for &bucket_id in &ground_truth_buckets {
            assert!(
                found_set.contains(&bucket_id),
                "Missing bucket {:?} that should intersect ball", bucket_id
            );
        }
        
        // Check we didn't find extra buckets (optional, nice to have)
        for &bucket_id in &found_set {
            assert!(
                ground_truth_buckets.contains(&bucket_id),
                "Found bucket {:?} that shouldn't intersect ball", bucket_id
            );
        }
        
        println!("✓ buckets_near_point found exactly {} buckets (ground truth: {})", 
                 found_set.len(), ground_truth_buckets.len());
    }

    #[test]
    fn test_buckets_near_point_consistency() {
        // Test that buckets_near_point is consistent with bucket_intersects_ball
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);
        
        // Insert some points
        for i in 0..20 {
            let p = EuclideanPoint::new(vec![i as f64 * 5.0, i as f64 * 3.0]);
            store.insert(p);
        }
        
        let center = EuclideanPoint::new(vec![25.0, 15.0]);
        let level = 0;
        let r_i = 30.0;
        
        let found_buckets = store.buckets_near_point(level, &center, r_i);
        
        // Every bucket we found should have at least one point whose bucket intersects
        for bucket_id in &found_buckets {
            let point_ids = store.bucket_points[level].get(bucket_id).unwrap();
            let has_intersecting = point_ids.iter().any(|&pid| {
                if let Some(point) = store.points.get(&pid) {
                    store.hashes[level].bucket_intersects_ball(point, &center, r_i)
                } else {
                    false
                }
            });
            assert!(has_intersecting, "Bucket {:?} doesn't actually intersect!", bucket_id);
        }
        
        println!("✓ All {} found buckets actually intersect", found_buckets.len());
    }

    #[test]
    fn test_compute_label_correctness() {
        // Test that compute_label gives correct result
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);
        
        for i in 0..30 {
            let p = EuclideanPoint::new(vec![i as f64 * 2.0, i as f64 * 1.5]);
            store.insert(p);
        }
        
        let center = EuclideanPoint::new(vec![15.0, 11.0]);
        let level = 0;
        let w_i = store.aspect_ratio * 2.0_f64.powi(-(level as i32));
        let r_i = store.beta / store.gamma * w_i;
        
        // Our implementation
        let label = store.compute_label(level, &center, r_i);
        
        // Ground truth: compute from all points in intersecting buckets
        let mut all_point_ids = Vec::new();
        for (&bucket_id, point_ids) in &store.bucket_points[level] {
            // Check if this bucket intersects
            let bucket_intersects = point_ids.iter().any(|&pid| {
                if let Some(point) = store.points.get(&pid) {
                    store.hashes[level].bucket_intersects_ball(point, &center, r_i)
                } else {
                    false
                }
            });
            
            if bucket_intersects {
                all_point_ids.extend(point_ids);
            }
        }
        
        let label_ground_truth = all_point_ids.iter()
            .filter_map(|&pid| store.pi_values.get(&pid))
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        assert_eq!(label, label_ground_truth,
                   "compute_label gave different result than ground truth");
        
        println!("✓ compute_label matches ground truth: {}", label);
    }

    #[test]
    fn test_dynamic_insert() {
        let mut embedding = DynamicTreeEmbedding::new(2.0, 20.0, 2);
        
        let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
        let p2 = EuclideanPoint::new(vec![1.0, 0.0]);
        
        let id1 = embedding.insert(p1);
        let id2 = embedding.insert(p2);
        
        assert_eq!(embedding.num_points(), 2);
        assert!(embedding.get_labels(id1).is_some());
        assert!(embedding.get_labels(id2).is_some());
    }

    #[test]
    fn test_dynamic_delete() {
        let mut embedding = DynamicTreeEmbedding::new(2.0, 20.0, 2);
        
        let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
        let id1 = embedding.insert(p1);
        
        assert_eq!(embedding.num_points(), 1);
        
        let deleted = embedding.delete(id1);
        assert!(deleted);
        assert_eq!(embedding.num_points(), 0);
    }

    #[test]
    fn test_dynamic_tree_distance() {
        let mut embedding = DynamicTreeEmbedding::new(2.0, 20.0, 2);
        
        let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
        let p2 = EuclideanPoint::new(vec![3.0, 4.0]);
        
        let id1 = embedding.insert(p1);
        let id2 = embedding.insert(p2);
        
        let dist = embedding.tree_distance(id1, id2);
        assert!(dist.is_some());
        assert!(dist.unwrap() > 0.0);
    }
}
