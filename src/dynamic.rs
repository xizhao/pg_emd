//! Dynamic tree embedding with efficient insertions and deletions.
//!
//! This module extends the static tree embedding (Algorithm 1) to support dynamic updates.
//!
//! ## Dynamic Algorithm (from paper Section 4.2)
//!
//! **Insertion of point p:**
//! 1. Assign random π-value to p
//! 2. Update bucket minimum π-values if necessary
//! 3. For each level i:
//!    - Find points q where bucket(q) intersects B̃(p, rᵢ)
//!    - Recompute labels for those points
//! 4. Complexity: O(n·m·d) worst case
//!    - Paper achieves Õ(n^ε + d) with better indexing
//!
//! **Deletion of point p:**
//! 1. Remove p from data structures
//! 2. Recompute bucket minimums if p was the minimum
//! 3. Update labels for affected points (similar to insertion)
//! 4. Complexity: O(n·m·d) worst case
//!
//! ## Optimization Opportunities
//!
//! Current implementation scans all points to find affected ones.
//! Could optimize with:
//! - Spatial indexing: HashMap<BucketId, Vec<PointId>> per level
//! - Only enumerate non-empty buckets
//! - Analytical range computation for affected buckets
//!
//! This would achieve the paper's Õ(n^ε + d) bound.

use crate::hashing::{BucketId, GridHash, MetricHash};
use crate::metric::EuclideanPoint;
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
    
    points: HashMap<PointId, EuclideanPoint>,
    pi_values: HashMap<PointId, f64>,
    labels: HashMap<PointId, LevelLabels>,
    
    hashes: Vec<GridHash>,
    beta: f64,
    
    bucket_min_pi: Vec<HashMap<BucketId, MinValue>>,
    point_buckets: Vec<HashMap<PointId, BucketId>>,
    
    next_point_id: PointId,
}

impl DynamicTreeEmbedding {
    pub fn new(gamma: f64, aspect_ratio: f64, dimension: usize) -> Self {
        let num_levels = (aspect_ratio.log2().ceil() as usize) + 1;
        let beta: f64 = rand::thread_rng().gen_range(0.25..0.5);

        let hashes: Vec<GridHash> = (0..num_levels)
            .map(|i| {
                let w_i = aspect_ratio * 2.0_f64.powi(-(i as i32));
                let tau_i = w_i / 2.0;
                let cell_size = tau_i / (dimension as f64).sqrt();
                GridHash::new(cell_size, dimension)
            })
            .collect();

        let bucket_min_pi = vec![HashMap::new(); num_levels];
        let point_buckets = vec![HashMap::new(); num_levels];

        Self {
            gamma,
            aspect_ratio,
            num_levels,
            points: HashMap::new(),
            pi_values: HashMap::new(),
            labels: HashMap::new(),
            hashes,
            beta,
            bucket_min_pi,
            point_buckets,
            next_point_id: 0,
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
            
            let label = self.compute_label_for_point(level, &point, r_i);
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

    fn update_affected_labels(&mut self, new_point_id: PointId, new_point: &EuclideanPoint) {
        for level in 0..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
            let r_i = self.beta / self.gamma * w_i;

            let affected_points: Vec<PointId> = self.points
                .iter()
                .filter(|(&pid, p)| {
                    pid != new_point_id && self.hashes[level].bucket_intersects_ball(p, new_point, r_i)
                })
                .map(|(&pid, _)| pid)
                .collect();

            for pid in affected_points {
                if let Some(point) = self.points.get(&pid) {
                    let new_label = self.compute_label_for_point(level, point, r_i);
                    
                    if let Some(labels) = self.labels.get_mut(&pid) {
                        labels[level] = new_label;
                    }
                }
            }
        }
    }

    fn update_affected_labels_after_delete(&mut self, deleted_point: &EuclideanPoint) {
        for level in 0..self.num_levels {
            let w_i = self.aspect_ratio * 2.0_f64.powi(-(level as i32));
            let r_i = self.beta / self.gamma * w_i;

            let affected_points: Vec<PointId> = self.points
                .iter()
                .filter(|(_, p)| self.hashes[level].bucket_intersects_ball(p, deleted_point, r_i))
                .map(|(&pid, _)| pid)
                .collect();

            for pid in affected_points {
                if let Some(point) = self.points.get(&pid) {
                    let new_label = self.compute_label_for_point(level, point, r_i);
                    
                    if let Some(labels) = self.labels.get_mut(&pid) {
                        labels[level] = new_label;
                    }
                }
            }
        }
    }

    fn compute_label_for_point(&self, level: usize, center: &EuclideanPoint, radius: f64) -> Label {
        self.points
            .iter()
            .filter(|(_, q)| self.hashes[level].bucket_intersects_ball(q, center, radius))
            .filter_map(|(&pid, _)| self.pi_values.get(&pid))
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    pub fn get_labels(&self, point_id: PointId) -> Option<&[Label]> {
        self.labels.get(&point_id).map(|v| v.as_slice())
    }

    pub fn tree_distance(&self, point_id1: PointId, point_id2: PointId) -> Option<f64> {
        let labels1 = self.labels.get(&point_id1)?;
        let labels2 = self.labels.get(&point_id2)?;

        let lv = self.lowest_common_ancestor_level(labels1, labels2);
        let w_lv = self.aspect_ratio * 2.0_f64.powi(-(lv as i32));
        Some(2.0 * w_lv)
    }

    fn lowest_common_ancestor_level(&self, labels1: &[Label], labels2: &[Label]) -> usize {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
