//! Metric decomposition via grid-based hashing.
//!
//! This module implements the bounded-diameter metric decomposition required by Algorithm 1.
//! The paper requires hash functions φᵢ where each bucket has diameter ≤ τᵢ.
//!
//! ## Key Insight: Diameter vs Cell Size
//!
//! For a d-dimensional grid with cell side length s:
//! - Euclidean diameter = s · √d (diagonal of hypercube)
//! - To satisfy diameter bound τ, we need: s · √d ≤ τ
//! - Therefore: s = τ / √d
//!
//! This is critical! Setting s = τ would violate the diameter bound.

use crate::metric::{Point, EuclideanPoint};

pub type BucketId = usize;

pub trait MetricHash<P: Point> {
    fn hash(&self, point: &P) -> BucketId;
    fn diameter(&self) -> f64;
    fn bucket_ids_in_ball(&self, center: &P, radius: f64) -> Vec<BucketId>;
}

pub struct GridHash {
    cell_size: f64,
    dimension: usize,
    offset: Vec<f64>,
}

impl GridHash {
    pub fn new(cell_size: f64, dimension: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let offset = (0..dimension).map(|_| rng.gen::<f64>() * cell_size).collect();
        Self {
            cell_size,
            dimension,
            offset,
        }
    }

    pub fn with_offset(cell_size: f64, dimension: usize, offset: Vec<f64>) -> Self {
        assert_eq!(offset.len(), dimension);
        Self {
            cell_size,
            dimension,
            offset,
        }
    }

    fn grid_coords(&self, point: &EuclideanPoint) -> Vec<i64> {
        point
            .coords()
            .iter()
            .zip(self.offset.iter())
            .map(|(c, o)| ((c + o) / self.cell_size).floor() as i64)
            .collect()
    }

    fn coords_to_id(&self, coords: &[i64]) -> BucketId {
        let mut hash: usize = 0;
        for &coord in coords {
            hash = hash.wrapping_mul(1000000007);
            hash = hash.wrapping_add(coord as usize);
        }
        hash
    }

    pub fn enumerate_buckets_in_ball(&self, center: &EuclideanPoint, radius: f64) -> Vec<BucketId> {
        let center_coords = self.grid_coords(center);
        let cells_radius = ((radius / self.cell_size).ceil() as i64) + 1;
        
        let mut result = Vec::new();
        self.enumerate_with_filter(&center_coords, cells_radius, 0, &mut vec![0; self.dimension], center, radius, &mut result);
        result
    }

    fn enumerate_with_filter(
        &self,
        center_coords: &[i64],
        radius: i64,
        dim: usize,
        current: &mut Vec<i64>,
        query_center: &EuclideanPoint,
        query_radius: f64,
        result: &mut Vec<BucketId>,
    ) {
        if dim == self.dimension {
            // Check if this bucket actually intersects the L2 ball
            if self.bucket_coords_intersect_ball(current, query_center, query_radius) {
                result.push(self.coords_to_id(current));
            }
            return;
        }

        let center_coord = center_coords[dim];
        for offset in -radius..=radius {
            current[dim] = center_coord + offset;
            self.enumerate_with_filter(center_coords, radius, dim + 1, current, query_center, query_radius, result);
        }
    }

    fn bucket_coords_intersect_ball(&self, coords: &[i64], center: &EuclideanPoint, radius: f64) -> bool {
        let mut dist_sq = 0.0;
        
        for (i, &k) in coords.iter().enumerate() {
            let box_min = k as f64 * self.cell_size - self.offset[i];
            let box_max = (k + 1) as f64 * self.cell_size - self.offset[i];
            let center_coord = center.coords()[i];

            let closest = if center_coord < box_min {
                box_min
            } else if center_coord > box_max {
                box_max
            } else {
                center_coord
            };

            dist_sq += (center_coord - closest).powi(2);
        }
        
        dist_sq.sqrt() <= radius
    }

    /// Tests if the bucket containing `point` intersects the ball B(center, radius).
    ///
    /// This is the key operation for computing B̃(p, r) efficiently:
    /// - Paper definition: B̃ᵢᴾ(p, r) = buksPᵢ(B(p, r))
    /// - Means: union of all points in buckets that intersect B(p, r)
    ///
    /// ## Algorithm
    /// 1. Find point's grid cell (bucket)
    /// 2. Compute axis-aligned bounding box of that cell
    /// 3. Find closest point on box to center
    /// 4. Check if distance ≤ radius
    ///
    /// ## Complexity
    /// - Time: O(d) where d is dimension
    /// - Much better than enumerating buckets: O((2r/cell_size)^d) which is exponential!
    ///
    /// ## Why This Works
    /// This implements exact box-sphere intersection in Euclidean space.
    /// A bucket (grid cell) is an axis-aligned hypercube, and we compute
    /// the minimum distance from center to any point in that hypercube.
    pub fn bucket_intersects_ball(&self, point: &EuclideanPoint, center: &EuclideanPoint, radius: f64) -> bool {
        let coords = self.grid_coords(point);
        let mut dist_sq = 0.0;

        // For each dimension, find the closest coordinate in the cell's interval
        for (i, &k) in coords.iter().enumerate() {
            let box_min = k as f64 * self.cell_size - self.offset[i];
            let box_max = (k + 1) as f64 * self.cell_size - self.offset[i];
            let center_coord = center.coords()[i];

            // Clamp center to box interval
            let closest = if center_coord < box_min {
                box_min
            } else if center_coord > box_max {
                box_max
            } else {
                center_coord
            };

            dist_sq += (center_coord - closest).powi(2);
        }

        dist_sq.sqrt() <= radius
    }
}

impl MetricHash<EuclideanPoint> for GridHash {
    fn hash(&self, point: &EuclideanPoint) -> BucketId {
        let coords = self.grid_coords(point);
        self.coords_to_id(&coords)
    }

    fn diameter(&self) -> f64 {
        self.cell_size * (self.dimension as f64).sqrt()
    }

    fn bucket_ids_in_ball(&self, center: &EuclideanPoint, radius: f64) -> Vec<BucketId> {
        let center_coords = self.grid_coords(center);
        let cells_radius = ((radius / self.cell_size).ceil() as i64) + 1;

        let mut bucket_ids = Vec::new();
        let mut stack = vec![center_coords.clone()];
        let mut visited = std::collections::HashSet::new();

        while let Some(coords) = stack.pop() {
            let id = self.coords_to_id(&coords);
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);

            let mut any_coord_in_range = true;
            for (i, &coord) in coords.iter().enumerate() {
                if (coord - center_coords[i]).abs() > cells_radius {
                    any_coord_in_range = false;
                    break;
                }
            }

            if !any_coord_in_range {
                continue;
            }

            bucket_ids.push(id);

            for dim in 0..self.dimension {
                for delta in [-1, 1] {
                    let mut new_coords = coords.clone();
                    new_coords[dim] += delta;
                    if (new_coords[dim] - center_coords[dim]).abs() <= cells_radius {
                        stack.push(new_coords);
                    }
                }
            }
        }

        bucket_ids
    }
}

pub struct SparsePartition {
    grid_hashes: Vec<GridHash>,
    #[allow(dead_code)]
    gamma: f64,
}

impl SparsePartition {
    pub fn new(cell_size: f64, dimension: usize, gamma: f64) -> Self {
        let num_grids = (gamma.log2().ceil() as usize).max(1);
        let grid_hashes = (0..num_grids)
            .map(|_| GridHash::new(cell_size, dimension))
            .collect();
        
        Self { grid_hashes, gamma }
    }

    pub fn hash(&self, point: &EuclideanPoint) -> Vec<BucketId> {
        self.grid_hashes.iter().map(|h| h.hash(point)).collect()
    }

    pub fn bucket_ids_in_ball(&self, center: &EuclideanPoint, radius: f64) -> Vec<Vec<BucketId>> {
        self.grid_hashes
            .iter()
            .map(|h| h.bucket_ids_in_ball(center, radius))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_hash_same_cell() {
        let hash = GridHash::with_offset(10.0, 2, vec![0.0, 0.0]);
        let p1 = EuclideanPoint::new(vec![1.0, 1.0]);
        let p2 = EuclideanPoint::new(vec![2.0, 2.0]);
        assert_eq!(hash.hash(&p1), hash.hash(&p2));
    }

    #[test]
    fn test_grid_hash_different_cells() {
        let hash = GridHash::with_offset(10.0, 2, vec![0.0, 0.0]);
        let p1 = EuclideanPoint::new(vec![1.0, 1.0]);
        let p2 = EuclideanPoint::new(vec![11.0, 1.0]);
        assert_ne!(hash.hash(&p1), hash.hash(&p2));
    }

    #[test]
    fn test_diameter() {
        let hash = GridHash::new(10.0, 2);
        assert!((hash.diameter() - 10.0 * 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_bucket_ids_in_ball() {
        let hash = GridHash::with_offset(10.0, 2, vec![0.0, 0.0]);
        let center = EuclideanPoint::new(vec![5.0, 5.0]);
        let bucket_ids = hash.bucket_ids_in_ball(&center, 5.0);
        assert!(!bucket_ids.is_empty());
    }
}
