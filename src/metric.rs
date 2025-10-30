//! Metric space abstractions for tree embedding.
//!
//! This module defines the point types and metric operations needed for the tree embedding algorithm.
//! The algorithm works for general metric spaces, but this implementation focuses on Euclidean spaces.

use std::fmt::Debug;

/// Trait for points in a metric space.
///
/// The tree embedding algorithm requires only:
/// - Distance function (satisfying metric properties)
/// - Dimension (for grid-based hashing)
pub trait Point: Clone + Debug + PartialEq {
    fn distance(&self, other: &Self) -> f64;
    fn dimension(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq)]
pub struct EuclideanPoint {
    coords: Vec<f64>,
}

impl EuclideanPoint {
    pub fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    pub fn random(dimension: usize, range: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let coords = (0..dimension).map(|_| rng.gen::<f64>() * range).collect();
        Self { coords }
    }
}

impl Point for EuclideanPoint {
    fn distance(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn dimension(&self) -> usize {
        self.coords.len()
    }
}

pub struct MetricBall<'a, P: Point> {
    center: &'a P,
    radius: f64,
}

impl<'a, P: Point> MetricBall<'a, P> {
    pub fn new(center: &'a P, radius: f64) -> Self {
        Self { center, radius }
    }

    pub fn contains(&self, point: &P) -> bool {
        self.center.distance(point) <= self.radius
    }

    pub fn intersects<I>(&self, points: I) -> Vec<usize>
    where
        I: IntoIterator<Item = (usize, &'a P)>,
    {
        points
            .into_iter()
            .filter(|(_, p)| self.contains(p))
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance() {
        let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
        let p2 = EuclideanPoint::new(vec![3.0, 4.0]);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metric_ball() {
        let center = EuclideanPoint::new(vec![0.0, 0.0]);
        let ball = MetricBall::new(&center, 5.0);
        
        let p1 = EuclideanPoint::new(vec![3.0, 4.0]);
        let p2 = EuclideanPoint::new(vec![5.0, 5.0]);
        
        assert!(ball.contains(&p1));
        assert!(!ball.contains(&p2));
    }
}
