//! PostgreSQL extension for Earth Mover's Distance.
//!
//! Provides SQL functions for computing approximate EMD between distributions.

use pgrx::prelude::*;
use crate::dynamic::DynamicTreeEmbedding;
use crate::emd::Distribution;
use crate::metric::EuclideanPoint;

pgrx::pg_module_magic!();

/// Compute approximate Earth Mover's Distance between two float arrays.
///
/// Returns O(log n)-approximate EMD, much faster than exact O(n³) computation.
///
/// Example:
/// ```sql
/// SELECT emd(
///     ARRAY[0.1, 0.2, 0.3, 0.4],
///     ARRAY[0.4, 0.3, 0.2, 0.1]
/// );
/// ```
#[pg_extern]
fn emd(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() {
        error!("Arrays must have same length");
    }

    if a.is_empty() {
        return 0.0;
    }

    let dimension = a.len();
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, dimension);

    // Insert point for distribution A
    let id_a = store.insert(EuclideanPoint::new(a));
    
    // Insert point for distribution B  
    let id_b = store.insert(EuclideanPoint::new(b));

    // Compute EMD between uniform distributions over single points
    let dist_a = Distribution::new(vec![(id_a, 1.0)]);
    let dist_b = Distribution::new(vec![(id_b, 1.0)]);

    store.emd_distance(&dist_a, &dist_b)
}

/// Compute EMD between two weighted distributions.
///
/// Each distribution is an array of (point, weight) pairs encoded as JSON.
///
/// Example:
/// ```sql
/// SELECT emd_weighted(
///     '[{"point": [1.0, 2.0], "weight": 0.5}, {"point": [3.0, 4.0], "weight": 0.5}]',
///     '[{"point": [10.0, 20.0], "weight": 1.0}]'
/// );
/// ```
#[pg_extern]
fn emd_weighted(dist_a_json: &str, dist_b_json: &str) -> f64 {
    use serde_json::Value;

    let dist_a_data: Vec<Value> = serde_json::from_str(dist_a_json)
        .unwrap_or_else(|_| error!("Invalid JSON for distribution A"));
    
    let dist_b_data: Vec<Value> = serde_json::from_str(dist_b_json)
        .unwrap_or_else(|_| error!("Invalid JSON for distribution B"));

    if dist_a_data.is_empty() || dist_b_data.is_empty() {
        return 0.0;
    }

    // Extract dimension from first point
    let dimension = dist_a_data[0]["point"]
        .as_array()
        .map(|arr| arr.len())
        .unwrap_or(0);

    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, dimension);

    // Insert all points from both distributions
    let mut points_a = Vec::new();
    for item in dist_a_data {
        let point_array = item["point"].as_array()
            .unwrap_or_else(|| error!("Missing 'point' field"));
        let weight = item["weight"].as_f64()
            .unwrap_or_else(|| error!("Missing 'weight' field"));
        
        let coords: Vec<f64> = point_array
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();
        
        let id = store.insert(EuclideanPoint::new(coords));
        points_a.push((id, weight));
    }

    let mut points_b = Vec::new();
    for item in dist_b_data {
        let point_array = item["point"].as_array()
            .unwrap_or_else(|| error!("Missing 'point' field"));
        let weight = item["weight"].as_f64()
            .unwrap_or_else(|| error!("Missing 'weight' field"));
        
        let coords: Vec<f64> = point_array
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();
        
        let id = store.insert(EuclideanPoint::new(coords));
        points_b.push((id, weight));
    }

    let dist_a = Distribution::new(points_a);
    let dist_b = Distribution::new(points_b);

    store.emd_distance(&dist_a, &dist_b)
}

/// Compute tree distance between two points (useful for debugging).
#[pg_extern]
fn tree_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if a.len() != b.len() {
        error!("Arrays must have same length");
    }

    let dimension = a.len();
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, dimension);

    let id_a = store.insert(EuclideanPoint::new(a));
    let id_b = store.insert(EuclideanPoint::new(b));

    store.tree_distance(id_a, id_b).unwrap_or(0.0)
}

#[cfg(test)]
pub mod pg_test {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_emd_basic() {
        let result = Spi::get_one::<f64>("SELECT emd(ARRAY[1.0, 2.0], ARRAY[3.0, 4.0])")
            .expect("SPI failed");
        assert!(result.unwrap() > 0.0);
    }

    #[pg_test]
    fn test_emd_identical() {
        let result = Spi::get_one::<f64>("SELECT emd(ARRAY[1.0, 2.0, 3.0], ARRAY[1.0, 2.0, 3.0])")
            .expect("SPI failed");
        assert_eq!(result.unwrap(), 0.0);
    }

    #[pg_test]
    fn test_tree_distance() {
        let result = Spi::get_one::<f64>("SELECT tree_distance(ARRAY[0.0, 0.0], ARRAY[10.0, 10.0])")
            .expect("SPI failed");
        assert!(result.unwrap() > 0.0);
    }
}
