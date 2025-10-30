//! PostgreSQL extension for Earth Mover's Distance.
//!
//! Provides SQL functions for computing approximate EMD between distributions.

use pgrx::prelude::*;
use crate::dynamic::DynamicTreeEmbedding;
use crate::emd::Distribution;
use crate::emd_1d::emd_1d_exact;
use crate::metric::EuclideanPoint;

pgrx::pg_module_magic!();

/// Compute EXACT Earth Mover's Distance for 1D histograms.
///
/// Treats input arrays as histograms where array[i] is the weight/mass at bin i.
/// Bins are positioned at integer coordinates 0, 1, 2, ..., n-1.
///
/// Uses the cumulative distribution function formula for O(n) EXACT computation.
/// This is much faster AND more accurate than tree embedding for 1D histograms.
///
/// Example:
/// ```sql
/// SELECT emd(
///     ARRAY[0.1, 0.2, 0.3, 0.4],  -- histogram with 4 bins
///     ARRAY[0.4, 0.3, 0.2, 0.1]   -- reversed histogram
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

    // For 1D histograms, use exact O(n) algorithm
    // EMD = Σ |CDF_A(i) - CDF_B(i)|
    emd_1d_exact(&a, &b)
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
    
    // Scale aspect_ratio to the actual data range
    let mut max_coord: f64 = 0.0;
    for &x in a.iter().chain(b.iter()) {
        max_coord = max_coord.max(x.abs());
    }
    let aspect_ratio = max_coord.max(10.0) * 2.0;
    
    let mut store = DynamicTreeEmbedding::new(1.5, aspect_ratio, dimension);

    let id_a = store.insert(EuclideanPoint::new(a));
    let id_b = store.insert(EuclideanPoint::new(b));

    store.tree_distance(id_a, id_b).unwrap_or(0.0)
}

// PostgreSQL integration tests removed - use SQL test files instead
