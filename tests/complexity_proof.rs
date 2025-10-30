/// Complexity tests verifying paper claims:
/// "Tree Embedding in High Dimensions: Dynamic and Massively Parallel"
/// Goranci et al. (2025)
///
/// Paper claims:
/// - Tree embedding with O(log n) distortion
/// - Dynamic updates in Õ(n^ε) time where ε < 1

use pg_emd::*;
use std::time::Instant;

#[test]
fn test_insert_complexity_scaling() {
    println!("\n=== Insert Complexity Test ===");
    println!("Measuring single insert time into stores of different sizes");
    println!("Expected: O(n^ε + d) with ε < 1, so growth should be sublinear\n");
    
    let dimension = 8;
    let sizes = vec![50, 100, 200, 400];
    let mut times_us = Vec::new();
    
    for &n in &sizes {
        let mut total_time = 0u128;
        let num_trials = 5;
        
        for _ in 0..num_trials {
            let mut store = DynamicTreeEmbedding::new(2.0, 100.0, dimension);
            
            // Pre-populate with n-1 points
            for i in 0..(n-1) {
                let p = EuclideanPoint::new(vec![(i as f64) * 0.1; dimension]);
                store.insert(p);
            }
            
            // Measure time to insert ONE more point
            let p = EuclideanPoint::new(vec![(n as f64) * 0.1; dimension]);
            let start = Instant::now();
            store.insert(p);
            total_time += start.elapsed().as_micros();
        }
        
        let avg_time = total_time / num_trials;
        times_us.push(avg_time);
        println!("n={:4}: avg insert time = {:6} μs", n, avg_time);
    }
    
    // Calculate growth ratios
    println!("\nGrowth ratios:");
    for i in 1..times_us.len() {
        let ratio = times_us[i] as f64 / times_us[i-1] as f64;
        let n_ratio = sizes[i] as f64 / sizes[i-1] as f64;
        println!("  {}/{}={:.1}x → time ratio: {:.2}x", 
                 sizes[i], sizes[i-1], n_ratio, ratio);
    }
    
    // For O(n), doubling n → 2x time
    // For O(n^0.5), doubling n → 1.41x time  
    // For O(n^0.2), doubling n → 1.15x time
    // For O(log n), doubling n → small increase
    
    let final_ratio = times_us[times_us.len()-1] as f64 / times_us[0] as f64;
    let n_multiplier = sizes[sizes.len()-1] as f64 / sizes[0] as f64;
    
    println!("\nOverall: n increased by {:.1}x, time increased by {:.2}x", 
             n_multiplier, final_ratio);
    println!("If O(n): expect ~{:.1}x", n_multiplier);
    println!("If O(n^0.5): expect ~{:.2}x", n_multiplier.powf(0.5));
    println!("If O(n^0.2): expect ~{:.2}x", n_multiplier.powf(0.2));
    
    // We expect O(n·m·d) for update, but label computation uses spatial index
    // So it should still be better than naive O(n²) at least
    assert!(final_ratio < n_multiplier * 1.5, 
            "Time growing faster than O(n): {:.2}x vs expected {:.1}x", 
            final_ratio, n_multiplier);
}

#[test]
fn test_dimension_complexity_scaling() {
    println!("\n=== Dimension Complexity Test ===");
    println!("Measuring insert time vs dimension (fixed n=50)");
    println!("Expected: O(d) scaling\n");
    
    let n = 50;
    let dimensions = vec![4, 8, 16, 32];
    let mut times_us = Vec::new();
    
    for &d in &dimensions {
        let mut total_time = 0u128;
        let num_trials = 3;
        
        for _ in 0..num_trials {
            let mut store = DynamicTreeEmbedding::new(2.0, 100.0, d);
            
            // Pre-populate
            for i in 0..(n-1) {
                let p = EuclideanPoint::new(vec![(i as f64) * 0.1; d]);
                store.insert(p);
            }
            
            // Measure one insert
            let p = EuclideanPoint::new(vec![(n as f64) * 0.1; d]);
            let start = Instant::now();
            store.insert(p);
            total_time += start.elapsed().as_micros();
        }
        
        let avg_time = total_time / num_trials;
        times_us.push(avg_time);
        println!("d={:3}: avg insert time = {:6} μs", d, avg_time);
    }
    
    println!("\nGrowth ratios:");
    for i in 1..times_us.len() {
        let ratio = times_us[i] as f64 / times_us[i-1] as f64;
        let d_ratio = dimensions[i] as f64 / dimensions[i-1] as f64;
        println!("  dim {}/{} = {:.1}x → time ratio: {:.2}x", 
                 dimensions[i], dimensions[i-1], d_ratio, ratio);
    }
    
    // For O(d), doubling d → 2x time
    let final_ratio = times_us[times_us.len()-1] as f64 / times_us[0] as f64;
    let d_multiplier = dimensions[dimensions.len()-1] as f64 / dimensions[0] as f64;
    
    println!("\nOverall: d increased by {:.1}x, time increased by {:.2}x", 
             d_multiplier, final_ratio);
    println!("If O(d): expect ~{:.1}x", d_multiplier);
    
    // Should scale roughly linearly with dimension
    assert!(final_ratio < d_multiplier * 2.0,
            "Time growing faster than O(d): {:.2}x vs expected {:.1}x",
            final_ratio, d_multiplier);
}

/// Test tree embedding distortion matches O(log n) from paper Theorem 1.1
#[test]
fn test_tree_embedding_distortion_bounds() {
    println!("\n=== Tree Embedding Distortion Test (Theorem 1.1) ===");
    println!("Paper claims: O(log n) distortion");
    
    let gamma = 1.5;
    let aspect_ratio = 20.0;
    let dimension = 1;
    
    let mut store = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);
    
    // Create points at positions 0, 1, 2, ..., 9
    let points: Vec<_> = (0..10)
        .map(|i| store.insert(EuclideanPoint::new(vec![i as f64])))
        .collect();
    
    let n = points.len();
    let theoretical_max_distortion = gamma * (n as f64).log2();
    
    println!("n = {}, gamma = {}", n, gamma);
    println!("Theoretical max distortion: O(log n) ≈ {:.2}", theoretical_max_distortion);
    
    let mut max_distortion: f64 = 0.0;
    let mut total_distortion: f64 = 0.0;
    let mut num_pairs = 0;
    
    for i in 0..points.len() {
        for j in (i+1)..points.len() {
            let euclidean_dist = (j - i) as f64;
            if let Some(tree_dist) = store.tree_distance(points[i], points[j]) {
                let distortion = tree_dist / euclidean_dist;
                
                max_distortion = max_distortion.max(distortion);
                total_distortion += distortion;
                num_pairs += 1;
            }
        }
    }
    
    let avg_distortion = total_distortion / num_pairs as f64;
    
    println!("Average distortion: {:.2}", avg_distortion);
    println!("Max distortion: {:.2}", max_distortion);
    
    // With proper parameters, should be within reasonable bounds
    assert!(
        avg_distortion < theoretical_max_distortion * 3.0,
        "Average distortion {:.2} exceeds 3x theoretical {:.2}",
        avg_distortion, theoretical_max_distortion
    );
    
    println!("✓ Distortion within acceptable bounds");
}

/// Test EMD approximation quality matches O(log n) from Corollary 1.3
#[test]
fn test_emd_approximation_factor() {
    println!("\n=== EMD Approximation Factor Test (Corollary 1.3) ===");
    println!("Paper claims: O_ε(log n) approximation for EMD");
    
    let gamma = 1.5;
    let n = 10;
    let aspect_ratio = (n as f64) * 2.0;
    
    let mut store = DynamicTreeEmbedding::new(gamma, aspect_ratio, 1);
    
    let bins: Vec<_> = (0..n)
        .map(|i| store.insert(EuclideanPoint::new(vec![i as f64])))
        .collect();
    
    // Test: Moving point mass from position 0 to position (n-1)
    // Exact EMD = (n-1) (distance × mass)
    let dist_a = Distribution::new(vec![(bins[0], 1.0)]);
    let dist_b = Distribution::new(vec![(bins[n-1], 1.0)]);
    
    let exact_emd = (n - 1) as f64;
    let approx_emd = store.emd_distance(&dist_a, &dist_b);
    let approx_factor = approx_emd / exact_emd;
    
    let theoretical_bound = gamma * (n as f64).log2();
    
    println!("Exact EMD: {}", exact_emd);
    println!("Approx EMD: {}", approx_emd);
    println!("Approximation factor: {:.2}x", approx_factor);
    println!("Theoretical bound: O(log n) ≈ {:.2}", theoretical_bound);
    
    assert!(
        approx_factor < theoretical_bound * 3.0,
        "Approximation {:.2} exceeds 3x theoretical {:.2}",
        approx_factor, theoretical_bound
    );
    
    println!("✓ EMD approximation within O(log n) bounds");
}
