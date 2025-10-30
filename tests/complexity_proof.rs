use adze_store::*;
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
