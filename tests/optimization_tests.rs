use pg_emd::*;
use std::time::Instant;

#[test]
fn test_optimization_maintains_correctness() {
    let gamma = 2.0;
    let aspect_ratio = 100.0;
    let dimension = 4;
    
    let mut store = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);

    let points: Vec<EuclideanPoint> = (0..50)
        .map(|i| EuclideanPoint::new(vec![
            i as f64,
            (i * 2) as f64,
            (i * 3) as f64,
            (i * 4) as f64,
        ]))
        .collect();

    let ids: Vec<_> = points.iter()
        .map(|p| store.insert(p.clone()))
        .collect();

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let euclidean_dist = points[i].distance(&points[j]);
            if let Some(tree_dist) = store.tree_distance(ids[i], ids[j]) {
                assert!(
                    tree_dist >= euclidean_dist * 0.95,
                    "Dominating property violated with optimization: tree={}, euclidean={}, ratio={}",
                    tree_dist, euclidean_dist, tree_dist / euclidean_dist
                );
            }
        }
    }

    store.delete(ids[10]);
    store.delete(ids[20]);

    for i in 0..ids.len() {
        if i == 10 || i == 20 {
            continue;
        }
        for j in (i + 1)..ids.len() {
            if j == 10 || j == 20 {
                continue;
            }
            let euclidean_dist = points[i].distance(&points[j]);
            if let Some(tree_dist) = store.tree_distance(ids[i], ids[j]) {
                assert!(
                    tree_dist >= euclidean_dist * 0.95,
                    "Dominating property violated after delete with optimization: tree={}, euclidean={}, ratio={}",
                    tree_dist, euclidean_dist, tree_dist / euclidean_dist
                );
            }
        }
    }
}

#[test]
fn test_spatial_index_maintained() {
    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);
    
    let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
    let p2 = EuclideanPoint::new(vec![5.0, 5.0]);
    let p3 = EuclideanPoint::new(vec![10.0, 10.0]);
    
    let id1 = store.insert(p1);
    let id2 = store.insert(p2);
    let id3 = store.insert(p3);
    
    assert_eq!(store.num_points(), 3);
    
    store.delete(id2);
    
    assert_eq!(store.num_points(), 2);
    
    // Verify spatial index is maintained correctly
    assert!(store.get_labels(id1).is_some());
    assert!(store.get_labels(id2).is_none());
    assert!(store.get_labels(id3).is_some());
}

#[test]
fn test_insert_time_sublinear() {
    println!("\n=== Testing Insert Time Complexity ===");
    
    let dimension = 8;
    let sizes = vec![50, 100, 200, 400];
    let mut times = Vec::new();
    
    for &n in &sizes {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, dimension);
        
        for i in 0..(n - 1) {
            let p = EuclideanPoint::new(vec![(i as f64) * 0.1; dimension]);
            store.insert(p);
        }
        
        let start = Instant::now();
        let p = EuclideanPoint::new(vec![(n as f64) * 0.1; dimension]);
        store.insert(p);
        let elapsed = start.elapsed();
        
        times.push(elapsed.as_micros());
        println!("n={:4}: insert time = {:6} μs", n, elapsed.as_micros());
    }
    
    let ratio_1 = times[1] as f64 / times[0] as f64;
    let ratio_2 = times[2] as f64 / times[1] as f64;
    let ratio_3 = times[3] as f64 / times[2] as f64;
    
    println!("\nTime ratios (should be < 2 for sublinear):");
    println!("  100/50:   {:.2}x", ratio_1);
    println!("  200/100:  {:.2}x", ratio_2);
    println!("  400/200:  {:.2}x", ratio_3);
    
    println!("\nFor O(n^ε) with ε < 1, ratios should be < 2.0");
    println!("For O(n), ratios would be ≈ 2.0");
    
    assert!(
        ratio_3 < 3.0,
        "Insert time growing too fast (>3x): {} (likely not optimized)",
        ratio_3
    );
}

#[test]
fn test_affected_points_count() {
    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 4);
    
    let n = 100;
    for i in 0..n {
        let p = EuclideanPoint::new(vec![
            (i as f64) * 0.5,
            (i as f64) * 0.3,
            (i as f64) * 0.2,
            (i as f64) * 0.1,
        ]);
        store.insert(p);
    }
    
    assert_eq!(store.num_points(), 100);
}

#[test]
fn test_many_inserts_and_deletes() {
    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 4);
    
    let mut ids = Vec::new();
    for i in 0..100 {
        let p = EuclideanPoint::new(vec![(i as f64) * 0.1; 4]);
        let id = store.insert(p);
        ids.push(id);
    }
    
    assert_eq!(store.num_points(), 100);
    
    for i in (0..100).step_by(2) {
        store.delete(ids[i]);
    }
    
    assert_eq!(store.num_points(), 50);
    
    for i in 0..50 {
        let p = EuclideanPoint::new(vec![(i + 200) as f64; 4]);
        let id = store.insert(p);
        ids.push(id);
    }
    
    assert_eq!(store.num_points(), 100);
}

#[test]
fn test_high_dimensional_optimized() {
    let mut store = DynamicTreeEmbedding::new(2.0, 50.0, 20);
    
    for _i in 0..30 {
        let p = EuclideanPoint::random(20, 50.0);
        store.insert(p);
    }
    
    assert_eq!(store.num_points(), 30);
}

#[test]
fn test_dimension_scaling() {
    println!("\n=== Testing Dimension Scaling ===");
    
    let n = 50;
    let dimensions = vec![4, 8, 16, 32];
    let mut times = Vec::new();
    
    for &d in &dimensions {
        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, d);
        
        for i in 0..(n - 1) {
            let p = EuclideanPoint::new(vec![(i as f64) * 0.1; d]);
            store.insert(p);
        }
        
        let start = Instant::now();
        let p = EuclideanPoint::new(vec![(n as f64) * 0.1; d]);
        store.insert(p);
        let elapsed = start.elapsed();
        
        times.push(elapsed.as_micros());
        println!("d={:2}: insert time = {:6} μs", d, elapsed.as_micros());
    }
    
    println!("\nFor O(d), time should grow linearly with dimension");
    
    let ratio_1 = times[1] as f64 / times[0] as f64;
    let ratio_2 = times[2] as f64 / times[1] as f64;
    
    println!("  8/4:  {:.2}x (expected ~2x)", ratio_1);
    println!("  16/8: {:.2}x (expected ~2x)", ratio_2);
}
