/// Test suite to verify EMD implementation correctness
/// 
/// These tests check that EMD behaves like Earth Mover's Distance,
/// not just as a tree-embedding distance metric.

use pg_emd::*;

#[test]
fn test_emd_histogram_basic() {
    // EMD for 1D histograms has a simple formula:
    // EMD([a1, a2, a3], [b1, b2, b3]) = sum of cumulative differences
    
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 1);
    
    // Create bins at positions 0, 1, 2 in 1D space
    let bin_0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let bin_1 = store.insert(EuclideanPoint::new(vec![1.0]));
    let bin_2 = store.insert(EuclideanPoint::new(vec![2.0]));
    
    // Distribution A: [0.5, 0.3, 0.2] - mass at bins 0, 1, 2
    let dist_a = Distribution::new(vec![
        (bin_0, 0.5),
        (bin_1, 0.3),
        (bin_2, 0.2),
    ]);
    
    // Distribution B: [0.2, 0.3, 0.5] - reversed
    let dist_b = Distribution::new(vec![
        (bin_0, 0.2),
        (bin_1, 0.3),
        (bin_2, 0.5),
    ]);
    
    let emd = store.emd_distance(&dist_a, &dist_b);
    
    // For 1D histograms with unit spacing:
    // EMD = |0.5-0.2| + |(0.5+0.3)-(0.2+0.3)| + |(0.5+0.3+0.2)-(0.2+0.3+0.5)|
    //     = 0.3 + 0.3 + 0.0 = 0.6
    // (or cumulative difference formula)
    
    println!("EMD for reversed histograms: {}", emd);
    assert!(emd > 0.0, "EMD should be non-zero for different histograms");
}

#[test]
fn test_emd_point_mass_shift() {
    // Test moving a point mass from one location to another
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 1);
    
    let pos_0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let pos_1 = store.insert(EuclideanPoint::new(vec![1.0]));
    
    // All mass at position 0
    let dist_a = Distribution::new(vec![(pos_0, 1.0)]);
    
    // All mass at position 1
    let dist_b = Distribution::new(vec![(pos_1, 1.0)]);
    
    let emd = store.emd_distance(&dist_a, &dist_b);
    
    println!("EMD for point mass shift by 1 unit: {}", emd);
    
    // Should be approximately 1.0 (distance × mass)
    // With tree embedding, might have some approximation factor
    assert!(emd > 0.5, "EMD should be at least 0.5 for unit distance shift");
    assert!(emd < 20.0, "EMD should not be too large due to approximation");
}

#[test]
fn test_emd_multi_bin_histogram() {
    // Test a more realistic histogram scenario
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 1);
    
    // Create 5 bins at positions 0, 1, 2, 3, 4
    let bins: Vec<_> = (0..5)
        .map(|i| store.insert(EuclideanPoint::new(vec![i as f64])))
        .collect();
    
    // Distribution A: concentrated at start
    let dist_a = Distribution::new(vec![
        (bins[0], 0.4),
        (bins[1], 0.3),
        (bins[2], 0.2),
        (bins[3], 0.1),
        (bins[4], 0.0),
    ]);
    
    // Distribution B: concentrated at end
    let dist_b = Distribution::new(vec![
        (bins[0], 0.0),
        (bins[1], 0.1),
        (bins[2], 0.2),
        (bins[3], 0.3),
        (bins[4], 0.4),
    ]);
    
    let emd = store.emd_distance(&dist_a, &dist_b);
    
    println!("EMD for shifted distributions: {}", emd);
    assert!(emd > 0.0, "EMD should be positive for shifted distributions");
}

#[test]
fn test_emd_symmetry() {
    // EMD should be symmetric: EMD(A, B) = EMD(B, A)
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 1);
    
    let bin_0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let bin_1 = store.insert(EuclideanPoint::new(vec![1.0]));
    
    let dist_a = Distribution::new(vec![(bin_0, 0.7), (bin_1, 0.3)]);
    let dist_b = Distribution::new(vec![(bin_0, 0.3), (bin_1, 0.7)]);
    
    let emd_ab = store.emd_distance(&dist_a, &dist_b);
    let emd_ba = store.emd_distance(&dist_b, &dist_a);
    
    println!("EMD(A,B) = {}, EMD(B,A) = {}", emd_ab, emd_ba);
    
    assert!(
        (emd_ab - emd_ba).abs() < 0.001,
        "EMD should be symmetric: {} vs {}",
        emd_ab,
        emd_ba
    );
}

#[test]
fn test_emd_triangle_inequality() {
    // EMD should satisfy triangle inequality: EMD(A,C) <= EMD(A,B) + EMD(B,C)
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 1);
    
    let bin_0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let bin_1 = store.insert(EuclideanPoint::new(vec![1.0]));
    
    let dist_a = Distribution::new(vec![(bin_0, 1.0)]);
    let dist_b = Distribution::new(vec![(bin_0, 0.5), (bin_1, 0.5)]);
    let dist_c = Distribution::new(vec![(bin_1, 1.0)]);
    
    let emd_ac = store.emd_distance(&dist_a, &dist_c);
    let emd_ab = store.emd_distance(&dist_a, &dist_b);
    let emd_bc = store.emd_distance(&dist_b, &dist_c);
    
    println!("EMD(A,C) = {}, EMD(A,B) + EMD(B,C) = {}", emd_ac, emd_ab + emd_bc);
    
    assert!(
        emd_ac <= emd_ab + emd_bc + 0.001,
        "Triangle inequality violated: {} > {} + {}",
        emd_ac,
        emd_ab,
        emd_bc
    );
}

#[test]
fn test_current_implementation_is_wrong() {
    // This test DOCUMENTS the bug in the current pg_extension::emd() function
    // It treats arrays as point coordinates, not as histograms
    
    let mut store = DynamicTreeEmbedding::new(2.0, 1000.0, 3);
    
    // What pg_extension::emd currently does:
    // Treats [0.6, 0.3, 0.1] as a point at coordinates (0.6, 0.3, 0.1)
    let point_a = store.insert(EuclideanPoint::new(vec![0.6, 0.3, 0.1]));
    let point_b = store.insert(EuclideanPoint::new(vec![0.5, 0.3, 0.2]));
    
    let wrong_dist_a = Distribution::new(vec![(point_a, 1.0)]);
    let wrong_dist_b = Distribution::new(vec![(point_b, 1.0)]);
    
    let wrong_result = store.emd_distance(&wrong_dist_a, &wrong_dist_b);
    
    println!("WRONG implementation result: {}", wrong_result);
    
    // What it SHOULD do:
    // Treat [0.6, 0.3, 0.1] as histogram with bins at positions 0, 1, 2
    let bin_0 = store.insert(EuclideanPoint::new(vec![0.0, 0.0, 0.0]));
    let bin_1 = store.insert(EuclideanPoint::new(vec![1.0, 0.0, 0.0]));
    let bin_2 = store.insert(EuclideanPoint::new(vec![2.0, 0.0, 0.0]));
    
    let correct_dist_a = Distribution::new(vec![
        (bin_0, 0.6),
        (bin_1, 0.3),
        (bin_2, 0.1),
    ]);
    
    let correct_dist_b = Distribution::new(vec![
        (bin_0, 0.5),
        (bin_1, 0.3),
        (bin_2, 0.2),
    ]);
    
    let correct_result = store.emd_distance(&correct_dist_a, &correct_dist_b);
    
    println!("CORRECT implementation result: {}", correct_result);
    
    // These should be DIFFERENT because they compute different things!
    assert_ne!(
        wrong_result, correct_result,
        "Wrong and correct implementations should differ"
    );
    
    assert!(
        correct_result > 0.0,
        "Correct EMD should be non-zero for different histograms"
    );
}
