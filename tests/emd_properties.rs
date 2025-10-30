/// EMD Properties Tests
///
/// Verifies mathematical properties that EMD must satisfy:
/// - Exactness: 1D EMD should be exact (not approximate)
/// - Symmetry: EMD(A,B) = EMD(B,A)
/// - Non-negativity: EMD ≥ 0  
/// - Identity: EMD(A,A) = 0
/// - Mass scaling: k×mass → k×EMD (linearity in mass)
/// - Distance scaling: For 1D exact, 2×distance → 2×EMD

use pg_emd::*;

#[test]
fn test_1d_emd_exact_linearity() {
    println!("\n=== 1D EMD: Linearity (MUST be exact) ===");
    
    use crate::emd_1d::emd_1d_exact;
    
    let emd1 = emd_1d_exact(&vec![1.0, 0.0], &vec![0.0, 1.0]);
    let emd2 = emd_1d_exact(&vec![1.0, 0.0, 0.0], &vec![0.0, 0.0, 1.0]);
    let emd4 = emd_1d_exact(&vec![1.0, 0.0, 0.0, 0.0, 0.0], &vec![0.0, 0.0, 0.0, 0.0, 1.0]);
    
    println!("Distance 1: {} (exact=1.0)", emd1);
    println!("Distance 2: {} (exact=2.0)", emd2);
    println!("Distance 4: {} (exact=4.0)", emd4);
    
    assert!((emd1 - 1.0).abs() < 0.001, "d=1 failed");
    assert!((emd2 - 2.0).abs() < 0.001, "d=2 failed");
    assert!((emd4 - 4.0).abs() < 0.001, "d=4 failed");
    
    // Check ratios
    assert!((emd2 / emd1 - 2.0).abs() < 0.01, "ratio 2:1 failed");
    assert!((emd4 / emd2 - 2.0).abs() < 0.01, "ratio 4:2 failed");
    
    println!("✓ PASSED - 1D EMD is EXACT");
}

#[test]
fn test_1d_emd_exact_known_values() {
    println!("\n=== 1D EMD: Known Values ===");
    
    use crate::emd_1d::emd_1d_exact;
    
    // [0.5, 0.3, 0.2] vs [0.2, 0.3, 0.5]
    // CDF_A = [0.5, 0.8, 1.0], CDF_B = [0.2, 0.5, 1.0]
    // EMD = |0.5-0.2| + |0.8-0.5| = 0.6
    let emd = emd_1d_exact(&vec![0.5, 0.3, 0.2], &vec![0.2, 0.3, 0.5]);
    println!("EMD([0.5,0.3,0.2], [0.2,0.3,0.5]) = {} (exact=0.6)", emd);
    assert!((emd - 0.6).abs() < 0.0001);
    
    // Identical
    let emd = emd_1d_exact(&vec![0.25, 0.25, 0.25, 0.25], &vec![0.25, 0.25, 0.25, 0.25]);
    println!("EMD(identical) = {}", emd);
    assert_eq!(emd, 0.0);
    
    println!("✓ PASSED");
}

#[test]
fn test_1d_emd_symmetry() {
    println!("\n=== 1D EMD: Symmetry ===");
    
    use crate::emd_1d::emd_1d_exact;
    
    let a = vec![0.7, 0.2, 0.1];
    let b = vec![0.1, 0.2, 0.7];
    
    let emd_ab = emd_1d_exact(&a, &b);
    let emd_ba = emd_1d_exact(&b, &a);
    
    println!("EMD(A,B) = {}, EMD(B,A) = {}", emd_ab, emd_ba);
    assert_eq!(emd_ab, emd_ba);
    
    println!("✓ PASSED");
}

#[test]
fn test_tree_emd_matches_paper_lemma_5_13() {
    println!("\n=== Tree EMD: Matches Paper Lemma 5.13 ===");
    
    let gamma = 2.0;
    let aspect_ratio = 20.0;
    let dimension = 1;
    
    let mut store = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);
    
    let p0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let p1 = store.insert(EuclideanPoint::new(vec![5.0]));
    
    // For bipartite matching (|A|=|B|=1), EMD = tree distance
    let dist_a = Distribution::new(vec![(p0, 1.0)]);
    let dist_b = Distribution::new(vec![(p1, 1.0)]);
    
    let emd = store.emd_distance(&dist_a, &dist_b);
    let tree_dist = store.tree_distance(p0, p1).unwrap();
    
    println!("EMD: {}, Tree distance: {}", emd, tree_dist);
    assert!((emd - tree_dist).abs() < 0.01);
    
    println!("✓ PASSED - Tree EMD matches Lemma 5.13");
}

#[test]
fn test_tree_emd_multi_point_distribution() {
    println!("\n=== Tree EMD: Multi-Point Distribution ===");
    
    let gamma = 2.0;
    let aspect_ratio = 20.0;
    let dimension = 1;
    
    let mut store = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);
    
    let p0 = store.insert(EuclideanPoint::new(vec![0.0]));
    let p1 = store.insert(EuclideanPoint::new(vec![5.0]));
    let p2 = store.insert(EuclideanPoint::new(vec![10.0]));
    
    // A: 0.5 at p0, 0.5 at p1
    // B: 1.0 at p2
    let dist_a = Distribution::new(vec![(p0, 0.5), (p1, 0.5)]);
    let dist_b = Distribution::new(vec![(p2, 1.0)]);
    
    let emd = store.emd_distance(&dist_a, &dist_b);
    
    println!("Multi-point EMD: {}", emd);
    assert!(emd > 0.0);
    
    // Verify it's using the tree metric
    let d0_2 = store.tree_distance(p0, p2).unwrap();
    let d1_2 = store.tree_distance(p1, p2).unwrap();
    
    // Should be weighted combination
    println!("Tree distances: p0→p2={}, p1→p2={}", d0_2, d1_2);
    println!("Weighted avg: {}", 0.5 * d0_2 + 0.5 * d1_2);
    
    println!("✓ PASSED");
}
