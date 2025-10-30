use pg_emd::*;

fn main() {
    println!("=== pg_emd: Earth Mover's Distance Examples ===\n");
    
    image_histogram_example();
    println!();
    document_topic_example();
    println!();
    streaming_distribution_example();
}

fn image_histogram_example() {
    println!("1. Image Histogram Comparison");
    println!("------------------------------");
    
    let mut store = DynamicTreeEmbedding::new(2.0, 255.0, 3);
    
    // Simulate RGB histograms (simplified to 10 bins per color)
    println!("Image A: Predominantly red");
    let hist_a: Vec<_> = vec![
        (100, 0, 0),   // Dark red
        (200, 0, 0),   // Bright red
        (150, 50, 50), // Light red
        (100, 20, 20), // Dark red-ish
    ]
    .into_iter()
    .map(|(r, g, b)| {
        store.insert(EuclideanPoint::new(vec![r as f64, g as f64, b as f64]))
    })
    .collect();
    
    println!("Image B: Predominantly blue");
    let hist_b: Vec<_> = vec![
        (0, 0, 100),   // Dark blue
        (0, 0, 200),   // Bright blue
        (50, 50, 150), // Light blue
        (20, 20, 100), // Dark blue-ish
    ]
    .into_iter()
    .map(|(r, g, b)| {
        store.insert(EuclideanPoint::new(vec![r as f64, g as f64, b as f64]))
    })
    .collect();
    
    println!("Image C: Similar to image A (slightly shifted red)");
    let hist_c: Vec<_> = vec![
        (110, 10, 10),
        (190, 10, 10),
        (140, 60, 60),
        (90, 30, 30),
    ]
    .into_iter()
    .map(|(r, g, b)| {
        store.insert(EuclideanPoint::new(vec![r as f64, g as f64, b as f64]))
    })
    .collect();
    
    let emd_ab = store.emd_between_sets(&hist_a, &hist_b);
    let emd_ac = store.emd_between_sets(&hist_a, &hist_c);
    
    println!("\nEMD Results:");
    println!("  EMD(A, B) = {:.2} (red vs blue - very different)", emd_ab);
    println!("  EMD(A, C) = {:.2} (red vs red - similar)", emd_ac);
    println!("  Ratio: {:.2}x", emd_ab / emd_ac);
    
    assert!(emd_ab > emd_ac, "Different colors should have higher EMD");
}

fn document_topic_example() {
    println!("2. Document Topic Distribution Comparison");
    println!("------------------------------------------");
    
    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 5);
    
    // Simulate topic distributions (5 topics: tech, sports, politics, health, entertainment)
    println!("Doc A: Tech article");
    let topics_a = vec![0.7, 0.1, 0.1, 0.05, 0.05]; // 70% tech
    
    println!("Doc B: Sports article");  
    let topics_b = vec![0.1, 0.7, 0.1, 0.05, 0.05]; // 70% sports
    
    println!("Doc C: Mixed tech/sports");
    let topics_c = vec![0.4, 0.4, 0.1, 0.05, 0.05]; // 40% tech, 40% sports
    
    // Convert to points (topics as dimensions)
    let id_a = store.insert(EuclideanPoint::new(topics_a));
    let id_b = store.insert(EuclideanPoint::new(topics_b));
    let id_c = store.insert(EuclideanPoint::new(topics_c));
    
    // Create distributions (single point with weight 1.0)
    let dist_a = Distribution::new(vec![(id_a, 1.0)]);
    let dist_b = Distribution::new(vec![(id_b, 1.0)]);
    let dist_c = Distribution::new(vec![(id_c, 1.0)]);
    
    let emd_ab = store.emd_distance(&dist_a, &dist_b);
    let emd_ac = store.emd_distance(&dist_a, &dist_c);
    let emd_bc = store.emd_distance(&dist_b, &dist_c);
    
    println!("\nEMD Results:");
    println!("  EMD(Tech, Sports) = {:.2}", emd_ab);
    println!("  EMD(Tech, Mixed)  = {:.2}", emd_ac);
    println!("  EMD(Sports, Mixed) = {:.2}", emd_bc);
    
    assert!(emd_ac < emd_ab, "Mixed doc closer to tech than sports doc");
}

fn streaming_distribution_example() {
    println!("3. Streaming Distribution Updates");
    println!("-----------------------------------");
    
    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 2);
    
    println!("Initial distribution: uniform 0-10");
    let mut current_dist: Vec<_> = (0..10)
        .map(|i| store.insert(EuclideanPoint::new(vec![i as f64, 0.0])))
        .collect();
    
    let baseline_dist = current_dist.clone();
    
    println!("\nTime 1: Add high values (distribution shifts right)");
    for i in 10..15 {
        let id = store.insert(EuclideanPoint::new(vec![i as f64, 0.0]));
        current_dist.push(id);
    }
    
    let emd_t1 = store.emd_between_sets(&baseline_dist, &current_dist);
    println!("  EMD from baseline: {:.2}", emd_t1);
    
    println!("\nTime 2: Remove low values (distribution shifts more right)");
    for i in 0..5 {
        store.delete(current_dist[i]);
    }
    let current_dist: Vec<_> = current_dist[5..].to_vec();
    
    let emd_t2 = store.emd_between_sets(&baseline_dist, &current_dist);
    println!("  EMD from baseline: {:.2}", emd_t2);
    
    assert!(emd_t2 > emd_t1, "More shift should increase EMD");
    
    println!("\n✓ Dynamic updates maintained efficiently!");
    println!("  (Each update in Õ(n^0.81) time, EMD computation in O(n))");
}
