use pg_emd::*;

#[test]
fn test_static_embedding_distortion() {
    let gamma = 2.0;
    let aspect_ratio = 100.0;
    
    let points: Vec<EuclideanPoint> = (0..50)
        .map(|i| EuclideanPoint::new(vec![i as f64, 0.0]))
        .collect();

    let embedding = TreeEmbedding::new(gamma, aspect_ratio);
    let result = embedding.compute_embedding(&points);

    let mut total_distortion = 0.0;
    let mut num_pairs = 0;

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let euclidean_dist = points[i].distance(&points[j]);
            let tree_dist = embedding.tree_distance(
                result.get_labels(i),
                result.get_labels(j),
            );

            assert!(
                tree_dist >= euclidean_dist * 0.95,
                "Dominating property violated: tree_dist={}, euclidean_dist={}",
                tree_dist,
                euclidean_dist
            );

            if euclidean_dist > 0.0 {
                let distortion = tree_dist / euclidean_dist;
                total_distortion += distortion;
                num_pairs += 1;
            }
        }
    }

    let avg_distortion = total_distortion / num_pairs as f64;
    let expected_max_distortion = gamma * gamma.log2() * (points.len() as f64).log2();
    
    println!("Average distortion: {:.2}", avg_distortion);
    println!("Expected max distortion: {:.2}", expected_max_distortion);
    
    assert!(
        avg_distortion < expected_max_distortion * 2.0,
        "Average distortion too high: {} > {}",
        avg_distortion,
        expected_max_distortion
    );
}

#[test]
fn test_dynamic_embedding_distortion() {
    let gamma = 2.0;
    let aspect_ratio = 100.0;
    let dimension = 2;
    
    let mut embedding = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);

    let points: Vec<EuclideanPoint> = (0..30)
        .map(|i| EuclideanPoint::new(vec![i as f64, (i * 2) as f64]))
        .collect();

    let ids: Vec<_> = points.iter()
        .map(|p| embedding.insert(p.clone()))
        .collect();

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            let euclidean_dist = points[i].distance(&points[j]);
            if let Some(tree_dist) = embedding.tree_distance(ids[i], ids[j]) {
                assert!(
                    tree_dist >= euclidean_dist * 0.95,
                    "Dominating property violated after insertion"
                );
            }
        }
    }

    embedding.delete(ids[5]);
    embedding.delete(ids[10]);

    for i in 0..ids.len() {
        if i == 5 || i == 10 {
            continue;
        }
        for j in (i + 1)..ids.len() {
            if j == 5 || j == 10 {
                continue;
            }
            let euclidean_dist = points[i].distance(&points[j]);
            if let Some(tree_dist) = embedding.tree_distance(ids[i], ids[j]) {
                assert!(
                    tree_dist >= euclidean_dist * 0.95,
                    "Dominating property violated after deletion"
                );
            }
        }
    }
}

#[test]
fn test_random_points_distortion() {
    let gamma = 2.0;
    let aspect_ratio = 50.0;
    let dimension = 3;
    
    let points: Vec<EuclideanPoint> = (0..20)
        .map(|_| EuclideanPoint::random(dimension, aspect_ratio))
        .collect();

    let embedding = TreeEmbedding::new(gamma, aspect_ratio);
    let result = embedding.compute_embedding(&points);

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let euclidean_dist = points[i].distance(&points[j]);
            let tree_dist = embedding.tree_distance(
                result.get_labels(i),
                result.get_labels(j),
            );

            if euclidean_dist > 1e-6 {
                assert!(
                    tree_dist >= euclidean_dist * 0.95,
                    "Dominating property violated for random points"
                );

                let distortion = tree_dist / euclidean_dist;
                let max_expected = gamma * gamma.log2() * (points.len() as f64).log2() * 10.0;
                assert!(
                    distortion < max_expected,
                    "Distortion too high: {} (max expected: {})",
                    distortion,
                    max_expected
                );
            }
        }
    }
}

#[test]
fn test_dynamic_many_operations() {
    let gamma = 2.0;
    let aspect_ratio = 100.0;
    let dimension = 2;
    
    let mut embedding = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);

    let mut points = Vec::new();
    let mut ids = Vec::new();

    for i in 0..20 {
        let p = EuclideanPoint::new(vec![i as f64 * 2.0, i as f64]);
        let id = embedding.insert(p.clone());
        points.push(p);
        ids.push(id);
    }

    assert_eq!(embedding.num_points(), 20);

    for i in (0..10).step_by(2) {
        embedding.delete(ids[i]);
    }

    assert_eq!(embedding.num_points(), 15);

    for i in 0..5 {
        let p = EuclideanPoint::new(vec![(i + 100) as f64, (i + 100) as f64]);
        let id = embedding.insert(p.clone());
        points.push(p);
        ids.push(id);
    }

    assert_eq!(embedding.num_points(), 20);

    let active_ids: Vec<_> = ids.iter().enumerate()
        .filter(|(i, _)| *i >= 10 || i % 2 == 1)
        .map(|(_, &id)| id)
        .collect();

    for &id1 in &active_ids {
        for &id2 in &active_ids {
            if id1 != id2 {
                let dist = embedding.tree_distance(id1, id2);
                assert!(dist.is_some());
            }
        }
    }
}

#[test]
fn test_high_dimensional_embedding() {
    let gamma = 2.0;
    let aspect_ratio = 50.0;
    let dimension = 10;
    
    let points: Vec<EuclideanPoint> = (0..15)
        .map(|_| EuclideanPoint::random(dimension, aspect_ratio))
        .collect();

    let embedding = TreeEmbedding::new(gamma, aspect_ratio);
    let result = embedding.compute_embedding(&points);

    assert_eq!(result.num_points(), points.len());

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let euclidean_dist = points[i].distance(&points[j]);
            let tree_dist = embedding.tree_distance(
                result.get_labels(i),
                result.get_labels(j),
            );

            if euclidean_dist > 1e-6 {
                assert!(tree_dist >= euclidean_dist * 0.95);
            }
        }
    }
}

#[test]
fn test_clustering_quality() {
    let gamma = 2.0;
    let aspect_ratio = 100.0;
    let _dimension = 2;
    
    let mut cluster1: Vec<EuclideanPoint> = (0..5)
        .map(|i| EuclideanPoint::new(vec![i as f64, i as f64]))
        .collect();
    
    let mut cluster2: Vec<EuclideanPoint> = (0..5)
        .map(|i| EuclideanPoint::new(vec![50.0 + i as f64, 50.0 + i as f64]))
        .collect();
    
    let mut points = Vec::new();
    points.append(&mut cluster1);
    points.append(&mut cluster2);

    let embedding = TreeEmbedding::new(gamma, aspect_ratio);
    let result = embedding.compute_embedding(&points);

    let within_cluster1 = embedding.tree_distance(
        result.get_labels(0),
        result.get_labels(4),
    );
    
    let between_clusters = embedding.tree_distance(
        result.get_labels(0),
        result.get_labels(5),
    );

    assert!(between_clusters > within_cluster1,
        "Between-cluster distance should be larger than within-cluster");
}
