use pg_emd::*;

fn main() {
    println!("=== pg_emd: Tree Embedding for EMD Examples ===\n");

    static_embedding_example();
    println!();
    dynamic_embedding_example();
    println!();
    clustering_example();
}

fn static_embedding_example() {
    println!("1. Static Tree Embedding Example");
    println!("---------------------------------");

    let points = vec![
        EuclideanPoint::new(vec![0.0, 0.0]),
        EuclideanPoint::new(vec![1.0, 1.0]),
        EuclideanPoint::new(vec![10.0, 10.0]),
        EuclideanPoint::new(vec![11.0, 11.0]),
    ];

    let gamma = 2.0;
    let aspect_ratio = 20.0;
    let embedding = TreeEmbedding::new(gamma, aspect_ratio);
    
    println!("Computing tree embedding for {} points...", points.len());
    let result = embedding.compute_embedding(&points);

    println!("\nDistance comparisons:");
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let euclidean = points[i].distance(&points[j]);
            let tree_dist = embedding.tree_distance(
                result.get_labels(i),
                result.get_labels(j),
            );
            let distortion = tree_dist / euclidean;
            
            println!(
                "  Points {} to {}: Euclidean={:.2}, Tree={:.2}, Distortion={:.2}x",
                i, j, euclidean, tree_dist, distortion
            );
        }
    }
}

fn dynamic_embedding_example() {
    println!("2. Dynamic Tree Embedding Example");
    println!("----------------------------------");

    let gamma = 2.0;
    let aspect_ratio = 50.0;
    let dimension = 2;
    
    let mut embedding = DynamicTreeEmbedding::new(gamma, aspect_ratio, dimension);

    println!("Inserting points dynamically...");
    let p1 = EuclideanPoint::new(vec![0.0, 0.0]);
    let p2 = EuclideanPoint::new(vec![5.0, 5.0]);
    let p3 = EuclideanPoint::new(vec![10.0, 10.0]);

    let id1 = embedding.insert(p1.clone());
    let id2 = embedding.insert(p2.clone());
    let id3 = embedding.insert(p3.clone());

    println!("Number of points: {}", embedding.num_points());

    if let Some(dist) = embedding.tree_distance(id1, id2) {
        println!("Tree distance between point 1 and 2: {:.2}", dist);
    }

    if let Some(dist) = embedding.tree_distance(id2, id3) {
        println!("Tree distance between point 2 and 3: {:.2}", dist);
    }

    println!("\nDeleting point 2...");
    embedding.delete(id2);
    println!("Number of points: {}", embedding.num_points());

    if let Some(dist) = embedding.tree_distance(id1, id3) {
        println!("Tree distance between point 1 and 3 (after deletion): {:.2}", dist);
    }

    println!("\nInserting a new point...");
    let p4 = EuclideanPoint::new(vec![2.5, 2.5]);
    let id4 = embedding.insert(p4);
    println!("Number of points: {}", embedding.num_points());

    if let Some(dist) = embedding.tree_distance(id1, id4) {
        println!("Tree distance between point 1 and new point: {:.2}", dist);
    }
}

fn clustering_example() {
    println!("3. Clustering with Tree Embeddings");
    println!("-----------------------------------");

    let mut cluster1 = vec![
        EuclideanPoint::new(vec![0.0, 0.0]),
        EuclideanPoint::new(vec![1.0, 1.0]),
        EuclideanPoint::new(vec![2.0, 2.0]),
    ];

    let mut cluster2 = vec![
        EuclideanPoint::new(vec![20.0, 20.0]),
        EuclideanPoint::new(vec![21.0, 21.0]),
        EuclideanPoint::new(vec![22.0, 22.0]),
    ];

    let mut points = Vec::new();
    points.append(&mut cluster1);
    points.append(&mut cluster2);

    let embedding = TreeEmbedding::new(2.0, 50.0);
    let result = embedding.compute_embedding(&points);

    println!("Cluster 1 points: 0, 1, 2");
    println!("Cluster 2 points: 3, 4, 5");
    println!();

    let within_c1 = embedding.tree_distance(
        result.get_labels(0),
        result.get_labels(2),
    );
    
    let within_c2 = embedding.tree_distance(
        result.get_labels(3),
        result.get_labels(5),
    );
    
    let between = embedding.tree_distance(
        result.get_labels(0),
        result.get_labels(3),
    );

    println!("Within cluster 1 distance: {:.2}", within_c1);
    println!("Within cluster 2 distance: {:.2}", within_c2);
    println!("Between clusters distance: {:.2}", between);
    println!();
    println!("Tree embeddings preserve cluster structure:");
    println!("  Between-cluster >> Within-cluster: {}", between > within_c1.max(within_c2));
}
