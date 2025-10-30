use adze_store::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn benchmark_static_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("static_embedding");
    
    for size in [10, 50, 100, 200].iter() {
        let points: Vec<EuclideanPoint> = (0..*size)
            .map(|i| EuclideanPoint::new(vec![(i as f64) * 0.1, (i as f64) * 0.2]))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                let embedding = TreeEmbedding::new(2.0, 100.0);
                b.iter(|| {
                    let result = embedding.compute_embedding(black_box(&points));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_dynamic_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_insertion");
    
    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let embedding = DynamicTreeEmbedding::new(2.0, 100.0, 2);
                        let points: Vec<EuclideanPoint> = (0..size)
                            .map(|i| EuclideanPoint::new(vec![(i as f64) * 0.1, (i as f64) * 0.2]))
                            .collect();
                        (embedding, points)
                    },
                    |(mut embedding, points)| {
                        for point in points {
                            black_box(embedding.insert(black_box(point)));
                        }
                    },
                );
            },
        );
    }
    
    group.finish();
}

fn benchmark_dynamic_deletion(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_deletion");
    
    for size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let mut embedding = DynamicTreeEmbedding::new(2.0, 100.0, 2);
                        let points: Vec<EuclideanPoint> = (0..size)
                            .map(|i| EuclideanPoint::new(vec![(i as f64) * 0.1, (i as f64) * 0.2]))
                            .collect();
                        let ids: Vec<_> = points.into_iter()
                            .map(|p| embedding.insert(p))
                            .collect();
                        (embedding, ids)
                    },
                    |(mut embedding, ids)| {
                        for id in ids {
                            black_box(embedding.delete(black_box(id)));
                        }
                    },
                );
            },
        );
    }
    
    group.finish();
}

fn benchmark_tree_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tree_distance");
    
    for size in [10, 50, 100].iter() {
        let points: Vec<EuclideanPoint> = (0..*size)
            .map(|i| EuclideanPoint::new(vec![(i as f64) * 0.1, (i as f64) * 0.2]))
            .collect();
        
        let embedding = TreeEmbedding::new(2.0, 100.0);
        let result = embedding.compute_embedding(&points);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| {
                    let mut total = 0.0;
                    for i in 0..size {
                        for j in (i + 1)..size {
                            total += embedding.tree_distance(
                                black_box(result.get_labels(i)),
                                black_box(result.get_labels(j)),
                            );
                        }
                    }
                    black_box(total);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_high_dimensional(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimensional");
    
    for dim in [2, 5, 10, 20].iter() {
        let points: Vec<EuclideanPoint> = (0..50)
            .map(|_| EuclideanPoint::random(*dim, 100.0))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, _| {
                let embedding = TreeEmbedding::new(2.0, 100.0);
                b.iter(|| {
                    let result = embedding.compute_embedding(black_box(&points));
                    black_box(result);
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_operations");
    
    group.bench_function("insert_delete_query", |b| {
        b.iter_with_setup(
            || DynamicTreeEmbedding::new(2.0, 100.0, 2),
            |mut embedding| {
                let mut ids = Vec::new();
                
                for i in 0..50 {
                    let p = EuclideanPoint::new(vec![(i as f64) * 0.5, (i as f64) * 0.3]);
                    let id = embedding.insert(black_box(p));
                    ids.push(id);
                }
                
                for i in (0..25).step_by(2) {
                    embedding.delete(black_box(ids[i]));
                }
                
                for i in 1..25 {
                    if i % 2 == 1 {
                        for j in (i + 1)..25 {
                            if j % 2 == 1 {
                                black_box(embedding.tree_distance(ids[i], ids[j]));
                            }
                        }
                    }
                }
                
                for i in 0..25 {
                    let p = EuclideanPoint::new(vec![(i + 100) as f64, (i + 100) as f64]);
                    black_box(embedding.insert(black_box(p)));
                }
            },
        );
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_static_embedding,
    benchmark_dynamic_insertion,
    benchmark_dynamic_deletion,
    benchmark_tree_distance_computation,
    benchmark_high_dimensional,
    benchmark_mixed_operations
);
criterion_main!(benches);
