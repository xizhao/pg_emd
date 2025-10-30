use adze_store::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, AxisScale};
use std::time::Duration;

/// Benchmark insert performance at different scales to verify O(n^ε + d) complexity.
/// 
/// If optimization works:
/// - Insert time should grow slowly with n (sublinear, like n^0.2)
/// - NOT linear O(n) like the old implementation
fn benchmark_insert_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_scaling");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    // Test at different scales: if O(n^ε), should grow slowly
    // If O(n·m·d), would grow linearly with n
    for n in [50, 100, 200, 400, 800].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            n,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        // Pre-populate with size-1 points
                        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 8);
                        for i in 0..(size - 1) {
                            let p = EuclideanPoint::new(vec![
                                (i as f64) * 0.1,
                                (i as f64) * 0.2,
                                (i as f64) * 0.3,
                                (i as f64) * 0.4,
                                (i as f64) * 0.5,
                                (i as f64) * 0.6,
                                (i as f64) * 0.7,
                                (i as f64) * 0.8,
                            ]);
                            store.insert(p);
                        }
                        (store, size as f64)
                    },
                    |(mut store, val)| {
                        // Insert ONE point into a store with size-1 points
                        // Time should be O(n^ε + d), NOT O(n)
                        let p = EuclideanPoint::new(vec![val; 8]);
                        black_box(store.insert(black_box(p)));
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark delete performance scaling
fn benchmark_delete_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete_scaling");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    for n in [50, 100, 200, 400, 800].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            n,
            |b, &size| {
                b.iter_with_setup(
                    || {
                        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 8);
                        let mut ids = Vec::new();
                        for i in 0..size {
                            let p = EuclideanPoint::new(vec![(i as f64) * 0.1; 8]);
                            let id = store.insert(p);
                            ids.push(id);
                        }
                        (store, ids[size / 2]) // Delete middle point
                    },
                    |(mut store, id)| {
                        black_box(store.delete(black_box(id)));
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Measure average insert time vs number of existing points
/// This directly shows the complexity: slope on log-log plot = exponent
fn measure_complexity_directly(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_measurement");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);
    
    println!("\n=== Complexity Measurement ===");
    println!("If O(n^ε + d): time should grow slowly (ε << 1)");
    println!("If O(n): time would double when n doubles");
    println!("\nn\tAvg Insert Time");
    
    for n in [100, 200, 400, 800, 1600].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(n),
            n,
            |b, &size| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 8);
                        
                        // Pre-populate
                        for i in 0..size {
                            let p = EuclideanPoint::new(vec![(i as f64) * 0.1; 8]);
                            store.insert(p);
                        }
                        
                        // Measure time for one more insert
                        let start = std::time::Instant::now();
                        let p = EuclideanPoint::new(vec![(size as f64) * 0.1; 8]);
                        black_box(store.insert(black_box(p)));
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }
    
    group.finish();
}

/// Test that dimension scaling is O(d) not O(n*d)
fn benchmark_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    
    let n = 200; // Fixed number of points
    
    for d in [4, 8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(d),
            d,
            |b, &dim| {
                b.iter_with_setup(
                    || {
                        let mut store = DynamicTreeEmbedding::new(2.0, 100.0, dim);
                        for i in 0..(n - 1) {
                            let p = EuclideanPoint::new(vec![(i as f64) * 0.1; dim]);
                            store.insert(p);
                        }
                        (store, dim)
                    },
                    |(mut store, dim)| {
                        let p = EuclideanPoint::new(vec![1.0; dim]);
                        black_box(store.insert(black_box(p)));
                    },
                );
            },
        );
    }
    
    group.finish();
}

/// Compare batch insert performance
fn benchmark_batch_inserts(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inserts");
    
    for batch_size in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                b.iter(|| {
                    let mut store = DynamicTreeEmbedding::new(2.0, 100.0, 8);
                    for i in 0..size {
                        let p = EuclideanPoint::new(vec![(i as f64) * 0.1; 8]);
                        black_box(store.insert(black_box(p)));
                    }
                    black_box(store);
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_insert_scaling,
    benchmark_delete_scaling,
    measure_complexity_directly,
    benchmark_dimension_scaling,
    benchmark_batch_inserts,
);
criterion_main!(benches);
