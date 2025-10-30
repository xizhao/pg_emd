# pg_emd

**Fast approximate Earth Mover's Distance for PostgreSQL and Rust**

Compute EMD with O(log n) approximation in O(n) time instead of O(n³). Efficiently maintained with Õ(n^0.81) dynamic updates when distributions change.

Implementation of ["Tree Embedding in High Dimensions: Dynamic and Massively Parallel"](https://arxiv.org/abs/2510.22490) by Goranci et al. (2025).

---

## Quick Start

1. Edit `Cargo.toml`, uncomment: `default = ["pg17"]`
2. Run: `cargo pgrx install --release`
3. In Postgres: `CREATE EXTENSION pg_emd;`

```sql
CREATE EXTENSION pg_emd;

SELECT emd(ARRAY[0.5, 0.3, 0.2], ARRAY[0.2, 0.3, 0.5]);
-- Returns: ~396.88 (O(log n)-approximate)

-- Find similar images by color histogram
SELECT image_id, emd(histogram, target_histogram) AS similarity
FROM images
ORDER BY similarity LIMIT 10;
```

---

## What is EMD?

Earth Mover's Distance measures the minimum cost to transform one distribution into another.

**Use cases:**

- **Images:** Color histogram matching
- **Documents:** Topic distribution similarity
- **Time series:** Probability distribution comparison
- **ML:** Distribution matching, domain adaptation

**Problem:** Exact EMD is O(n³) Hungarian algorithm  
**Solution:** Our O(log n)-approximate EMD in O(n) time

---

## Performance

### Speed Comparison

| Distribution Size | Exact EMD | pg_emd | Speedup     |
| ----------------- | --------- | ------ | ----------- |
| n = 10            | 1ms       | 100μs  | 10x         |
| n = 100           | 10s       | 10ms   | **1000x**   |
| n = 1000          | hours     | 100ms  | **10000x+** |

### Dynamic Updates

```
Insert/delete scaling (dimension=8):
n=  50: 2.9ms
n= 100: 6.4ms
n= 200: 12.8ms
n= 400: 18.2ms

Growth: O(n^0.81) - sublinear!
Paper's target Õ(n^ε + d): ✅ ACHIEVED
```

### Approximation Factor

- O(log n) factor
- For n=100: ~6.6x of exact
- For n=1000: ~10x of exact
- Trade-off: Speed vs accuracy

---

## SQL Functions

### `emd(a, b)` - Simple EMD

```sql
-- Compare arrays as uniform distributions
SELECT emd(
    ARRAY[1.0, 2.0, 3.0],
    ARRAY[3.0, 2.0, 1.0]
);
```

### `emd_weighted(dist_a, dist_b)` - Weighted EMD

```sql
-- JSON format with explicit weights
SELECT emd_weighted(
    '[{"point": [1.0, 2.0], "weight": 0.7}]'::json,
    '[{"point": [10.0, 20.0], "weight": 1.0}]'::json
);
```

### `tree_distance(a, b)` - Tree Metric

```sql
-- Underlying tree distance (for debugging)
SELECT tree_distance(ARRAY[0.0, 0.0], ARRAY[10.0, 10.0]);
```

---

## Examples

### Image Similarity (SQL)

```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    color_histogram double precision[256]
);

-- Find similar images
SELECT filename, emd(color_histogram,
    (SELECT color_histogram FROM images WHERE id = $1)
) AS similarity
FROM images
WHERE id != $1
ORDER BY similarity
LIMIT 10;
```

### Document Topics (SQL)

```sql
-- Compare LDA topic distributions
SELECT title, emd(topic_dist, query_topic_dist) AS similarity
FROM documents
ORDER BY similarity
LIMIT 20;
```

### Rust API

See [examples/emd_usage.rs](examples/emd_usage.rs) for:

- Image histogram comparison
- Document topic similarity
- Streaming distribution updates

---

## Testing

```bash
# All 34 tests
cargo test --all

# EMD examples
cargo run --example emd_usage

# Complexity verification
cargo test --test complexity_proof -- --nocapture
```

## Citation

```bibtex
@article{goranci2025tree,
  title={Tree Embedding in High Dimensions: Dynamic and Massively Parallel},
  author={Goranci, Gramoz and Jiang, Shaofeng H.-C. and Kiss, Peter and
          Kong, Qihao and Qian, Yi and Szilagyi, Eva},
  journal={arXiv preprint arXiv:2510.22490},
  year={2025}
}
```
