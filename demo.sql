-- pg_emd Extension Demo
-- Fast approximate Earth Mover's Distance for PostgreSQL

-- 1. Basic EMD: Compare two simple distributions
SELECT emd(
    ARRAY[0.5, 0.3, 0.2],  -- Distribution A
    ARRAY[0.2, 0.3, 0.5]   -- Distribution B
) AS emd_basic;
-- Result: ~3.9 (O(log n)-approximate distance)

-- 2. Image Color Histogram Comparison
-- Simulating RGB histograms
SELECT emd(
    ARRAY[0.6, 0.3, 0.1],  -- Image 1: reddish
    ARRAY[0.1, 0.6, 0.3]   -- Image 2: greenish  
) AS color_similarity;

-- 3. Document Topic Distribution
-- LDA topic weights
SELECT emd(
    ARRAY[0.7, 0.2, 0.1],  -- Doc 1: mostly topic 0
    ARRAY[0.1, 0.2, 0.7]   -- Doc 2: mostly topic 2
) AS topic_distance;

-- 4. Weighted EMD with explicit points and weights
SELECT emd_weighted(
    '[{"point": [1.0, 2.0], "weight": 0.7}, {"point": [3.0, 4.0], "weight": 0.3}]'::json,
    '[{"point": [10.0, 20.0], "weight": 1.0}]'::json
) AS weighted_emd;

-- 5. Tree embedding distance (underlying metric)
SELECT tree_distance(
    ARRAY[0.0, 0.0],
    ARRAY[10.0, 10.0]
) AS tree_dist;

-- 6. Performance comparison: exact vs approximate
-- This would take minutes with exact O(n³) Hungarian algorithm
-- but completes in milliseconds with our O(n) tree embedding
SELECT emd(
    ARRAY[0.1, 0.15, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05, 0.05],  -- n=10
    ARRAY[0.05, 0.05, 0.05, 0.07, 0.08, 0.1, 0.15, 0.2, 0.15, 0.1]   -- reversed
) AS large_array_emd;

-- List all available functions
\df emd*
\df tree_distance
