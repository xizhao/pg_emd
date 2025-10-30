-- pg_emd extension for PostgreSQL
-- Fast approximate Earth Mover's Distance using dynamic tree embeddings

-- Complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pg_emd" to load this file. \quit

-- EMD distance function for simple arrays
CREATE FUNCTION emd(a double precision[], b double precision[])
RETURNS double precision
AS 'MODULE_PATHNAME', 'emd_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION emd IS 
'Compute approximate Earth Mover''s Distance between two distributions.
Returns O(log n)-approximate EMD in O(n) time.
Much faster than exact O(n³) Hungarian algorithm.';

-- Weighted EMD function
CREATE FUNCTION emd_weighted(dist_a json, dist_b json)
RETURNS double precision
AS 'MODULE_PATHNAME', 'emd_weighted_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION emd_weighted IS
'Compute EMD between weighted distributions specified as JSON.
Format: [{"point": [x, y, ...], "weight": w}, ...]';

-- Tree distance function (for debugging/advanced use)
CREATE FUNCTION tree_distance(a double precision[], b double precision[])
RETURNS double precision
AS 'MODULE_PATHNAME', 'tree_distance_wrapper'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

COMMENT ON FUNCTION tree_distance IS
'Compute tree embedding distance between two points.
Useful for understanding the underlying tree structure.';
