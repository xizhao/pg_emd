//! # pg_drift: Dynamic Tree Embeddings for EMD
//!
//! Implementation of the dynamic tree embedding algorithm from:
//! "Tree Embedding in High Dimensions: Dynamic and Massively Parallel"
//! by Goranci et al. (arXiv:2510.22490, 2025)
//!
//! ## Purpose
//!
//! This library provides efficient data structures for:
//! - **Earth Mover's Distance (EMD):** O(log n)-approximate EMD in O(n) time
//! - **k-median clustering:** Dynamic clustering with Õ(n^ε) updates
//! - **Metric optimization:** Transform hard metric problems to tree problems
//!
//! ## Algorithm Overview (Algorithm 1 from paper)
//!
//! The core algorithm creates a probabilistic tree embedding with O(Γ log Γ · log n) distortion:
//!
//! 1. **Metric Decomposition**: Partition space using bounded-diameter hash functions φᵢ
//!    - Each level i has diameter bound τᵢ = wᵢ/2
//!    - Scale: wᵢ = aspect_ratio · 2⁻ⁱ
//!
//! 2. **Random Map π**: Assign each point a uniform random value in [0,1]
//!    - Critical: Do NOT sort these values (destroys randomness)
//!
//! 3. **Label Computation**: For each level i and point p:
//!    - Sample β ∈ [1/4, 1/2]
//!    - Set rᵢ = β/Γ · wᵢ
//!    - Compute B̃ᵢᴾ(p, rᵢ) = union of points in buckets intersecting B(p, rᵢ)
//!    - Set label ℓₚ⁽ⁱ⁾ = min{π(q) : q ∈ B̃ᵢᴾ(p, rᵢ)}
//!
//! 4. **Tree Distance**: Points p, q have tree distance based on their lowest common ancestor
//!
//! ## Distortion Guarantees (Lemma 3.1)
//!
//! - **Dominating Property**: ∀p, q: distₜ(p, q) ≥ dist(p, q)
//! - **Expected Expansion**: E[distₜ(p, q)] ≤ O(Γ log Γ) · log n · dist(p, q)
//!
//! ## Key Implementation Details
//!
//! ### Metric Hashing (GridHash)
//! - Grid-based partitioning with cell_size = τᵢ / √d
//! - Ensures Euclidean diameter ≤ τᵢ (not just side length!)
//! - Random offset provides probabilistic guarantees
//!
//! ### B̃ Computation (Critical!)
//! - B̃ is NOT "points in the ball"
//! - B̃ is "all points whose bucket intersects the ball"
//! - Points outside the ball can be included if their bucket intersects
//! - This is essential for the distortion analysis
//!
//! ### Bucket-Ball Intersection
//! - Uses axis-aligned box-to-sphere intersection test
//! - O(d) time per point
//! - Avoids exponential O((2r)^d) bucket enumeration
//!
//! ### Dynamic Updates
//! - Insert: O(n·m·d) - recompute labels for affected points
//! - Delete: O(n·m·d) - update bucket minimums and nearby labels
//! - Could optimize to Õ(n^ε + d) with spatial indexing

pub mod metric;
pub mod hashing;
pub mod tree_embedding;
pub mod dynamic;
pub mod emd;

// PostgreSQL extension module (only compiled when building with pgrx features)
#[cfg(any(feature = "pg12", feature = "pg13", feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
pub mod pg_extension;

pub use metric::{Point, EuclideanPoint};
pub use tree_embedding::TreeEmbedding;
pub use dynamic::DynamicTreeEmbedding;
pub use emd::Distribution;
