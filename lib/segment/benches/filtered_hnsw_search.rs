//! Filtered HNSW graph-search benchmark.
//!
//! This is the path most exposed to the batched `ConditionChecker` refactor:
//! every graph neighbour goes through `FilteredScorer::score_points` →
//! `ScorerFilters::check_batched` → `OptimizedFilter`, while the ACORN variant
//! still calls per-neighbour `check_vector` interleaved with traversal.
//!
//! The filter goes through a real struct payload index, so the per-neighbour
//! check runs the production `OptimizedFilter` leaf-dispatch path.
//!
//! Knobs:
//! - algorithm: `Hnsw` (batched `score_points`) vs `Acorn` (per-neighbour
//!   `check_vector`, 2-hop expansion);
//! - filters: none / one indexed field (1..90% selectivity) / two indexed
//!   fields ANDed.

#[cfg(not(target_os = "windows"))]
mod prof;

mod fixture;

use std::hint::black_box;

use common::counter::hardware_counter::HardwareCounterCell;
use common::cow::SimpleCow;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ordered_float::OrderedFloat;
use rand::SeedableRng;
use rand::rngs::StdRng;
use segment::fixtures::index_fixtures::random_vector;
use segment::fixtures::payload_context_fixture::create_struct_payload_index;
use segment::fixtures::payload_fixtures::{FLT_KEY, INT_KEY};
use segment::index::PayloadIndexRead;
use segment::index::hnsw_index::graph_layers::SearchAlgorithm;
use segment::index::hnsw_index::point_scorer::FilteredScorer;
use segment::spaces::simple::CosineMetric;
use segment::types::{Condition, FieldCondition, Filter, Range};
use segment::vector_storage::DEFAULT_STOPPED;
use tempfile::Builder;

// Match `hnsw_search_graph` exactly so the cached graph on disk is reused.
const NUM_VECTORS: usize = 1_000_000;
const DIM: usize = 64;
const M: usize = 16;
const TOP: usize = 10;
const EF_CONSTRUCT: usize = 100;
const EF: usize = 100;
const USE_HEURISTIC: bool = true;

type Metric = CosineMetric;

/// A condition selecting ~`pct`% of points via the indexed `FLT_KEY` field
/// (a single uniform value in `0.0..10.0`).
fn flt_condition(pct: u64) -> Condition {
    let lt = pct as f64 * 10.0 / 100.0;
    Condition::Field(FieldCondition::new_range(
        FLT_KEY.parse().unwrap(),
        Range {
            lt: Some(OrderedFloat(lt)),
            gt: None,
            gte: None,
            lte: None,
        },
    ))
}

/// A condition selecting ~half of the points via the indexed `INT_KEY` field
/// (1..=3 values, each uniform in `0..500`; `any value < 150` ≈ 50% of points).
fn int_condition() -> Condition {
    Condition::Field(FieldCondition::new_range(
        INT_KEY.parse().unwrap(),
        Range {
            lt: Some(OrderedFloat(150.0)),
            gt: None,
            gte: None,
            lte: None,
        },
    ))
}

/// (bench id, filter) pairs: no filter, one indexed condition at various
/// selectivities, and two ANDed indexed conditions.
fn scenarios() -> Vec<(String, Option<Filter>)> {
    let mut scenarios = vec![("none".to_string(), None)];
    for pct in [1, 10, 50, 90] {
        let filter = Filter::new_must(flt_condition(pct));
        scenarios.push((format!("one-{pct:02}pct"), Some(filter)));
    }
    let two = Filter {
        should: None,
        min_should: None,
        must: Some(vec![flt_condition(50), int_condition()]),
        must_not: None,
    };
    scenarios.push(("two".to_string(), Some(two)));
    scenarios
}

fn filtered_hnsw_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered-hnsw-search");

    let (vector_holder, graph_layers) =
        fixture::make_cached_graph::<Metric>(NUM_VECTORS, DIM, M, EF_CONSTRUCT, USE_HEURISTIC);

    let deleted_points = common::bitvec::BitVec::repeat(false, NUM_VECTORS);

    // Real struct payload index over the same `0..NUM_VECTORS` id space, so the
    // filter context is the production `OptimizedFilter`, not a synthetic shim.
    let dir = Builder::new().prefix("filtered_hnsw").tempdir().unwrap();
    let struct_index = create_struct_payload_index(dir.path(), NUM_VECTORS, 42);
    let hw_counter = HardwareCounterCell::new();

    for (name, filter) in scenarios() {
        struct_index.with_view(|view| {
            for algorithm in [SearchAlgorithm::Hnsw, SearchAlgorithm::Acorn] {
                let algo_name = match algorithm {
                    SearchAlgorithm::Hnsw => "hnsw",
                    SearchAlgorithm::Acorn => "acorn",
                };
                let id = BenchmarkId::new(algo_name, &name);
                let mut rng = StdRng::seed_from_u64(42);
                group.bench_with_input(id, &name, |b, _| {
                    b.iter(|| {
                        let query = random_vector(&mut rng, DIM);
                        // The filter context is rebuilt per query, as in the
                        // production search path.
                        let filter_context = filter.as_ref().map(|f| {
                            SimpleCow::Owned(view.filter_context(f, &hw_counter).unwrap())
                        });
                        let scorer = FilteredScorer::new(
                            query.into(),
                            vector_holder.storage(),
                            vector_holder.quantized_vectors(),
                            filter_context,
                            &deleted_points,
                            HardwareCounterCell::new(),
                        )
                        .unwrap();

                        black_box(
                            graph_layers
                                .search(TOP, EF, algorithm, scorer, None, &DEFAULT_STOPPED)
                                .unwrap(),
                        );
                    })
                });
            }
        });
    }

    group.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = filtered_hnsw_benchmark
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = filtered_hnsw_benchmark
}

criterion_main!(benches);
