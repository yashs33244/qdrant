//! Benchmark: `OptimizedFilter::check_batched` (in-place partition).
//!
//! Leaves are `TestBits`: a `BitVec` of `[0, NUM_POINTS)` bits set at random with
//! match probability `p`, a first-class `ConditionCheckerEnum::TestBits` variant
//! (no boxed `dyn`). Nested filters go in via `ConditionCheckerEnum::Filter`.
//!
//! Each iteration copies the window into a fresh buffer, then partitions it in
//! place with `check_batched(&mut buf, keep = true)` and `black_box`es the
//! resulting `[0..r]` passers.
//!
//! Dimensions:
//!   - selectivity `p` ∈ {0.1, 0.5, 0.9}: per-leaf match probability.
//!   - batch size N ∈ {1, 5, 10, 50, 100} ids per timed iteration, drawn as a
//!     moving window over a shared random id pool.

use std::hint::black_box;

use bitvec::vec::BitVec;
use common::condition_checker::{ConditionChecker, Rest, Select};
use common::types::PointOffsetType;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::prelude::StdRng;
use rand::{RngExt, SeedableRng};
use segment::index::condition_checker::{ConditionCheckerEnum, TestBits};
use segment::index::query_optimization::optimized_filter::OptimizedFilter;

#[cfg(not(target_os = "windows"))]
mod prof;

const NUM_POINTS: usize = 1 << 16;
const QUERY_POOL: usize = 1 << 15;
const NS: [usize; 5] = [1, 5, 10, 50, 100];
/// Per-leaf match probabilities.
const PS: [f64; 3] = [0.1, 0.5, 0.9];
const POOL_SEED: u64 = 0x5eed_f117;
/// Every shape rebuilds from this seed, so its leaves are deterministic.
const FILTER_SEED: u64 = 0x0f11_7e57;

// ======================================================================
// filter shapes
// ======================================================================

type Filter = OptimizedFilter<'static>;
type Builder = Box<dyn Fn(&mut StdRng, f64) -> Filter>;

fn random_bits(rng: &mut StdRng, p: f64) -> BitVec {
    (0..NUM_POINTS)
        .map(|_| rng.random_range(0.0..1.0) < p)
        .collect()
}

fn leaf(rng: &mut StdRng, p: f64) -> ConditionCheckerEnum<'static> {
    ConditionCheckerEnum::TestBits(TestBits(random_bits(rng, p)))
}

fn leaves(rng: &mut StdRng, n: usize, p: f64) -> Vec<ConditionCheckerEnum<'static>> {
    (0..n).map(|_| leaf(rng, p)).collect()
}

fn filter(
    should: Vec<ConditionCheckerEnum<'static>>,
    min_should: Vec<ConditionCheckerEnum<'static>>,
    min_should_count: usize,
    must: Vec<ConditionCheckerEnum<'static>>,
    must_not: Vec<ConditionCheckerEnum<'static>>,
) -> Filter {
    OptimizedFilter::new(should, min_should, min_should_count, must, must_not)
}

fn nest(inner: Filter) -> ConditionCheckerEnum<'static> {
    ConditionCheckerEnum::Filter(inner)
}

fn builders() -> Vec<(&'static str, Builder)> {
    vec![
        (
            "should_1",
            Box::new(|r, p| filter(leaves(r, 1, p), vec![], 0, vec![], vec![])),
        ),
        (
            "should_2",
            Box::new(|r, p| filter(leaves(r, 2, p), vec![], 0, vec![], vec![])),
        ),
        (
            "should_3",
            Box::new(|r, p| filter(leaves(r, 3, p), vec![], 0, vec![], vec![])),
        ),
        (
            "must_1",
            Box::new(|r, p| filter(vec![], vec![], 0, leaves(r, 1, p), vec![])),
        ),
        (
            "must_2",
            Box::new(|r, p| filter(vec![], vec![], 0, leaves(r, 2, p), vec![])),
        ),
        (
            "must_3",
            Box::new(|r, p| filter(vec![], vec![], 0, leaves(r, 3, p), vec![])),
        ),
        (
            "must_not_1",
            Box::new(|r, p| filter(vec![], vec![], 0, vec![], leaves(r, 1, p))),
        ),
        (
            "must_not_2",
            Box::new(|r, p| filter(vec![], vec![], 0, vec![], leaves(r, 2, p))),
        ),
        (
            "must_not_3",
            Box::new(|r, p| filter(vec![], vec![], 0, vec![], leaves(r, 3, p))),
        ),
        (
            "min_should_2of3",
            Box::new(|r, p| filter(vec![], leaves(r, 3, p), 2, vec![], vec![])),
        ),
        (
            "combined",
            Box::new(|r, p| {
                filter(
                    leaves(r, 1, p),
                    leaves(r, 3, p),
                    2,
                    leaves(r, 1, p),
                    leaves(r, 1, p),
                )
            }),
        ),
        // Nested: three single-leaf `must` filters.
        (
            "nested_must_3",
            Box::new(|r, p| {
                let sub = |r: &mut StdRng, p: f64| {
                    nest(filter(vec![], vec![], 0, leaves(r, 1, p), vec![]))
                };
                filter(
                    vec![],
                    vec![],
                    0,
                    vec![sub(r, p), sub(r, p), sub(r, p)],
                    vec![],
                )
            }),
        ),
        // Nested: `should` of two 2-leaf `should` filters.
        (
            "nested_should",
            Box::new(|r, p| {
                let sub = |r: &mut StdRng, p: f64| {
                    nest(filter(leaves(r, 2, p), vec![], 0, vec![], vec![]))
                };
                filter(vec![sub(r, p), sub(r, p)], vec![], 0, vec![], vec![])
            }),
        ),
        // Nested: `must` chain 3 levels deep.
        (
            "nested_deep_must",
            Box::new(|r, p| {
                let l3 = filter(vec![], vec![], 0, leaves(r, 1, p), vec![]);
                let l2 = filter(vec![], vec![], 0, vec![nest(l3)], vec![]);
                filter(vec![], vec![], 0, vec![nest(l2)], vec![])
            }),
        ),
        // Nested + mixed: must_not + min_should under filters.
        (
            "nested_mixed",
            Box::new(|r, p| {
                let a = nest(filter(vec![], vec![], 0, vec![], leaves(r, 2, p)));
                let b = nest(filter(vec![], leaves(r, 3, p), 2, vec![], vec![]));
                filter(vec![b], vec![], 0, vec![a], vec![])
            }),
        ),
        // A filter nested under `must_not` — this is what makes the inner filter be
        // evaluated with `keep = false` (the inverted / no-rotate path).
        (
            "mn_must_2",
            Box::new(|r, p| {
                let inner = filter(vec![], vec![], 0, leaves(r, 2, p), vec![]);
                filter(vec![], vec![], 0, vec![], vec![nest(inner)])
            }),
        ),
        (
            "mn_should_2",
            Box::new(|r, p| {
                let inner = filter(leaves(r, 2, p), vec![], 0, vec![], vec![]);
                filter(vec![], vec![], 0, vec![], vec![nest(inner)])
            }),
        ),
        (
            "mn_min_should",
            Box::new(|r, p| {
                let inner = filter(vec![], leaves(r, 3, p), 2, vec![], vec![]);
                filter(vec![], vec![], 0, vec![], vec![nest(inner)])
            }),
        ),
    ]
}

// ======================================================================
// harness
// ======================================================================

fn build_filter(build: &Builder, p: f64) -> Filter {
    build(&mut StdRng::seed_from_u64(FILTER_SEED), p)
}

/// A pool of random ids in `[0, NUM_POINTS)`.
fn query_pool(rng: &mut StdRng) -> Vec<PointOffsetType> {
    (0..QUERY_POOL)
        .map(|_| rng.random_range(0..NUM_POINTS) as PointOffsetType)
        .collect()
}

/// Next contiguous window of `n` ids, advancing `cursor`.
fn window<'a>(ids: &'a [PointOffsetType], cursor: &mut usize, n: usize) -> &'a [PointOffsetType] {
    let span = ids.len() - n + 1;
    let start = *cursor % span;
    *cursor = start + 1;
    &ids[start..start + n]
}

fn all(c: &mut Criterion) {
    let ids = query_pool(&mut StdRng::seed_from_u64(POOL_SEED));
    let mut group = c.benchmark_group("optimized_filter");

    for (name, build) in builders() {
        for &p in &PS {
            let pp = (p * 100.0) as u32;
            verify(name, &build, p, &ids);

            for &n in &NS {
                let mut filt = build_filter(&build, p);
                let mut buf: Vec<PointOffsetType> = Vec::with_capacity(n);
                let mut cursor = 0;
                group.bench_function(format!("{name}/p{pp:02}/n{n:03}"), |b| {
                    b.iter(|| {
                        let w = window(&ids, &mut cursor, n);
                        buf.clear();
                        buf.extend_from_slice(w);
                        let r = filt
                            .check_batched(&mut buf, Select::Match, Rest::Drop)
                            .unwrap();
                        black_box(&buf[..r]);
                    })
                });
            }
        }
    }

    group.finish();
}

/// Assert `check_batched` (both polarities) agrees with the per-id `check`.
fn verify(name: &str, build: &Builder, p: f64, ids: &[PointOffsetType]) {
    for &n in &NS {
        let w = &ids[..n];
        let base = build_filter(build, p);
        let expect: Vec<bool> = w.iter().map(|&id| base.check(id).unwrap()).collect();

        let sorted = |keep: bool| {
            let mut v: Vec<PointOffsetType> = w
                .iter()
                .zip(&expect)
                .filter(|&(_, &e)| e == keep)
                .map(|(&id, _)| id)
                .collect();
            v.sort_unstable();
            v
        };

        for select in [Select::Match, Select::NonMatch] {
            for rest in [Rest::Keep, Rest::Drop] {
                let mut f = build_filter(build, p);
                let mut buf = w.to_vec();
                let r = f.check_batched(&mut buf, select, rest).unwrap();
                buf[..r].sort_unstable();
                assert_eq!(
                    buf[..r],
                    sorted(select == Select::Match)[..],
                    "{name}/p{p}/n{n}: {select:?} {rest:?}"
                );
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(prof::FlamegraphProfiler::new(100));
    targets = all
}

#[cfg(target_os = "windows")]
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = all
}

criterion_main!(benches);
