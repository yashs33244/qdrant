//! Benchmark: per-point (old) vs batched (new) field-index reads.
//!
//! Compares the single-point read methods against the batched counterparts that
//! were added to amortise on-disk IO across a candidate batch:
//!   - numeric:   NumericIndexRead::check_values_any   vs  for_each_matching_value
//!   - map:       MapIndexRead::check_values_any       vs  for_each_matching_value
//!   - geo:       GeoIndexRead::check_values_any       vs  for_each_matching_value
//!   - full-text: InvertedIndex::check_match           vs  check_match_batch
//!
//! Backends (per index):
//!   - in-RAM   (in-memory index; its batched method is the trait DEFAULT — a plain
//!     per-point loop — so batched ≈ old there, by construction)
//!   - mmap     (OnDisk*<MmapFile>)
//!   - io_uring  (OnDisk*<IoUringFile>, Linux only)
//!
//! Dimensions:
//!   - selectivity s ∈ {10%, 50%, 90%}: fraction of points matching the predicate.
//!     For the value-scan indexes (numeric/map/geo) every candidate's values are
//!     read regardless of the predicate, so selectivity only flips a branch. For
//!     full-text it sets the *posting-list size*: the old path re-reads that
//!     posting per point, the batched path loads it once — so the win grows with s.
//!   - batch size N ∈ {1, 10, 100}: points handled per timed iteration.
//!     "old" calls the single-point method N times in a loop; "batched" handles
//!     all N in one call (same point ids, drawn from a shared random pool).
//!
//! Size: ~256 MiB on-disk payload by default; tune with `QDRANT_BENCH_MIB=<mib>`.
//! NB: 256 MiB usually fits in the OS page cache, so mmap/io_uring reads are
//! *warm* — this measures per-call + IO-batching overhead, not cold-disk latency.
//! The io_uring index is opened over the files the mmap build wrote, so its pages
//! are warm too. For cold IO, drop caches between runs and/or size beyond RAM.

use std::hint::black_box;

use common::bitvec::BitVec;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
#[cfg(target_os = "linux")]
use common::universal_io::{IoUringFile, IoUringFs};
use common::universal_io::{MmapFile, MmapFs, Populate};
use criterion::measurement::Measurement;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use rand::prelude::StdRng;
use rand::{RngExt, SeedableRng};
use tempfile::Builder;

#[cfg(not(target_os = "windows"))]
mod prof;

const NS: [usize; 3] = [1, 10, 100];
/// (label, matching fraction). Label is used in benchmark ids.
const SELECTIVITIES: [(&str, f64); 3] = [("10", 0.10), ("50", 0.50), ("90", 0.90)];
const QUERY_POOL: usize = 1 << 16;
const SEED: u64 = 0x5eed_1234;

/// Target on-disk size of the point→values payload, in bytes (env-tunable).
fn target_bytes() -> usize {
    let mib = std::env::var("QDRANT_BENCH_MIB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);
    mib * 1024 * 1024
}

fn seeded_rng() -> StdRng {
    StdRng::seed_from_u64(SEED)
}

/// A pool of random point ids in `[0, num_points)` to query.
fn query_pool(num_points: usize, rng: &mut StdRng) -> Vec<PointOffsetType> {
    (0..QUERY_POOL)
        .map(|_| rng.random_range(0..num_points) as PointOffsetType)
        .collect()
}

/// Next contiguous window of `n` ids from the pool, advancing `cursor`.
fn window<'a>(ids: &'a [PointOffsetType], cursor: &mut usize, n: usize) -> &'a [PointOffsetType] {
    let span = ids.len() - n + 1;
    let start = *cursor % span;
    *cursor = start + 1;
    &ids[start..start + n]
}

/// Register `old` and `batched` bench functions for every N in `NS`.
///
/// Each closure takes a slice of point ids and returns the match count (returned
/// so the optimiser can't elide the work).
fn register<M: Measurement>(
    group: &mut BenchmarkGroup<'_, M>,
    prefix: &str,
    ids: &[PointOffsetType],
    old: impl Fn(&[PointOffsetType]) -> u64,
    batched: impl Fn(&[PointOffsetType]) -> u64,
) {
    for &n in &NS {
        let mut cursor = 0usize;
        group.bench_function(format!("{prefix}/n{n:03}/old"), |b| {
            b.iter(|| {
                let w = window(ids, &mut cursor, n);
                black_box(old(w));
            })
        });

        let mut cursor = 0usize;
        group.bench_function(format!("{prefix}/n{n:03}/batched"), |b| {
            b.iter(|| {
                let w = window(ids, &mut cursor, n);
                black_box(batched(w));
            })
        });
    }
}

fn all(c: &mut Criterion) {
    let target = target_bytes();
    numeric::bench(c, target);
    map::bench(c, target);
    geo::bench(c, target);
    full_text::bench(c, target);
}

// ======================================================================
// numeric  (f64 uniform in [0,1), 1 value/point)
// ======================================================================
mod numeric {
    use segment::common::operation_error::OperationResult;
    use segment::index::field_index::numeric_index::NumericIndexRead;
    use segment::index::field_index::numeric_index::mutable_numeric_index::InMemoryNumericIndex;
    use segment::index::field_index::numeric_index::on_disk_numeric_index::OnDiskNumericIndex;

    use super::*;

    // point→values layout per point: 16 B range entry (MmapRange) + 8 B value.
    const BYTES_PER_POINT: usize = 16 + 8;

    // P(v > 1 - frac) = frac for v ~ U(0,1): match fraction == frac.
    fn threshold(frac: f64) -> f64 {
        1.0 - frac
    }

    fn make_in_memory(num_points: usize, rng: &mut StdRng) -> InMemoryNumericIndex<f64> {
        (0..num_points)
            .map(|i| Ok((i as PointOffsetType, rng.random_range(0.0..1.0f64))))
            .collect::<OperationResult<InMemoryNumericIndex<f64>>>()
            .unwrap()
    }

    pub fn bench(c: &mut Criterion, target_bytes: usize) {
        let num_points = target_bytes / BYTES_PER_POINT;
        let mut rng = seeded_rng();
        let ids = query_pool(num_points, &mut rng);
        let hw = HardwareCounterCell::new();
        let deleted = BitVec::repeat(false, num_points);

        let mut group = c.benchmark_group("numeric");

        // Build the in-memory index once; in-RAM benches borrow it, then it's
        // consumed by the mmap build.
        let in_memory = make_in_memory(num_points, &mut rng);

        // ---- in-RAM (baseline). InMemoryNumericIndex exposes only the single
        // inherent check; the batched method is the trait default loop (== old). ----
        for (sel, frac) in SELECTIVITIES {
            let t = threshold(frac);
            let mut cursor = 0usize;
            for &n in &NS {
                group.bench_function(format!("inram/s{sel}/n{n:03}/old"), |b| {
                    b.iter(|| {
                        let w = window(&ids, &mut cursor, n);
                        let mut count = 0u64;
                        for &id in w {
                            if in_memory.check_values_any(id, |v| *v > t) {
                                count += 1;
                            }
                        }
                        black_box(count);
                    })
                });
            }
        }

        // ---- mmap (build, consuming the in-memory index) ----
        let dir = Builder::new().prefix("num-bench").tempdir().unwrap();
        let mmap = OnDiskNumericIndex::<f64, MmapFile>::build(
            &MmapFs,
            in_memory,
            dir.path(),
            Populate::Blocking,
            &deleted,
        )
        .unwrap();

        // ---- io_uring (open the same files; Linux only) ----
        #[cfg(target_os = "linux")]
        let uring = OnDiskNumericIndex::<f64, IoUringFile>::open(
            &IoUringFs,
            dir.path(),
            Populate::No,
            &deleted,
        )
        .unwrap()
        .unwrap();

        for (sel, frac) in SELECTIVITIES {
            let t = threshold(frac);

            macro_rules! register_index {
                ($name:expr, $idx:expr) => {
                    register(
                        &mut group,
                        &format!("{}/s{sel}", $name),
                        &ids,
                        |w| {
                            let mut count = 0u64;
                            for &id in w {
                                if $idx.check_values_any(id, |v| *v > t, &hw) {
                                    count += 1;
                                }
                            }
                            count
                        },
                        |w| {
                            let mut count = 0u64;
                            $idx.for_each_matching_value(
                                w.iter().map(|&id| ((), id)),
                                |v| *v > t,
                                &hw,
                                |_, matched| {
                                    if matched {
                                        count += 1;
                                    }
                                },
                            )
                            .unwrap();
                            count
                        },
                    );
                };
            }

            register_index!("mmap", mmap);
            #[cfg(target_os = "linux")]
            register_index!("io_uring", uring);
        }

        group.finish();
    }
}

// ======================================================================
// map  (keyword / str, 1 value/point, 1000-cardinality categorical)
// ======================================================================
mod map {
    use ahash::HashMap;
    use ecow::EcoString;
    use segment::index::field_index::map_index::MapIndex;
    use segment::index::field_index::map_index::on_disk_map_index::OnDiskMapIndex;
    use segment::index::field_index::map_index::read_ops::MapIndexRead;
    use segment::types::Memory;

    use super::*;

    // 13-byte values ("category-NNNN") → 16 B range + (4 + 13) B value ≈ 33 B/point.
    const BYTES_PER_POINT: usize = 16 + 4 + 13;
    const NUM_DISTINCT: usize = 1_000;

    // Values are zero-padded so lexicographic order == numeric order; `v <= cutoff`
    // then matches a `frac` prefix of the (uniformly assigned) categories.
    fn cutoff(frac: f64) -> String {
        let last = ((frac * NUM_DISTINCT as f64) as usize).saturating_sub(1);
        format!("category-{last:04}")
    }

    #[allow(clippy::type_complexity)]
    fn gen_data(
        num_points: usize,
        rng: &mut StdRng,
    ) -> (
        Vec<Vec<EcoString>>,
        HashMap<EcoString, Vec<PointOffsetType>>,
    ) {
        let vocab: Vec<EcoString> = (0..NUM_DISTINCT)
            .map(|i| EcoString::from(format!("category-{i:04}")))
            .collect();

        let mut point_to_values: Vec<Vec<EcoString>> = Vec::with_capacity(num_points);
        let mut values_to_points: HashMap<EcoString, Vec<PointOffsetType>> = HashMap::default();
        for point_id in 0..num_points as PointOffsetType {
            let v = vocab[rng.random_range(0..NUM_DISTINCT)].clone();
            values_to_points
                .entry(v.clone())
                .or_default()
                .push(point_id);
            point_to_values.push(vec![v]);
        }
        (point_to_values, values_to_points)
    }

    pub fn bench(c: &mut Criterion, target_bytes: usize) {
        let num_points = target_bytes / BYTES_PER_POINT;
        let mut rng = seeded_rng();
        let ids = query_pool(num_points, &mut rng);
        let hw = HardwareCounterCell::new();
        let deleted = BitVec::repeat(false, num_points);

        let mut group = c.benchmark_group("map");

        let (p2v, v2p) = gen_data(num_points, &mut rng);

        // ---- build the shared on-disk files once (via mmap) ----
        let dir = Builder::new().prefix("map-bench").tempdir().unwrap();
        let mmap = OnDiskMapIndex::<str, MmapFile>::build(
            &MmapFs,
            dir.path(),
            p2v,
            v2p,
            Populate::Blocking,
            &deleted,
            false,
        )
        .unwrap();
        // in-RAM: same files loaded fully into RAM (batched == default loop).
        let ram = MapIndex::<str>::new_immutable(dir.path(), Memory::Pinned, &deleted)
            .unwrap()
            .unwrap();
        #[cfg(target_os = "linux")]
        let uring = OnDiskMapIndex::<str, IoUringFile>::open(
            &IoUringFs,
            dir.path(),
            Populate::No,
            &deleted,
        )
        .unwrap()
        .unwrap();

        for (sel, frac) in SELECTIVITIES {
            let cut = cutoff(frac);
            let pred = |v: &str| v <= cut.as_str();

            macro_rules! register_index {
                ($name:expr, $idx:expr) => {
                    register(
                        &mut group,
                        &format!("{}/s{sel}", $name),
                        &ids,
                        |w| {
                            let mut count = 0u64;
                            for &id in w {
                                if $idx.check_values_any(id, &hw, pred).unwrap() {
                                    count += 1;
                                }
                            }
                            count
                        },
                        |w| {
                            let mut count = 0u64;
                            $idx.for_each_matching_value(
                                w.iter().map(|&id| ((), id)),
                                &hw,
                                pred,
                                |_, matched| {
                                    if matched {
                                        count += 1;
                                    }
                                },
                            )
                            .unwrap();
                            count
                        },
                    );
                };
            }

            register_index!("inram", ram);
            register_index!("mmap", mmap);
            #[cfg(target_os = "linux")]
            register_index!("io_uring", uring);
        }

        group.finish();
    }
}

// ======================================================================
// geo  (1 GeoPoint/point, uniform global lat/lon)
// ======================================================================
mod geo {
    use segment::index::field_index::geo_index::GeoIndexRead;
    use segment::index::field_index::geo_index::mutable_geo_index::InMemoryGeoIndex;
    use segment::index::field_index::geo_index::on_disk_geo_index::OnDiskGeoIndex;
    use segment::types::{CheckGeoPoint, GeoBoundingBox, GeoPoint};

    use super::*;

    // 16 B range entry + 16 B GeoPoint (2×f64).
    const BYTES_PER_POINT: usize = 16 + 16;

    // Full longitude, a latitude band of height `frac·180` centred on the equator.
    // Points are uniform in (lon, lat), so the matching fraction ≈ frac.
    fn bbox(frac: f64) -> GeoBoundingBox {
        let half_lat = frac * 90.0;
        GeoBoundingBox {
            top_left: GeoPoint::new_unchecked(-180.0, half_lat),
            bottom_right: GeoPoint::new_unchecked(180.0, -half_lat),
        }
    }

    fn make_in_memory(num_points: usize, rng: &mut StdRng) -> InMemoryGeoIndex {
        let hw = HardwareCounterCell::new();
        let mut idx = InMemoryGeoIndex::new();
        for i in 0..num_points as PointOffsetType {
            let lon = rng.random_range(-180.0..180.0f64);
            let lat = rng.random_range(-90.0..90.0f64);
            idx.add_many_geo_points(i, vec![GeoPoint::new_unchecked(lon, lat)], &hw)
                .unwrap();
        }
        idx
    }

    pub fn bench(c: &mut Criterion, target_bytes: usize) {
        let num_points = target_bytes / BYTES_PER_POINT;
        let mut rng = seeded_rng();
        let ids = query_pool(num_points, &mut rng);
        let hw = HardwareCounterCell::new();
        let deleted = BitVec::repeat(false, num_points);

        let mut group = c.benchmark_group("geo");

        let in_memory = make_in_memory(num_points, &mut rng);

        // ---- in-RAM (InMemoryGeoIndex implements GeoIndexRead; batched == default
        // loop). Runs before the mmap build consumes the in-memory index. ----
        for (sel, frac) in SELECTIVITIES {
            let bb = bbox(frac);
            register(
                &mut group,
                &format!("inram/s{sel}"),
                &ids,
                |w| {
                    let mut count = 0u64;
                    let check = |p: &GeoPoint| bb.check_point(p);
                    for &id in w {
                        if GeoIndexRead::check_values_any(&in_memory, id, &hw, &check).unwrap() {
                            count += 1;
                        }
                    }
                    count
                },
                |w| {
                    let mut count = 0u64;
                    in_memory
                        .for_each_matching_value(
                            w.iter().map(|&id| ((), id)),
                            &hw,
                            |p: &GeoPoint| bb.check_point(p),
                            |_, matched| {
                                if matched {
                                    count += 1;
                                }
                            },
                        )
                        .unwrap();
                    count
                },
            );
        }

        // ---- mmap (build, consuming the in-memory index) ----
        let dir = Builder::new().prefix("geo-bench").tempdir().unwrap();
        let mmap = OnDiskGeoIndex::<MmapFile>::build(
            &MmapFs,
            in_memory,
            dir.path(),
            Populate::Blocking,
            &deleted,
        )
        .unwrap();
        #[cfg(target_os = "linux")]
        let uring =
            OnDiskGeoIndex::<IoUringFile>::open(&IoUringFs, dir.path(), Populate::No, &deleted)
                .unwrap()
                .unwrap();

        for (sel, frac) in SELECTIVITIES {
            let bb = bbox(frac);

            macro_rules! register_index {
                ($name:expr, $idx:expr) => {
                    register(
                        &mut group,
                        &format!("{}/s{sel}", $name),
                        &ids,
                        |w| {
                            let mut count = 0u64;
                            for &id in w {
                                if $idx
                                    .check_values_any(id, &hw, |p: &GeoPoint| bb.check_point(p))
                                    .unwrap()
                                {
                                    count += 1;
                                }
                            }
                            count
                        },
                        |w| {
                            let mut count = 0u64;
                            $idx.for_each_matching_value(
                                w.iter().map(|&id| ((), id)),
                                &hw,
                                |p: &GeoPoint| bb.check_point(p),
                                |_, matched| {
                                    if matched {
                                        count += 1;
                                    }
                                },
                            )
                            .unwrap();
                            count
                        },
                    );
                };
            }

            register_index!("mmap", mmap);
            #[cfg(target_os = "linux")]
            register_index!("io_uring", uring);
        }

        group.finish();
    }
}

// ======================================================================
// full-text  (ids-only postings; batched = load posting once vs per-point)
// ======================================================================
//
// `OnDiskInvertedIndex` (where the real `check_match_batch` lives) is under a
// private module, so we drive it through the public `OnDiskFullTextIndex<S>`
// wrapper (forwards verbatim) and the `FullTextIndex` enum for in-RAM.
//
// Selectivity is set by injecting marker tokens `s10`/`s50`/`s90` into 10/50/90%
// of documents; querying a marker matches that fraction and, crucially, makes the
// posting that big — so the per-point re-read cost of the old path scales with s.
//
// Sizing here is the postings file, not point→values. NB: this is the heaviest
// index to build (millions of docs) — lower `QDRANT_BENCH_MIB` while iterating.
mod full_text {
    use segment::data_types::index::{TextIndexParams, TextIndexType, TokenizerType};
    use segment::index::field_index::full_text_index::FullTextIndex;
    use segment::index::field_index::full_text_index::full_text_index_read::FullTextIndexRead;
    use segment::index::field_index::full_text_index::on_disk_text_index::OnDiskFullTextIndex;
    use segment::index::field_index::{FieldIndexBuilderTrait, ValueIndexer};
    use segment::types::Memory;

    use super::*;

    const VOCAB: usize = 10_000;
    const TOKENS_PER_DOC: usize = 24;
    // postings.dat ≈ num_docs * TOKENS_PER_DOC * ~1.6 B/id (random tokens) plus the
    // marker postings (Σ frac · num_docs); rough — rescale via QDRANT_BENCH_MIB.
    const BYTES_PER_DOC: usize = TOKENS_PER_DOC * 16 / 10;

    fn config() -> TextIndexParams {
        TextIndexParams {
            r#type: TextIndexType::Text,
            tokenizer: TokenizerType::Word,
            phrase_matching: Some(false), // ids-only postings — smallest/fastest
            memory: Some(Memory::Cold),
            ..TextIndexParams::default()
        }
    }

    fn gen_docs(num_docs: usize, rng: &mut StdRng) -> Vec<String> {
        let mut docs: Vec<String> = (0..num_docs)
            .map(|_| {
                let mut toks: Vec<String> = (0..TOKENS_PER_DOC)
                    .map(|_| format!("w{}", rng.random_range(0..VOCAB)))
                    .collect();
                // Inject selectivity markers independently per document.
                for (label, frac) in SELECTIVITIES {
                    if rng.random_range(0.0..1.0f64) < frac {
                        toks.push(format!("s{label}"));
                    }
                }
                toks.join(" ")
            })
            .collect();
        // Guarantee every marker exists (so parse never returns None, even tiny).
        if let Some(first) = docs.first_mut() {
            first.insert_str(0, "s10 s50 s90 ");
        }
        docs
    }

    /// `is_on_disk = true` writes on-disk files (reopened typed); `false` yields an
    /// in-RAM immutable index.
    fn build(
        dir: std::path::PathBuf,
        cfg: &TextIndexParams,
        docs: &[String],
        is_on_disk: bool,
    ) -> FullTextIndex {
        let hw = HardwareCounterCell::new();
        let empty = BitVec::new();
        let mut builder = FullTextIndex::builder_mmap(dir, cfg.clone(), is_on_disk, &empty);
        FieldIndexBuilderTrait::init(&mut builder).unwrap();
        for (id, doc) in docs.iter().enumerate() {
            builder
                .add_many(id as PointOffsetType, vec![doc.clone()], &hw)
                .unwrap();
        }
        FieldIndexBuilderTrait::finalize(builder).unwrap()
    }

    /// Register old/batched for one backend across all selectivities. `parse`
    /// produces the marker query for this backend (token ids are per-index).
    fn register_index<I: FullTextIndexRead>(
        group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
        name: &str,
        index: &I,
        ids: &[PointOffsetType],
        hw: &HardwareCounterCell,
    ) {
        for (sel, _frac) in SELECTIVITIES {
            let q = index
                .parse_text_query(&format!("s{sel}"), hw)
                .unwrap()
                .unwrap();
            register(
                group,
                &format!("{name}/s{sel}"),
                ids,
                |w| {
                    let mut count = 0u64;
                    for &id in w {
                        if index.check_match(&q, id).unwrap() {
                            count += 1;
                        }
                    }
                    count
                },
                |w| {
                    let mut count = 0u64;
                    index
                        .check_match_batch(&q, w.iter().map(|&id| ((), id)), |_, matched| {
                            if matched {
                                count += 1;
                            }
                        })
                        .unwrap();
                    count
                },
            );
        }
    }

    pub fn bench(c: &mut Criterion, target_bytes: usize) {
        let num_docs = target_bytes / BYTES_PER_DOC;
        let mut rng = seeded_rng();
        let ids = query_pool(num_docs, &mut rng);
        let hw = HardwareCounterCell::new();
        let empty = BitVec::new();
        let cfg = config();
        let docs = gen_docs(num_docs, &mut rng);

        let mut group = c.benchmark_group("fulltext");

        // ---- on-disk files built once; opened as mmap + io_uring ----
        let dir = Builder::new().prefix("ft-bench").tempdir().unwrap();
        let _written = build(dir.path().to_path_buf(), &cfg, &docs, true);

        let mmap = OnDiskFullTextIndex::<MmapFile>::open(
            &MmapFs,
            dir.path().to_path_buf(),
            cfg.clone(),
            Populate::Blocking,
            &empty,
        )
        .unwrap()
        .unwrap();
        register_index(&mut group, "mmap", &mmap, &ids, &hw);

        #[cfg(target_os = "linux")]
        {
            let uring = OnDiskFullTextIndex::<IoUringFile>::open(
                &IoUringFs,
                dir.path().to_path_buf(),
                cfg.clone(),
                Populate::No,
                &empty,
            )
            .unwrap()
            .unwrap();
            register_index(&mut group, "io_uring", &uring, &ids, &hw);
        }

        // ---- in-RAM (immutable index loaded into RAM; default-loop batched) ----
        let ram_dir = Builder::new().prefix("ft-ram").tempdir().unwrap();
        let ram = build(ram_dir.path().to_path_buf(), &cfg, &docs, false);
        register_index(&mut group, "inram", &ram, &ids, &hw);

        group.finish();
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
