//! Shared property tests for the per-index [`ConditionChecker::check_batched`]
//! implementations (geo / numeric / full-text / map).
//!
//! Each index builds its checker through the common
//! [`PayloadFieldIndexRead::condition_checker`] seam, so a single randomized
//! property runner covers all of them: `check_batched` must partition an id
//! list exactly the way calling [`ConditionChecker::check`] per id would.

use common::bitvec::BitVec;
use common::condition_checker::{ConditionChecker as _, Rest, Select};
use common::counter::hardware_accumulator::HwMeasurementAcc;
use common::counter::hardware_counter::HardwareCounterCell;
use common::types::PointOffsetType;
use itertools::Itertools as _;
use ordered_float::OrderedFloat;
use rand::prelude::StdRng;
use rand::seq::SliceRandom as _;
use rand::{RngExt, SeedableRng};
use serde_json::{Value, json};
use tempfile::TempDir;

use crate::data_types::index::{TextIndexParams, TextIndexType, TokenizerType};
use crate::index::field_index::full_text_index::FullTextIndex;
use crate::index::field_index::geo_index::GeoIndex;
use crate::index::field_index::map_index::MapIndex;
use crate::index::field_index::numeric_index::NumericIndex;
use crate::index::field_index::{FieldIndex, FieldIndexBuilderTrait, PayloadFieldIndexRead as _};
use crate::json_path::JsonPath;
use crate::types::{FieldCondition, FloatPayloadType, GeoPoint, GeoRadius, Match, Range};

const NUM_POINTS: PointOffsetType = 128;
const KEY: &str = "field";
const TRIALS: usize = 16;

#[test]
fn geo() {
    // A ~6000 km radius around the origin catches a fraction of the globe-wide
    // points, giving a genuine mix of matches and non-matches.
    let condition = FieldCondition::new_geo_radius(
        JsonPath::new(KEY),
        GeoRadius {
            center: GeoPoint::new_unchecked(0.0, 0.0),
            radius: OrderedFloat(6_000_000.0),
        },
    );
    run(1, &condition, |rng, on_disk| {
        let rows = gen_rows(rng, |rng| {
            vec![json!({
                "lon": rng.random_range(-180.0..180.0),
                "lat": rng.random_range(-90.0..90.0),
            })]
        });
        let dir = tempfile::tempdir().unwrap();
        let index = build_index(
            GeoIndex::builder_mmap(dir.path(), on_disk, &empty_deleted()),
            &rows,
        );
        (FieldIndex::GeoIndex(index), dir)
    });
}

#[test]
fn numeric() {
    let condition = FieldCondition::new_range(
        JsonPath::new(KEY),
        Range {
            lt: Some(OrderedFloat(75.0)),
            gt: None,
            gte: Some(OrderedFloat(25.0)),
            lte: None,
        },
    );
    run(2, &condition, |rng, on_disk| {
        let rows = gen_rows(rng, |rng| match rng.random_bool(0.2) {
            true => vec![],
            false => vec![Value::from(rng.random_range(0.0..100.0))],
        });
        let dir = tempfile::tempdir().unwrap();
        let builder = NumericIndex::<FloatPayloadType, FloatPayloadType>::builder_mmap(
            dir.path(),
            on_disk,
            &empty_deleted(),
        );
        (FieldIndex::FloatIndex(build_index(builder, &rows)), dir)
    });
}

#[test]
fn full_text() {
    let words = ["alpha", "beta", "gamma", "delta"];
    let condition = FieldCondition::new_match(JsonPath::new(KEY), Match::new_text("alpha"));
    run(3, &condition, |rng, on_disk| {
        let rows = gen_rows(rng, |rng| {
            let doc = words.iter().filter(|_| rng.random_bool(0.5)).join(" ");
            vec![Value::from(doc)]
        });
        let config = TextIndexParams {
            r#type: TextIndexType::Text,
            tokenizer: TokenizerType::Word,
            lowercase: Some(true),
            ..Default::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let builder = FullTextIndex::builder_mmap(
            dir.path().to_path_buf(),
            config,
            on_disk,
            &empty_deleted(),
        );
        (FieldIndex::FullTextIndex(build_index(builder, &rows)), dir)
    });
}

#[test]
fn map_keyword() {
    let colors = ["red", "green", "blue"];
    let condition = FieldCondition::new_match(JsonPath::new(KEY), "red".to_string().into());
    run(4, &condition, |rng, on_disk| {
        let rows = gen_rows(rng, |rng| match rng.random_bool(0.2) {
            true => vec![],
            false => vec![Value::from(colors[rng.random_range(0..colors.len())])],
        });
        let dir = tempfile::tempdir().unwrap();
        let builder =
            MapIndex::<str>::builder_immutable(dir.path(), on_disk, &empty_deleted(), false);
        (FieldIndex::KeywordIndex(build_index(builder, &rows)), dir)
    });
}

/// Runs the [`assert_check_batched`] property against both mmap storage
/// variants: `on_disk = false` (the per-item `for_each_matching_value` default)
/// and `on_disk = true` (the batched `values_iter_batch` path).
fn run(
    seed: u64,
    condition: &FieldCondition,
    build: impl Fn(&mut StdRng, bool) -> (FieldIndex, TempDir),
) {
    let mut rng = StdRng::seed_from_u64(seed);
    for on_disk in [false, true] {
        let (index, _dir) = build(&mut rng, on_disk);
        assert_check_batched(&index, condition, &mut rng);
    }
}

/// The core property: `check_batched` yields the same partition as per-id
/// [`ConditionChecker::check`], for every `(Select, Rest)`, over random id lists.
fn assert_check_batched(index: &FieldIndex, condition: &FieldCondition, rng: &mut StdRng) {
    let mut checker = index
        .condition_checker(condition, HwMeasurementAcc::new())
        .unwrap()
        .unwrap();

    // Sanity: the condition genuinely splits the points, so the partitioning
    // path is actually exercised (not a trivial all-match / all-none checker).
    let matches = (0..NUM_POINTS)
        .filter(|&id| checker.check(id).unwrap())
        .count();
    assert!(
        0 < matches && matches < NUM_POINTS as usize,
        "degenerate condition"
    );

    for _ in 0..TRIALS {
        // A shuffled, random-length subset of the point ids.
        let mut ids: Vec<PointOffsetType> = (0..NUM_POINTS).collect();
        ids.shuffle(rng);
        ids.truncate(rng.random_range(0..=NUM_POINTS as usize));

        for select in [Select::Match, Select::NonMatch] {
            let expected = sorted(
                ids.iter()
                    .copied()
                    .filter(|&id| checker.check(id).unwrap() == select.is_match()),
            );

            for rest in [Rest::Keep, Rest::Drop] {
                let mut buf = ids.clone();
                let partition = checker.check_batched(&mut buf, select, rest).unwrap();

                // Left side holds exactly the ids landing on the selected side.
                assert_eq!(
                    sorted(buf[..partition].iter().copied()),
                    expected,
                    "select={select:?} rest={rest:?}",
                );
                // `Keep` preserves every input id somewhere in the buffer.
                if rest == Rest::Keep {
                    assert_eq!(
                        sorted(buf),
                        sorted(ids.iter().copied()),
                        "select={select:?} Keep"
                    );
                }
            }
        }
    }
}

fn gen_rows(rng: &mut StdRng, mut value: impl FnMut(&mut StdRng) -> Vec<Value>) -> Vec<Vec<Value>> {
    (0..NUM_POINTS).map(|_| value(rng)).collect()
}

fn build_index<B: FieldIndexBuilderTrait>(
    mut builder: B,
    rows: &[Vec<Value>],
) -> B::FieldIndexType {
    let hw_counter = HardwareCounterCell::new();
    builder.init().unwrap();
    for (id, values) in rows.iter().enumerate() {
        let refs: Vec<&Value> = values.iter().collect();
        builder
            .add_point(id as PointOffsetType, &refs, &hw_counter)
            .unwrap();
    }
    builder.finalize().unwrap()
}

fn empty_deleted() -> BitVec {
    BitVec::repeat(false, NUM_POINTS as usize)
}

fn sorted(ids: impl IntoIterator<Item = PointOffsetType>) -> Vec<PointOffsetType> {
    ids.into_iter().sorted_unstable().collect()
}
