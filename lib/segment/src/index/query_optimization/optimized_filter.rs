use common::condition_checker::{CheckItem, ConditionChecker, Rest, Select};
use common::iterator_ext::IteratorExt;
use common::types::PointOffsetType;

use crate::common::operation_error::{OperationError, OperationResult};
use crate::index::condition_checker::ConditionCheckerEnum;

pub struct OptimizedFilter<'a> {
    /// At least one of those conditions should match, if not empty.
    should: Vec<ConditionCheckerEnum<'a>>,
    /// At least minimum amount of given conditions should match
    min_should: Vec<ConditionCheckerEnum<'a>>,
    min_should_count: usize,
    /// All conditions must match
    must: Vec<ConditionCheckerEnum<'a>>,
    /// All conditions must NOT match
    must_not: Vec<ConditionCheckerEnum<'a>>,

    scratch: Vec<usize>,
}

impl<'a> OptimizedFilter<'a> {
    pub fn new(
        should: Vec<ConditionCheckerEnum<'a>>,
        min_should: Vec<ConditionCheckerEnum<'a>>,
        min_should_count: usize,
        must: Vec<ConditionCheckerEnum<'a>>,
        must_not: Vec<ConditionCheckerEnum<'a>>,
    ) -> Self {
        OptimizedFilter {
            should,
            min_should,
            min_should_count,
            must,
            must_not,
            scratch: Vec::new(),
        }
    }

    /// A filter that matches a point iff the single given checker matches it.
    pub fn from_checker(checker: ConditionCheckerEnum<'a>) -> Self {
        Self::new(Vec::new(), Vec::new(), 0, vec![checker], Vec::new())
    }
}

impl ConditionChecker for OptimizedFilter<'_> {
    type Error = OperationError;

    fn check(&self, point_id: PointOffsetType) -> OperationResult<bool> {
        let OptimizedFilter {
            should,
            min_should,
            min_should_count,
            must,
            must_not,
            scratch: _,
        } = self;

        // `should`: at least one matches, if not empty.
        if !should.is_empty()
            && !should
                .iter()
                .try_any(|condition| condition.check(point_id))?
        {
            return Ok(false);
        }

        // `min_should`: at least `min_count` match.
        let mut remaining = *min_should_count;
        let mut min_should_iter = min_should.iter();
        while remaining > 0 {
            let Some(condition) = min_should_iter.next() else {
                // Not enough conditions to match `min_count`
                return Ok(false);
            };
            if condition.check(point_id)? {
                remaining -= 1;
            }
        }

        // `must`: all match.
        for condition in must {
            if !condition.check(point_id)? {
                return Ok(false);
            }
        }

        // `must_not`: none match.
        for condition in must_not {
            if condition.check(point_id)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn check_batched<K: CheckItem>(
        &mut self,
        ids: &mut [K],
        select: Select,
        rest: Rest,
    ) -> OperationResult<usize> {
        match select {
            Select::Match => self.select_match(ids, rest),
            Select::NonMatch => self.select_non_match(ids),
        }
    }
}

impl<'a> OptimizedFilter<'a> {
    fn select_match<K: CheckItem>(&mut self, ids: &mut [K], rest: Rest) -> OperationResult<usize> {
        let OptimizedFilter {
            should,
            min_should,
            min_should_count,
            must,
            must_not,
            scratch: min_should_bounds,
        } = self;

        // Survivors to the front. `rest` decides if the failing side survives.
        let mut hi = ids.len();

        // `must`: matches to the front (compact unless the drops must survive).
        for child in must.iter_mut() {
            hi = child.check_batched(&mut ids[..hi], Select::Match, rest)?;
        }

        // `must_not`: non-matches (survivors) to the front.
        for child in must_not.iter_mut() {
            hi = child.check_batched(&mut ids[..hi], Select::NonMatch, rest)?;
        }

        // `should`: union of matches. Every child but the last must keep its
        // non-matches for the next; the last may drop them unless `rest`.
        if !should.is_empty() {
            let last = should.len() - 1;
            let mut matched = 0;
            for (i, child) in should.iter_mut().enumerate() {
                let child_rest = if i != last { Rest::Keep } else { rest };
                matched += child.check_batched(&mut ids[matched..hi], Select::Match, child_rest)?;
            }
            hi = matched;
        }

        // `min_should`: counting sort, survivors (count >= k) to the front.
        if *min_should_count > 0 {
            let k = *min_should_count;
            if k > min_should.len() {
                hi = 0;
            } else {
                counting_sort(
                    min_should,
                    &mut ids[..hi],
                    k,
                    Select::Match,
                    min_should_bounds,
                )?;
                hi = min_should_bounds[0];
            }
        }

        Ok(hi)
    }

    fn select_non_match<K: CheckItem>(&mut self, ids: &mut [K]) -> OperationResult<usize> {
        let OptimizedFilter {
            should,
            min_should,
            min_should_count,
            must,
            must_not,
            scratch: min_should_bounds,
        } = self;

        // Failers to the front (`!F`, an OR of the negated clauses). Every
        // child partitions (survivors continue to the next clause).
        let mut f = 0;

        // `!must` = OR of `!cond`: accumulate the ids failing any `must`.
        for child in must.iter_mut() {
            f += child.check_batched(&mut ids[f..], Select::NonMatch, Rest::Keep)?;
        }

        // `!must_not` = OR of `cond`: accumulate the ids matching any `must_not`.
        for child in must_not.iter_mut() {
            f += child.check_batched(&mut ids[f..], Select::Match, Rest::Keep)?;
        }

        // `!should` = AND of `!cond`: narrow to the ids matching none.
        if !should.is_empty() {
            let mut u = ids.len();
            for child in should.iter_mut() {
                u = f + child.check_batched(&mut ids[f..u], Select::NonMatch, Rest::Keep)?;
            }
            f = u;
        }

        // `!min_should` = fewer than `k` match = at least `len - k + 1` don't.
        if *min_should_count > 0 {
            let k = *min_should_count;
            let len = min_should.len();
            if k > len {
                f = ids.len(); // unsatisfiable clause: every alive id fails it
            } else {
                counting_sort(
                    min_should,
                    &mut ids[f..],
                    len - k + 1,
                    Select::NonMatch,
                    min_should_bounds,
                )?;
                f += min_should_bounds[0];
            }
        }

        Ok(f)
    }
}

fn counting_sort<K: CheckItem>(
    children: &mut [ConditionCheckerEnum<'_>],
    ids: &mut [K],
    threshold: usize,
    select: Select,
    scratch: &mut Vec<usize>,
) -> OperationResult<()> {
    let m = ids.len();
    scratch.clear();
    scratch.resize(threshold, 0);
    for child in children.iter_mut() {
        for i in 0..threshold {
            let start = scratch[i];
            let end = if i + 1 < threshold { scratch[i + 1] } else { m };
            scratch[i] += child.check_batched(&mut ids[start..end], select, Rest::Keep)?;
        }
    }
    Ok(())
}
