// Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bucket-aligned prefill chunks.
//!
//! A step runs the compiled program of the smallest token bucket that holds its tokens (with data parallelism, its
//! busiest rank's tokens times the rank count), so a step a few tokens past a bucket pays for the whole next one. With
//! the model step's measured time per bucket ([`StepCost`], from [`EngineCore::set_step_cost`]) and chunked prefill
//! on, [`trim_prefill`] cuts a step past a bucket boundary down to the bucket below by taking the excess tokens from
//! its newest prefill chunks, when that is predicted to be faster: the lower bucket's time plus the cut tokens' share
//! of the bucket they would fill on their own must stay under [`TRIM_MARGIN`] of the time of the bucket the step
//! would run. The cut tokens are scheduled by the next step. A trimmed chunk keeps the pages allocated for it, and a
//! chunk that ended on its request's recurrent-snapshot boundary hands the boundary back to the request, so its next
//! chunk ends there and takes the save.
//!
//! [`EngineCore::set_step_cost`]: crate::engine::EngineCore::set_step_cost

use crate::engine::dp::RankState;
use crate::engine::request::{Handle, RequestTable};
use crate::engine::sched::schedule::{Entry, RankStep};
use crate::engine::sched::SchedParams;
use crate::engine::CoreError;

/// A trim is taken only when its predicted time stays under this share of the untrimmed step's.
pub const TRIM_MARGIN: f64 = 0.95;

/// The model step's measured time per compiled token bucket.
#[derive(Clone, Debug)]
pub struct StepCost {
    buckets: Vec<i64>,
    ms: Vec<f64>,
}

impl StepCost {
    /// A table over the compiled token `buckets` (ascending) and their step times `ms` in milliseconds.
    pub fn new(buckets: &[i64], ms: &[f64]) -> Result<Self, String> {
        if buckets.is_empty() || buckets.len() != ms.len() {
            return Err("the step-time table needs one time per bucket and at least one bucket".into());
        }
        if buckets[0] <= 0 || buckets.windows(2).any(|w| w[1] <= w[0]) {
            return Err("step-time buckets must be positive and strictly ascending".into());
        }
        if ms.iter().any(|&t| !t.is_finite() || t <= 0.0) {
            return Err("step times must be positive and finite".into());
        }
        Ok(Self {
            buckets: buckets.to_vec(),
            ms: ms.to_vec(),
        })
    }

    /// The index of the smallest bucket holding `tokens`, or `None` past the table.
    fn index(&self, tokens: i64) -> Option<usize> {
        let i = self.buckets.partition_point(|&b| b < tokens);
        (i < self.buckets.len()).then_some(i)
    }

    /// Whether cutting a step of bucket `index` down to the bucket below pays off when `deferred` tokens (counted as
    /// a step's tokens are: the busiest rank's times the rank count) move to later steps, each charged its share of
    /// the bucket those tokens would fill on their own.
    fn pays(&self, index: usize, deferred: i64) -> bool {
        let Some(j) = self.index(deferred) else {
            return false;
        };
        let later = deferred as f64 * self.ms[j] / self.buckets[j] as f64;
        self.ms[index - 1] + later < TRIM_MARGIN * self.ms[index]
    }
}

/// Whether `e` may give tokens back: a chunk of more than one token that verifies no drafts, of a request with no
/// async placeholders and no image spans.
fn trimmable(reqs: &RequestTable, e: &Entry) -> Result<bool, CoreError> {
    let r = reqs
        .get(e.handle)
        .ok_or_else(|| CoreError::Invariant(format!("stale handle {:#x} in a rank step", e.handle)))?;
    Ok(e.scheduled > 1
        && e.drafts.is_empty()
        && r.placeholders == 0
        && r.spans.is_empty()
        && e.computed + e.scheduled <= r.num_tokens())
}

/// A chunk that ended at `end` was trimmed: when it held its request's snapshot save, the save is cancelled and the
/// boundary handed back to the request, so its next chunk ends there and saves.
fn give_back_save(
    rank: &mut RankState,
    step: &mut RankStep,
    reqs: &mut RequestTable,
    h: Handle,
    end: i64,
) -> Result<(), CoreError> {
    let Some(pos) = step.saves.iter().position(|&(x, _)| x == h) else {
        return Ok(());
    };
    let (_, slot) = step.saves.remove(pos);
    let index = rank
        .snapshots
        .as_mut()
        .ok_or_else(|| CoreError::Invariant("a snapshot save without a snapshot index".into()))?;
    let key = index
        .pending_key(slot)
        .ok_or_else(|| CoreError::Invariant(format!("snapshot save slot {slot} is not pending")))?;
    index.cancel(slot);
    let r = reqs
        .get_mut(h)
        .ok_or_else(|| CoreError::Invariant(format!("stale handle {h:#x} in a snapshot save")))?;
    r.save_target = Some((end, key));
    Ok(())
}

/// Cuts a schedule (one [`RankStep`] per rank, before its tokens are applied) down to the token bucket below its own
/// when `cost` predicts that is faster (see the module docs). Only with chunked prefill. The tokens above the lower
/// bucket's share of every rank come from that rank's newest trimmable chunks, each keeping at least one token; a
/// schedule whose excess those chunks cannot cover is left as it is.
pub fn trim_prefill(
    cost: &StepCost,
    ranks: &mut [RankState],
    steps: &mut [RankStep],
    reqs: &mut RequestTable,
    p: &SchedParams,
) -> Result<(), CoreError> {
    if !p.chunked_prefill {
        return Ok(());
    }
    if ranks.len() != steps.len() {
        return Err(CoreError::Invariant(format!(
            "{} rank steps for {} ranks",
            steps.len(),
            ranks.len()
        )));
    }
    let dp = ranks.len() as i64;
    let tokens: Vec<i64> = steps
        .iter()
        .map(|s| s.entries.iter().map(|e| e.scheduled).sum())
        .collect();
    let busiest = tokens.iter().copied().max().unwrap_or(0);
    let Some(index) = cost.index(busiest * dp) else {
        return Ok(());
    };
    if index == 0 {
        return Ok(());
    }
    let cap = cost.buckets[index - 1] / dp;
    let mut deferred = 0i64;
    for (step, &t) in steps.iter().zip(&tokens) {
        let excess = t - cap;
        if excess <= 0 {
            continue;
        }
        let mut room = 0;
        for e in &step.entries {
            if trimmable(reqs, e)? {
                room += e.scheduled - 1;
            }
        }
        if room < excess {
            return Ok(());
        }
        deferred = deferred.max(excess);
    }
    if deferred <= 0 || !cost.pays(index, deferred * dp) {
        return Ok(());
    }
    for ((rank, step), &t) in ranks.iter_mut().zip(steps.iter_mut()).zip(&tokens) {
        let mut excess = t - cap;
        let mut k = step.entries.len();
        while excess > 0 && k > 0 {
            k -= 1;
            let e = &step.entries[k];
            if !trimmable(reqs, e)? {
                continue;
            }
            let (handle, end, cut) = (e.handle, e.computed + e.scheduled, excess.min(e.scheduled - 1));
            step.entries[k].scheduled -= cut;
            step.budget_remaining += cut;
            excess -= cut;
            give_back_save(rank, step, reqs, handle, end)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trim_pays_only_when_the_lower_bucket_plus_the_deferred_share_is_cheaper() {
        let c = StepCost::new(&[4, 8, 16, 32, 64], &[10.0, 10.0, 10.0, 40.0, 80.0]).unwrap();
        assert_eq!(c.index(23), Some(3));
        assert_eq!(c.index(65), None);
        // 32 -> 16: 10 + 7 * 10 / 8 = 18.75 < 0.95 * 40.
        assert!(c.pays(3, 7));
        // 16 -> 8: 10 + 2 * 10 / 4 = 15 is not under 0.95 * 10.
        assert!(!c.pays(2, 2));
        let flat = StepCost::new(&[4, 8, 16, 32], &[10.0; 4]).unwrap();
        assert!(!flat.pays(3, 7));
    }

    #[test]
    fn bad_tables_are_refused() {
        assert!(StepCost::new(&[], &[]).is_err());
        assert!(StepCost::new(&[8, 4], &[1.0, 1.0]).is_err());
        assert!(StepCost::new(&[4], &[0.0]).is_err());
        assert!(StepCost::new(&[4, 8], &[1.0]).is_err());
    }
}
