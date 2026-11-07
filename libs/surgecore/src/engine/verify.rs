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

//! The load-aware verification budget of speculative steps.
//!
//! Each synchronous schedule decides how many of every running request's drafts the step verifies, from none to all
//! of them. The inputs are a step-time table (milliseconds per compiled token bucket of the model step, measured by
//! the runner after compile) and per-request acceptance estimates. The decision starts from the step's tokens without
//! drafts and adds draft positions in decreasing estimated cumulative acceptance, one token each, and keeps the count
//! that maximises the expected emitted tokens per millisecond of the step's bucket. Verifying nothing is a valid
//! answer. Only the table's buckets count: a count whose step would need a bucket past the table is never chosen.
//!
//! The estimates are conditional acceptance rates per draft position (the chance that draft `j` is accepted once
//! drafts `0..j` were): an exponential moving average per request, started from a global prior that is itself a moving
//! average of each step's mean observation. A commit observes, for each request that verified drafts, the positions up
//! to and including its first rejection; positions it did not observe relax toward the prior. The table gets an online
//! correction added to every bucket: a measured step's time minus the table's time, averaged over the first
//! [`WARMUP_SAMPLES`] steps and then a moving average; a corrected time never drops below half the table's.

use crate::engine::dp::RankState;
use crate::engine::request::RequestTable;
use crate::engine::sched::SchedParams;
use crate::engine::CoreError;

/// Measured steps whose mean sets the step-time correction before it becomes a moving average.
pub const WARMUP_SAMPLES: u32 = 8;

/// Sums and counts of one step's acceptance observations per draft position.
pub type StepObservations = Vec<(f64, f64)>;

/// The verification budget: the step-time table, its online correction and the global acceptance prior.
#[derive(Clone, Debug)]
pub struct VerifyBudget {
    buckets: Vec<i64>,
    ms: Vec<f64>,
    offset: f64,
    samples: u32,
    correction: f64,
    alpha: f64,
    relax: f64,
    prior: Vec<f64>,
}

/// One request that may verify drafts.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    /// Its data-parallel rank.
    pub rank: usize,
    /// The most drafts it can verify.
    pub drafts: usize,
    /// Its conditional acceptance per draft position.
    pub accept: &'a [f64],
}

impl VerifyBudget {
    /// A budget over the compiled token `buckets` (ascending) and their step times `ms`, for `k` drafts per request.
    /// `prior` starts every position's acceptance estimate; `alpha` weighs each observation in the moving averages;
    /// `correction` weighs each measured step time in the table's correction (0 turns it off).
    pub fn new(buckets: &[i64], ms: &[f64], k: usize, prior: f64, alpha: f64, correction: f64) -> Result<Self, String> {
        if buckets.is_empty() || buckets.len() != ms.len() {
            return Err("the step-time table needs one time per bucket and at least one bucket".into());
        }
        if buckets[0] <= 0 || buckets.windows(2).any(|w| w[1] <= w[0]) {
            return Err("step-time buckets must be positive and strictly ascending".into());
        }
        if ms.iter().any(|&t| !t.is_finite() || t <= 0.0) {
            return Err("step times must be positive and finite".into());
        }
        if k == 0 {
            return Err("a verification budget needs speculative tokens".into());
        }
        for (name, v) in [("prior", prior), ("alpha", alpha), ("correction", correction)] {
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("{name} must lie in [0, 1], got {v}"));
            }
        }
        Ok(Self {
            buckets: buckets.to_vec(),
            ms: ms.to_vec(),
            offset: 0.0,
            samples: 0,
            correction,
            alpha,
            relax: alpha / 4.0,
            prior: vec![prior; k],
        })
    }

    /// Drafts per request the budget was built for.
    pub fn k(&self) -> usize {
        self.prior.len()
    }

    /// The global conditional acceptance prior per draft position.
    pub fn prior(&self) -> &[f64] {
        &self.prior
    }

    /// The online correction added to every bucket's time (ms).
    pub fn offset(&self) -> f64 {
        self.offset
    }

    fn bucket(&self, tokens: i64) -> Option<usize> {
        let i = self.buckets.partition_point(|&b| b < tokens);
        (i < self.buckets.len()).then_some(i)
    }

    /// The predicted time (ms) of a step of `tokens` tokens: its bucket's table time plus the correction, never below
    /// half the table time; `None` past the table.
    pub fn step_ms(&self, tokens: i64) -> Option<f64> {
        self.bucket(tokens)
            .map(|i| (self.ms[i] + self.offset).max(0.5 * self.ms[i]))
    }

    /// Feeds one measured step time back into the correction. The first [`WARMUP_SAMPLES`] samples set it to their
    /// mean difference from the table (the host's share of a step, which the table does not hold); later samples
    /// move it by `correction`, and those above four times the prediction (stalls) are ignored.
    pub fn observe_step(&mut self, tokens: i64, measured_ms: f64) {
        let (Some(i), Some(pred)) = (self.bucket(tokens), self.step_ms(tokens)) else {
            return;
        };
        if self.correction <= 0.0 || !measured_ms.is_finite() || measured_ms <= 0.0 {
            return;
        }
        let w = if self.samples < WARMUP_SAMPLES {
            1.0 / (self.samples + 1) as f64
        } else if measured_ms > 4.0 * pred {
            return;
        } else {
            self.correction
        };
        self.samples += 1;
        self.offset += w * ((measured_ms - self.ms[i]) - self.offset);
    }

    /// Updates a request's estimates `est` after it verified `verified` drafts and `accepted` were accepted, and
    /// adds the observations to `step` (one `(sum, count)` per position). An empty `est` starts from the prior.
    /// Positions it did not observe (past its first rejection, or every position when it verified none) relax toward
    /// the prior, so a request whose share fell to zero is tried again.
    pub fn observe_request(&self, est: &mut Vec<f64>, verified: usize, accepted: usize, step: &mut StepObservations) {
        if est.len() < self.k() {
            est.clone_from(&self.prior);
        }
        let accepted = accepted.min(verified);
        let seen = (accepted + 1).min(verified).min(self.k());
        for j in 0..self.k() {
            if j < seen {
                let obs = if j < accepted { 1.0 } else { 0.0 };
                est[j] += self.alpha * (obs - est[j]);
                step[j].0 += obs;
                step[j].1 += 1.0;
            } else {
                est[j] += self.relax * (self.prior[j] - est[j]);
            }
        }
    }

    /// Moves the prior toward one step's mean observation per position.
    pub fn observe_prior(&mut self, step: &StepObservations) {
        for (p, &(sum, n)) in self.prior.iter_mut().zip(step) {
            if n > 0.0 {
                *p += self.alpha * (sum / n - *p);
            }
        }
    }

    /// The draft count each candidate verifies. `base_rows` holds each rank's step tokens without drafts,
    /// `base_emitted` the tokens the step emits without drafts, `dp` the ranks (a step's bucket is `dp` times its
    /// busiest rank's tokens) and `cap` a rank's token budget. A step whose base already lies past the table keeps
    /// every draft.
    pub fn decide(
        &self,
        base_rows: &[i64],
        base_emitted: f64,
        cands: &[Candidate<'_>],
        dp: i64,
        cap: i64,
    ) -> Vec<usize> {
        let tokens = |rows: &[i64]| dp * rows.iter().copied().max().unwrap_or(0);
        let mut rows = base_rows.to_vec();
        let Some(t0) = self.step_ms(tokens(&rows)) else {
            return cands.iter().map(|c| c.drafts).collect();
        };
        // Per rank, cumulative acceptance per (candidate, position) in decreasing order; a stable sort keeps each
        // candidate's positions in order, so the chosen positions are prefixes.
        let mut items: Vec<Vec<(f64, usize)>> = vec![Vec::new(); rows.len()];
        for (i, c) in cands.iter().enumerate() {
            let mut cum = 1.0;
            for j in 0..c.drafts {
                cum *= c.accept.get(j).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                items[c.rank].push((cum, i));
            }
        }
        for list in &mut items {
            list.sort_by(|a, b| b.0.total_cmp(&a.0));
        }
        let mut next = vec![0usize; rows.len()];
        let mut picks: Vec<usize> = Vec::new();
        let mut expected = base_emitted;
        let mut best = (expected / t0, 0usize);
        loop {
            // The highest next position on a rank below the busiest one (it leaves the step's bucket alone), else
            // the highest next position anywhere. With one rank this is the global order.
            let busiest = rows.iter().copied().max().unwrap_or(0);
            let mut choice: Option<(bool, f64, usize)> = None;
            for (r, list) in items.iter().enumerate() {
                if next[r] >= list.len() || rows[r] >= cap {
                    continue;
                }
                let key = (rows[r] < busiest, list[next[r]].0, r);
                if choice.is_none_or(|c| (key.0, key.1) > (c.0, c.1)) {
                    choice = Some(key);
                }
            }
            let Some((_, cum, r)) = choice else {
                break;
            };
            rows[r] += 1;
            let Some(t) = self.step_ms(tokens(&rows)) else {
                break;
            };
            picks.push(items[r][next[r]].1);
            next[r] += 1;
            expected += cum;
            if expected / t > best.0 * (1.0 + 1e-9) {
                best = (expected / t, picks.len());
            }
        }
        let mut shares = vec![0usize; cands.len()];
        for &i in &picks[..best.1] {
            shares[i] += 1;
        }
        shares
    }
}

/// Cuts every running request's drafts to its share of this step's verification budget and returns the drafts the
/// runner should produce for the next step (the largest share of the same decision with `k` drafts for every request
/// that samples this step; `k` when no request samples).
pub fn apply(v: &VerifyBudget, ranks: &[RankState], reqs: &mut RequestTable, p: &SchedParams) -> Result<i64, CoreError> {
    let k = v.k();
    let mut base = vec![0i64; ranks.len()];
    let mut emitted = 0.0;
    // (handle, rank, drafts it can verify, drafts it could verify next step)
    let mut rows: Vec<(u64, usize, usize, usize)> = Vec::new();
    for rank in ranks {
        for &h in &rank.running {
            let r = reqs
                .get(h)
                .ok_or_else(|| CoreError::Invariant(format!("stale handle {h:#x} in a rank list")))?;
            let room = p.max_model_len - 1 - r.computed;
            let mut n = r.num_tokens() + r.placeholders - r.computed;
            if p.long_prefill_threshold > 0 {
                n = n.min(p.long_prefill_threshold);
            }
            n = n.min(room).max(0);
            if n == 0 {
                continue;
            }
            base[rank.index] += n;
            if r.computed + n < r.num_tokens() {
                continue;
            }
            emitted += 1.0;
            let spare = (room - n).max(0) as usize;
            rows.push((h, rank.index, r.drafts.len().min(spare), k.min(spare)));
        }
    }
    for b in &mut base {
        *b = (*b).min(p.token_budget);
    }
    let dp = ranks.len() as i64;
    let (shares, next) = {
        let accept = |h: u64| -> &[f64] {
            match reqs.get(h) {
                Some(r) if r.accept.len() >= k => &r.accept,
                _ => v.prior(),
            }
        };
        let now: Vec<Candidate<'_>> = rows
            .iter()
            .map(|&(h, rank, d, _)| Candidate {
                rank,
                drafts: d,
                accept: accept(h),
            })
            .collect();
        let shares = v.decide(&base, emitted, &now, dp, p.token_budget);
        let later: Vec<Candidate<'_>> = rows
            .iter()
            .map(|&(h, rank, _, d)| Candidate {
                rank,
                drafts: d,
                accept: accept(h),
            })
            .collect();
        let next = if later.is_empty() {
            k
        } else {
            v.decide(&base, emitted, &later, dp, p.token_budget)
                .into_iter()
                .max()
                .unwrap_or(0)
        };
        (shares, next)
    };
    for (&(h, _, _, _), &s) in rows.iter().zip(&shares) {
        if let Some(r) = reqs.get_mut(h) {
            r.drafts.truncate(s);
        }
    }
    Ok(next as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sched::commit::Output;
    use crate::engine::sched::Policy;
    use crate::engine::{request::NewRequest, request::Sampling, EngineConfig, EngineCore};
    use crate::kv::group::{Group, Kind};

    fn budget(buckets: &[i64], ms: &[f64]) -> VerifyBudget {
        VerifyBudget::new(buckets, ms, 3, 0.8, 0.5, 0.5).unwrap()
    }

    fn cands(accept: &[[f64; 3]]) -> Vec<Candidate<'_>> {
        accept
            .iter()
            .map(|a| Candidate {
                rank: 0,
                drafts: 3,
                accept: a,
            })
            .collect()
    }

    #[test]
    fn fills_the_free_bucket_and_stops_where_time_grows() {
        // 4 decode rows; up to 8 rows cost the same 10 ms, 16 rows cost 20 ms.
        let v = budget(&[4, 8, 16], &[10.0, 10.0, 20.0]);
        let acc = [[0.9; 3]; 4];
        // 8 rows: (4 + 4 * 0.9) / 10 = 0.76 beats 16 rows: (4 + 4 * (0.9 + 0.81 + 0.729)) / 20 = 0.688.
        assert_eq!(v.decide(&[4], 4.0, &cands(&acc), 1, 64), vec![1, 1, 1, 1]);
    }

    #[test]
    fn verifies_nothing_when_verification_costs_more_than_it_returns() {
        let v = budget(&[4, 8], &[10.0, 30.0]);
        let acc = [[0.5; 3]; 4];
        assert_eq!(v.decide(&[4], 4.0, &cands(&acc), 1, 64), vec![0, 0, 0, 0]);
    }

    #[test]
    fn shares_follow_the_estimates() {
        let v = budget(&[4, 6, 16], &[10.0, 11.0, 40.0]);
        let acc = [[0.95; 3], [0.2; 3], [0.2; 3], [0.2; 3]];
        // 6 rows: (4 + 0.95 + 0.9025) / 11 is best; the 7th row needs the 40 ms bucket.
        assert_eq!(v.decide(&[4], 4.0, &cands(&acc), 1, 64), vec![2, 0, 0, 0]);
    }

    #[test]
    fn never_needs_a_bucket_past_the_table_or_the_budget() {
        let v = budget(&[4, 6], &[10.0, 10.0]);
        let acc = [[1.0; 3]; 4];
        assert_eq!(v.decide(&[4], 4.0, &cands(&acc), 1, 64), vec![2, 0, 0, 0]);
        let v = budget(&[4, 64], &[10.0, 10.0]);
        assert_eq!(v.decide(&[4], 4.0, &cands(&acc), 1, 5), vec![1, 0, 0, 0]);
        // A base already past the table keeps every draft.
        assert_eq!(v.decide(&[100], 4.0, &cands(&acc), 1, 1000), vec![3, 3, 3, 3]);
    }

    #[test]
    fn data_parallel_steps_use_the_busiest_rank() {
        // 2 ranks, 2 rows each; the step's bucket is 2 x the busiest rank's rows.
        let v = budget(&[4, 6, 8], &[10.0, 10.0, 30.0]);
        let acc = [[0.9; 3]; 4];
        let mut c = cands(&acc);
        c[2].rank = 1;
        c[3].rank = 1;
        // One draft on rank 0 makes the step 6 tokens (free); a second row on rank 0 makes it 8 (30 ms), so each
        // rank verifies one draft: 2 x 3 rows = 6 tokens.
        assert_eq!(v.decide(&[2, 2], 4.0, &c, 2, 64), vec![1, 0, 1, 0]);
    }

    #[test]
    fn estimates_update_up_to_the_first_rejection() {
        let mut v = budget(&[4], &[10.0]);
        let mut est = Vec::new();
        let mut step = vec![(0.0, 0.0); 3];
        v.observe_request(&mut est, 3, 1, &mut step);
        close(&est, &[0.9, 0.4, 0.8]);
        assert_eq!(step, vec![(1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]);
        let mut other = vec![0.5; 3];
        v.observe_request(&mut other, 2, 2, &mut step);
        // Position 2 was not verified: it relaxes toward the prior by alpha / 4.
        close(&other, &[0.75, 0.75, 0.5 + 0.125 * 0.3]);
        v.observe_prior(&step);
        close(v.prior(), &[0.9, 0.65, 0.8]);
    }

    fn close(have: &[f64], want: &[f64]) {
        assert_eq!(have.len(), want.len());
        for (h, w) in have.iter().zip(want) {
            assert!((h - w).abs() < 1e-12, "{have:?} != {want:?}");
        }
    }

    #[test]
    fn measured_step_times_correct_the_table() {
        let mut v = budget(&[4, 8], &[10.0, 20.0]);
        // Warm-up: the mean difference of the first samples, however far from the table.
        v.observe_step(4, 50.0);
        assert_eq!(v.offset(), 40.0);
        for _ in 1..WARMUP_SAMPLES {
            v.observe_step(8, 20.0);
        }
        assert_eq!(v.offset(), 5.0);
        assert_eq!(v.step_ms(8), Some(25.0));
        v.observe_step(4, 100.0);
        assert_eq!(v.offset(), 5.0, "an outlier after the warm-up is ignored");
        v.observe_step(9, 5.0);
        assert_eq!(v.offset(), 5.0, "a step past the table is ignored");
        v.observe_step(4, 17.0);
        assert_eq!(v.offset(), 6.0, "then a moving average with weight `correction`");
        assert!(VerifyBudget::new(&[8, 4], &[1.0, 1.0], 3, 0.8, 0.5, 0.5).is_err());
        assert!(VerifyBudget::new(&[4], &[1.0], 0, 0.8, 0.5, 0.5).is_err());
    }

    fn core() -> EngineCore {
        EngineCore::new(&EngineConfig {
            max_rows: 4,
            max_model_len: 64,
            num_pages: 64,
            dp_size: 1,
            groups: vec![Group {
                kind: Kind::Full,
                page_size: 4,
                type_index: 0,
            }],
            token_budget: 32,
            long_prefill_threshold: 0,
            chunked_prefill: false,
            token_safety_margin: None,
            policy: Policy::Fcfs,
            prefix_caching: false,
            snapshot_slots: 0,
            async_scheduling: false,
            req_buckets: Vec::new(),
            num_spec_tokens: 3,
            use_eagle: false,
            include_finished_set: false,
        })
        .unwrap()
    }

    fn admit(core: &mut EngineCore, n: usize) -> Vec<u64> {
        let prompt = [1, 2, 3, 4];
        let reqs: Vec<NewRequest<'_>> = (0..n)
            .map(|i| NewRequest {
                prompt: &prompt,
                max_tokens: 32,
                min_tokens: 0,
                eos: None,
                ignore_eos: false,
                priority: 0,
                arrival: i as f64,
                flags: 0,
                stop_ids: &[],
                spans: &[],
                dp_rank: None,
                hash_extra: 0,
                sampling: Sampling::default(),
            })
            .collect();
        core.add(&reqs).unwrap()
    }

    /// Commits `plan` with one sampled token per accepted draft plus one, and `new` drafts for every entry.
    fn commit(core: &mut EngineCore, plan: &crate::engine::plan::Plan, accepted: &[i32], new: usize) {
        let mut sampled = Vec::new();
        let mut offsets = vec![0];
        let mut drafts = Vec::new();
        let mut doff = vec![0];
        for &a in accepted {
            sampled.extend(std::iter::repeat_n(7, (a.max(0) + 1) as usize));
            offsets.push(sampled.len() as i32);
            drafts.extend(std::iter::repeat_n(9, new));
            doff.push(drafts.len() as i32);
        }
        let out = Output {
            sampled: &sampled,
            sampled_offsets: &offsets,
            present: None,
            accepted: Some(accepted),
            drafts: Some((&drafts, &doff)),
        };
        core.commit(plan, &out).unwrap();
    }

    #[test]
    fn the_core_cuts_drafts_to_the_budget_and_learns_from_commits() {
        let mut core = core();
        let hs = admit(&mut core, 4);
        let plan = core.schedule().unwrap();
        assert_eq!(plan.num_draft_tokens_next, 3, "no budget: the configured drafts");
        commit(&mut core, &plan, &[-1; 4], 3);
        // Without a budget every request verifies its 3 drafts.
        let plan = core.schedule().unwrap();
        assert_eq!(plan.scheduled, vec![4; 4]);
        commit(&mut core, &plan, &[3, 3, 3, 3], 3);

        // 4 rows are free up to 8; 16 rows cost 4x. Prior 0.8 gives one draft per request now and next step.
        core.set_verify_budget(&[4, 8, 16], &[10.0, 10.0, 40.0], 0.8, 0.5, 0.0)
            .unwrap();
        let plan = core.schedule().unwrap();
        assert_eq!(plan.scheduled, vec![2; 4]);
        assert_eq!(plan.draft_offsets, vec![0, 1, 2, 3, 4]);
        assert_eq!(plan.num_draft_tokens_next, 1);
        // Request 0 accepts, the others reject: request 0 alone verifies more next.
        commit(&mut core, &plan, &[1, 0, 0, 0], 3);
        close(&core.request(hs[0]).unwrap().accept, &[0.9, 0.8, 0.8]);
        close(&core.request(hs[1]).unwrap().accept, &[0.4, 0.8, 0.8]);
        // Cumulative estimates 0.9, 0.72, 0.576 (request 0) and 0.4, 0.32, 0.256: the four free rows go to request
        // 0's three positions and request 1's first.
        let plan = core.schedule().unwrap();
        assert_eq!(plan.scheduled, vec![4, 2, 1, 1]);
        commit(&mut core, &plan, &[3, 0, 0, 0], 3);
        core.check().unwrap();

        // Verification that costs more than it returns: no drafts, and the runner skips the drafter.
        core.set_verify_budget(&[4, 8], &[10.0, 100.0], 0.8, 0.5, 0.0).unwrap();
        let plan = core.schedule().unwrap();
        assert_eq!(plan.scheduled, vec![1; 4]);
        assert!(plan.draft_tokens.is_empty());
        assert_eq!(plan.num_draft_tokens_next, 0);
        assert_eq!(plan.flags & crate::engine::plan::PLAN_HAS_VERIFY, 0);
        commit(&mut core, &plan, &[-1; 4], 0);
        core.check().unwrap();
    }
}
