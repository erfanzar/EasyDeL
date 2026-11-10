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

//! The engine core: request table, scheduler, data-parallel ranks and step records.
//!
//! [`EngineCore`] owns every live request and one [`dp::RankState`] per data-parallel rank. Its calls are
//! [`add`](EngineCore::add) (admission, with routing to a rank), [`schedule`](EngineCore::schedule) (one [`Plan`]),
//! [`commit`](EngineCore::commit) (a plan's sampled tokens, in dispatch order, giving [`Events`]) and
//! [`finish`](EngineCore::finish) (aborts and stop strings). The decisions are those of eSurge's former Python
//! scheduler (single-rank, async and data-parallel), pinned by goldens. Synchronous speculative steps carry each
//! request's drafts, and [`verify`] can cut them per step to a load-aware verification budget
//! ([`set_verify_budget`](EngineCore::set_verify_budget)).
//!
//! This module stays free of PyO3; its bindings live in the extension module's `engine` binding file.

pub mod dp;
pub mod plan;
pub mod record;
pub mod request;
pub mod sched;
pub mod verify;

#[cfg(test)]
mod async_spec;
#[cfg(test)]
mod golden;

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

use crate::kv::group::Group;
use crate::kv::snapshot::SnapshotIndex;
use crate::kv::{KvConfig, KvCore, KvError};
use dp::{rank_buckets, route, RankState};
use plan::{
    Events, Plan, KIND_NEW, KIND_RUNNING, PLAN_ASYNC_SAFE, PLAN_HAS_PREFILL, PLAN_HAS_VERIFY, PLAN_PAGES_CHANGED,
    PLAN_STRUCTURED_PENDING, STATS_GLOBAL,
};
use request::{FinishReason, Handle, NewRequest, Request, RequestTable, Status, FLAG_GRAMMAR};
use sched::commit::{commit_plan, Output};
use sched::trim::{trim_prefill, StepCost};
use sched::{free_request, schedule::schedule_rank, Policy, Queue, SchedParams, TokenBudget};
use verify::VerifyBudget;

/// An engine-core error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// A bad argument; nothing changed.
    Value(String),
    /// An internal check failed; the core is no longer usable.
    Invariant(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(m) | Self::Invariant(m) => f.write_str(m),
        }
    }
}

impl From<KvError> for CoreError {
    fn from(err: KvError) -> Self {
        Self::Invariant(format!("KV core: {err}"))
    }
}

/// Configuration numbers of an [`EngineCore`].
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Running rows over all ranks.
    pub max_rows: usize,
    /// Longest sequence.
    pub max_model_len: i64,
    /// KV pages over all ranks (divisible by `dp_size` when it is above 1).
    pub num_pages: usize,
    /// Data-parallel ranks.
    pub dp_size: usize,
    /// Cache groups, in order (at least one).
    pub groups: Vec<Group>,
    /// Per-step token budget of each rank.
    pub token_budget: i64,
    /// Per-step prefill cap (0 is off).
    pub long_prefill_threshold: i64,
    /// Whether a prefill may span steps when it does not fit the remaining budget.
    pub chunked_prefill: bool,
    /// Tokens reserved per running request in a KV-aware budget (`None` is off).
    pub token_safety_margin: Option<i64>,
    /// The policy.
    pub policy: Policy,
    /// Prefix caching.
    pub prefix_caching: bool,
    /// Recurrent-snapshot slots per rank (0 is off; on only with prefix caching).
    pub snapshot_slots: i64,
    /// Async scheduling with output placeholders.
    pub async_scheduling: bool,
    /// Request-count buckets.
    pub req_buckets: Vec<i64>,
    /// Speculative tokens per step (0 is off): the drafts a request may carry, and the placeholder drafts an async
    /// step gives each request that samples.
    pub num_spec_tokens: i64,
    /// EAGLE-style drafting: every allocation reserves `num_spec_tokens` lookahead slots and a prefix hit drops its
    /// last page (only with `num_spec_tokens > 0`).
    pub use_eagle: bool,
    /// Report the requests freed since the previous commit in [`Events::finished_all`].
    pub include_finished_set: bool,
    /// Data-parallel admission alignment, in steps (0 is off; used only with `dp_size > 1`). A data-parallel step
    /// runs one program sized by its busiest rank, so a prefill admitted on one rank makes every rank run a
    /// prefill-sized step. While a rank with waiting requests has no free row, no rank admits, so the ranks start
    /// their prefills in the same step. A rank with waiting requests and nothing running admits at once, and no hold
    /// lasts more than this many consecutive steps.
    pub dp_admission_hold: i64,
}

/// The request-count ladder of a single-rank core: the positive buckets, sorted and deduplicated, then `max_running`
/// appended when it is not the last one (as the former scheduler built it).
fn single_rank_buckets(ladder: &[i64], max_running: usize) -> Vec<usize> {
    let mut out: Vec<usize> = ladder.iter().filter(|&&b| b > 0).map(|&b| b as usize).collect();
    out.sort_unstable();
    out.dedup();
    if out.last() != Some(&max_running) {
        out.push(max_running);
    }
    out
}

/// The engine core.
pub struct EngineCore {
    params: SchedParams,
    num_groups: usize,
    pages_per_rank: usize,
    reqs: RequestTable,
    ranks: Vec<RankState>,
    inflight: VecDeque<u64>,
    next_plan: u64,
    verify: Option<VerifyBudget>,
    /// `(plan id, step tokens, time schedule returned)` of the synchronous plan whose step time is measured.
    timing: Option<(u64, i64, Instant)>,
    /// The longest admission hold in steps (see [`EngineConfig::dp_admission_hold`]); 0 is off.
    admission_hold: i64,
    /// Consecutive steps admission has been held so far.
    admission_held: i64,
    /// The model step's measured time per token bucket; with it, steps trim prefill chunks to the bucket below when
    /// that is predicted to be faster (see [`sched::trim`]).
    step_cost: Option<StepCost>,
}

fn value<T>(msg: impl Into<String>) -> Result<T, CoreError> {
    Err(CoreError::Value(msg.into()))
}

impl EngineCore {
    /// A core over `cfg`.
    pub fn new(cfg: &EngineConfig) -> Result<Self, CoreError> {
        if cfg.groups.is_empty() {
            return value("at least one cache group is needed");
        }
        if cfg.dp_size == 0 || cfg.max_rows == 0 || cfg.num_pages == 0 {
            return value("dp_size, max_rows and num_pages must be positive");
        }
        if cfg.token_budget <= 0 || cfg.max_model_len <= 1 {
            return value("token_budget must be positive and max_model_len above 1");
        }
        if cfg.dp_size > 1 && cfg.num_pages % cfg.dp_size != 0 {
            return value(format!(
                "num_pages={} must be divisible by dp_size={}",
                cfg.num_pages, cfg.dp_size
            ));
        }
        let pages_per_rank = cfg.num_pages / cfg.dp_size;
        let rows_per_rank = (cfg.max_rows / cfg.dp_size).max(1);
        if cfg.num_spec_tokens < 0 {
            return value("num_spec_tokens must not be negative");
        }
        if cfg.dp_admission_hold < 0 {
            return value("dp_admission_hold must not be negative");
        }
        let page_size = cfg.groups[0].page_size;
        let eagle = cfg.use_eagle && cfg.num_spec_tokens > 0;
        let lookahead = if eagle { cfg.num_spec_tokens } else { 0 };
        let mut max_chunk = cfg.token_budget;
        if cfg.long_prefill_threshold > 0 && cfg.long_prefill_threshold < max_chunk {
            max_chunk = cfg.long_prefill_threshold;
        }
        max_chunk += lookahead;
        let params = SchedParams {
            max_model_len: cfg.max_model_len,
            token_budget: cfg.token_budget,
            long_prefill_threshold: cfg.long_prefill_threshold.max(0),
            chunked_prefill: cfg.chunked_prefill,
            policy: cfg.policy,
            async_scheduling: cfg.async_scheduling,
            page_size,
            max_chunk,
            lookahead,
            num_spec_tokens: cfg.num_spec_tokens,
        };
        let ladder = if cfg.dp_size > 1 {
            rank_buckets(&cfg.req_buckets, rows_per_rank)
        } else {
            single_rank_buckets(&cfg.req_buckets, rows_per_rank)
        };
        let mut ranks = Vec::with_capacity(cfg.dp_size);
        for index in 0..cfg.dp_size {
            let kv = KvCore::new(KvConfig {
                num_pages: pages_per_rank,
                groups: cfg.groups.clone(),
                max_model_len: cfg.max_model_len,
                caching: cfg.prefix_caching,
                eagle,
            })
            .map_err(|e| CoreError::Value(e.to_string()))?;
            let snapshots = if cfg.prefix_caching && cfg.snapshot_slots > 0 {
                Some(
                    SnapshotIndex::new(cfg.snapshot_slots, index as i64 * cfg.snapshot_slots)
                        .map_err(|e| CoreError::Value(e.to_string()))?,
                )
            } else {
                None
            };
            let budget = cfg
                .token_safety_margin
                .map(|m| TokenBudget::new(cfg.token_budget, page_size, m));
            let mut rank = RankState {
                index,
                waiting: Queue::new(cfg.policy),
                running: Vec::new(),
                kv,
                snapshots,
                budget,
                finished: Vec::new(),
                max_running: rows_per_rank,
                buckets: ladder.clone(),
                current_bucket: 0,
                page_offset: (index * pages_per_rank) as u32,
                finished_set: cfg.include_finished_set.then(Vec::new),
            };
            rank.current_bucket = rank.select_bucket(0);
            ranks.push(rank);
        }
        Ok(Self {
            params,
            num_groups: cfg.groups.len(),
            pages_per_rank,
            reqs: RequestTable::default(),
            ranks,
            inflight: VecDeque::new(),
            next_plan: 0,
            verify: None,
            timing: None,
            admission_hold: if cfg.dp_size > 1 { cfg.dp_admission_hold } else { 0 },
            admission_held: 0,
            step_cost: None,
        })
    }

    /// Whether this step admits waiting requests: always, unless data-parallel admission alignment holds them.
    ///
    /// Admission is held while a rank with waiting requests has no free row and another such rank has one, unless a
    /// rank with waiting requests runs nothing or the hold has lasted `admission_hold` consecutive steps.
    fn admission_gate(&mut self) -> bool {
        if self.admission_hold <= 0 {
            return true;
        }
        let (mut any_free, mut blocked, mut idle) = (false, false, false);
        for rank in self.ranks.iter().filter(|rank| !rank.waiting.is_empty()) {
            let free = rank.running.len() < rank.max_running;
            any_free |= free;
            blocked |= !free;
            idle |= rank.running.is_empty();
        }
        let admit = !any_free || !blocked || idle || self.admission_held >= self.admission_hold;
        self.admission_held = if admit { 0 } else { self.admission_held + 1 };
        admit
    }

    /// Turns on the load-aware verification budget (see [`verify`]) for synchronous speculative steps: `buckets`
    /// are the compiled token buckets of the model step (ascending) and `ms` their measured step times. Replaces a
    /// previous budget (per-request estimates are kept).
    pub fn set_verify_budget(
        &mut self,
        buckets: &[i64],
        ms: &[f64],
        prior: f64,
        alpha: f64,
        correction: f64,
    ) -> Result<(), CoreError> {
        if self.params.num_spec_tokens <= 0 {
            return value("a verification budget needs num_spec_tokens > 0");
        }
        if self.params.async_scheduling {
            return value("the verification budget runs on synchronous steps only");
        }
        let k = self.params.num_spec_tokens as usize;
        self.verify = Some(VerifyBudget::new(buckets, ms, k, prior, alpha, correction).map_err(CoreError::Value)?);
        self.timing = None;
        Ok(())
    }

    /// The verification budget, when one is set.
    pub fn verify_budget(&self) -> Option<&VerifyBudget> {
        self.verify.as_ref()
    }

    /// Turns on bucket-aligned prefill chunks (see [`sched::trim`]; only with chunked prefill): `buckets` are the
    /// compiled token buckets of the model step (ascending) and `ms` their measured step times. Replaces a previous
    /// table.
    pub fn set_step_cost(&mut self, buckets: &[i64], ms: &[f64]) -> Result<(), CoreError> {
        self.step_cost = Some(StepCost::new(buckets, ms).map_err(CoreError::Value)?);
        Ok(())
    }

    /// Data-parallel ranks.
    pub fn dp_size(&self) -> usize {
        self.ranks.len()
    }

    /// Cache groups.
    pub fn num_groups(&self) -> usize {
        self.num_groups
    }

    /// KV pages per rank.
    pub fn pages_per_rank(&self) -> usize {
        self.pages_per_rank
    }

    /// Plans returned by `schedule` with entries and not yet committed.
    pub fn num_inflight(&self) -> usize {
        self.inflight.len()
    }

    /// The live request of `h`.
    pub fn request(&self, h: Handle) -> Option<&Request> {
        self.reqs.get(h)
    }

    /// The rank of the live request `h`.
    pub fn rank_of(&self, h: Handle) -> Option<usize> {
        self.reqs.get(h).map(|r| r.rank)
    }

    /// `(rank, status, num_prompt, num_tokens, computed, placeholders, cached_tokens)` of the live request `h`;
    /// status 0 waiting, 1 running, 2 preempted.
    pub fn request_info(&self, h: Handle) -> Option<[i64; 7]> {
        self.reqs.get(h).map(|r| {
            let status = match r.status {
                Status::Running => 1,
                Status::Preempted => 2,
                _ => 0,
            };
            [
                r.rank as i64,
                status,
                r.num_prompt,
                r.num_tokens(),
                r.computed,
                r.placeholders,
                r.cached_tokens,
            ]
        })
    }

    /// The speculative drafts the live request `h` carries into its next step.
    pub fn drafts_of(&self, h: Handle) -> Option<&[i32]> {
        self.reqs.get(h).map(|r| r.drafts.as_slice())
    }

    fn validate(&self, r: &NewRequest<'_>) -> Result<(), CoreError> {
        if r.prompt.is_empty() {
            return value("a prompt must hold at least one token");
        }
        if r.prompt.len() as i64 >= self.params.max_model_len {
            return value(format!(
                "prompt of {} tokens does not fit max_model_len={}",
                r.prompt.len(),
                self.params.max_model_len
            ));
        }
        if r.max_tokens <= 0 {
            return value("max_tokens must be positive");
        }
        if let Some(rank) = r.dp_rank {
            if rank >= self.ranks.len() {
                return value(format!("dp_rank {rank} is out of range"));
            }
        }
        let mut prev = 0;
        for &(start, stop) in r.spans {
            if start < prev || stop <= start || stop > r.prompt.len() as i64 {
                return value("image spans must be sorted, non-empty and inside the prompt");
            }
            if stop - start > self.params.token_budget {
                return value(format!(
                    "an image span of {} tokens exceeds the per-step token budget {}",
                    stop - start,
                    self.params.token_budget
                ));
            }
            prev = stop;
        }
        Ok(())
    }

    /// Admits requests in order (each is routed after the previous one is queued). Atomic: when one request fails
    /// validation nothing is admitted.
    pub fn add(&mut self, requests: &[NewRequest<'_>]) -> Result<Vec<Handle>, CoreError> {
        for r in requests {
            self.validate(r)?;
        }
        let mut out = Vec::with_capacity(requests.len());
        for r in requests {
            let probe = Request {
                handle: self.reqs.next_handle(),
                rank: 0,
                status: Status::Waiting,
                tokens: r.prompt.to_vec(),
                num_prompt: r.prompt.len() as i64,
                computed: 0,
                placeholders: 0,
                max_tokens: r.max_tokens,
                min_tokens: r.min_tokens,
                eos: r.eos,
                ignore_eos: r.ignore_eos,
                stop_ids: r.stop_ids.to_vec(),
                priority: r.priority,
                arrival: r.arrival,
                flags: r.flags,
                spans: r.spans.to_vec(),
                hash_extra: r.hash_extra,
                sampling: r.sampling,
                kv_slot: None,
                save_target: None,
                cached_tokens: -1,
                stop_reason: None,
                drafts: Vec::new(),
                accept: Vec::new(),
                inflight_drafts: 0,
            };
            let rank = match r.dp_rank {
                Some(rank) => rank,
                None if self.ranks.len() == 1 => 0,
                None => route(&self.ranks, &self.reqs, &probe),
            };
            let h = self.reqs.insert(|handle| Request { rank, handle, ..probe });
            self.ranks[rank].waiting.push(h, (r.priority, r.arrival));
            out.push(h);
        }
        Ok(out)
    }

    /// Schedules one step over every rank.
    pub fn schedule(&mut self) -> Result<Plan, CoreError> {
        let p = self.params;
        let ng = self.num_groups;
        let mut plan = Plan {
            id: self.next_plan,
            new_page_offsets: vec![0],
            rank_offsets: vec![0],
            draft_offsets: vec![0],
            new_token_offsets: vec![0],
            stats: vec![0; STATS_GLOBAL + 4 * self.ranks.len()],
            ..Plan::default()
        };
        self.next_plan += 1;
        plan.num_draft_tokens_next = p.num_spec_tokens;
        if let Some(v) = self.verify.as_ref() {
            plan.num_draft_tokens_next = verify::apply(v, &self.ranks, &mut self.reqs, &p)?;
        }
        let mut grammar_pending = false;
        let spec_async = p.async_scheduling && p.num_spec_tokens > 0;
        let mut all_single = true;
        let mut prefill = false;
        let mut pages_changed = false;
        let mut verify = false;
        let admit = self.admission_gate();
        // Every rank is scheduled before any step's tokens are applied, so the step can be trimmed as a whole (a
        // rank's schedule reads only its own requests, so the order of decisions is unchanged).
        let mut steps = Vec::with_capacity(self.ranks.len());
        let mut finishes = Vec::with_capacity(self.ranks.len());
        for rank in self.ranks.iter_mut() {
            let mut finished = std::mem::take(&mut rank.finished);
            steps.push(schedule_rank(rank, &mut self.reqs, &p, admit, &mut finished)?);
            finishes.push(finished);
        }
        if let Some(cost) = self.step_cost.as_ref() {
            trim_prefill(cost, &mut self.ranks, &mut steps, &mut self.reqs, &p)?;
        }
        for ((rank, step), finished) in self.ranks.iter_mut().zip(steps).zip(finishes) {
            let base = plan.handles.len();
            for e in &step.entries {
                let r = self
                    .reqs
                    .get_mut(e.handle)
                    .ok_or_else(|| CoreError::Invariant("scheduled a dead request".into()))?;
                r.computed += e.scheduled;
            }
            let mut rank_tokens = 0i64;
            for e in &step.entries {
                let r = self.reqs.get_mut(e.handle).expect("checked above");
                if e.kind != KIND_NEW {
                    let len = r.num_tokens();
                    let lo = e.computed.clamp(0, len);
                    let hi = (e.computed + e.scheduled - e.drafts.len() as i64).clamp(lo, len);
                    plan.new_tokens.extend_from_slice(&r.tokens[lo as usize..hi as usize]);
                }
                plan.new_token_offsets.push(plan.new_tokens.len() as i32);
                plan.draft_tokens.extend_from_slice(&e.drafts);
                plan.draft_offsets.push(plan.draft_tokens.len() as i32);
                verify |= !e.drafts.is_empty();
                let cur_spec = e.drafts.len() as i64;
                let inflight_before = if p.async_scheduling && r.placeholders > 0 {
                    r.inflight_drafts
                } else {
                    -1
                };
                plan.inflight_drafts.push(inflight_before);
                let emit = if p.async_scheduling {
                    grammar_pending |= r.flags & FLAG_GRAMMAR != 0 && r.placeholders > 0;
                    if r.computed == r.num_tokens() + r.placeholders + cur_spec {
                        // Optimistic advance: every draft counts as accepted until this step's commit.
                        r.placeholders += 1 + cur_spec;
                        r.inflight_drafts += cur_spec;
                        r.drafts = vec![-1; p.num_spec_tokens as usize];
                        true
                    } else {
                        false
                    }
                } else {
                    r.computed >= r.num_tokens()
                };
                all_single &= if spec_async {
                    // A verify step may overlap only when every entry is a running decode window of 1 + k (or 1
                    // without drafts) whose next window of 1 + k still fits the model length.
                    let k = p.num_spec_tokens;
                    e.kind == KIND_RUNNING
                        && emit
                        && ((cur_spec == 0 && e.scheduled == 1) || (cur_spec == k && e.scheduled == 1 + k))
                        && r.computed + 1 + k < p.max_model_len
                } else {
                    e.scheduled == 1
                };
                plan.spec_slots.push(if emit { (1 + cur_spec) as i32 } else { 0 });
                prefill |= e.computed < r.num_prompt;
                pages_changed |= e.kind != KIND_RUNNING;
                rank_tokens += e.scheduled;
                plan.handles.push(e.handle);
                plan.scheduled.push(e.scheduled as i32);
                plan.computed.push(e.computed as i32);
                plan.kinds.push(e.kind);
                plan.emits.push(u8::from(emit));
                for gi in 0..ng {
                    for &pg in &e.pages[gi] {
                        plan.new_pages.push(rank.global_page(pg) as i32);
                        pages_changed = true;
                    }
                    plan.new_page_offsets.push(plan.new_pages.len() as i32);
                }
            }
            let entry_of = |h: Handle| {
                step.entries
                    .iter()
                    .position(|e| e.handle == h)
                    .map(|i| (base + i) as i32)
                    .ok_or_else(|| CoreError::Invariant("a snapshot pair of an unscheduled request".into()))
            };
            for &(h, slot) in &step.restores {
                plan.snapshot_restores.push([entry_of(h)?, slot as i32]);
            }
            for &(h, slot) in &step.saves {
                plan.snapshot_saves.push([entry_of(h)?, slot as i32]);
            }
            if rank.index == 0 {
                plan.common_prefix_pages = step.common_prefix.iter().map(|&v| v as i32).collect();
            }
            plan.req_bucket += rank.current_bucket as i64;
            pages_changed |= !step.preempted.is_empty() || !finished.is_empty();
            plan.preempted.extend_from_slice(&step.preempted);
            plan.finished.extend_from_slice(&finished);
            plan.rank_offsets.push(plan.handles.len() as i32);
            let k = STATS_GLOBAL + 4 * rank.index;
            let free = rank.kv.pool().num_free() as i64;
            plan.stats[k] = rank_tokens;
            plan.stats[k + 1] = rank.running.len() as i64;
            plan.stats[k + 2] = rank.waiting.len() as i64;
            plan.stats[k + 3] = free;
            plan.stats[0] += rank_tokens;
            plan.stats[1] += rank.running.len() as i64;
            plan.stats[2] += rank.waiting.len() as i64;
            plan.stats[3] += free;
            plan.stats[4] += step.budget_initial;
            plan.stats[5] += step.budget_remaining;
            plan.stats[6] += step.preempted.len() as i64;
            plan.stats[7] = plan.stats[7].max(rank_tokens);
        }
        if p.async_scheduling && !plan.is_empty() && all_single && !grammar_pending {
            plan.flags |= PLAN_ASYNC_SAFE;
        }
        if prefill {
            plan.flags |= PLAN_HAS_PREFILL;
        }
        if pages_changed {
            plan.flags |= PLAN_PAGES_CHANGED;
        }
        if verify {
            plan.flags |= PLAN_HAS_VERIFY;
        }
        if grammar_pending {
            plan.flags |= PLAN_STRUCTURED_PENDING;
        }
        if !plan.is_empty() {
            // A step is timed only when it runs alone (no plan in flight), from here to its commit.
            if self.verify.is_some() && self.inflight.is_empty() {
                let tokens = self.ranks.len() as i64 * plan.stats[7];
                self.timing = Some((plan.id, tokens, Instant::now()));
            }
            self.inflight.push_back(plan.id);
        }
        Ok(plan)
    }

    /// Commits a plan's runner output (see [`Output`]). Plans with entries commit in the order `schedule` returned
    /// them. Every commit, of a plan with or without entries, reports the requests freed since the previous commit
    /// when the core keeps the finished set.
    pub fn commit(&mut self, plan: &Plan, out: &Output<'_>) -> Result<Events, CoreError> {
        let mut events = Events::new();
        if !plan.is_empty() {
            if self.inflight.front() != Some(&plan.id) {
                return value(format!(
                    "plan {} committed out of order (next in flight: {:?})",
                    plan.id,
                    self.inflight.front()
                ));
            }
            if plan.rank_offsets.len() != self.ranks.len() + 1 {
                return value("the plan belongs to another core");
            }
            out.validate(plan.len()).map_err(CoreError::Value)?;
            self.inflight.pop_front();
            if let Some(v) = self.verify.as_mut() {
                if let Some((id, tokens, start)) = self.timing {
                    if id == plan.id {
                        v.observe_step(tokens, start.elapsed().as_secs_f64() * 1e3);
                        self.timing = None;
                    }
                }
                let mut step = vec![(0.0, 0.0); v.k()];
                for i in 0..plan.len() {
                    if !out.is_present(i) || plan.emits[i] == 0 {
                        continue;
                    }
                    let verified = plan.drafts(i).len();
                    let accepted = match out.accepted_of(i) {
                        Some(a) => a.max(0) as usize,
                        None if verified == 0 => 0,
                        None => continue,
                    };
                    if let Some(r) = self.reqs.get_mut(plan.handles[i]) {
                        v.observe_request(&mut r.accept, verified, accepted, &mut step);
                    }
                }
                v.observe_prior(&step);
            }
            commit_plan(&mut self.ranks, &mut self.reqs, &self.params, plan, out, &mut events)?;
        }
        for rank in &mut self.ranks {
            if let Some(set) = rank.finished_set.as_mut() {
                events.finished_all.append(set);
            }
        }
        Ok(events)
    }

    /// [`commit`](Self::commit) with at most one sampled token per entry (`next_token[i]`, -1 for none), every entry
    /// present and no speculation (the golden replay's form).
    #[cfg(test)]
    pub fn commit_tokens(&mut self, plan: &Plan, next_token: &[i32]) -> Result<Events, CoreError> {
        if next_token.len() != plan.len() {
            return value(format!(
                "next_token has {} values for a plan of {} entries",
                next_token.len(),
                plan.len()
            ));
        }
        let (sampled, offsets) = Output::flatten_single(next_token);
        self.commit(plan, &Output::sampled(&sampled, &offsets))
    }

    /// Finishes live requests (aborts, stop strings) in order; returns the handles that were live.
    pub fn finish(&mut self, handles: &[Handle], reason: FinishReason) -> Result<Vec<Handle>, CoreError> {
        let mut out = Vec::new();
        for &h in handles {
            let Some(r) = self.reqs.get_mut(h) else {
                continue;
            };
            let rank = &mut self.ranks[r.rank];
            if r.status == Status::Running {
                if let Some(pos) = rank.running.iter().position(|&x| x == h) {
                    rank.running.remove(pos);
                }
            } else {
                rank.waiting.remove(&[h]);
            }
            r.status = reason.status();
            free_request(
                &mut rank.kv,
                &mut self.reqs,
                h,
                &mut rank.finished,
                rank.finished_set.as_mut(),
            )?;
            out.push(h);
        }
        Ok(out)
    }

    /// Forgets every published page and snapshot, rank by rank; stops (`false`) at the first rank whose pages are
    /// in use, as the former per-rank reset did.
    pub fn reset_prefix_cache(&mut self) -> bool {
        for rank in &mut self.ranks {
            if !rank.kv.reset() {
                return false;
            }
            if let Some(s) = rank.snapshots.as_mut() {
                s.reset();
            }
        }
        true
    }

    /// Running, waiting, free pages and pending prefill tokens, then the same four per rank.
    pub fn counts(&self) -> Vec<i64> {
        let mut out = vec![0i64; 4];
        for rank in &self.ranks {
            let c = [
                rank.running.len() as i64,
                rank.waiting.len() as i64,
                rank.kv.pool().num_free() as i64,
                rank.pending_prefill(&self.reqs),
            ];
            for (o, v) in out.iter_mut().zip(c) {
                *o += v;
            }
            out.extend_from_slice(&c);
        }
        out
    }

    /// A hash of every live request's status, tokens, counters and global page ids (never page keys).
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        for r in self.reqs.iter() {
            mix(r.handle);
            mix(r.status as u64);
            mix(r.computed as u64);
            mix(r.placeholders as u64);
            for &t in &r.tokens {
                mix(t as u64);
            }
            let rank = &self.ranks[r.rank];
            if let Some(kv) = r.kv_slot.and_then(|s| rank.kv.get(s)) {
                for g in &kv.pages {
                    mix(g.len() as u64);
                    for &pg in g {
                        mix(u64::from(rank.global_page(pg)));
                    }
                }
            }
        }
        h
    }

    /// Checks the invariants: every rank's KV core (reference counts, free list, prefix map), every running and
    /// waiting request's rank and status, the running caps, and that the table holds exactly the queued and running
    /// requests.
    pub fn check(&self) -> Result<(), CoreError> {
        let mut listed = 0usize;
        for rank in &self.ranks {
            rank.kv.check().map_err(CoreError::Invariant)?;
            if rank.running.len() > rank.max_running {
                return Err(CoreError::Invariant(format!(
                    "rank {}: {} running over {} rows",
                    rank.index,
                    rank.running.len(),
                    rank.max_running
                )));
            }
            for &h in &rank.running {
                let r = self
                    .reqs
                    .get(h)
                    .ok_or_else(|| CoreError::Invariant(format!("dead running handle {h:#x}")))?;
                if r.rank != rank.index || r.status != Status::Running {
                    return Err(CoreError::Invariant(format!(
                        "running request {h:#x} in state {:?}",
                        r.status
                    )));
                }
                if r.computed > self.params.max_model_len {
                    return Err(CoreError::Invariant(format!(
                        "request {h:#x} computed past max_model_len"
                    )));
                }
            }
            for h in rank.waiting.iter() {
                let r = self
                    .reqs
                    .get(h)
                    .ok_or_else(|| CoreError::Invariant(format!("dead waiting handle {h:#x}")))?;
                if r.rank != rank.index || !matches!(r.status, Status::Waiting | Status::Preempted) {
                    return Err(CoreError::Invariant(format!(
                        "waiting request {h:#x} in state {:?}",
                        r.status
                    )));
                }
            }
            listed += rank.running.len() + rank.waiting.len();
        }
        if listed != self.reqs.len() {
            return Err(CoreError::Invariant(format!(
                "{} live requests but {listed} queued or running",
                self.reqs.len()
            )));
        }
        Ok(())
    }
}
