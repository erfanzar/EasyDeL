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

//! Data-parallel ranks, in process: one [`RankState`] per rank, and admission routing.
//!
//! Each rank owns its waiting queue, its running list, its KV core over `pages_per_rank` rank-local page ids
//! (`1..pages_per_rank`, with local page 0 the null page), its recurrent-snapshot slot block and its token budget.
//! Page ids leave the core as `rank * pages_per_rank + local` (local 0 stays 0). Every rank has the full per-step
//! token budget.

use crate::engine::request::{Handle, Request, RequestTable};
use crate::engine::sched::{Queue, TokenBudget};
use crate::kv::snapshot::SnapshotIndex;
use crate::kv::{KvCore, Tokens};

/// One data-parallel rank's scheduling state.
pub struct RankState {
    /// The rank's index.
    pub index: usize,
    /// Waiting (new and preempted) requests.
    pub waiting: Queue,
    /// Running requests, in admission order.
    pub running: Vec<Handle>,
    /// The rank's KV core (rank-local page ids).
    pub kv: KvCore,
    /// The rank's recurrent-snapshot index (prefix caching with snapshot slots only).
    pub snapshots: Option<SnapshotIndex>,
    /// The KV-aware token budget (only with a token safety margin).
    pub budget: Option<TokenBudget>,
    /// Requests finished since the last schedule.
    pub finished: Vec<Handle>,
    /// Running-request cap of the rank (its row block).
    pub max_running: usize,
    /// Request-count buckets, ascending, the last equal to `max_running`.
    pub buckets: Vec<usize>,
    /// The bucket selected for the current running count.
    pub current_bucket: usize,
    /// First global page id of the rank (`rank * pages_per_rank`).
    pub page_offset: u32,
    /// Requests freed since the last commit, in free order (kept only with `include_finished_set`).
    pub finished_set: Option<Vec<Handle>>,
}

/// The request-count bucket ladder of a rank: the buckets in `(0, max_running]`, deduplicated, ending at
/// `max_running`.
pub fn rank_buckets(ladder: &[i64], max_running: usize) -> Vec<usize> {
    let mut out: Vec<usize> = ladder
        .iter()
        .filter(|&&b| b > 0 && b as usize <= max_running)
        .map(|&b| b as usize)
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.last() != Some(&max_running) {
        out.push(max_running);
    }
    out
}

impl RankState {
    /// The smallest bucket holding `n` requests, else the largest.
    pub fn select_bucket(&self, n: usize) -> usize {
        if n == 0 {
            return self.buckets[0];
        }
        self.buckets
            .iter()
            .copied()
            .find(|&b| n <= b)
            .unwrap_or(*self.buckets.last().expect("a rank has at least one bucket"))
    }

    /// Selects the bucket for `desired` running requests; whether that many fit the rank.
    pub fn ensure_capacity(&mut self, desired: usize) -> bool {
        let bucket = self.select_bucket(desired);
        self.current_bucket = bucket;
        desired <= self.max_running && desired <= bucket
    }

    /// The global id of a rank-local page id.
    pub fn global_page(&self, local: u32) -> u32 {
        if local == 0 {
            0
        } else {
            self.page_offset + local
        }
    }

    /// Prompt tokens the rank still has to prefill: the running requests' rest, the waiting requests' prompts.
    pub fn pending_prefill(&self, reqs: &RequestTable) -> i64 {
        let running: i64 = self
            .running
            .iter()
            .filter_map(|&h| reqs.get(h))
            .map(|r| (r.num_prompt - r.computed).max(0))
            .sum();
        let waiting: i64 = self
            .waiting
            .iter()
            .filter_map(|h| reqs.get(h))
            .map(|r| r.num_prompt)
            .sum();
        running + waiting
    }
}

/// Whether a rank can take one more request without queueing behind its row cap (running plus waiting below it).
fn has_room(rank: &RankState) -> bool {
    rank.running.len() + rank.waiting.len() < rank.max_running
}

/// The rank a new request goes to.
///
/// Among ranks with room (running plus waiting below the row cap; every rank when none has room): the rank with
/// the most prefix-cached tokens (read-only probes; ties to the lower rank), else the rank with the fewest pending
/// prefill tokens, then the fewest queued requests (ties to the lower rank).
///
/// A full rank is never chosen for its prefix while another rank has room: a prompt prefix shared by every request
/// (a common system prompt) would otherwise send all traffic to the one rank holding it and leave the others idle.
pub fn route(ranks: &[RankState], reqs: &RequestTable, request: &Request) -> usize {
    let any_room = ranks.iter().any(has_room);
    let eligible = |rank: &RankState| !any_room || has_room(rank);
    let mut best: Option<usize> = None;
    let mut best_tokens = 0;
    for rank in ranks {
        if !eligible(rank) || !rank.kv.caching() || request.no_prefix_hit() {
            continue;
        }
        let tokens = Tokens {
            tokens: &request.tokens,
            offset: 0,
        };
        // A failing probe is skipped, as the former rank probe skipped a rank that raised.
        let Ok(cached) = rank
            .kv
            .probe(None, Some(tokens), request.hash_extra, request.num_tokens() - 1)
        else {
            continue;
        };
        if cached > best_tokens {
            best_tokens = cached;
            best = Some(rank.index);
        }
    }
    if let Some(rank) = best {
        return rank;
    }
    // `min_by_key` keeps the first of equal keys, so ties go to the lower rank.
    ranks
        .iter()
        .min_by_key(|rank| {
            (
                !eligible(rank),
                rank.pending_prefill(reqs),
                rank.running.len() + rank.waiting.len(),
            )
        })
        .map_or(0, |rank| rank.index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::request::{NewRequest, Sampling};
    use crate::engine::sched::commit::Output;
    use crate::engine::sched::Policy;
    use crate::engine::{EngineConfig, EngineCore};
    use crate::kv::group::{Group, Kind};

    #[test]
    fn bucket_ladder() {
        assert_eq!(rank_buckets(&[], 4), vec![4]);
        assert_eq!(rank_buckets(&[8, 2, 2, 1], 4), vec![1, 2, 4]);
        assert_eq!(rank_buckets(&[1, 4], 4), vec![1, 4]);
    }

    /// Two ranks of two rows each, page size 4.
    fn dp_core(prefix_caching: bool) -> EngineCore {
        EngineCore::new(&EngineConfig {
            max_rows: 4,
            max_model_len: 64,
            num_pages: 64,
            dp_size: 2,
            groups: vec![Group {
                kind: Kind::Full,
                page_size: 4,
                type_index: 0,
            }],
            token_budget: 64,
            long_prefill_threshold: 0,
            chunked_prefill: false,
            token_safety_margin: None,
            policy: Policy::Fcfs,
            prefix_caching,
            snapshot_slots: 0,
            async_scheduling: false,
            req_buckets: Vec::new(),
            num_spec_tokens: 0,
            use_eagle: false,
            include_finished_set: false,
            dp_admission_hold: 0,
        })
        .unwrap()
    }

    /// Admits one request and returns the rank it was routed to.
    fn add(core: &mut EngineCore, prompt: &[i32]) -> usize {
        let req = NewRequest {
            prompt,
            max_tokens: 32,
            min_tokens: 0,
            eos: None,
            ignore_eos: true,
            priority: 0,
            arrival: 0.0,
            flags: 0,
            stop_ids: &[],
            spans: &[],
            dp_rank: None,
            hash_extra: 0,
            sampling: Sampling::default(),
        };
        let h = core.add(&[req]).unwrap()[0];
        core.rank_of(h).unwrap()
    }

    /// Schedules one step and commits one sampled token for every entry that samples.
    fn step(core: &mut EngineCore) {
        let plan = core.schedule().unwrap();
        let mut sampled = Vec::new();
        let mut offsets = vec![0];
        for &e in &plan.emits {
            if e != 0 {
                sampled.push(7);
            }
            offsets.push(sampled.len() as i32);
        }
        let out = Output {
            sampled: &sampled,
            sampled_offsets: &offsets,
            present: None,
            accepted: None,
            drafts: None,
        };
        core.commit(&plan, &out).unwrap();
        core.check().unwrap();
    }

    /// A shared 16-token prefix followed by a request-unique tail.
    fn shared(tail: i32) -> Vec<i32> {
        let mut p: Vec<i32> = (1..=16).collect();
        p.extend([100 + tail, 200 + tail]);
        p
    }

    #[test]
    fn a_shared_prefix_follows_its_rank_only_while_that_rank_has_room() {
        let mut core = dp_core(true);
        assert_eq!(
            add(&mut core, &shared(0)),
            0,
            "nothing cached, both ranks empty: the lower rank"
        );
        step(&mut core); // publishes the shared prefix on rank 0
        assert_eq!(
            add(&mut core, &shared(1)),
            0,
            "rank 0 holds the prefix and has a free row"
        );
        assert_eq!(
            add(&mut core, &shared(2)),
            1,
            "rank 0 is full: the prefix must not queue it there"
        );
        assert_eq!(add(&mut core, &shared(3)), 1, "rank 1 still has a row");
        assert_eq!(
            add(&mut core, &shared(4)),
            0,
            "every rank is full: back to the rank holding the prefix"
        );
    }

    #[test]
    fn without_prefix_hits_the_least_loaded_rank_wins_and_full_ranks_are_skipped() {
        let mut core = dp_core(false);
        let p = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(add(&mut core, &p), 0, "a tie goes to the lower rank");
        assert_eq!(
            add(&mut core, &p),
            1,
            "rank 0 has 8 pending prefill tokens, rank 1 none"
        );
        step(&mut core); // both prefills run: no pending prefill anywhere, one running request per rank
        assert_eq!(add(&mut core, &p), 0, "equal pending prefill and queue: the lower rank");
        step(&mut core); // rank 0 is now full (2 running), rank 1 has 1 running and no pending prefill
        assert_eq!(
            add(&mut core, &[9, 10]),
            1,
            "rank 0 is full: equal pending prefill must not send it there"
        );
        assert_eq!(
            add(&mut core, &p),
            0,
            "every rank is full: the fewest pending prefill tokens"
        );
    }
}
