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

/// The rank a new request goes to: the rank with the most prefix-cached tokens (read-only probes; ties to the
/// lower rank), else the rank with the fewest pending prefill tokens (ties to the lower rank). This is the former
/// Python DP scheduler's rule, which the engine goldens pin; requests past their prefill count no prefill tokens.
pub fn route(ranks: &[RankState], reqs: &RequestTable, request: &Request) -> usize {
    let mut best: Option<usize> = None;
    let mut best_tokens = 0;
    for rank in ranks {
        if !rank.kv.caching() || request.no_prefix_hit() {
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
    let mut pick = 0;
    let mut pick_pending = i64::MAX;
    for rank in ranks {
        let pending = rank.pending_prefill(reqs);
        if pending < pick_pending {
            pick_pending = pending;
            pick = rank.index;
        }
    }
    pick
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_ladder() {
        assert_eq!(rank_buckets(&[], 4), vec![4]);
        assert_eq!(rank_buckets(&[8, 2, 2, 1], 4), vec![1, 2, 4]);
        assert_eq!(rank_buckets(&[1, 4], 4), vec![1, 4]);
    }
}
