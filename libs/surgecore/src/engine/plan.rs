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

//! A step's decisions ([`Plan`]) and the outcome of committing its tokens ([`Events`]).
//!
//! Plan entries are in decision order: rank-major, and within a rank the running requests in running order, then
//! the requests admitted this step in admission order (the order of today's `num_scheduled_tokens`).

use crate::engine::request::Handle;

/// Entry kind: a request that was already running.
pub const KIND_RUNNING: u8 = 0;
/// Entry kind: a request scheduled for the first time.
pub const KIND_NEW: u8 = 1;
/// Entry kind: a preempted request scheduled again.
pub const KIND_RESUMED: u8 = 2;

/// Plan flag: every entry is one token, async scheduling is on and no grammar row has a pending token, so the next
/// step may be dispatched before this one's tokens are on the host (model-level checks stay with the caller).
pub const PLAN_ASYNC_SAFE: u32 = 1;
/// Plan flag: some entry prefills prompt tokens.
pub const PLAN_HAS_PREFILL: u32 = 2;
/// Plan flag: some entry verifies speculative drafts (never set without speculation).
pub const PLAN_HAS_VERIFY: u32 = 4;
/// Plan flag: page rows changed (new pages, new or resumed rows, preemptions or finishes).
pub const PLAN_PAGES_CHANGED: u32 = 8;
/// Plan flag: a structured-output (grammar) request still had an unresolved async placeholder when it was
/// scheduled (the former `pending_structured_output_tokens`).
pub const PLAN_STRUCTURED_PENDING: u32 = 16;

/// Number of global entries at the head of [`Plan::stats`]; four per-rank entries follow.
pub const STATS_GLOBAL: usize = 8;

/// One step's scheduling decisions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// The plan's id (consecutive per core).
    pub id: u64,
    /// `PLAN_*` bits.
    pub flags: u32,
    /// Scheduled requests.
    pub handles: Vec<Handle>,
    /// Tokens scheduled per entry.
    pub scheduled: Vec<i32>,
    /// Computed tokens per entry before this step (the prefix hit for a new or resumed entry).
    pub computed: Vec<i32>,
    /// `KIND_*` per entry.
    pub kinds: Vec<u8>,
    /// Whether the entry samples a token this step.
    pub emits: Vec<u8>,
    /// Global page ids per entry and group: all of a new or resumed entry's pages, the newly allocated ones of a
    /// running entry. Entry `i`, group `g` is `new_pages[new_page_offsets[i*G+g]..new_page_offsets[i*G+g+1]]`.
    pub new_pages: Vec<i32>,
    /// Offsets into `new_pages` (`n * G + 1` values).
    pub new_page_offsets: Vec<i32>,
    /// Entry offsets per rank (`dp + 1` values).
    pub rank_offsets: Vec<i32>,
    /// The speculative drafts each entry verifies this step, flat (the former `scheduled_spec_decode_tokens`).
    pub draft_tokens: Vec<i32>,
    /// Offsets into `draft_tokens` (`n + 1` values); an entry without drafts has an empty range.
    pub draft_offsets: Vec<i32>,
    /// The known tokens each running or resumed entry feeds this step, flat: its tokens from `computed` up to
    /// `computed + scheduled - drafts` (the former `CachedRequestData.new_token_ids`; empty for a new entry).
    pub new_tokens: Vec<i32>,
    /// Offsets into `new_tokens` (`n + 1` values).
    pub new_token_offsets: Vec<i32>,
    /// Leading pages per group shared by every request rank 0 scheduled (the former `num_common_prefix_pages`).
    pub common_prefix_pages: Vec<i32>,
    /// The request-count bucket selected after scheduling, summed over ranks (the former `suggested_bucket`).
    pub req_bucket: i64,
    /// Requests preempted this step.
    pub preempted: Vec<Handle>,
    /// Requests finished since the previous plan (stops, finishes and aborts, zero-token drops).
    pub finished: Vec<Handle>,
    /// `(entry, slot)` recurrent-snapshot restores (global slots).
    pub snapshot_restores: Vec<[i32; 2]>,
    /// `(entry, slot)` recurrent-snapshot saves (global slots), in reservation order per rank.
    pub snapshot_saves: Vec<[i32; 2]>,
    /// Scheduled tokens, running, waiting, free pages, initial and remaining token budget, preemptions, the busiest
    /// rank's tokens; then per rank: scheduled tokens, running, waiting, free pages.
    pub stats: Vec<i64>,
}

impl Plan {
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether the plan schedules nothing.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Number of cache groups the page offsets cover.
    pub fn num_groups(&self) -> usize {
        if self.handles.is_empty() {
            0
        } else {
            (self.new_page_offsets.len() - 1) / self.handles.len()
        }
    }

    /// The global page ids of entry `i` in group `g`.
    pub fn pages(&self, i: usize, g: usize) -> &[i32] {
        let k = i * self.num_groups() + g;
        &self.new_pages[self.new_page_offsets[k] as usize..self.new_page_offsets[k + 1] as usize]
    }

    /// The drafts entry `i` verifies this step.
    pub fn drafts(&self, i: usize) -> &[i32] {
        if self.draft_offsets.len() <= i + 1 {
            return &[];
        }
        &self.draft_tokens[self.draft_offsets[i] as usize..self.draft_offsets[i + 1] as usize]
    }

    /// Whether a commit has to apply the plan's bookkeeping (it schedules, preempts or reports finishes).
    pub fn needs_commit(&self) -> bool {
        !self.handles.is_empty()
    }
}

/// The outcome of a commit, in plan order: one entry per request that received tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Events {
    /// Requests that received tokens.
    pub handles: Vec<Handle>,
    /// Offsets into `tokens` (`E + 1` values).
    pub token_offsets: Vec<i32>,
    /// The tokens emitted, truncated at a stop.
    pub tokens: Vec<i32>,
    /// Position of each entry's first token among the step's sampled tokens (the logprob row).
    pub logprob_starts: Vec<i32>,
    /// 0 running, 1 stop (EOS or stop id), 2 length, 3 abort.
    pub finish: Vec<i8>,
    /// The stop token that finished the request, -1 otherwise.
    pub stop_token: Vec<i32>,
    /// Prefix-cache hit tokens of the request's first admission.
    pub cached_tokens: Vec<i32>,
    /// Requests freed since the previous commit, in free order (only with `include_finished_set`; the former
    /// `EngineCoreOutputs.finished_requests`).
    pub finished_all: Vec<Handle>,
}

impl Events {
    /// An empty event set.
    pub fn new() -> Self {
        Self {
            token_offsets: vec![0],
            ..Self::default()
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Whether no request received tokens.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// The tokens of entry `i`.
    pub fn tokens_of(&self, i: usize) -> &[i32] {
        &self.tokens[self.token_offsets[i] as usize..self.token_offsets[i + 1] as usize]
    }
}
