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

//! Scheduling: the waiting queues, the token budget, one rank's schedule step and the commit of a step's tokens.

pub mod budget;
pub mod commit;
pub mod queue;
pub mod schedule;

pub use budget::{align_prefill_chunk, TokenBudget};
pub use queue::{Policy, Queue};

use crate::engine::request::{Handle, Request, RequestTable};
use crate::engine::CoreError;
use crate::kv::KvCore;

/// The request's slot in its rank's KV core, opened on first use.
pub fn kv_slot(kv: &mut KvCore, r: &mut Request) -> usize {
    match r.kv_slot {
        Some(slot) => slot,
        None => {
            let slot = kv.open(r.hash_extra);
            r.kv_slot = Some(slot);
            slot
        }
    }
}

/// Frees a finished request: its pages, then its page keys, then its table entry. Records it in `finished` (reported
/// by the next plan) and, when the rank keeps one, in `finished_set` (reported by the next commit). The caller has
/// already removed it from its rank's running list or waiting queue.
pub fn free_request(
    kv: &mut KvCore,
    reqs: &mut RequestTable,
    h: Handle,
    finished: &mut Vec<Handle>,
    finished_set: Option<&mut Vec<Handle>>,
) -> Result<(), CoreError> {
    let r = reqs
        .remove(h)
        .ok_or_else(|| CoreError::Invariant(format!("freeing an unknown request {h:#x}")))?;
    if let Some(slot) = r.kv_slot {
        if !kv.free(slot)? {
            kv.forget_keys(slot)?;
        }
    }
    finished.push(h);
    if let Some(set) = finished_set {
        set.push(h);
    }
    Ok(())
}

/// The scheduling numbers every rank shares.
#[derive(Clone, Copy, Debug)]
pub struct SchedParams {
    /// Longest sequence (prompt plus output).
    pub max_model_len: i64,
    /// Per-step token budget of one rank.
    pub token_budget: i64,
    /// Per-step prefill cap (0 is off).
    pub long_prefill_threshold: i64,
    /// Whether a prefill may span steps when it does not fit the remaining budget.
    pub chunked_prefill: bool,
    /// The policy.
    pub policy: Policy,
    /// Whether steps overlap with output placeholders.
    pub async_scheduling: bool,
    /// Tokens per page of the first cache group (the page-key granularity).
    pub page_size: i64,
    /// The most token slots one step adds to a prefilling request: the prefill cap (else the budget) plus the
    /// lookahead.
    pub max_chunk: i64,
    /// Speculative lookahead slots every allocation reserves (the draft count with an EAGLE drafter, else 0).
    pub lookahead: i64,
    /// Placeholder drafts an async step gives each request that samples (the configured draft count).
    pub num_spec_tokens: i64,
}
