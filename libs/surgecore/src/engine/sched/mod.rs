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
pub mod trim;

pub use budget::{align_prefill_chunk, TokenBudget};
pub use queue::{Policy, Queue};

use crate::engine::request::{Handle, Request, RequestTable};
use crate::engine::CoreError;
use crate::kv::snapshot::SnapshotIndex;
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
    /// Decode snapshots are on (see [`EngineConfig::decode_snapshots`](crate::engine::EngineConfig)).
    pub decode_snapshots: bool,
    /// Snapshot-aware routing is on (see [`EngineConfig::snapshot_routing`](crate::engine::EngineConfig)).
    pub snapshot_routing: bool,
}

/// Decode snapshots: retires the request's rolling snapshot slot when it leaves its rows (freed or preempted).
///
/// The slot is committed under the page key of the last page boundary a committed, listed step wrote into it; it is
/// dropped when nothing valid was written. When a plan still in flight writes the slot, its content will not match
/// that boundary once the plan runs: the slot is then dropped when that plan commits (`retiring`).
pub fn retire_decode_snapshot(
    kv: &KvCore,
    snaps: &mut Option<SnapshotIndex>,
    retiring: &mut Vec<(i64, u64)>,
    r: &mut Request,
    page_size: i64,
    oldest_inflight: Option<u64>,
) {
    let Some((slot, boundary)) = r.decode_snap.take() else {
        return;
    };
    let Some(index) = snaps.as_mut() else {
        return;
    };
    if oldest_inflight.is_some_and(|f| r.decode_plan >= f) {
        retiring.push((slot, r.decode_plan));
        return;
    }
    if boundary > 0 && page_size > 0 && boundary % page_size == 0 {
        let j = (boundary / page_size - 1) as usize;
        let known = r.kv_slot.and_then(|s| kv.page_key_at(s, j, &r.tokens).ok().flatten());
        let key = known.or_else(|| {
            let end = (boundary as usize).min(r.tokens.len());
            kv.chain_keys(&r.tokens[..end], r.hash_extra).get(j).copied()
        });
        if let Some(key) = key {
            if index.commit_keyed(slot, key) {
                return;
            }
        }
    }
    index.cancel(slot);
}
