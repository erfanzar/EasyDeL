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

//! One rank's schedule step.
//!
//! The running pass comes first: each running request gets its next chunk (capped by the long-prefill threshold, the
//! budget, `max_model_len - 1`, image-span alignment and its snapshot boundary), and a failed page allocation
//! preempts (FCFS: the last running request; priority: the first request with the highest `(priority, arrival)`,
//! un-scheduled and refunded when it was already scheduled this step). Waiting requests are admitted only when
//! nothing was preempted: a prefix hit (through a ready recurrent snapshot when snapshots are on), the non-chunked
//! skip rule, the zero-token drop, image-span alignment, the snapshot boundary, and, while requests run, the rule
//! that the pages for the whole remaining prompt must be free. The order of every list operation is the former
//! Python scheduler's, quirks included, so the decisions are the same.

use crate::engine::dp::RankState;
use crate::engine::plan::{KIND_NEW, KIND_RESUMED, KIND_RUNNING};
use crate::engine::request::{Handle, Request, RequestTable, Status};
use crate::engine::sched::{align_prefill_chunk, free_request, kv_slot, Policy, Queue, SchedParams};
use crate::engine::CoreError;
use crate::kv::group::GroupPages;
use crate::kv::hash::PageKey;
use crate::kv::snapshot::SnapshotIndex;
use crate::kv::{KvCore, Tokens};

/// One scheduled request of a rank step.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The request.
    pub handle: Handle,
    /// Tokens scheduled.
    pub scheduled: i64,
    /// Computed tokens before the step (the prefix hit for a new or resumed request).
    pub computed: i64,
    /// `KIND_*`.
    pub kind: u8,
    /// Rank-local page ids per group: all of them for a new or resumed request, the new ones otherwise.
    pub pages: GroupPages,
    /// The speculative drafts this step verifies (the request's drafts trimmed to what was scheduled).
    pub drafts: Vec<i32>,
}

/// The decisions of one rank's schedule step.
#[derive(Clone, Debug, Default)]
pub struct RankStep {
    /// Scheduled requests in decision order.
    pub entries: Vec<Entry>,
    /// Preempted requests in preemption order.
    pub preempted: Vec<Handle>,
    /// `(request, slot)` snapshot restores.
    pub restores: Vec<(Handle, i64)>,
    /// `(request, slot)` snapshot saves in reservation order.
    pub saves: Vec<(Handle, i64)>,
    /// The step's initial token budget.
    pub budget_initial: i64,
    /// The token budget left.
    pub budget_remaining: i64,
    /// Leading pages per group shared by every scheduled request (rank 0 only; empty on other ranks).
    pub common_prefix: Vec<i64>,
}

fn get(reqs: &RequestTable, h: Handle) -> Result<&Request, CoreError> {
    reqs.get(h)
        .ok_or_else(|| CoreError::Invariant(format!("stale handle {h:#x} in a rank list")))
}

fn get_mut(reqs: &mut RequestTable, h: Handle) -> Result<&mut Request, CoreError> {
    reqs.get_mut(h)
        .ok_or_else(|| CoreError::Invariant(format!("stale handle {h:#x} in a rank list")))
}

fn empty_pages(kv: &KvCore) -> GroupPages {
    vec![Vec::new(); kv.num_groups()]
}

/// The longest prefix hit of `r` (at most `max_hit` tokens, and at most all but its last token). Computes and keeps
/// the request's page keys first; a request that takes no hits gets none.
pub fn lookup(kv: &mut KvCore, r: &mut Request, max_hit: Option<i64>) -> Result<(GroupPages, i64), CoreError> {
    if !kv.caching() {
        return Ok((empty_pages(kv), 0));
    }
    let slot = kv_slot(kv, r);
    let mut max_len = r.num_tokens() - 1;
    if let Some(m) = max_hit {
        max_len = max_len.min(m);
    }
    let tokens = if kv.num_keys(slot)? == 0 {
        Some(Tokens {
            tokens: &r.tokens,
            offset: 0,
        })
    } else {
        None
    };
    Ok(kv.lookup(slot, tokens, r.no_prefix_hit(), max_len, None, None)?)
}

fn page_keys(kv: &mut KvCore, r: &mut Request) -> Result<Vec<PageKey>, CoreError> {
    let slot = kv_slot(kv, r);
    let tokens = if kv.num_keys(slot)? == 0 {
        Some(Tokens {
            tokens: &r.tokens,
            offset: 0,
        })
    } else {
        None
    };
    Ok(kv.ensure_keys(slot, tokens)?.to_vec())
}

/// The longest prefix hit whose boundary page has a ready recurrent snapshot, and the request's save target (its last
/// full-page boundary before its last token, when that lies past the hit and the request has no image spans).
fn snapshot_prefix_hit(
    rank: &mut RankState,
    r: &mut Request,
    ps: i64,
) -> Result<(GroupPages, i64, Option<PageKey>), CoreError> {
    let keys = page_keys(&mut rank.kv, r)?;
    let last = (keys.len() as i64).min((r.num_tokens() - 1).div_euclid(ps));
    let mut hit = (empty_pages(&rank.kv), 0, None);
    let mut upper = last * ps;
    let snaps = rank
        .snapshots
        .as_ref()
        .ok_or_else(|| CoreError::Invariant("snapshot hit without a snapshot index".into()))?;
    for np in (1..=last).rev() {
        let key = keys[np as usize - 1];
        if np * ps > upper || !snaps.is_ready(key) {
            continue;
        }
        let (pages, tokens) = lookup(&mut rank.kv, r, Some(np * ps))?;
        if tokens == np * ps {
            hit = (pages, tokens, Some(key));
            break;
        }
        upper = tokens;
    }
    r.save_target = None;
    if last * ps > hit.1 && r.spans.is_empty() {
        r.save_target = Some((last * ps, keys[last as usize - 1]));
    }
    Ok(hit)
}

/// Ends a chunk on the request's snapshot boundary; drops a target that is behind or already snapshotted.
fn clip_to_snapshot(snaps: &Option<SnapshotIndex>, r: &mut Request, computed: i64, n: i64) -> i64 {
    let (Some((boundary, key)), Some(index)) = (r.save_target, snaps.as_ref()) else {
        return n;
    };
    if computed >= boundary || index.contains(key) {
        r.save_target = None;
        return n;
    }
    n.min(boundary - computed)
}

/// Reserves a snapshot slot when the chunk ends exactly on the request's boundary.
fn record_save(snaps: &mut Option<SnapshotIndex>, r: &mut Request, end: i64, step: &mut RankStep) {
    let (Some((boundary, key)), Some(index)) = (r.save_target, snaps.as_mut()) else {
        return;
    };
    if end != boundary {
        return;
    }
    r.save_target = None;
    let exclude: Vec<i64> = step.restores.iter().map(|&(_, s)| s).collect();
    if let Some(slot) = index.reserve(key, &exclude) {
        step.saves.push((r.handle, slot));
    }
}

fn cancel_save(snaps: &mut Option<SnapshotIndex>, h: Handle, step: &mut RankStep) {
    if let Some(pos) = step.saves.iter().position(|&(x, _)| x == h) {
        let (_, slot) = step.saves.remove(pos);
        if let Some(index) = snaps.as_mut() {
            index.cancel(slot);
        }
    }
}

/// Free pages a running request still draws before it holds its whole prompt (0 once it decodes).
fn outstanding_prompt_pages(kv: &KvCore, r: &Request, p: &SchedParams) -> Result<i64, CoreError> {
    if r.computed >= r.num_prompt.max(r.num_tokens() - 1) {
        return Ok(0);
    }
    Ok(kv.needed(r.kv_slot, r.num_tokens() + p.lookahead, None, Some(p.max_chunk))?)
}

fn take_budget(rank: &mut RankState, budget: &mut i64, n: i64) {
    match rank.budget.as_mut() {
        Some(b) => {
            b.consume(n);
            *budget = b.remaining;
        }
        None => *budget -= n,
    }
}

/// Schedules one rank. Requests dropped at admission are freed and recorded in `finished`.
pub fn schedule_rank(
    rank: &mut RankState,
    reqs: &mut RequestTable,
    p: &SchedParams,
    finished: &mut Vec<Handle>,
) -> Result<RankStep, CoreError> {
    let mut step = RankStep::default();
    let mut token_budget = match rank.budget {
        Some(mut b) => {
            let mut demand = 0;
            for &h in &rank.running {
                let r = get(reqs, h)?;
                demand += (r.num_tokens() + r.drafts.len() as i64 + r.placeholders - r.computed)
                    .min(p.max_model_len - 1 - r.computed)
                    .max(0);
            }
            let free = rank.kv.pool().num_free() as i64;
            let budget = b.begin_cycle(free, rank.running.len() as i64, demand);
            rank.budget = Some(b);
            budget
        }
        None => p.token_budget,
    };
    step.budget_initial = token_budget;

    // Running pass.
    let mut req_index: i64 = 0;
    rank.ensure_capacity(rank.running.len());
    while (req_index as usize) < rank.running.len() && token_budget > 0 {
        let h = rank.running[req_index as usize];
        let r = get_mut(reqs, h)?;
        if p.async_scheduling && r.computed >= r.num_prompt && r.num_output() + r.placeholders >= r.max_tokens {
            req_index += 1;
            continue;
        }
        let mut n = r.num_tokens() + r.drafts.len() as i64 + r.placeholders - r.computed;
        if p.long_prefill_threshold > 0 && p.long_prefill_threshold < n {
            n = p.long_prefill_threshold;
        }
        n = n.min(token_budget);
        n = n.min(p.max_model_len - 1 - r.computed);
        if n > 0 && !r.spans.is_empty() {
            n = align_prefill_chunk(&r.spans, r.computed, n, token_budget);
        }
        if n > 0 {
            let computed = r.computed;
            n = clip_to_snapshot(&rank.snapshots, r, computed, n);
        }
        if n <= 0 {
            req_index += 1;
            continue;
        }

        let max_attempts = rank.running.len() + 1;
        let mut attempts = 0usize;
        let mut new_pages = None;
        loop {
            let r = get_mut(reqs, h)?;
            let slot = kv_slot(&mut rank.kv, r);
            if let Some(pages) = rank
                .kv
                .allocate(slot, r.computed, n, 0, None, p.lookahead, None, None)?
            {
                new_pages = Some(pages);
                break;
            }
            attempts += 1;
            if attempts >= max_attempts {
                break;
            }
            let victim = match p.policy {
                Policy::Priority => {
                    let mut best = 0usize;
                    let mut best_key = get(reqs, rank.running[0])?.queue_key();
                    for (i, &c) in rank.running.iter().enumerate().skip(1) {
                        let key = get(reqs, c)?.queue_key();
                        if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
                            best = i;
                            best_key = key;
                        }
                    }
                    let v = rank.running.remove(best);
                    if let Some(pos) = step.entries.iter().position(|e| e.handle == v) {
                        let e = step.entries.remove(pos);
                        cancel_save(&mut rank.snapshots, v, &mut step);
                        match rank.budget.as_mut() {
                            Some(b) => {
                                b.release(e.scheduled);
                                token_budget = b.remaining;
                            }
                            None => token_budget += e.scheduled,
                        }
                        req_index -= 1;
                    }
                    v
                }
                Policy::Fcfs => rank
                    .running
                    .pop()
                    .ok_or_else(|| CoreError::Invariant("preemption with nothing running".into()))?,
            };
            let vr = get_mut(reqs, victim)?;
            if let Some(slot) = vr.kv_slot {
                if rank.kv.free(slot)? {
                    vr.kv_slot = None;
                }
            }
            vr.status = Status::Preempted;
            vr.computed = 0;
            let key = vr.queue_key();
            rank.waiting.prepend(victim, key);
            step.preempted.push(victim);
            if victim == h {
                break;
            }
        }
        let Some(pages) = new_pages else {
            req_index += 1;
            continue;
        };
        let r = get_mut(reqs, h)?;
        let computed = r.computed;
        // The drafts covered by this chunk are trimmed to what was scheduled and verified this step.
        let mut drafts = Vec::new();
        if !r.drafts.is_empty() {
            let k = n + computed - r.num_tokens();
            if k > 0 {
                r.drafts.truncate(k as usize);
                drafts.clone_from(&r.drafts);
            }
        }
        step.entries.push(Entry {
            handle: h,
            scheduled: n,
            computed,
            kind: KIND_RUNNING,
            pages,
            drafts,
        });
        record_save(&mut rank.snapshots, r, computed + n, &mut step);
        take_budget(rank, &mut token_budget, n);
        req_index += 1;
    }

    // Admission, only when nothing was preempted.
    let mut skipped = Queue::new(p.policy);
    if step.preempted.is_empty() {
        if let Some(mut b) = rank.budget {
            let remaining = token_budget;
            let free = rank.kv.pool().num_free() as i64;
            let admission = b.begin_cycle(free, rank.running.len() as i64, 0);
            b.consume((admission - remaining).max(0));
            token_budget = b.remaining;
            rank.budget = Some(b);
        }
        let mut reserved: Option<i64> = None;
        while !rank.waiting.is_empty() && token_budget > 0 {
            if !rank.ensure_capacity(rank.running.len() + 1) {
                break;
            }
            let h = rank.waiting.peek().expect("non-empty queue");
            let r = get_mut(reqs, h)?;
            let mut restore_key = None;
            let (hits, n_hit, num_computed) = if r.computed == 0 && rank.snapshots.is_some() {
                let (pages, tokens, key) = snapshot_prefix_hit(rank, r, p.page_size)?;
                restore_key = key;
                (pages, tokens, tokens)
            } else if r.computed == 0 {
                let (pages, tokens) = lookup(&mut rank.kv, r, None)?;
                (pages, tokens, tokens)
            } else {
                (empty_pages(&rank.kv), 0, r.computed)
            };
            let mut n = r.num_tokens() - num_computed;
            if p.long_prefill_threshold > 0 && p.long_prefill_threshold < n {
                n = p.long_prefill_threshold;
            }
            if !p.chunked_prefill && n > token_budget {
                let too_large = n >= p.token_budget;
                let batch_empty = step.entries.is_empty();
                if !too_large && !batch_empty {
                    let key = r.queue_key();
                    rank.waiting.pop();
                    skipped.push(h, key);
                    continue;
                }
            }
            n = n.min(token_budget);
            if n <= 0 {
                rank.waiting.pop();
                r.status = Status::Aborted;
                free_request(&mut rank.kv, reqs, h, finished, rank.finished_set.as_mut())?;
                continue;
            }
            if !r.spans.is_empty() {
                let aligned = align_prefill_chunk(&r.spans, num_computed, n, token_budget);
                if aligned <= 0 {
                    let key = r.queue_key();
                    rank.waiting.pop();
                    skipped.push(h, key);
                    continue;
                }
                n = aligned;
            }
            n = clip_to_snapshot(&rank.snapshots, r, num_computed, n);
            if !rank.running.is_empty() {
                let res = match reserved {
                    Some(v) => v,
                    None => {
                        let mut total = 0;
                        for &x in &rank.running {
                            total += outstanding_prompt_pages(&rank.kv, get(reqs, x)?, p)?;
                        }
                        reserved = Some(total);
                        total
                    }
                };
                let r = get(reqs, h)?;
                let needed = rank
                    .kv
                    .needed(r.kv_slot, r.num_tokens() + p.lookahead, Some(&hits), Some(p.max_chunk))?;
                if needed + res > rank.kv.pool().num_free() as i64 {
                    break;
                }
            }
            let r = get_mut(reqs, h)?;
            let slot = kv_slot(&mut rank.kv, r);
            if rank
                .kv
                .allocate(slot, r.computed, n, n_hit, Some(&hits), p.lookahead, None, None)?
                .is_none()
            {
                break;
            }
            rank.waiting.pop();
            rank.running.push(h);
            let kind = match r.status {
                Status::Waiting => KIND_NEW,
                Status::Preempted => KIND_RESUMED,
                s => return Err(CoreError::Invariant(format!("admitting a request in state {s:?}"))),
            };
            let pages = rank.kv.page_ids(slot)?.to_vec();
            take_budget(rank, &mut token_budget, n);
            let r = get_mut(reqs, h)?;
            r.status = Status::Running;
            r.computed = num_computed;
            if let Some(v) = reserved.as_mut() {
                *v += outstanding_prompt_pages(&rank.kv, r, p)?;
            }
            if let Some(key) = restore_key {
                let slot = rank
                    .snapshots
                    .as_mut()
                    .and_then(|s| s.lookup(key))
                    .ok_or_else(|| CoreError::Invariant("the snapshot of a prefix hit vanished".into()))?;
                step.restores.push((h, slot));
            }
            record_save(&mut rank.snapshots, r, num_computed + n, &mut step);
            if r.cached_tokens < 0 {
                r.cached_tokens = num_computed;
            }
            step.entries.push(Entry {
                handle: h,
                scheduled: n,
                computed: num_computed,
                kind,
                pages,
                drafts: Vec::new(),
            });
            if token_budget <= 0 {
                break;
            }
        }
    }
    if !skipped.is_empty() {
        rank.waiting.prepend_all(skipped);
    }

    rank.ensure_capacity(rank.running.len());
    let total: i64 = step.entries.iter().map(|e| e.scheduled).sum();
    if total > p.token_budget {
        return Err(CoreError::Invariant(format!(
            "rank {} scheduled {total} tokens over its budget {}",
            rank.index, p.token_budget
        )));
    }
    if token_budget < 0 {
        return Err(CoreError::Invariant(format!("negative token budget {token_budget}")));
    }
    if rank.running.len() > rank.current_bucket || step.entries.len() > rank.running.len() {
        return Err(CoreError::Invariant(format!(
            "rank {}: {} running, {} scheduled, bucket {}",
            rank.index,
            rank.running.len(),
            step.entries.len(),
            rank.current_bucket
        )));
    }
    step.budget_remaining = token_budget;
    if rank.index == 0 {
        // The former scheduler's representative request: the first running entry, else resumed, else new.
        let rep = [KIND_RUNNING, KIND_RESUMED, KIND_NEW]
            .iter()
            .find_map(|&k| step.entries.iter().find(|e| e.kind == k));
        step.common_prefix = match rep {
            Some(e) => {
                let slot = get(reqs, e.handle)?.kv_slot;
                rank.kv.common_prefix(slot, step.entries.len() as i64)
            }
            None => vec![0; rank.kv.num_groups()],
        };
    }
    Ok(step)
}
