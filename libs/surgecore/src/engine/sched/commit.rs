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

//! The commit of one step's runner output.
//!
//! Per rank, in plan order: the step's snapshot saves are committed (or cancelled for an entry absent from the
//! runner output), then each entry that is present and whose request is still live takes its new drafts, gives back
//! its rejected drafts (`scheduled drafts - accepted` computed tokens), appends its sampled tokens one by one with
//! the stop check after each (truncating at a stop), and gives back its async placeholders. A request that was
//! running when the commit started publishes its pages up to `computed - placeholders`, and a stopped request is
//! freed. Running lists and queues drop the stopped requests after the rank's entries, as the former Python
//! scheduler did.

use crate::engine::dp::RankState;
use crate::engine::plan::{Events, Plan};
use crate::engine::request::{Handle, RequestTable, Status};
use crate::engine::sched::{free_request, SchedParams};
use crate::engine::CoreError;
use crate::kv::Tokens;

/// One plan's runner output, per plan entry in plan order.
#[derive(Clone, Copy, Debug)]
pub struct Output<'a> {
    /// Every entry's sampled tokens, flat: the accepted drafts then the bonus token (empty for a prefill chunk or
    /// an absent entry).
    pub sampled: &'a [i32],
    /// Offsets into `sampled` (`n + 1` values).
    pub sampled_offsets: &'a [i32],
    /// Per entry: 1 present; 0 absent from the runner output (nothing of it is applied and its snapshot save is
    /// cancelled); 2 listed by the runner but without a sampled row (its snapshot save is committed, nothing else
    /// is applied). `None` is every entry present.
    pub present: Option<&'a [u8]>,
    /// Per entry, the accepted-draft count, -1 when absent; `None` is absent for every entry.
    pub accepted: Option<&'a [i32]>,
    /// The new drafts, flat, with `n + 1` offsets; `None` when the runner returned none (the request keeps its
    /// drafts). A present entry with an empty range gets no drafts.
    pub drafts: Option<(&'a [i32], &'a [i32])>,
}

fn check_offsets(name: &str, offsets: &[i32], n: usize, total: usize) -> Result<(), String> {
    if offsets.len() != n + 1
        || offsets[0] != 0
        || offsets[n] as usize != total
        || offsets.windows(2).any(|w| w[1] < w[0])
    {
        return Err(format!(
            "{name} must hold {} non-decreasing offsets from 0 to {total}",
            n + 1
        ));
    }
    Ok(())
}

impl<'a> Output<'a> {
    /// Sampled tokens only: every entry present, no acceptance counts, no new drafts.
    pub fn sampled(sampled: &'a [i32], sampled_offsets: &'a [i32]) -> Self {
        Self {
            sampled,
            sampled_offsets,
            present: None,
            accepted: None,
            drafts: None,
        }
    }

    /// One token per entry (-1 for none) as flat tokens and offsets.
    #[cfg(test)]
    pub fn flatten_single(next_token: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut sampled = Vec::with_capacity(next_token.len());
        let mut offsets = Vec::with_capacity(next_token.len() + 1);
        offsets.push(0);
        for &t in next_token {
            if t >= 0 {
                sampled.push(t);
            }
            offsets.push(sampled.len() as i32);
        }
        (sampled, offsets)
    }

    /// Checks the shapes against a plan of `n` entries.
    pub fn validate(&self, n: usize) -> Result<(), String> {
        check_offsets("sampled_offsets", self.sampled_offsets, n, self.sampled.len())?;
        if let Some(p) = self.present {
            if p.len() != n || p.iter().any(|&v| v > 2) {
                return Err(format!("present must hold {n} values in 0..=2"));
            }
        }
        if let Some(a) = self.accepted {
            if a.len() != n {
                return Err(format!("accepted has {} values for a plan of {n} entries", a.len()));
            }
        }
        if let Some((d, off)) = self.drafts {
            check_offsets("draft_offsets", off, n, d.len())?;
        }
        Ok(())
    }

    /// Whether entry `i` is listed in the runner output (its snapshot save is committed).
    pub fn is_listed(&self, i: usize) -> bool {
        self.present.is_none_or(|p| p[i] != 0)
    }

    /// Whether entry `i`'s output is applied.
    pub fn is_present(&self, i: usize) -> bool {
        self.present.is_none_or(|p| p[i] == 1)
    }

    /// The sampled tokens of entry `i`.
    pub fn sampled_of(&self, i: usize) -> &'a [i32] {
        &self.sampled[self.sampled_offsets[i] as usize..self.sampled_offsets[i + 1] as usize]
    }

    /// The accepted-draft count of entry `i`, if given.
    pub fn accepted_of(&self, i: usize) -> Option<i64> {
        self.accepted.and_then(|a| (a[i] >= 0).then_some(i64::from(a[i])))
    }

    /// The new drafts of entry `i`, if the runner returned drafts.
    pub fn drafts_of(&self, i: usize) -> Option<&'a [i32]> {
        self.drafts.map(|(d, off)| &d[off[i] as usize..off[i + 1] as usize])
    }
}

/// Applies `out` to the plan's requests and appends their events.
pub fn commit_plan(
    ranks: &mut [RankState],
    reqs: &mut RequestTable,
    p: &SchedParams,
    plan: &Plan,
    out: &Output<'_>,
    events: &mut Events,
) -> Result<(), CoreError> {
    for (k, rank) in ranks.iter_mut().enumerate() {
        let (lo, hi) = (plan.rank_offsets[k] as usize, plan.rank_offsets[k + 1] as usize);
        for save in &plan.snapshot_saves {
            let entry = save[0] as usize;
            if (lo..hi).contains(&entry) {
                if let Some(index) = rank.snapshots.as_mut() {
                    if out.is_listed(entry) {
                        index.commit(i64::from(save[1]));
                    } else {
                        index.cancel(i64::from(save[1]));
                    }
                }
            }
        }
        let mut stopped_running: Vec<Handle> = Vec::new();
        let mut stopped_waiting: Vec<Handle> = Vec::new();
        for i in lo..hi {
            let h = plan.handles[i];
            let Some(r) = reqs.get_mut(h) else {
                continue;
            };
            if !out.is_present(i) {
                continue;
            }
            let sampled = out.sampled_of(i);
            if let Some(d) = out.drafts_of(i) {
                r.drafts = d.to_vec();
            }
            if let Some(a) = out.accepted_of(i) {
                let rejected = (plan.drafts(i).len() as i64 - a).max(0);
                if rejected > 0 {
                    r.computed = (r.computed - rejected).max(0);
                }
            }
            let before = r.status;
            let mut stopped = false;
            let mut emitted = 0usize;
            for &token in sampled {
                r.tokens.push(token);
                emitted += 1;
                if r.check_stop(p.max_model_len) {
                    stopped = true;
                    break;
                }
            }
            if p.async_scheduling && emitted > 0 {
                r.placeholders -= emitted as i64;
                if r.placeholders < 0 {
                    return Err(CoreError::Invariant(format!(
                        "request {h:#x} has negative placeholders"
                    )));
                }
            }
            if before == Status::Running && rank.kv.caching() {
                let n = (r.computed - r.placeholders).min(r.num_tokens());
                if let Some(slot) = r.kv_slot {
                    let tokens = Tokens {
                        tokens: &r.tokens,
                        offset: 0,
                    };
                    rank.kv.publish(slot, n, Some(tokens))?;
                }
            }
            let (code, stop, cached) = (r.status.finish_code(), r.stop_reason.unwrap_or(-1), r.cached_tokens);
            if stopped {
                r.drafts.clear();
                free_request(&mut rank.kv, reqs, h, &mut rank.finished, rank.finished_set.as_mut())?;
                if before == Status::Running {
                    stopped_running.push(h);
                } else {
                    stopped_waiting.push(h);
                }
            }
            if emitted > 0 {
                events.handles.push(h);
                events.tokens.extend_from_slice(&sampled[..emitted]);
                events.token_offsets.push(events.tokens.len() as i32);
                // The entry's first position in the full sampled list, before stop truncation.
                events.logprob_starts.push(out.sampled_offsets[i]);
                events.finish.push(code);
                events.stop_token.push(stop);
                events.cached_tokens.push(cached as i32);
            }
        }
        if !stopped_running.is_empty() {
            rank.running.retain(|h| !stopped_running.contains(h));
        }
        if !stopped_waiting.is_empty() {
            rank.waiting.remove(&stopped_waiting);
        }
    }
    Ok(())
}
