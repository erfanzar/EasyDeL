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

//! Golden replay: every case of `tests/data/engine_goldens.npz`, captured from eSurge's former Python scheduler,
//! must be reproduced decision for decision.
//!
//! A case is one varint stream: a config block, then the calls in order. `ADD` admits a request (and names the rank
//! the former scheduler routed it to), `FINISH` finishes one (abort or stop string), `SCHEDULE` carries the expected
//! plan (per entry: request, scheduled tokens, computed tokens, whether it samples, its kind and its page ids per
//! group; then the preempted and finished sets and the snapshot restore and save pairs), and `COMMIT` carries the
//! sampled tokens of the oldest uncommitted plan and the expected events.

use std::collections::{BTreeMap, VecDeque};

use super::plan::{Plan, PLAN_STRUCTURED_PENDING};
use super::record::{decode, encode, VarintReader};
use super::request::{FinishReason, Handle, NewRequest, Sampling};
use super::sched::commit::Output;
use super::sched::Policy;
use super::{CoreError, EngineConfig, EngineCore};
use crate::kv::group::{Group, Kind};

const OP_ADD: i64 = 1;
const OP_FINISH: i64 = 2;
const OP_SCHEDULE: i64 = 3;
const OP_COMMIT: i64 = 4;
const OP_ADD2: i64 = 5;
const OP_SCHEDULE2: i64 = 6;
const OP_COMMIT2: i64 = 7;

/// The members of a stored (uncompressed) npz archive: name to `.npy` bytes.
fn npz_members(bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut pos = 0usize;
    let u16_at = |p: usize| u16::from_le_bytes([bytes[p], bytes[p + 1]]) as usize;
    let u32_at = |p: usize| u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]) as usize;
    while pos + 30 <= bytes.len() && u32_at(pos) == 0x0403_4b50 {
        assert_eq!(u16_at(pos + 8), 0, "fixture members must be stored, not compressed");
        let size = u32_at(pos + 18);
        let name_len = u16_at(pos + 26);
        let extra_len = u16_at(pos + 28);
        let name = String::from_utf8(bytes[pos + 30..pos + 30 + name_len].to_vec()).unwrap();
        let start = pos + 30 + name_len + extra_len;
        out.insert(
            name.trim_end_matches(".npy").to_string(),
            bytes[start..start + size].to_vec(),
        );
        pos = start + size;
    }
    out
}

/// The `(header text, data bytes)` of one `.npy` member.
fn npy(member: &[u8]) -> (String, &[u8]) {
    assert_eq!(&member[..6], b"\x93NUMPY");
    let (len, at) = if member[6] == 1 {
        (u16::from_le_bytes([member[8], member[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([member[8], member[9], member[10], member[11]]) as usize,
            12,
        )
    };
    let header = String::from_utf8(member[at..at + len].to_vec()).unwrap();
    (header, &member[at + len..])
}

/// A 0-d little-endian unicode array as a string.
fn npy_str(member: &[u8]) -> String {
    let (header, data) = npy(member);
    assert!(header.contains("'<U"), "expected a unicode array: {header}");
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .filter(|&c| c != 0)
        .map(|c| char::from_u32(c).unwrap())
        .collect()
}

/// The fixture's cases as `(name, log)`, with its header text.
pub(crate) fn load_cases(file: &str) -> (String, Vec<(String, Vec<u8>)>) {
    let path = format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(path).expect("golden fixture");
    let members = npz_members(&bytes);
    let header = npy_str(&members["header"]);
    let names_json = npy_str(&members["names"]);
    let names: Vec<String> = names_json
        .trim_matches(|c| c == '[' || c == ']')
        .split(", ")
        .map(|s| s.trim_matches('"').to_string())
        .collect();
    let (_, off_bytes) = npy(&members["offsets"]);
    let offsets: Vec<usize> = off_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let (_, log) = npy(&members["log"]);
    assert_eq!(names.len() + 1, offsets.len());
    let cases = names
        .into_iter()
        .enumerate()
        .map(|(i, n)| (n, log[offsets[i]..offsets[i + 1]].to_vec()))
        .collect();
    (header, cases)
}

fn err(e: CoreError) -> String {
    e.to_string()
}

fn config(c: &[i64]) -> EngineConfig {
    let ng = c[12] as usize;
    let groups = (0..ng)
        .map(|g| {
            let d = &c[13 + 4 * g..17 + 4 * g];
            let kind = match d[0] {
                0 => Kind::Full,
                1 => Kind::Sliding(d[2]),
                _ => Kind::Chunked(d[2]),
            };
            Group {
                kind,
                page_size: d[1],
                type_index: d[3],
            }
        })
        .collect();
    EngineConfig {
        max_rows: c[0] as usize,
        max_model_len: c[1],
        num_pages: c[2] as usize,
        dp_size: c[3] as usize,
        groups,
        token_budget: c[4],
        long_prefill_threshold: c[5],
        chunked_prefill: c[6] != 0,
        token_safety_margin: (c[11] >= 0).then_some(c[11]),
        policy: Policy::from_code(c[7]).unwrap(),
        prefix_caching: c[8] != 0,
        snapshot_slots: c[9],
        async_scheduling: c[10] != 0,
        req_buckets: Vec::new(),
        num_spec_tokens: 0,
        use_eagle: false,
        include_finished_set: false,
    }
}

struct Ids(BTreeMap<Handle, i64>);

impl Ids {
    fn of(&self, h: Handle) -> i64 {
        *self.0.get(&h).unwrap_or(&-1)
    }

    fn sorted(&self, hs: &[Handle]) -> Vec<i64> {
        let mut v: Vec<i64> = hs.iter().map(|&h| self.of(h)).collect();
        v.sort_unstable();
        v
    }

    fn pairs(&self, plan: &Plan, pairs: &[[i32; 2]]) -> Vec<i64> {
        let mut v: Vec<(i64, i64)> = pairs
            .iter()
            .map(|p| (self.of(plan.handles[p[0] as usize]), i64::from(p[1])))
            .collect();
        v.sort_unstable();
        v.into_iter().flat_map(|(a, b)| [a, b]).collect()
    }
}

/// Replays one case; returns the number of schedule steps.
pub(crate) fn replay(log: &[u8]) -> Result<usize, String> {
    let mut r = VarintReader::new(log);
    let n = r.next_len().map_err(err)?;
    let c = r.next_vec(n).map_err(err)?;
    let cfg = config(&c);
    let ng = cfg.groups.len();
    let mut core = EngineCore::new(&cfg).map_err(err)?;
    let mut handle_of: BTreeMap<i64, Handle> = BTreeMap::new();
    let mut ids = Ids(BTreeMap::new());
    let mut pending: VecDeque<Plan> = VecDeque::new();
    let mut steps = 0usize;
    while !r.done() {
        let op = r.next_i64().map_err(err)?;
        match op {
            OP_ADD => {
                let idx = r.next_i64().map_err(err)?;
                let n = r.next_len().map_err(err)?;
                let prompt: Vec<i32> = r.next_vec(n).map_err(err)?.into_iter().map(|v| v as i32).collect();
                let v = r.next_vec(6).map_err(err)?;
                let ns = r.next_len().map_err(err)?;
                let stops: Vec<i32> = r.next_vec(ns).map_err(err)?.into_iter().map(|v| v as i32).collect();
                let nsp = r.next_len().map_err(err)?;
                let flat = r.next_vec(2 * nsp).map_err(err)?;
                let spans: Vec<(i64, i64)> = flat.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                let rank = r.next_i64().map_err(err)? as usize;
                let req = NewRequest {
                    prompt: &prompt,
                    max_tokens: v[0],
                    min_tokens: 0,
                    eos: (v[1] >= 0).then_some(v[1] as i32),
                    ignore_eos: v[2] != 0,
                    priority: v[3],
                    arrival: v[4] as f64,
                    flags: if v[5] != 0 {
                        super::request::FLAG_NO_PREFIX_HIT
                    } else {
                        0
                    },
                    stop_ids: &stops,
                    spans: &spans,
                    dp_rank: None,
                    hash_extra: 0,
                    sampling: Sampling::default(),
                };
                let h = core.add(&[req]).map_err(err)?[0];
                handle_of.insert(idx, h);
                ids.0.insert(h, idx);
                let got = core.rank_of(h).unwrap();
                if got != rank {
                    return Err(format!(
                        "step {steps}: request {idx} routed to rank {got}, expected {rank}"
                    ));
                }
            }
            OP_FINISH => {
                let k = r.next_len().map_err(err)?;
                let victims: Vec<Handle> = r.next_vec(k).map_err(err)?.into_iter().map(|i| handle_of[&i]).collect();
                let reason = FinishReason::from_code(r.next_i64().map_err(err)?).unwrap();
                core.finish(&victims, reason).map_err(err)?;
            }
            OP_SCHEDULE => {
                let plan = core.schedule().map_err(err)?;
                let n = r.next_len().map_err(err)?;
                if plan.len() != n {
                    return Err(format!(
                        "step {steps}: {} entries, expected {n} ({:?} scheduled {:?})",
                        plan.len(),
                        plan.handles.iter().map(|&h| ids.of(h)).collect::<Vec<_>>(),
                        plan.scheduled
                    ));
                }
                for i in 0..n {
                    let e = r.next_vec(5).map_err(err)?;
                    let got = [
                        ids.of(plan.handles[i]),
                        i64::from(plan.scheduled[i]),
                        i64::from(plan.computed[i]),
                        i64::from(plan.emits[i]),
                        i64::from(plan.kinds[i]),
                    ];
                    if got[..] != e[..] {
                        return Err(format!(
                            "step {steps} entry {i}: (id, scheduled, computed, emits, kind) {got:?}, expected {e:?}"
                        ));
                    }
                    for g in 0..ng {
                        let k = r.next_len().map_err(err)?;
                        let want = r.next_vec(k).map_err(err)?;
                        let have: Vec<i64> = plan.pages(i, g).iter().map(|&p| i64::from(p)).collect();
                        if have != want {
                            return Err(format!(
                                "step {steps} entry {i} (request {}) group {g}: pages {have:?}, expected {want:?}",
                                got[0]
                            ));
                        }
                    }
                }
                for (what, have) in [("preempted", &plan.preempted), ("finished", &plan.finished)] {
                    let k = r.next_len().map_err(err)?;
                    let want = r.next_vec(k).map_err(err)?;
                    let have = ids.sorted(have);
                    if have != want {
                        return Err(format!("step {steps}: {what} {have:?}, expected {want:?}"));
                    }
                }
                for (what, pairs) in [("restores", &plan.snapshot_restores), ("saves", &plan.snapshot_saves)] {
                    let k = r.next_len().map_err(err)?;
                    let want = r.next_vec(2 * k).map_err(err)?;
                    let have = ids.pairs(&plan, pairs);
                    if have != want {
                        return Err(format!("step {steps}: snapshot {what} {have:?}, expected {want:?}"));
                    }
                }
                if decode(&encode(&plan)).map_err(err)? != plan {
                    return Err(format!("step {steps}: the step record does not round-trip"));
                }
                if !plan.is_empty() || !plan.preempted.is_empty() || !plan.finished.is_empty() {
                    pending.push_back(plan);
                }
                steps += 1;
            }
            OP_COMMIT => {
                let n = r.next_len().map_err(err)?;
                let tokens: Vec<i32> = r.next_vec(n).map_err(err)?.into_iter().map(|v| v as i32).collect();
                let plan = pending
                    .pop_front()
                    .ok_or_else(|| format!("step {steps}: commit without a pending plan"))?;
                if plan.len() != n {
                    return Err(format!(
                        "step {steps}: commit of {n} tokens for a plan of {}",
                        plan.len()
                    ));
                }
                let events = core.commit_tokens(&plan, &tokens).map_err(err)?;
                let ne = r.next_len().map_err(err)?;
                if events.len() != ne {
                    return Err(format!(
                        "step {steps}: {} events, expected {ne} (for {:?})",
                        events.len(),
                        events.handles.iter().map(|&h| ids.of(h)).collect::<Vec<_>>()
                    ));
                }
                for i in 0..ne {
                    let idx = r.next_i64().map_err(err)?;
                    let k = r.next_len().map_err(err)?;
                    let want_tokens = r.next_vec(k).map_err(err)?;
                    let tail = r.next_vec(3).map_err(err)?;
                    let have_tokens: Vec<i64> = events.tokens_of(i).iter().map(|&t| i64::from(t)).collect();
                    let have = [
                        i64::from(events.finish[i]),
                        i64::from(events.stop_token[i]),
                        i64::from(events.cached_tokens[i]),
                    ];
                    if ids.of(events.handles[i]) != idx || have_tokens != want_tokens || have[..] != tail[..] {
                        return Err(format!(
                            "step {steps} event {i}: request {} tokens {have_tokens:?} (finish, stop, cached) \
                             {have:?}; expected request {idx} tokens {want_tokens:?} {tail:?}",
                            ids.of(events.handles[i])
                        ));
                    }
                }
            }
            other => return Err(format!("unknown op {other}")),
        }
        core.check().map_err(|e| format!("step {steps}: invariant: {e}"))?;
    }
    if !pending.is_empty() || core.num_inflight() != 0 {
        return Err("plans left uncommitted at the end of the case".into());
    }
    Ok(steps)
}

fn config_ext(c: &[i64]) -> EngineConfig {
    let mut cfg = config(c);
    let at = 13 + 4 * cfg.groups.len();
    cfg.num_spec_tokens = c[at];
    cfg.use_eagle = c[at + 1] != 0;
    cfg.include_finished_set = c[at + 2] != 0;
    let nb = c[at + 3] as usize;
    cfg.req_buckets = c[at + 4..at + 4 + nb].to_vec();
    cfg
}

fn ints(v: Vec<i64>) -> Vec<i32> {
    v.into_iter().map(|x| x as i32).collect()
}

/// Replays one case of the extended fixture (drafts, runner-output gaps, multimodal keys, buckets, the finished
/// set, the structured-output flag); returns the number of schedule steps.
pub(crate) fn replay_ext(log: &[u8]) -> Result<usize, String> {
    let mut r = VarintReader::new(log);
    let n = r.next_len().map_err(err)?;
    let c = r.next_vec(n).map_err(err)?;
    let cfg = config_ext(&c);
    let ng = cfg.groups.len();
    let mut core = EngineCore::new(&cfg).map_err(err)?;
    let mut handle_of: BTreeMap<i64, Handle> = BTreeMap::new();
    let mut ids = Ids(BTreeMap::new());
    let mut pending: VecDeque<Plan> = VecDeque::new();
    let mut steps = 0usize;
    while !r.done() {
        let op = r.next_i64().map_err(err)?;
        match op {
            OP_ADD2 => {
                let idx = r.next_i64().map_err(err)?;
                let n = r.next_len().map_err(err)?;
                let prompt = ints(r.next_vec(n).map_err(err)?);
                let v = r.next_vec(6).map_err(err)?;
                let ns = r.next_len().map_err(err)?;
                let stops = ints(r.next_vec(ns).map_err(err)?);
                let nsp = r.next_len().map_err(err)?;
                let flat = r.next_vec(2 * nsp).map_err(err)?;
                let spans: Vec<(i64, i64)> = flat.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                let h4 = r.next_vec(4).map_err(err)?;
                let hi = ((h4[0] as u64) << 32) | h4[1] as u64;
                let lo = ((h4[2] as u64) << 32) | h4[3] as u64;
                let rank = r.next_i64().map_err(err)? as usize;
                let req = NewRequest {
                    prompt: &prompt,
                    max_tokens: v[0],
                    min_tokens: 0,
                    eos: (v[1] >= 0).then_some(v[1] as i32),
                    ignore_eos: v[2] != 0,
                    priority: v[3],
                    arrival: v[4] as f64,
                    flags: v[5] as u32,
                    stop_ids: &stops,
                    spans: &spans,
                    dp_rank: None,
                    hash_extra: (u128::from(hi) << 64) | u128::from(lo),
                    sampling: Sampling::default(),
                };
                let h = core.add(&[req]).map_err(err)?[0];
                handle_of.insert(idx, h);
                ids.0.insert(h, idx);
                let got = core.rank_of(h).unwrap();
                if got != rank {
                    return Err(format!(
                        "step {steps}: request {idx} routed to rank {got}, expected {rank}"
                    ));
                }
            }
            OP_FINISH => {
                let k = r.next_len().map_err(err)?;
                let victims: Vec<Handle> = r.next_vec(k).map_err(err)?.into_iter().map(|i| handle_of[&i]).collect();
                let reason = FinishReason::from_code(r.next_i64().map_err(err)?).unwrap();
                core.finish(&victims, reason).map_err(err)?;
            }
            OP_SCHEDULE2 => {
                let plan = core.schedule().map_err(err)?;
                let n = r.next_len().map_err(err)?;
                if plan.len() != n {
                    return Err(format!(
                        "step {steps}: {} entries, expected {n} ({:?} scheduled {:?})",
                        plan.len(),
                        plan.handles.iter().map(|&h| ids.of(h)).collect::<Vec<_>>(),
                        plan.scheduled
                    ));
                }
                for i in 0..n {
                    let e = r.next_vec(5).map_err(err)?;
                    let got = [
                        ids.of(plan.handles[i]),
                        i64::from(plan.scheduled[i]),
                        i64::from(plan.computed[i]),
                        i64::from(plan.emits[i]),
                        i64::from(plan.kinds[i]),
                    ];
                    if got[..] != e[..] {
                        return Err(format!(
                            "step {steps} entry {i}: (id, scheduled, computed, emits, kind) {got:?}, expected {e:?}"
                        ));
                    }
                    for g in 0..ng {
                        let k = r.next_len().map_err(err)?;
                        let want = r.next_vec(k).map_err(err)?;
                        let have: Vec<i64> = plan.pages(i, g).iter().map(|&p| i64::from(p)).collect();
                        if have != want {
                            return Err(format!(
                                "step {steps} entry {i} (request {}) group {g}: pages {have:?}, expected {want:?}",
                                got[0]
                            ));
                        }
                    }
                    let k = r.next_len().map_err(err)?;
                    let want = ints(r.next_vec(k).map_err(err)?);
                    if plan.drafts(i) != want.as_slice() {
                        return Err(format!(
                            "step {steps} entry {i} (request {}): drafts {:?}, expected {want:?}",
                            got[0],
                            plan.drafts(i)
                        ));
                    }
                    let k = r.next_len().map_err(err)?;
                    let want = ints(r.next_vec(k).map_err(err)?);
                    let o = &plan.new_token_offsets;
                    let have = &plan.new_tokens[o[i] as usize..o[i + 1] as usize];
                    if have != want.as_slice() {
                        return Err(format!(
                            "step {steps} entry {i} (request {}): new tokens {have:?}, expected {want:?}",
                            got[0]
                        ));
                    }
                }
                for (what, have) in [("preempted", &plan.preempted), ("finished", &plan.finished)] {
                    let k = r.next_len().map_err(err)?;
                    let want = r.next_vec(k).map_err(err)?;
                    let have = ids.sorted(have);
                    if have != want {
                        return Err(format!("step {steps}: {what} {have:?}, expected {want:?}"));
                    }
                }
                for (what, pairs) in [("restores", &plan.snapshot_restores), ("saves", &plan.snapshot_saves)] {
                    let k = r.next_len().map_err(err)?;
                    let want = r.next_vec(2 * k).map_err(err)?;
                    let have = ids.pairs(&plan, pairs);
                    if have != want {
                        return Err(format!("step {steps}: snapshot {what} {have:?}, expected {want:?}"));
                    }
                }
                let tail = r.next_vec(3).map_err(err)?;
                let cp = r.next_vec(tail[2] as usize).map_err(err)?;
                let structured = i64::from(plan.flags & PLAN_STRUCTURED_PENDING != 0);
                if structured != tail[0] || plan.req_bucket != tail[1] {
                    return Err(format!(
                        "step {steps}: (structured pending, req_bucket) ({structured}, {}), expected ({}, {})",
                        plan.req_bucket, tail[0], tail[1]
                    ));
                }
                let have_cp: Vec<i64> = plan.common_prefix_pages.iter().map(|&v| i64::from(v)).collect();
                if have_cp != cp {
                    return Err(format!(
                        "step {steps}: common prefix pages {have_cp:?}, expected {cp:?}"
                    ));
                }
                if decode(&encode(&plan)).map_err(err)? != plan {
                    return Err(format!("step {steps}: the step record does not round-trip"));
                }
                if !plan.is_empty() || !plan.preempted.is_empty() || !plan.finished.is_empty() {
                    pending.push_back(plan);
                }
                steps += 1;
            }
            OP_COMMIT2 => {
                let n = r.next_len().map_err(err)?;
                let mut present = Vec::with_capacity(n);
                let mut sampled = Vec::new();
                let mut offsets = vec![0i32];
                let mut accepted = Vec::with_capacity(n);
                for _ in 0..n {
                    present.push(r.next_i64().map_err(err)? as u8);
                    let k = r.next_len().map_err(err)?;
                    sampled.extend(ints(r.next_vec(k).map_err(err)?));
                    offsets.push(sampled.len() as i32);
                    accepted.push(r.next_i64().map_err(err)? as i32);
                }
                let spec_none = r.next_i64().map_err(err)? != 0;
                let mut drafts = Vec::new();
                let mut draft_offsets = vec![0i32];
                if !spec_none {
                    for _ in 0..n {
                        let k = r.next_len().map_err(err)?;
                        drafts.extend(ints(r.next_vec(k).map_err(err)?));
                        draft_offsets.push(drafts.len() as i32);
                    }
                }
                let plan = pending
                    .pop_front()
                    .ok_or_else(|| format!("step {steps}: commit without a pending plan"))?;
                if plan.len() != n {
                    return Err(format!(
                        "step {steps}: commit of {n} entries for a plan of {}",
                        plan.len()
                    ));
                }
                let out = Output {
                    sampled: &sampled,
                    sampled_offsets: &offsets,
                    present: Some(&present),
                    accepted: Some(&accepted),
                    drafts: (!spec_none).then_some((drafts.as_slice(), draft_offsets.as_slice())),
                };
                let events = core.commit(&plan, &out).map_err(err)?;
                let ne = r.next_len().map_err(err)?;
                if events.len() != ne {
                    return Err(format!(
                        "step {steps}: {} events, expected {ne} (for {:?})",
                        events.len(),
                        events.handles.iter().map(|&h| ids.of(h)).collect::<Vec<_>>()
                    ));
                }
                for i in 0..ne {
                    let idx = r.next_i64().map_err(err)?;
                    let k = r.next_len().map_err(err)?;
                    let want_tokens = r.next_vec(k).map_err(err)?;
                    let tail = r.next_vec(4).map_err(err)?;
                    let have_tokens: Vec<i64> = events.tokens_of(i).iter().map(|&t| i64::from(t)).collect();
                    let have = [
                        i64::from(events.finish[i]),
                        i64::from(events.stop_token[i]),
                        i64::from(events.cached_tokens[i]),
                    ];
                    let lp_ok = tail[3] < 0 || i64::from(events.logprob_starts[i]) == tail[3];
                    let same = ids.of(events.handles[i]) == idx && have_tokens == want_tokens && have[..] == tail[..3];
                    if !same || !lp_ok {
                        return Err(format!(
                            "step {steps} event {i}: request {} tokens {have_tokens:?} (finish, stop, cached) \
                             {have:?} logprob start {}; expected request {idx} tokens {want_tokens:?} {tail:?}",
                            ids.of(events.handles[i]),
                            events.logprob_starts[i]
                        ));
                    }
                }
                let k = r.next_len().map_err(err)?;
                let want = r.next_vec(k).map_err(err)?;
                let have = ids.sorted(&events.finished_all);
                if have != want {
                    return Err(format!("step {steps}: finished_all {have:?}, expected {want:?}"));
                }
            }
            other => return Err(format!("unknown op {other}")),
        }
        core.check().map_err(|e| format!("step {steps}: invariant: {e}"))?;
    }
    if !pending.is_empty() || core.num_inflight() != 0 {
        return Err("plans left uncommitted at the end of the case".into());
    }
    Ok(steps)
}

#[test]
fn fixture_header_names_source_and_commit_only() {
    for (file, count) in [("engine_goldens.npz", 108), ("engine_goldens_ext.npz", 53)] {
        let (header, cases) = load_cases(file);
        assert_eq!(
            header,
            r#"{"commit": "c499246765d4", "source": "easydel.inference.esurge.scheduler"}"#
        );
        assert_eq!(cases.len(), count);
    }
}

#[test]
fn every_golden_case_reproduces_bitwise() {
    let (_, cases) = load_cases("engine_goldens.npz");
    let mut failures = Vec::new();
    let mut steps = 0;
    for (name, log) in &cases {
        match replay(log) {
            Ok(n) => steps += n,
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    assert_eq!(steps, 6760);
}

#[test]
fn every_extended_golden_case_reproduces_bitwise() {
    let (_, cases) = load_cases("engine_goldens_ext.npz");
    let mut failures = Vec::new();
    let mut steps = 0;
    for (name, log) in &cases {
        match replay_ext(log) {
            Ok(n) => steps += n,
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    assert_eq!(steps, 2866);
}
