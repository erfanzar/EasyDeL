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

//! Async scheduling with speculative drafts in flight, against synchronous scheduling.
//!
//! A simulated device runs every plan when it is scheduled, in dispatch order, as the accelerator does. Its model is
//! deterministic per position (the true token of request `i` at position `p`), and its drafter proposes `k` drafts
//! whose leading `acc(i, L)` are right (0..k, a function of the committed length `L`). A step that was scheduled while
//! the request's previous sampling step was still uncommitted starts at the plan's optimistic `computed` minus that
//! step's rejected drafts: the correction the device makes. The device writes every fed token into a paged KV store
//! through the core's page tables and, before it writes, checks that every position the step attends to holds the
//! right committed token. Each case runs synchronously and asynchronously (the next plan is scheduled before the
//! previous commit whenever the previous plan is `PLAN_ASYNC_SAFE`); both must emit the reference token streams with
//! the reference finish events.

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::engine::plan::{Plan, KIND_RUNNING, PLAN_ASYNC_SAFE};
use crate::engine::request::{Handle, NewRequest, Sampling};
use crate::engine::sched::commit::Output;
use crate::engine::sched::Policy;
use crate::engine::{EngineConfig, EngineCore};
use crate::kv::group::{Group, Kind};
use crate::kv::pool::NULL_PAGE;

const EOS: i32 = 2;

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

#[derive(Clone, Debug)]
struct Spec {
    prompt: Vec<i32>,
    max_tokens: i64,
    eos_at: Option<i64>,
    arrive: usize,
    priority: i64,
}

#[derive(Clone, Debug)]
struct Case {
    k: i64,
    groups: Vec<Group>,
    page_size: i64,
    num_pages: usize,
    dp: usize,
    budget: i64,
    max_model_len: i64,
    caching: bool,
    policy: Policy,
    reqs: Vec<Spec>,
    seed: u64,
}

impl Case {
    fn truth(&self, i: usize, pos: i64) -> i32 {
        let s = &self.reqs[i];
        let plen = s.prompt.len() as i64;
        if pos < plen {
            return s.prompt[pos as usize];
        }
        if s.eos_at == Some(pos - plen) {
            return EOS;
        }
        10 + (mix(self.seed ^ i as u64, pos as u64) % 200) as i32
    }

    fn acc(&self, i: usize, len: i64) -> i64 {
        (mix(self.seed ^ (1000 + i as u64), len as u64) % (self.k as u64 + 1)) as i64
    }

    fn drafts_for(&self, i: usize, len: i64) -> Vec<i32> {
        let a = self.acc(i, len);
        (0..self.k)
            .map(|j| if j < a { self.truth(i, len + j) } else { 5000 + j as i32 })
            .collect()
    }

    /// The tokens, finish code and stop token request `i` must end with.
    fn reference(&self, i: usize) -> (Vec<i32>, i8, i32) {
        let s = &self.reqs[i];
        let plen = s.prompt.len() as i64;
        let mut out = Vec::new();
        for o in 0.. {
            let t = self.truth(i, plen + o);
            out.push(t);
            if plen + o + 1 >= self.max_model_len || o + 1 >= s.max_tokens {
                return (out, 2, -1);
            }
            if t == EOS {
                return (out, 1, -1);
            }
        }
        unreachable!()
    }

    fn config(&self, async_scheduling: bool) -> EngineConfig {
        EngineConfig {
            max_rows: 8,
            max_model_len: self.max_model_len,
            num_pages: self.num_pages,
            dp_size: self.dp,
            groups: self.groups.clone(),
            token_budget: self.budget,
            long_prefill_threshold: 0,
            chunked_prefill: true,
            token_safety_margin: None,
            policy: self.policy,
            prefix_caching: self.caching,
            snapshot_slots: 0,
            async_scheduling,
            req_buckets: Vec::new(),
            num_spec_tokens: self.k,
            use_eagle: false,
            include_finished_set: false,
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Stats {
    preempted: usize,
    stale: usize,
    safe_plans: usize,
    mid_window_stops: usize,
    accepted: Vec<usize>,
}

struct Dev {
    tokens: Vec<i32>,
    drafts: Vec<i32>,
    last_rej: i64,
}

struct StepOut {
    sampled: Vec<i32>,
    offsets: Vec<i32>,
    accepted: Vec<i32>,
    drafts: Vec<i32>,
    draft_offsets: Vec<i32>,
    sampled_len: Vec<usize>,
}

type Outcome = BTreeMap<usize, (Vec<i32>, i8, i32)>;

struct Sim<'a> {
    case: &'a Case,
    async_: bool,
    core: EngineCore,
    idx: HashMap<Handle, usize>,
    dev: Vec<Dev>,
    inflight: HashMap<usize, usize>,
    store: HashMap<u32, Vec<Option<(i64, i32)>>>,
    out: Outcome,
    stats: Stats,
}

/// The request's page ids per group (global; the null page where a window dropped one), as the runner sees them.
fn tables(core: &EngineCore, h: Handle) -> Vec<Vec<u32>> {
    let r = core.reqs.get(h).expect("a scheduled request is live");
    let rank = &core.ranks[r.rank];
    let slot = r.kv_slot.expect("a scheduled request holds a KV slot");
    rank.kv
        .get(slot)
        .expect("an open KV slot")
        .pages
        .iter()
        .map(|g| g.iter().map(|&p| rank.global_page(p)).collect())
        .collect()
}

impl<'a> Sim<'a> {
    fn new(case: &'a Case, async_: bool) -> Self {
        Self {
            case,
            async_,
            core: EngineCore::new(&case.config(async_)).unwrap(),
            idx: HashMap::new(),
            dev: case
                .reqs
                .iter()
                .map(|s| Dev {
                    tokens: s.prompt.clone(),
                    drafts: Vec::new(),
                    last_rej: 0,
                })
                .collect(),
            inflight: HashMap::new(),
            store: HashMap::new(),
            out: Outcome::new(),
            stats: Stats {
                accepted: vec![0; case.k as usize + 1],
                ..Stats::default()
            },
        }
    }

    fn page(&self, table: &[u32], pos: i64, what: &str) -> u32 {
        let page = table
            .get((pos / self.case.page_size) as usize)
            .copied()
            .unwrap_or(NULL_PAGE);
        assert_ne!(page, NULL_PAGE, "{what} position {pos} has no page");
        page
    }

    #[allow(clippy::needless_range_loop)] // entries are addressed by index across several plan columns
    fn execute(&mut self, plan: &Plan, stale: &[bool]) -> StepOut {
        let k = self.case.k;
        let ps = self.case.page_size;
        let mut writes = Vec::new();
        let mut entries = Vec::new();
        for i in 0..plan.len() {
            let h = plan.handles[i];
            let ri = self.idx[&h];
            let drafts = plan.drafts(i);
            let d = drafts.len() as i64;
            let s = i64::from(plan.scheduled[i]);
            let c = i64::from(plan.computed[i]);
            let dev = &self.dev[ri];
            let s0 = if stale[i] { c - dev.last_rej } else { c };
            let len = dev.tokens.len() as i64;
            let emits = plan.emits[i] == 1;
            if emits {
                assert_eq!(
                    s0 + s - d,
                    len,
                    "entry {i} (request {ri}) does not feed up to its committed length"
                );
            } else {
                assert!(s0 + s - d < len, "a non-sampling entry feeds past the committed tokens");
            }
            assert_eq!(plan.spec_slots[i], if emits { (1 + d) as i32 } else { 0 });
            for (j, &t) in drafts.iter().enumerate() {
                assert!(j < dev.drafts.len(), "request {ri} verifies a draft it was never given");
                if self.async_ {
                    assert!(t == -1 || t == dev.drafts[j]);
                } else {
                    assert_eq!(t, dev.drafts[j]);
                }
            }
            if stale[i] {
                // The request's previous step is still uncommitted: a full window of placeholder drafts.
                assert!(
                    s == 1 + k && d == k && drafts.iter().all(|&t| t == -1),
                    "stale entry {i}: {s} {drafts:?}"
                );
                self.stats.stale += 1;
            }
            let tabs = tables(&self.core, h);
            for p in s0..s0 + s {
                let t = if p < len {
                    dev.tokens[p as usize]
                } else {
                    dev.drafts[(p - len) as usize]
                };
                for table in &tabs {
                    writes.push((self.page(table, p, "a written"), (p % ps) as usize, (p, t)));
                }
            }
            entries.push((ri, s0, tabs));
        }
        // Every layer writes the step's KV before it attends.
        for (page, off, v) in writes {
            self.store.entry(page).or_insert_with(|| vec![None; ps as usize])[off] = Some(v);
        }
        for (ri, s0, tabs) in &entries {
            for (g, table) in tabs.iter().enumerate() {
                let lo = match self.case.groups[g].kind {
                    Kind::Full => 0,
                    Kind::Sliding(w) => (s0 - w + 1).max(0),
                    Kind::Chunked(c) => s0 / c * c,
                };
                for p in lo..*s0 {
                    let page = self.page(table, p, "an attended");
                    let have = self.store.get(&page).and_then(|v| v[(p % ps) as usize]);
                    let want = (p, self.dev[*ri].tokens[p as usize]);
                    assert_eq!(
                        have,
                        Some(want),
                        "request {ri} group {g} reads a wrong KV at position {p}"
                    );
                }
            }
        }
        let mut o = StepOut {
            sampled: Vec::new(),
            offsets: vec![0],
            accepted: Vec::new(),
            drafts: Vec::new(),
            draft_offsets: vec![0],
            sampled_len: Vec::new(),
        };
        for i in 0..plan.len() {
            let ri = self.idx[&plan.handles[i]];
            let d = plan.drafts(i).len();
            let mut emitted = Vec::new();
            let mut accepted = -1;
            if plan.emits[i] == 1 {
                let case = self.case;
                let dev = &mut self.dev[ri];
                let len = dev.tokens.len() as i64;
                let mut a = 0usize;
                while a < d && dev.drafts[a] == case.truth(ri, len + a as i64) {
                    a += 1;
                }
                emitted.extend_from_slice(&dev.drafts[..a]);
                emitted.push(case.truth(ri, len + a as i64));
                dev.tokens.extend_from_slice(&emitted);
                dev.last_rej = (d - a) as i64;
                dev.drafts = case.drafts_for(ri, dev.tokens.len() as i64);
                o.drafts.extend_from_slice(&dev.drafts);
                if d > 0 {
                    accepted = a as i32;
                    self.stats.accepted[a] += 1;
                }
            }
            o.sampled_len.push(emitted.len());
            o.sampled.extend_from_slice(&emitted);
            o.offsets.push(o.sampled.len() as i32);
            o.accepted.push(accepted);
            o.draft_offsets.push(o.drafts.len() as i32);
        }
        o
    }

    fn commit(&mut self, plan: &Plan, o: &StepOut) {
        let out = Output {
            sampled: &o.sampled,
            sampled_offsets: &o.offsets,
            present: None,
            accepted: Some(&o.accepted),
            drafts: Some((&o.drafts, &o.draft_offsets)),
        };
        let ev = self.core.commit(plan, &out).unwrap();
        self.core.check().unwrap();
        for e in 0..ev.len() {
            let h = ev.handles[e];
            let ri = self.idx[&h];
            let toks = ev.tokens_of(e);
            let entry = plan.handles.iter().position(|&x| x == h).unwrap();
            if toks.len() < o.sampled_len[entry] {
                self.stats.mid_window_stops += 1;
            }
            let rec = self.out.entry(ri).or_insert_with(|| (Vec::new(), 0, -1));
            assert_eq!(rec.1, 0, "request {ri} received tokens after it finished");
            rec.0.extend_from_slice(toks);
            if ev.finish[e] != 0 {
                rec.1 = ev.finish[e];
                rec.2 = ev.stop_token[e];
            }
        }
        for i in 0..plan.len() {
            if plan.emits[i] == 1 {
                *self.inflight.get_mut(&self.idx[&plan.handles[i]]).unwrap() -= 1;
            }
        }
    }

    fn run(mut self) -> (Outcome, Stats) {
        let case = self.case;
        let mut pending: VecDeque<(Plan, StepOut)> = VecDeque::new();
        let mut admitted = 0usize;
        for step in 0..20_000usize {
            for (i, s) in case.reqs.iter().enumerate().filter(|(_, s)| s.arrive == step) {
                let req = NewRequest {
                    prompt: &s.prompt,
                    max_tokens: s.max_tokens,
                    min_tokens: 0,
                    eos: Some(EOS),
                    ignore_eos: false,
                    priority: s.priority,
                    arrival: i as f64,
                    flags: 0,
                    stop_ids: &[],
                    spans: &[],
                    dp_rank: None,
                    hash_extra: 0,
                    sampling: Sampling::default(),
                };
                let h = self.core.add(&[req]).unwrap()[0];
                self.idx.insert(h, i);
                admitted += 1;
            }
            if admitted == case.reqs.len() && self.core.reqs.is_empty() && pending.is_empty() {
                assert_eq!(self.core.num_inflight(), 0);
                return (self.out, self.stats);
            }
            let plan = self.core.schedule().unwrap();
            self.core.check().unwrap();
            self.stats.preempted += plan.preempted.len();
            let safe = plan.flags & PLAN_ASYNC_SAFE != 0;
            if !plan.is_empty() {
                if safe {
                    self.stats.safe_plans += 1;
                    for i in 0..plan.len() {
                        let d = plan.drafts(i).len() as i64;
                        let s = i64::from(plan.scheduled[i]);
                        assert!(plan.kinds[i] == KIND_RUNNING && plan.emits[i] == 1);
                        assert!((d == case.k && s == 1 + case.k) || (d == 0 && s == 1));
                        assert!(i64::from(plan.computed[i]) + 2 * s <= case.max_model_len);
                    }
                }
                let stale: Vec<bool> = plan
                    .handles
                    .iter()
                    .map(|h| self.inflight.get(&self.idx[h]).copied().unwrap_or(0) > 0)
                    .collect();
                let o = self.execute(&plan, &stale);
                for i in 0..plan.len() {
                    if plan.emits[i] == 1 {
                        *self.inflight.entry(self.idx[&plan.handles[i]]).or_insert(0) += 1;
                    }
                }
                pending.push_back((plan, o));
            }
            let keep = usize::from(self.async_ && safe && !pending.is_empty());
            while pending.len() > keep {
                let (p, o) = pending.pop_front().unwrap();
                self.commit(&p, &o);
            }
        }
        panic!("the case did not finish in 20000 steps");
    }
}

fn spec(prompt: Vec<i32>, max_tokens: i64, eos_at: Option<i64>, arrive: usize) -> Spec {
    Spec {
        prompt,
        max_tokens,
        eos_at,
        arrive,
        priority: 0,
    }
}

fn requests(seed: u64) -> Vec<Spec> {
    let tok =
        |n: usize, salt: u64| -> Vec<i32> { (0..n).map(|j| 10 + (mix(seed ^ salt, j as u64) % 300) as i32).collect() };
    let shared = tok(12, 7);
    let with = |tail: Vec<i32>| [shared.clone(), tail].concat();
    let mut reqs = vec![
        spec(with(tok(2, 1)), 24, None, 0),
        spec(with(tok(1, 2)), 18, Some(5), 0),
        spec(tok(7, 3), 30, Some(13), 2),
        spec(with(tok(3, 4)), 9, None, 4),
        spec(tok(40, 5), 80, None, 6),
        spec(tok(5, 6), 1, None, 6),
        spec(with(tok(4, 8)), 20, Some(0), 9),
        spec(tok(9, 9), 26, Some(17), 13),
        spec(with(tok(2, 10)), 33, None, 13),
    ];
    for (i, r) in reqs.iter_mut().enumerate() {
        r.priority = (i as i64 * 5) % 3;
    }
    reqs
}

fn cases() -> Vec<(String, Case)> {
    let full = Group {
        kind: Kind::Full,
        page_size: 4,
        type_index: 0,
    };
    let sliding = Group {
        kind: Kind::Sliding(12),
        page_size: 4,
        type_index: 1,
    };
    let mut out = Vec::new();
    let mut seed = 1u64;
    for k in [1i64, 3] {
        for (gname, groups) in [("full", vec![full]), ("full+sliding", vec![full, sliding])] {
            let tight = if groups.len() == 1 { 40 } else { 52 };
            for (pname, pages, budget) in [("roomy", 512, 64), ("tight", tight, 64), ("budget", 512, 14)] {
                for caching in [false, true] {
                    for policy in [Policy::Fcfs, Policy::Priority] {
                        seed += 1;
                        let name = format!("k{k}-{gname}-{pname}-cache{}-{policy:?}", u8::from(caching));
                        out.push((
                            name,
                            Case {
                                k,
                                groups: groups.clone(),
                                page_size: 4,
                                num_pages: pages,
                                dp: 1,
                                budget,
                                max_model_len: 96,
                                caching,
                                policy,
                                reqs: requests(seed),
                                seed,
                            },
                        ));
                    }
                }
            }
        }
        seed += 1;
        out.push((
            format!("k{k}-dp2"),
            Case {
                k,
                groups: vec![full, sliding],
                page_size: 4,
                num_pages: 256,
                dp: 2,
                budget: 32,
                max_model_len: 96,
                caching: true,
                policy: Policy::Fcfs,
                reqs: requests(seed),
                seed,
            },
        ));
    }
    out
}

#[test]
fn async_speculation_matches_sync_and_the_reference() {
    let mut total = Stats {
        accepted: vec![0; 4],
        ..Stats::default()
    };
    let mut tight_preempted = 0;
    for (name, case) in cases() {
        let reference: Outcome = (0..case.reqs.len()).map(|i| (i, case.reference(i))).collect();
        let (sync_out, _) = Sim::new(&case, false).run();
        let (async_out, stats) = Sim::new(&case, true).run();
        assert_eq!(sync_out, reference, "{name}: synchronous scheduling");
        assert_eq!(async_out, reference, "{name}: async scheduling");
        if name.contains("tight") {
            tight_preempted += stats.preempted;
        }
        total.preempted += stats.preempted;
        total.stale += stats.stale;
        total.safe_plans += stats.safe_plans;
        total.mid_window_stops += stats.mid_window_stops;
        for (t, a) in total.accepted.iter_mut().zip(&stats.accepted) {
            *t += a;
        }
    }
    // The cases exercise what they are meant to: overlapped verify steps, preemption, stops inside a window and
    // every acceptance count.
    assert!(total.stale > 100 && total.safe_plans > 100, "{total:?}");
    assert!(tight_preempted > 0, "{total:?}");
    assert!(total.mid_window_stops > 0, "{total:?}");
    assert!(total.accepted.iter().all(|&n| n > 0), "{total:?}");
}

#[test]
fn a_finished_request_drops_its_in_flight_window() {
    // One request that stops on an accepted draft while its next window is in flight: the window's results are
    // dropped, its pages return to the pool, and the next request reuses them safely.
    let full = Group {
        kind: Kind::Full,
        page_size: 4,
        type_index: 0,
    };
    let mut reqs = vec![
        spec((10..20).collect(), 40, Some(6), 0),
        spec((30..36).collect(), 12, None, 30),
    ];
    reqs[0].priority = 0;
    let case = Case {
        k: 3,
        groups: vec![full],
        page_size: 4,
        num_pages: 12,
        dp: 1,
        budget: 32,
        max_model_len: 64,
        caching: false,
        policy: Policy::Fcfs,
        reqs,
        seed: 99,
    };
    let reference: Outcome = (0..2).map(|i| (i, case.reference(i))).collect();
    let (out, stats) = Sim::new(&case, true).run();
    assert_eq!(out, reference);
    assert!(stats.stale > 0, "{stats:?}");
}
