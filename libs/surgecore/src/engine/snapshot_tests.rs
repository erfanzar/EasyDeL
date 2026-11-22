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

//! The opt-in recurrent-snapshot behaviours (decode snapshots, snapshot-aware routing; both off in the goldens).
//!
//! Every case runs against a device model: each recurrent row and each snapshot slot holds the exact token prefix its
//! state was computed from (restores load before a step's tokens are fed, saves store after, as the runner does), and
//! the device samples token `1000 + position`. A restore whose slot does not hold exactly the prefix the core claims,
//! or a row whose length differs from the core's computed count, fails the test. Cases run synchronously and with
//! async scheduling (the next plan dispatched before the previous one commits).

use std::collections::{HashMap, HashSet, VecDeque};

use super::plan::{Plan, KIND_NEW, KIND_RUNNING, PLAN_ASYNC_SAFE};
use super::request::{FinishReason, Handle, NewRequest, Sampling};
use super::sched::commit::Output;
use super::sched::Policy;
use super::{EngineConfig, EngineCore, SNAPSHOT_STATS};
use crate::kv::group::{Group, Kind};

const PS: i64 = 4;
const RESTORES: usize = 0;
const KEYED: usize = 2;
const EVICTIONS: usize = 3;
const CANCELS: usize = 4;
const READY: usize = 5;
const PENDING: usize = 6;
const FREE: usize = 7;
const LOST: usize = 8;
const LOST_TOKENS: usize = 9;

#[derive(Clone, Copy, Debug)]
struct Cfg {
    dp: usize,
    rows: usize,
    pages: usize,
    slots: i64,
    decode: bool,
    routing: bool,
    async_: bool,
    policy: Policy,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            dp: 1,
            rows: 4,
            pages: 128,
            slots: 4,
            decode: false,
            routing: false,
            async_: false,
            policy: Policy::Fcfs,
        }
    }
}

fn token(pos: usize) -> i32 {
    1000 + pos as i32
}

/// The engine core and a model of the device it drives.
struct Sim {
    core: EngineCore,
    cfg: Cfg,
    /// Every token the device knows per request: its prompt, then the tokens it sampled.
    seq: HashMap<Handle, Vec<i32>>,
    /// The prefix each live recurrent row was computed from.
    row: HashMap<Handle, Vec<i32>>,
    /// The prefix each written snapshot slot was computed from.
    slots: HashMap<i32, Vec<i32>>,
    /// Dispatched plans not committed yet, with their sampled tokens.
    pending: VecDeque<(Plan, Vec<i32>, Vec<i32>)>,
    /// The first admission of each request: (hit tokens, restored).
    admitted: HashMap<Handle, (i64, bool)>,
    prompt_len: HashMap<Handle, i64>,
    last_end: HashMap<Handle, i64>,
    finished: HashSet<Handle>,
    restores: usize,
    preempted: usize,
    /// Preempted requests whose next decode chunk would have ended on a page boundary (a decode save).
    boundary_victims: usize,
    clock: f64,
}

impl Sim {
    fn new(cfg: Cfg) -> Self {
        let core = EngineCore::new(&EngineConfig {
            max_rows: cfg.rows,
            max_model_len: 64,
            num_pages: cfg.pages,
            dp_size: cfg.dp,
            groups: vec![Group {
                kind: Kind::Full,
                page_size: PS,
                type_index: 0,
            }],
            token_budget: 64,
            long_prefill_threshold: 0,
            chunked_prefill: false,
            token_safety_margin: None,
            policy: cfg.policy,
            prefix_caching: true,
            snapshot_slots: cfg.slots,
            async_scheduling: cfg.async_,
            req_buckets: Vec::new(),
            num_spec_tokens: 0,
            use_eagle: false,
            include_finished_set: false,
            dp_admission_hold: 0,
            decode_snapshots: cfg.decode,
            snapshot_routing: cfg.routing,
        })
        .unwrap();
        Self {
            core,
            cfg,
            seq: HashMap::new(),
            row: HashMap::new(),
            slots: HashMap::new(),
            pending: VecDeque::new(),
            admitted: HashMap::new(),
            prompt_len: HashMap::new(),
            last_end: HashMap::new(),
            finished: HashSet::new(),
            restores: 0,
            preempted: 0,
            boundary_victims: 0,
            clock: 0.0,
        }
    }

    fn add(&mut self, prompt: &[i32], max_tokens: i64, rank: Option<usize>, priority: i64) -> Handle {
        let req = NewRequest {
            prompt,
            max_tokens,
            min_tokens: 0,
            eos: None,
            ignore_eos: true,
            priority,
            arrival: self.clock,
            flags: 0,
            stop_ids: &[],
            spans: &[],
            dp_rank: rank,
            hash_extra: 0,
            sampling: Sampling::default(),
        };
        self.clock += 1.0;
        let h = self.core.add(&[req]).unwrap()[0];
        self.seq.insert(h, prompt.to_vec());
        self.prompt_len.insert(h, prompt.len() as i64);
        h
    }

    /// Runs a plan on the device: restores, then each entry's tokens (sampling where it emits), then saves.
    fn dispatch(&mut self, plan: &Plan) -> (Vec<i32>, Vec<i32>) {
        let restored: HashMap<usize, i32> = plan.snapshot_restores.iter().map(|&[e, s]| (e as usize, s)).collect();
        let mut sampled = Vec::new();
        let mut offsets = vec![0];
        for i in 0..plan.len() {
            let h = plan.handles[i];
            let computed = plan.computed[i] as usize;
            let end = computed + plan.scheduled[i] as usize;
            if plan.kinds[i] != KIND_RUNNING {
                let state = match restored.get(&i) {
                    Some(slot) => {
                        let st = self
                            .slots
                            .get(slot)
                            .cloned()
                            .expect("a restore of a slot never written");
                        assert_eq!(
                            st.as_slice(),
                            &self.seq[&h][..computed],
                            "slot {slot} restored for {h:#x} at {computed} holds another prefix"
                        );
                        self.restores += 1;
                        st
                    }
                    None => {
                        assert_eq!(computed, 0, "{h:#x} starts at {computed} without a snapshot restore");
                        Vec::new()
                    }
                };
                if plan.kinds[i] == KIND_NEW {
                    self.admitted
                        .entry(h)
                        .or_insert((computed as i64, restored.contains_key(&i)));
                }
                self.row.insert(h, state);
            }
            let feed = self.seq[&h][computed..end].to_vec();
            let row = self.row.get_mut(&h).expect("a running request without a row");
            assert_eq!(
                row.len(),
                computed,
                "row of {h:#x} holds {} tokens, the core says {computed}",
                row.len()
            );
            row.extend_from_slice(&feed);
            if plan.emits[i] != 0 {
                let s = self.seq.get_mut(&h).unwrap();
                assert_eq!(
                    s.len(),
                    end,
                    "{h:#x} samples at {end} but the device knows {} tokens",
                    s.len()
                );
                s.push(token(end));
                sampled.push(token(end));
            }
            offsets.push(sampled.len() as i32);
            self.last_end.insert(h, end as i64);
        }
        for &[e, slot] in &plan.snapshot_saves {
            let h = plan.handles[e as usize];
            self.slots.insert(slot, self.row[&h].clone());
        }
        (sampled, offsets)
    }

    fn commit_one(&mut self) {
        let (plan, sampled, offsets) = self.pending.pop_front().unwrap();
        self.core.commit(&plan, &Output::sampled(&sampled, &offsets)).unwrap();
        self.core.check().unwrap();
    }

    fn step(&mut self) {
        let plan = self.core.schedule().unwrap();
        self.core.check().unwrap();
        for &v in &plan.preempted {
            self.preempted += 1;
            if let Some(&end) = self.last_end.get(&v) {
                if end >= self.prompt_len[&v] && (end + 1) % PS == 0 {
                    self.boundary_victims += 1;
                }
            }
            self.row.remove(&v);
        }
        self.finished.extend(plan.finished.iter().copied());
        let safe = plan.flags & PLAN_ASYNC_SAFE != 0;
        if !plan.is_empty() {
            let (sampled, offsets) = self.dispatch(&plan);
            self.pending.push_back((plan, sampled, offsets));
        }
        let keep = usize::from(self.cfg.async_ && safe && !self.pending.is_empty());
        while self.pending.len() > keep {
            self.commit_one();
        }
        self.check_rolling_cap();
    }

    fn idle(&self) -> bool {
        let c = self.core.counts();
        c[0] + c[1] == 0 && self.pending.is_empty()
    }

    fn run_until_idle(&mut self) {
        for _ in 0..5000 {
            if self.idle() {
                // One more schedule reports the last finishes.
                let plan = self.core.schedule().unwrap();
                self.finished.extend(plan.finished.iter().copied());
                return;
            }
            self.step();
        }
        panic!("requests still live after 5000 steps");
    }

    fn step_until(&mut self, done: impl Fn(&Self) -> bool) {
        for _ in 0..500 {
            self.step();
            if done(self) {
                return;
            }
        }
        panic!("the condition never held");
    }

    /// Whether the last dispatched, uncommitted plan writes a snapshot of `h`.
    fn inflight_save_of(&self, h: Handle) -> bool {
        self.pending
            .back()
            .is_some_and(|(p, _, _)| p.snapshot_saves.iter().any(|&[e, _]| p.handles[e as usize] == h))
    }

    fn check_rolling_cap(&self) {
        for rank in &self.core.ranks {
            if let Some(index) = rank.snapshots.as_ref() {
                assert!(index.num_unkeyed() as i64 <= (index.num_slots() / 2).max(1));
            }
        }
    }

    /// Snapshot counters summed over ranks.
    fn stats(&self) -> [i64; SNAPSHOT_STATS] {
        let mut out = [0i64; SNAPSHOT_STATS];
        for chunk in self.core.snapshot_stats().chunks(SNAPSHOT_STATS) {
            for (o, v) in out.iter_mut().zip(chunk) {
                *o += v;
            }
        }
        out
    }

    /// At rest: no slot pending (rolling or retiring), every slot accounted for.
    fn check_at_rest(&self) {
        assert!(self.idle());
        let s = self.stats();
        assert_eq!(s[PENDING], 0, "a slot is still pending at rest");
        assert_eq!(s[READY] + s[PENDING] + s[FREE], self.cfg.slots * self.cfg.dp as i64);
        assert!(self.core.ranks.iter().all(|r| r.retiring.is_empty()));
    }

    /// The first `n` tokens `h` saw, then a new turn.
    fn follow_up(&self, h: Handle, n: usize) -> Vec<i32> {
        let s = &self.seq[&h];
        assert!(s.len() >= n, "{h:#x} has {} tokens, {n} asked", s.len());
        let mut p = s[..n].to_vec();
        p.extend([50, 51, 52]);
        p
    }
}

fn both(mut f: impl FnMut(bool)) {
    f(false);
    f(true);
}

fn prompt6() -> Vec<i32> {
    (1..=6).collect()
}

#[test]
fn without_decode_snapshots_a_follow_up_hits_only_the_old_prompt() {
    both(|a| {
        let mut s = Sim::new(Cfg {
            async_: a,
            ..Cfg::default()
        });
        let h = s.add(&prompt6(), 7, None, 0);
        s.run_until_idle();
        let p = s.follow_up(h, 13);
        let f = s.add(&p, 1, None, 0);
        s.run_until_idle();
        assert_eq!(s.admitted[&f], (4, true), "async={a}");
        s.check_at_rest();
    });
}

/// Prompt 6, 7 outputs: the chunk ending at 12 writes the rolling slot and samples the 7th output, so the request
/// stops in the plan that wrote its slot. The slot is committed under boundary 12 (in the async case the commit's
/// `oldest_inflight` is read after the committed plan left the in-flight queue). Boundary 8 was written earlier into
/// the same slot and is not findable afterwards: a follow-up ending inside page [8, 12) restores the prompt snapshot.
#[test]
fn a_request_that_stops_in_the_plan_writing_its_slot_commits_that_boundary() {
    both(|a| {
        let mut s = Sim::new(Cfg {
            decode: true,
            async_: a,
            ..Cfg::default()
        });
        let h = s.add(&prompt6(), 7, None, 0);
        s.run_until_idle();
        assert_eq!(s.seq[&h].len(), 13);
        let st = s.stats();
        assert_eq!(st[KEYED], 1, "async={a}: one decode save at the finish");
        s.check_at_rest();
        let p = s.follow_up(h, 13);
        let f = s.add(&p, 1, None, 0);
        s.run_until_idle();
        assert_eq!(s.admitted[&f], (12, true), "async={a}");
        let p = s.follow_up(h, 10);
        let g = s.add(&p, 1, None, 0);
        s.run_until_idle();
        assert_eq!(
            s.admitted[&g],
            (4, true),
            "async={a}: no snapshot of boundary 8 survives"
        );
        s.check_at_rest();
    });
}

/// An abort between steps commits the last boundary a committed step wrote (boundary 8 here; the steps ending at 9
/// and 10 wrote nothing).
#[test]
fn an_aborted_request_keeps_the_last_boundary_a_committed_step_wrote() {
    both(|a| {
        let mut s = Sim::new(Cfg {
            decode: true,
            async_: a,
            ..Cfg::default()
        });
        let h = s.add(&prompt6(), 30, None, 0);
        s.step_until(|s| s.last_end.get(&h) == Some(&10));
        assert!(!s.inflight_save_of(h));
        assert_eq!(s.core.finish(&[h], FinishReason::Abort).unwrap(), vec![h]);
        s.run_until_idle();
        assert_eq!(s.stats()[KEYED], 1, "async={a}");
        let p = s.follow_up(h, 11);
        let f = s.add(&p, 1, None, 0);
        s.run_until_idle();
        assert_eq!(s.admitted[&f], (8, true), "async={a}");
        s.check_at_rest();
    });
}

/// An abort while an in-flight plan writes the rolling slot: the slot cannot hold the boundary the commits recorded
/// once that plan runs, so it retires and is dropped when that plan commits. Until then the prefix cache cannot be
/// reset (the slot would go back to the free list while written).
#[test]
fn an_abort_with_the_slot_write_in_flight_drops_the_slot_and_blocks_reset_until_commit() {
    let mut s = Sim::new(Cfg {
        decode: true,
        async_: true,
        ..Cfg::default()
    });
    let h = s.add(&prompt6(), 30, None, 0);
    s.step_until(|s| s.inflight_save_of(h));
    assert_eq!(s.last_end[&h], 8);
    assert_eq!(s.core.finish(&[h], FinishReason::Abort).unwrap(), vec![h]);
    assert_eq!(s.core.ranks[0].retiring.len(), 1);
    assert!(
        !s.core.reset_prefix_cache(),
        "a retiring slot is still written by an in-flight plan"
    );
    s.run_until_idle();
    let st = s.stats();
    assert_eq!(st[KEYED], 0);
    assert!(st[CANCELS] >= 1);
    s.check_at_rest();
    let p = s.follow_up(h, 9);
    let f = s.add(&p, 1, None, 0);
    s.run_until_idle();
    assert_eq!(s.admitted[&f], (4, true), "only the prompt snapshot remains");
    assert!(s.core.reset_prefix_cache());
    assert_eq!(s.stats()[READY], 0);
}

#[test]
fn reset_is_refused_while_a_live_request_holds_a_rolling_slot() {
    both(|a| {
        let mut s = Sim::new(Cfg {
            decode: true,
            async_: a,
            ..Cfg::default()
        });
        let h = s.add(&prompt6(), 30, None, 0);
        s.step_until(|s| s.last_end.get(&h).is_some_and(|&e| e > 9));
        assert!(s.core.reqs.get(h).unwrap().decode_snap.is_some());
        assert!(!s.core.reset_prefix_cache());
        s.run_until_idle();
        s.check_at_rest();
        assert!(s.core.reset_prefix_cache());
    });
}

/// Rank 0 holds the KV pages of 16 tokens of the follow-up but its snapshot was evicted; rank 1 holds a shorter hit
/// of 8 tokens with a ready snapshot.
fn routing_case(routing: bool, a: bool) -> (Sim, Handle) {
    let mut s = Sim::new(Cfg {
        dp: 2,
        rows: 8,
        slots: 1,
        routing,
        async_: a,
        ..Cfg::default()
    });
    s.add(&(1..=18).collect::<Vec<_>>(), 2, Some(0), 0);
    s.run_until_idle();
    s.add(&(500..518).collect::<Vec<_>>(), 2, Some(0), 0);
    s.run_until_idle();
    let mut p: Vec<i32> = (1..=8).collect();
    p.extend([70, 71]);
    s.add(&p, 2, Some(1), 0);
    s.run_until_idle();
    let lost_before = s.stats()[LOST];
    assert_eq!(lost_before, 0);
    let mut f: Vec<i32> = (1..=16).collect();
    f.extend([90, 91]);
    let h = s.add(&f, 1, None, 0);
    (s, h)
}

#[test]
fn snapshot_routing_prefers_a_ready_snapshot_over_a_longer_hit_without_one() {
    both(|a| {
        let (mut s, h) = routing_case(true, a);
        assert_eq!(s.core.rank_of(h), Some(1), "async={a}");
        s.run_until_idle();
        assert_eq!(s.admitted[&h], (8, true));
        assert_eq!(s.stats()[LOST], 0);
        s.check_at_rest();
    });
}

#[test]
fn kv_routing_follows_pages_whose_snapshot_is_gone_and_counts_the_lost_hit() {
    both(|a| {
        let (mut s, h) = routing_case(false, a);
        assert_eq!(
            s.core.rank_of(h),
            Some(0),
            "async={a}: rank 0 holds 16 tokens of KV pages"
        );
        s.run_until_idle();
        assert_eq!(s.admitted[&h], (0, false), "but no snapshot: nothing is reused");
        let st = s.stats();
        assert_eq!((st[LOST], st[LOST_TOKENS]), (1, 16));
        s.check_at_rest();
    });
}

/// Lost hits count only KV hits a missing snapshot blocked: never a cold admission, never two identical prompts
/// admitted in the same step (the first one's pages are not published yet), never a hit taken in full.
#[test]
fn lost_hits_stay_zero_without_a_blocked_kv_hit() {
    both(|a| {
        let mut s = Sim::new(Cfg {
            rows: 8,
            async_: a,
            ..Cfg::default()
        });
        for base in [100, 200, 300] {
            s.add(&(base..base + 10).collect::<Vec<_>>(), 3, None, 0);
        }
        let twin: Vec<i32> = (400..410).collect();
        s.add(&twin, 3, None, 0);
        s.add(&twin, 3, None, 0);
        s.run_until_idle();
        let resend = s.add(&twin, 1, None, 0);
        s.run_until_idle();
        assert_eq!(s.admitted[&resend], (8, true));
        let st = s.stats();
        assert_eq!((st[LOST], st[LOST_TOKENS]), (0, 0), "async={a}");
        s.check_at_rest();
    });
}

fn system_prompt(i: i32) -> Vec<i32> {
    // A shared 12-token system prompt and a unique 6-token tail: the prompt snapshot lands at 16.
    let mut p: Vec<i32> = (1..=12).collect();
    p.extend((0..6).map(|k| 100 + 10 * i + k));
    p
}

/// Multi-turn traffic: 16 conversations in waves of 4 concurrent first turns (18-token prompt, 24 outputs: 42 tokens,
/// last decode boundary 40), each wave's follow-ups sent once that wave finished. 8 slots on one rank. Returns
/// (evictions, follow-ups hitting the prompt snapshot (16), follow-ups hitting the whole history (40), reused tokens).
fn multi_turn_workload(decode: bool) -> (i64, usize, usize, i64) {
    let mut s = Sim::new(Cfg {
        rows: 8,
        pages: 512,
        slots: 8,
        decode,
        ..Cfg::default()
    });
    let (mut prompt_hits, mut full_hits, mut reused) = (0, 0, 0);
    for wave in 0..4 {
        let first: Vec<Handle> = (0..4)
            .map(|i| s.add(&system_prompt(4 * wave + i), 24, None, 0))
            .collect();
        s.run_until_idle();
        for h in first {
            let p = s.follow_up(h, 42);
            let f = s.add(&p, 1, None, 0);
            s.run_until_idle();
            let hit = s.admitted[&f].0;
            reused += hit;
            prompt_hits += usize::from(hit == 16);
            full_hits += usize::from(hit == 40);
        }
    }
    s.check_at_rest();
    (s.stats()[EVICTIONS], prompt_hits, full_hits, reused)
}

/// 16 prompts sent through 8 LRU slots in a cycle, then resent in the same order: every resend misses (each one's
/// snapshot was evicted by the 8 after it). Returns (evictions, resend hits /16).
fn cyclic_resend_workload(decode: bool) -> (i64, usize) {
    let mut s = Sim::new(Cfg {
        rows: 8,
        pages: 512,
        slots: 8,
        decode,
        ..Cfg::default()
    });
    let prompts: Vec<Vec<i32>> = (0..16).map(system_prompt).collect();
    for p in &prompts {
        s.add(p, 24, None, 0);
    }
    s.run_until_idle();
    let mut hits = 0;
    for p in &prompts {
        let r = s.add(p, 1, None, 0);
        s.run_until_idle();
        hits += usize::from(s.admitted[&r].0 == 16);
    }
    s.check_at_rest();
    (s.stats()[EVICTIONS], hits)
}

#[test]
fn rolling_slots_trade_prompt_snapshots_for_whole_history_hits() {
    let off = multi_turn_workload(false);
    let on = multi_turn_workload(true);
    eprintln!("multi-turn (16 conversations, waves of 4, 8 slots): (evictions, prompt hits /16, full-history hits /16, reused tokens)");
    eprintln!("  decode snapshots off: {off:?}");
    eprintln!("  decode snapshots on:  {on:?}");
    assert_eq!(
        (off.1, off.2),
        (16, 0),
        "without decode snapshots every follow-up reuses only the old prompt"
    );
    assert_eq!(
        on.2, 16,
        "with decode snapshots every follow-up reuses its whole first turn"
    );
    assert!(on.3 > off.3);
    let cyc_off = cyclic_resend_workload(false);
    let cyc_on = cyclic_resend_workload(true);
    eprintln!("cyclic resend (16 prompts, 8 slots): (evictions, resend hits /16)");
    eprintln!("  decode snapshots off: {cyc_off:?}");
    eprintln!("  decode snapshots on:  {cyc_on:?}");
    assert_eq!(cyc_off.1, 0, "a cycle longer than the LRU pool never hits");
    assert!(cyc_on.0 >= cyc_off.0, "rolling slots only add evictions");
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

#[derive(Default, Debug)]
struct Tally {
    restores: usize,
    keyed: i64,
    evictions: i64,
    preempted: usize,
    boundary_victims: usize,
    aborts: usize,
}

fn random_workload(cfg: Cfg, seed: u64, t: &mut Tally) {
    let mut s = Sim::new(cfg);
    let mut rng = Rng(seed);
    let system: Vec<i32> = (1..=8).collect();
    let mut started: Vec<Handle> = Vec::new();
    for _ in 0..300 {
        if started.len() < 30 && rng.below(3) == 0 {
            let done: Vec<Handle> = started.iter().copied().filter(|h| s.finished.contains(h)).collect();
            let mut p = if !done.is_empty() && rng.below(2) == 0 {
                let h = done[rng.below(done.len())];
                let seq = &s.seq[&h];
                let lo = s.prompt_len[&h] as usize;
                let n = (lo + rng.below(seq.len() - lo + 1)).min(36);
                seq[..n].to_vec()
            } else {
                system.clone()
            };
            for _ in 0..1 + rng.below(8) {
                p.push(100 + rng.below(50) as i32);
            }
            p.truncate(40);
            let max_tokens = 1 + rng.below(18) as i64;
            let priority = rng.below(3) as i64;
            started.push(s.add(&p, max_tokens, None, priority));
        }
        if rng.below(25) == 0 {
            let live: Vec<Handle> = started.iter().copied().filter(|h| !s.finished.contains(h)).collect();
            if !live.is_empty() {
                let h = live[rng.below(live.len())];
                if !s.core.finish(&[h], FinishReason::Abort).unwrap().is_empty() {
                    t.aborts += 1;
                }
            }
        }
        s.step();
    }
    s.run_until_idle();
    s.check_at_rest();
    let st = s.stats();
    t.restores += s.restores;
    assert_eq!(st[RESTORES] as usize, s.restores);
    t.keyed += st[KEYED];
    t.evictions += st[EVICTIONS];
    t.preempted += s.preempted;
    t.boundary_victims += s.boundary_victims;
}

/// Random traffic (shared system prompt, follow-ups of finished requests, priorities, aborts, a page pool small
/// enough to preempt) over every policy, sync/async, one or two ranks and both flags: every restore loads exactly the
/// prefix it claims, rolling slots stay under half of a rank, and nothing stays pending at rest.
#[test]
fn random_workloads_restore_only_the_state_their_prefix_was_computed_from() {
    let mut total = Tally::default();
    for policy in [Policy::Fcfs, Policy::Priority] {
        for async_ in [false, true] {
            for dp in [1usize, 2] {
                for decode in [false, true] {
                    for routing in [false, true] {
                        let cfg = Cfg {
                            dp,
                            rows: 8,
                            pages: 40,
                            slots: 4,
                            decode,
                            routing,
                            async_,
                            policy,
                        };
                        let mut t = Tally::default();
                        for seed in 1..=6u64 {
                            random_workload(cfg, seed * 7919 + dp as u64, &mut t);
                        }
                        total.restores += t.restores;
                        total.keyed += t.keyed;
                        total.evictions += t.evictions;
                        total.preempted += t.preempted;
                        total.boundary_victims += t.boundary_victims;
                        total.aborts += t.aborts;
                    }
                }
            }
        }
    }
    eprintln!("random workloads: {total:?}");
    assert!(total.restores > 0 && total.keyed > 0 && total.aborts > 0);
    assert!(
        total.preempted > 0 && total.boundary_victims > 0,
        "preemption at a decode boundary was exercised"
    );
}
