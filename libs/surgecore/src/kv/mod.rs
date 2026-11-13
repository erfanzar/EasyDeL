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

//! The KV core: page pool, prefix-cache page keys, cache-group managers and recurrent snapshots.
//!
//! [`KvCore`] owns one page pool and the per-request page state of every request that touched it, in a slab of
//! [`ReqKv`] entries addressed by an index the caller keeps (the Python binding maps request ids to it). Allocation is
//! transactional: a failed allocation leaves the pool and the request as they were (pages that fell behind a sliding
//! window before the capacity check stay released, as before the port). Pages are published as prefix hits only by
//! [`KvCore::publish`], after their KV is computed; lookups and probes never change state.
//!
//! This module stays free of PyO3; its bindings live in the extension module's `kv` binding file.

pub mod group;
pub mod hash;
pub mod pool;
pub mod search;
pub mod snapshot;

use std::fmt;

use group::{cdiv, Group, GroupPages, Kind};
use hash::{KeyHasher, PageKey, ROOT};
use pool::{pages_per_shard, PagePool, NULL_PAGE};
use search::HitSearch;

/// A KV-core error; each kind maps to the Python exception the former Python KV core raised in the same situation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// `ValueError`.
    Value(String),
    /// `IndexError`.
    Index(String),
    /// `RuntimeError`.
    Runtime(String),
    /// `AssertionError`.
    Assertion(String),
    /// `KeyError`.
    Key(String),
}

impl fmt::Display for KvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(m) | Self::Index(m) | Self::Runtime(m) | Self::Assertion(m) | Self::Key(m) => f.write_str(m),
        }
    }
}

/// Tokens handed to a call that may need to compute page keys: `tokens[i]` is the request's token `offset + i`.
#[derive(Clone, Copy)]
pub struct Tokens<'a> {
    /// The token values.
    pub tokens: &'a [i32],
    /// The position of `tokens[0]` in the request.
    pub offset: usize,
}

/// One request's KV state.
#[derive(Default, Clone)]
pub struct ReqKv {
    /// Page ids per cache group (null page 0 where a window dropped a page).
    pub pages: Vec<Vec<u32>>,
    /// Published-page count per group; `None` until the request's first allocation attaches its hit.
    pub cached: Vec<Option<i64>>,
    /// The chained keys of the request's full pages, as far as they are known.
    pub keys: Vec<PageKey>,
    /// The request's 128-bit multimodal identity (0 for text), mixed into every page key.
    pub extra: u128,
}

impl ReqKv {
    fn holds_nothing(&self) -> bool {
        self.keys.is_empty() && self.pages.iter().all(Vec::is_empty) && self.cached.iter().all(Option::is_none)
    }
}

/// Configuration numbers of a [`KvCore`].
pub struct KvConfig {
    /// Pages in the pool, the null page included.
    pub num_pages: usize,
    /// The cache groups, in order.
    pub groups: Vec<Group>,
    /// Token slots are capped at this length.
    pub max_model_len: i64,
    /// Prefix caching on (forced off without groups).
    pub caching: bool,
    /// Reserve one hit page for EAGLE hidden-state recomputation.
    pub eagle: bool,
}

/// The KV core of one pool.
pub struct KvCore {
    groups: Vec<Group>,
    search: HitSearch,
    pool: PagePool,
    caching: bool,
    eagle: bool,
    max_model_len: i64,
    hasher: KeyHasher,
    reqs: Vec<Option<ReqKv>>,
    free_slots: Vec<usize>,
}

fn check_page_ids(groups: &[Vec<u32>], num_pages: usize) -> Result<(), KvError> {
    for g in groups {
        if let Some(&p) = g.iter().find(|&&p| p as usize >= num_pages) {
            return Err(KvError::Value(format!("page id {p} is outside the pool")));
        }
    }
    Ok(())
}

impl KvCore {
    /// A core over `cfg`; fails on an empty pool or a group layout prefix caching cannot serve.
    pub fn new(cfg: KvConfig) -> Result<Self, KvError> {
        let caching = cfg.caching && !cfg.groups.is_empty();
        for g in &cfg.groups {
            if g.page_size <= 0 {
                return Err(KvError::Value(format!(
                    "page_size must be positive, got {}",
                    g.page_size
                )));
            }
            match g.kind {
                Kind::Sliding(w) | Kind::Chunked(w) if w <= 0 => {
                    return Err(KvError::Value(format!("window or chunk must be positive, got {w}")));
                }
                _ => {}
            }
        }
        let search = HitSearch::select(&cfg.groups, caching)?;
        Ok(Self {
            pool: PagePool::new(cfg.num_pages, caching)?,
            groups: cfg.groups,
            search,
            caching,
            eagle: cfg.eagle,
            max_model_len: cfg.max_model_len,
            hasher: KeyHasher::new(),
            reqs: Vec::new(),
            free_slots: Vec::new(),
        })
    }

    /// Whether prefix caching is on.
    pub fn caching(&self) -> bool {
        self.caching
    }

    /// The pool.
    pub fn pool(&self) -> &PagePool {
        &self.pool
    }

    /// Number of cache groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// A fresh request slot for a request whose multimodal identity is `extra` (0 for text).
    pub fn open(&mut self, extra: u128) -> usize {
        let entry = ReqKv {
            pages: vec![Vec::new(); self.groups.len()],
            cached: vec![None; self.groups.len()],
            keys: Vec::new(),
            extra,
        };
        match self.free_slots.pop() {
            Some(s) => {
                self.reqs[s] = Some(entry);
                s
            }
            None => {
                self.reqs.push(Some(entry));
                self.reqs.len() - 1
            }
        }
    }

    fn req(&self, slot: usize) -> Result<&ReqKv, KvError> {
        self.reqs
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| KvError::Value(format!("unknown request slot {slot}")))
    }

    fn req_mut(&mut self, slot: usize) -> Result<&mut ReqKv, KvError> {
        self.reqs
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| KvError::Value(format!("unknown request slot {slot}")))
    }

    /// The request's state (`None` for a closed or unknown slot).
    pub fn get(&self, slot: usize) -> Option<&ReqKv> {
        self.reqs.get(slot).and_then(Option::as_ref)
    }

    /// Live request slots.
    pub fn num_open(&self) -> usize {
        self.reqs.iter().filter(|r| r.is_some()).count()
    }

    fn chain_page_size(&self) -> usize {
        self.groups.first().map_or(1, |g| g.page_size as usize)
    }

    /// Number of known page keys of the request.
    pub fn num_keys(&self, slot: usize) -> Result<usize, KvError> {
        Ok(self.req(slot)?.keys.len())
    }

    /// The request's page keys; computes them from `tokens` (all of the request's tokens) when none are known.
    pub fn ensure_keys(&mut self, slot: usize, tokens: Option<Tokens<'_>>) -> Result<&[PageKey], KvError> {
        let ps = self.chain_page_size();
        if self.req(slot)?.keys.is_empty() {
            if let Some(t) = tokens {
                let extra = self.req(slot)?.extra;
                let keys = self.hasher.chain(t.tokens, ps, extra);
                self.req_mut(slot)?.keys = keys;
            }
        }
        Ok(&self.req(slot)?.keys)
    }

    fn range(&self, hint: Option<i64>, dp: Option<i64>) -> Option<(u32, u32)> {
        self.pool.shard_range(hint, dp)
    }

    /// The longest prefix hit for the request (at most `max_len` tokens). Computes and keeps the request's page keys
    /// first when none are known; `no_hit` (prompt logprobs requested) then returns no hit.
    #[allow(clippy::too_many_arguments)]
    pub fn lookup(
        &mut self,
        slot: usize,
        tokens: Option<Tokens<'_>>,
        no_hit: bool,
        max_len: i64,
        hint: Option<i64>,
        dp: Option<i64>,
    ) -> Result<(GroupPages, i64), KvError> {
        if !self.caching {
            return Ok((vec![Vec::new(); self.groups.len()], 0));
        }
        self.ensure_keys(slot, tokens)?;
        if no_hit {
            return Ok((vec![Vec::new(); self.groups.len()], 0));
        }
        let range = self.range(hint, dp);
        let keys = &self.req(slot)?.keys;
        self.search
            .find(&self.groups, &self.pool, keys, max_len, range, self.eagle)
    }

    /// The hit token count of a lookup that keeps nothing: the request's known keys when it has a slot with keys,
    /// otherwise keys computed from `tokens` into scratch (with the slot's identity, else `extra`).
    pub fn probe(
        &self,
        slot: Option<usize>,
        tokens: Option<Tokens<'_>>,
        extra: u128,
        max_len: i64,
    ) -> Result<i64, KvError> {
        if !self.caching {
            return Ok(0);
        }
        let req = slot.and_then(|s| self.get(s));
        let extra = req.map_or(extra, |r| r.extra);
        let known = req.map(|r| &r.keys).filter(|k| !k.is_empty());
        let scratch;
        let keys: &[PageKey] = match known {
            Some(k) => k,
            None => {
                let t = tokens.ok_or_else(|| KvError::Value("tokens are needed to compute the page keys".into()))?;
                scratch = self.hasher.chain(t.tokens, self.chain_page_size(), extra);
                &scratch
            }
        };
        Ok(self
            .search
            .find(&self.groups, &self.pool, keys, max_len, None, self.eagle)?
            .1)
    }

    fn remove_skipped(&mut self, slot: usize, computed: i64) -> Result<(), KvError> {
        let mut req = self.reqs[slot].take().expect("live slot");
        let mut result = Ok(());
        for (g, group) in self.groups.iter().enumerate() {
            result = group.remove_skipped(&mut self.pool, &mut req.pages[g], req.cached[g], computed);
            if result.is_err() {
                break;
            }
        }
        self.reqs[slot] = Some(req);
        result
    }

    fn infer_shard(&self, req: &ReqKv, dp: i64) -> Option<i64> {
        let pps = pages_per_shard(self.pool.num_pages(), dp)? as i64;
        let mut inferred = None;
        for &p in req.pages.iter().flatten() {
            if p == NULL_PAGE {
                continue;
            }
            let shard = (i64::from(p) / pps).min(dp - 1);
            match inferred {
                None => inferred = Some(shard),
                Some(s) if s != shard => return None,
                _ => {}
            }
        }
        inferred
    }

    /// Transactional allocation of slots for `n_new` more tokens after `computed + n_hit` (the request's computed
    /// tokens plus its new prefix hit `hits`), plus `lookahead` speculative slots. Returns the newly allocated pages
    /// per group, or `None` (nothing changed) when the pool cannot hold them.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        slot: usize,
        computed: i64,
        n_new: i64,
        n_hit: i64,
        hits: Option<&GroupPages>,
        lookahead: i64,
        hint: Option<i64>,
        dp: Option<i64>,
    ) -> Result<Option<GroupPages>, KvError> {
        if n_new == 0 {
            return Err(KvError::Value("num_new_tokens must be greater than 0".into()));
        }
        self.req(slot)?;
        let ng = self.groups.len();
        let empty: GroupPages = vec![Vec::new(); ng];
        let hits = hits.unwrap_or(&empty);
        if hits.len() != ng {
            return Err(KvError::Value(format!(
                "expected hit pages for {ng} cache groups, got {}",
                hits.len()
            )));
        }
        check_page_ids(hits, self.pool.num_pages())?;
        self.remove_skipped(slot, computed)?;
        let need = (computed + n_hit + n_new + lookahead).min(self.max_model_len);
        let mut to_alloc = 0i64;
        {
            let req = self.req(slot)?;
            for (g, group) in self.groups.iter().enumerate() {
                let evictable = hits[g]
                    .iter()
                    .filter(|&&p| p != NULL_PAGE && self.pool.ref_cnt(p) == 0)
                    .count() as i64;
                to_alloc += cdiv(need, group.page_size) - hits[g].len() as i64 - req.pages[g].len() as i64 + evictable;
            }
        }
        if to_alloc > self.pool.num_free() as i64 {
            return Ok(None);
        }
        if self.caching {
            self.pool.touch(hits);
        } else if hits.iter().any(|h| !h.is_empty()) {
            return Err(KvError::Assertion(
                "Computed pages should be empty when prefix caching is disabled".into(),
            ));
        }
        {
            let req = self.req_mut(slot)?;
            for (g, hit) in hits.iter().enumerate() {
                match req.cached[g] {
                    None => {
                        if !req.pages[g].is_empty() {
                            return Err(KvError::Assertion("pages attached before the prefix hit".into()));
                        }
                        req.pages[g].extend_from_slice(hit);
                        req.cached[g] = Some(hit.len() as i64);
                    }
                    Some(_) => {
                        if !hit.is_empty() {
                            return Err(KvError::Assertion("a prefix hit for a request that holds pages".into()));
                        }
                    }
                }
            }
        }
        let mut hint = hint;
        if let Some(d) = dp.filter(|&d| d > 1) {
            if let Some(s) = self.infer_shard(self.req(slot)?, d) {
                hint = Some(s);
            }
        }
        let range = self.range(hint, dp);
        let mut new_pages: GroupPages = Vec::with_capacity(ng);
        let mut failed = None;
        for g in 0..ng {
            let have = self.req(slot)?.pages[g].len() as i64;
            let n = cdiv(need, self.groups[g].page_size) - have;
            if n <= 0 {
                new_pages.push(Vec::new());
                continue;
            }
            match self.pool.get_new(n as usize, range) {
                Ok(pages) => {
                    self.req_mut(slot)?.pages[g].extend_from_slice(&pages);
                    new_pages.push(pages);
                }
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        let Some(err) = failed else {
            return Ok(Some(new_pages));
        };
        if !matches!(err, KvError::Value(_)) {
            return Err(err);
        }
        // Roll back the groups allocated by this call, then the attached hit.
        for (g, pages) in new_pages.iter().enumerate() {
            if pages.is_empty() {
                continue;
            }
            let list = &mut self.req_mut(slot)?.pages[g];
            list.truncate(list.len() - pages.len());
            self.pool.release(pages.iter().rev().copied())?;
        }
        let req = self.req_mut(slot)?;
        for (g, hit) in hits.iter().enumerate() {
            let n = hit.len();
            if n == 0 {
                if req.pages[g].is_empty() && req.cached[g].unwrap_or(0) == 0 {
                    req.cached[g] = None;
                }
                continue;
            }
            if req.pages[g].is_empty() {
                continue;
            }
            let k = n.min(req.pages[g].len());
            req.pages[g].drain(..k);
            let c = req.cached[g].unwrap_or(0);
            req.cached[g] = if c <= n as i64 { None } else { Some(c - n as i64) };
        }
        if self.caching {
            for h in hits {
                self.pool.release(h.iter().copied())?;
            }
        }
        Ok(None)
    }

    /// Free pages the request still draws before it holds `num_tokens` slots (read-only; see `Group::needed`).
    pub fn needed(
        &self,
        slot: Option<usize>,
        num_tokens: i64,
        hits: Option<&GroupPages>,
        max_step: Option<i64>,
    ) -> Result<i64, KvError> {
        let req = slot.and_then(|s| self.get(s));
        let num_tokens = num_tokens.min(self.max_model_len);
        let mut total = 0;
        for (g, group) in self.groups.iter().enumerate() {
            let held: &[u32] = req.map_or(&[], |r| &r.pages[g]);
            let hit: &[u32] = hits.and_then(|h| h.get(g)).map_or(&[], Vec::as_slice);
            total += group.needed(&self.pool, held, hit, num_tokens, max_step);
        }
        Ok(total)
    }

    fn close_if_empty(&mut self, slot: usize) -> bool {
        if self.get(slot).is_some_and(ReqKv::holds_nothing) {
            self.reqs[slot] = None;
            self.free_slots.push(slot);
            return true;
        }
        false
    }

    /// Frees every page of the request, newest first per group, and forgets its published-page counts. Closes the
    /// slot when the request then holds nothing (no pages and no keys); returns whether it did.
    pub fn free(&mut self, slot: usize) -> Result<bool, KvError> {
        for g in 0..self.groups.len() {
            let req = self.req_mut(slot)?;
            let pages = std::mem::take(&mut req.pages[g]);
            req.cached[g] = None;
            self.pool.release(pages.iter().rev().copied())?;
        }
        Ok(self.close_if_empty(slot))
    }

    /// Forgets the request's page keys; closes its slot when it then holds nothing (returns whether it did).
    pub fn forget_keys(&mut self, slot: usize) -> Result<bool, KvError> {
        self.req_mut(slot)?.keys = Vec::new();
        Ok(self.close_if_empty(slot))
    }

    /// The token range `[lo, hi)` a [`publish`](Self::publish) of `n` computed tokens must be given to compute the
    /// page keys it is missing (`None`: no tokens needed).
    pub fn publish_tokens_needed(&self, slot: usize, n: i64) -> Result<Option<(usize, usize)>, KvError> {
        let req = self.req(slot)?;
        let mut known = req.keys.len();
        let (mut lo, mut hi) = (usize::MAX, 0usize);
        for (g, group) in self.groups.iter().enumerate() {
            let Some(cached) = req.cached[g] else { break };
            let full = n.div_euclid(group.page_size);
            if cached >= full {
                continue;
            }
            if (known as i64) < cached {
                break;
            }
            let ps = group.page_size as usize;
            let end = (full.max(0) as usize).min(req.pages[g].len());
            for j in cached as usize..end.max(cached as usize) {
                if j >= known {
                    lo = lo.min(j * ps);
                    hi = hi.max((j + 1) * ps);
                    known = j + 1;
                }
            }
        }
        Ok((hi > 0).then_some((lo, hi)))
    }

    /// Publishes the request's pages made full by its first `n` computed tokens, in every group, as prefix hits.
    /// `tokens` must cover the range [`publish_tokens_needed`](Self::publish_tokens_needed) reports.
    pub fn publish(&mut self, slot: usize, n: i64, tokens: Option<Tokens<'_>>) -> Result<(), KvError> {
        if !self.caching {
            return Ok(());
        }
        let mut req = self
            .reqs
            .get_mut(slot)
            .and_then(|r| r.take())
            .ok_or_else(|| KvError::Key(format!("unknown request slot {slot}")))?;
        let result = self.publish_into(&mut req, n, tokens);
        self.reqs[slot] = Some(req);
        result
    }

    fn page_tokens<'a>(tokens: Option<Tokens<'a>>, start: usize, len: usize) -> Result<&'a [i32], KvError> {
        let t = tokens.ok_or_else(|| KvError::Value("tokens are needed to compute the page keys".into()))?;
        let lo = start
            .checked_sub(t.offset)
            .ok_or_else(|| KvError::Value("tokens do not cover the pages to publish".into()))?;
        let lo = lo.min(t.tokens.len());
        let hi = (lo + len).min(t.tokens.len());
        Ok(&t.tokens[lo..hi])
    }

    fn publish_into(&mut self, req: &mut ReqKv, n: i64, tokens: Option<Tokens<'_>>) -> Result<(), KvError> {
        for (g, group) in self.groups.iter().enumerate() {
            let Some(cached) = req.cached[g] else {
                return Err(KvError::Key("request has no published-page count".into()));
            };
            let full = n.div_euclid(group.page_size);
            // A publish of fewer tokens than already published changes nothing: the count never goes back.
            if cached >= full {
                continue;
            }
            if (req.keys.len() as i64) < cached {
                return Err(KvError::Value(format!(
                    "page_hashes length ({}) must be >= num_cached_pages ({cached})",
                    req.keys.len()
                )));
            }
            let ps = group.page_size as usize;
            let mut parent = if cached == 0 {
                ROOT
            } else {
                req.keys[cached as usize - 1]
            };
            let end = (full.max(0) as usize).min(req.pages[g].len());
            for j in cached as usize..end.max(cached as usize) {
                let page = req.pages[g][j];
                if page != NULL_PAGE && self.pool.key_of(page).is_some() {
                    return Err(KvError::Runtime(format!(
                        "Expected page_hash to be None for new full page at index {}, but it has one",
                        j - cached as usize
                    )));
                }
                let key = if j < req.keys.len() {
                    req.keys[j]
                } else {
                    let slice = Self::page_tokens(tokens, j * ps, ps)?;
                    if page != NULL_PAGE && slice.len() != ps {
                        return Err(KvError::Runtime(format!(
                            "Expected {ps} tokens, got {} at {j}th page",
                            slice.len()
                        )));
                    }
                    let key = self.hasher.page_key(parent, slice, req.extra);
                    req.keys.push(key);
                    key
                };
                if page != NULL_PAGE {
                    self.pool.publish(page, key, g as u32)?;
                }
                parent = key;
            }
            req.cached[g] = Some(full);
        }
        Ok(())
    }

    /// Page ids per group of the request.
    pub fn page_ids(&self, slot: usize) -> Result<&[Vec<u32>], KvError> {
        Ok(&self.req(slot)?.pages)
    }

    /// Leading pages per group shared by at least `num_scheduled` requests (full-attention groups; 0 otherwise).
    pub fn common_prefix(&self, slot: Option<usize>, num_scheduled: i64) -> Vec<i64> {
        if matches!(self.search, HitSearch::NoCache) {
            return vec![0; self.groups.len()];
        }
        let req = slot.and_then(|s| self.get(s));
        self.groups
            .iter()
            .enumerate()
            .map(|(g, group)| group.common_prefix(&self.pool, req.map_or(&[], |r| &r.pages[g]), num_scheduled))
            .collect()
    }

    /// Releases the request's unpublished pages past those its first `keep_tokens` slots need, in every group, tail
    /// first. Returns the number of pages released.
    pub fn release_tail(&mut self, slot: usize, keep_tokens: i64) -> Result<usize, KvError> {
        let mut released = 0;
        for g in 0..self.groups.len() {
            let keep = cdiv(keep_tokens.max(0), self.groups[g].page_size).max(0) as usize;
            let req = self.req_mut(slot)?;
            let floor = keep.max(req.cached[g].unwrap_or(0).max(0) as usize);
            if req.pages[g].len() <= floor {
                continue;
            }
            let tail: Vec<u32> = req.pages[g].drain(floor..).rev().collect();
            released += tail.iter().filter(|&&p| p != NULL_PAGE).count();
            self.pool.release(tail.into_iter())?;
        }
        Ok(released)
    }

    /// Forgets every published page; refused (`false`) while any page is in use.
    pub fn reset(&mut self) -> bool {
        self.pool.reset()
    }

    /// Checks the invariants: the pool's own (free list, prefix map, null page) and that every non-null page's
    /// reference count equals the number of request page-list entries holding it.
    pub fn check(&self) -> Result<(), String> {
        self.pool.check()?;
        let mut held = vec![0u32; self.pool.num_pages()];
        for req in self.reqs.iter().flatten() {
            for &p in req.pages.iter().flatten() {
                held[p as usize] += 1;
            }
        }
        for (p, &count) in held.iter().enumerate().skip(1) {
            if self.pool.ref_cnt(p as u32) != count {
                return Err(format!(
                    "page {p}: ref_cnt {} but {count} page-list entries",
                    self.pool.ref_cnt(p as u32)
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core(groups: Vec<Group>, num_pages: usize) -> KvCore {
        KvCore::new(KvConfig {
            num_pages,
            groups,
            max_model_len: 1024,
            caching: true,
            eagle: false,
        })
        .unwrap()
    }

    fn full(ps: i64) -> Group {
        Group {
            kind: Kind::Full,
            page_size: ps,
            type_index: 0,
        }
    }

    #[test]
    fn publish_after_compute_then_hit_and_rollback() {
        let mut kv = core(vec![full(4)], 16);
        let toks: Vec<i32> = (0..13).collect();
        let t = Tokens {
            tokens: &toks,
            offset: 0,
        };
        let a = kv.open(0);
        let (hit, n) = kv.lookup(a, Some(t), false, 12, None, None).unwrap();
        assert_eq!(n, 0);
        let pages = kv.allocate(a, 0, 13, 0, Some(&hit), 0, None, None).unwrap().unwrap();
        assert_eq!(pages[0], vec![1, 2, 3, 4]);
        let b = kv.open(0);
        assert_eq!(kv.lookup(b, Some(t), false, 12, None, None).unwrap().1, 0);
        assert_eq!(kv.publish_tokens_needed(a, 13).unwrap(), None);
        kv.publish(a, 13, None).unwrap();
        let (hit, n) = kv.lookup(b, Some(t), false, 12, None, None).unwrap();
        assert_eq!((hit[0].clone(), n), (vec![1, 2, 3], 12));
        // An allocation larger than the pool fails and leaves everything as it was.
        assert_eq!(kv.allocate(b, 0, 200, 12, Some(&hit), 0, None, None).unwrap(), None);
        assert_eq!(kv.page_ids(b).unwrap()[0], Vec::<u32>::new());
        kv.check().unwrap();
        kv.allocate(b, 0, 1, 12, Some(&hit), 0, None, None).unwrap().unwrap();
        assert_eq!(kv.page_ids(b).unwrap()[0], vec![1, 2, 3, 5]);
        assert_eq!(kv.common_prefix(Some(b), 2), vec![3]);
        kv.check().unwrap();
        assert!(!kv.reset());
        assert!(!kv.free(a).unwrap());
        assert!(!kv.free(b).unwrap());
        assert!(kv.forget_keys(a).unwrap());
        assert_eq!(kv.num_open(), 1);
        kv.check().unwrap();
        assert!(kv.reset());
    }

    #[test]
    fn a_shorter_publish_never_lowers_the_published_count() {
        let mut kv = core(vec![full(4)], 16);
        let toks: Vec<i32> = (0..13).collect();
        let t = Tokens {
            tokens: &toks,
            offset: 0,
        };
        let a = kv.open(0);
        kv.allocate(a, 0, 13, 0, None, 13, None, None).unwrap().unwrap();
        kv.publish(a, 12, Some(t)).unwrap();
        // 5 computed tokens cover 1 full page; the 3 pages already published stay published.
        kv.publish(a, 5, Some(t)).unwrap();
        assert_eq!(kv.publish_tokens_needed(a, 12).unwrap(), None);
        kv.publish(a, 12, Some(t)).unwrap();
        kv.check().unwrap();
        let b = kv.open(0);
        assert_eq!(kv.lookup(b, Some(t), false, 12, None, None).unwrap().1, 12);
    }

    #[test]
    fn release_tail_keeps_published_pages() {
        let mut kv = core(vec![full(2)], 16);
        let toks: Vec<i32> = (0..4).collect();
        let a = kv.open(0);
        kv.allocate(a, 0, 4, 0, None, 4, None, None).unwrap().unwrap();
        assert_eq!(kv.page_ids(a).unwrap()[0].len(), 4);
        kv.publish(
            a,
            4,
            Some(Tokens {
                tokens: &toks,
                offset: 0,
            }),
        )
        .unwrap();
        assert_eq!(kv.release_tail(a, 1).unwrap(), 2);
        assert_eq!(kv.page_ids(a).unwrap()[0].len(), 2);
        kv.check().unwrap();
    }
}
