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

//! The page pool: reference counts, the LRU free list, the null page and the prefix map.
//!
//! Pages are a struct of arrays indexed by page id. The free list is a doubly linked list through `prev`/`next`
//! with two sentinel slots past the last page; the front is evicted first. Page 0 is the null page: it is taken off
//! the free list at construction, is never allocated or freed, and its reference count is not tracked. The prefix
//! map sends a published `(page key, cache group)` to the pages holding it, in publication order; a lookup takes the
//! first one (optionally the first inside a data-parallel page range).

use std::collections::HashMap;

use super::hash::PageKey;
use super::KvError;

/// The null page's id.
pub const NULL_PAGE: u32 = 0;
const UNLINKED: u32 = u32::MAX;

/// Pages per data-parallel rank, or `None` when `dp <= 1` or the pages do not split evenly (`dp == 0` counts as 1).
pub fn pages_per_shard(num_pages: usize, dp: i64) -> Option<usize> {
    let dp = if dp == 0 { 1 } else { dp };
    if dp <= 1 {
        return None;
    }
    let dp = dp as usize;
    if num_pages == 0 || num_pages % dp != 0 {
        None
    } else {
        Some(num_pages / dp)
    }
}

/// The KV page pool of one core (one data-parallel rank's pages, or every page when there is one rank).
pub struct PagePool {
    num_pages: usize,
    caching: bool,
    ref_cnt: Vec<u32>,
    key: Vec<Option<(PageKey, u32)>>,
    prev: Vec<u32>,
    next: Vec<u32>,
    num_free: usize,
    map: HashMap<(PageKey, u32), Vec<u32>>,
}

impl PagePool {
    /// A pool of `num_pages` pages (page 0 is the null page). Fails when `num_pages` is 0.
    pub fn new(num_pages: usize, caching: bool) -> Result<Self, KvError> {
        if num_pages == 0 {
            return Err(KvError::Value("num_pages must be a positive integer".into()));
        }
        let head = num_pages as u32;
        let tail = head + 1;
        let mut prev = vec![UNLINKED; num_pages + 2];
        let mut next = vec![UNLINKED; num_pages + 2];
        let mut last = head;
        for p in 0..num_pages as u32 {
            next[last as usize] = p;
            prev[p as usize] = last;
            last = p;
        }
        next[last as usize] = tail;
        prev[tail as usize] = last;
        let mut pool = Self {
            num_pages,
            caching,
            ref_cnt: vec![0; num_pages],
            key: vec![None; num_pages],
            prev,
            next,
            num_free: num_pages,
            map: HashMap::new(),
        };
        let null = pool.next[head as usize];
        pool.unlink(null);
        Ok(pool)
    }

    fn head(&self) -> u32 {
        self.num_pages as u32
    }

    fn tail(&self) -> u32 {
        self.num_pages as u32 + 1
    }

    fn unlink(&mut self, p: u32) {
        let (a, b) = (self.prev[p as usize], self.next[p as usize]);
        self.next[a as usize] = b;
        self.prev[b as usize] = a;
        self.prev[p as usize] = UNLINKED;
        self.next[p as usize] = UNLINKED;
        self.num_free -= 1;
    }

    fn append(&mut self, p: u32) {
        let tail = self.tail();
        let last = self.prev[tail as usize];
        self.next[last as usize] = p;
        self.prev[p as usize] = last;
        self.next[p as usize] = tail;
        self.prev[tail as usize] = p;
        self.num_free += 1;
    }

    /// Total pages, the null page included.
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Pages on the free list.
    pub fn num_free(&self) -> usize {
        self.num_free
    }

    /// `1 - free / num_pages`.
    pub fn usage(&self) -> f64 {
        1.0 - (self.num_free as f64 / self.num_pages as f64)
    }

    /// Distinct published `(page key, group)` entries.
    pub fn num_cached(&self) -> usize {
        self.map.len()
    }

    /// The reference count of page `p` (not tracked for the null page).
    pub fn ref_cnt(&self, p: u32) -> u32 {
        self.ref_cnt[p as usize]
    }

    /// Whether page `p` is on the free list.
    pub fn is_free(&self, p: u32) -> bool {
        self.prev[p as usize] != UNLINKED
    }

    /// The published key and group of page `p`.
    pub fn key_of(&self, p: u32) -> Option<(PageKey, u32)> {
        self.key[p as usize]
    }

    /// The free list, front (evicted first) to back.
    pub fn free_order(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.num_free);
        let mut p = self.next[self.head() as usize];
        while p != self.tail() {
            out.push(p);
            p = self.next[p as usize];
        }
        out
    }

    /// Reference counts of pages 1.. (the null page is left out).
    pub fn refs(&self) -> Vec<u32> {
        self.ref_cnt[1..].to_vec()
    }

    /// The `[lo, hi)` page range of the hinted data-parallel rank, when a hint applies.
    pub fn shard_range(&self, hint: Option<i64>, dp: Option<i64>) -> Option<(u32, u32)> {
        let (hint, dp) = (hint?, dp?);
        let pps = pages_per_shard(self.num_pages, dp)?;
        let shard = hint.rem_euclid(dp) as usize;
        let lo = if shard == 0 { 1 } else { shard * pps };
        Some((lo as u32, (shard * pps + pps) as u32))
    }

    /// The published page of `key` in every group of `groups` (first published; first inside `range`), into `out`.
    /// Returns `false` on any miss.
    pub fn get_cached(&self, key: PageKey, groups: &[u32], range: Option<(u32, u32)>, out: &mut Vec<u32>) -> bool {
        out.clear();
        for &g in groups {
            let Some(pages) = self.map.get(&(key, g)) else {
                return false;
            };
            let found = match range {
                None => pages.first().copied(),
                Some((lo, hi)) => pages.iter().copied().find(|&p| lo <= p && p < hi),
            };
            match found {
                Some(p) => out.push(p),
                None => return false,
            }
        }
        true
    }

    fn evict(&mut self, p: u32) {
        if let Some(k) = self.key[p as usize] {
            if let Some(pages) = self.map.get_mut(&k) {
                self.key[p as usize] = None;
                if let Some(pos) = pages.iter().position(|&q| q == p) {
                    pages.remove(pos);
                }
                if pages.is_empty() {
                    self.map.remove(&k);
                }
            }
        }
    }

    /// Takes `n` pages off the free list (from the front, or the first ones inside `range`), evicts their prefix
    /// entries and gives each a reference. Fails, with nothing changed, when there are not enough pages.
    pub fn get_new(&mut self, n: usize, range: Option<(u32, u32)>) -> Result<Vec<u32>, KvError> {
        if n > self.num_free {
            return Err(KvError::Value(format!("Cannot get {n} free pages from the pool")));
        }
        let mut ret = Vec::with_capacity(n);
        match range {
            None => {
                for _ in 0..n {
                    let p = self.next[self.head() as usize];
                    self.unlink(p);
                    ret.push(p);
                }
            }
            Some((lo, hi)) => {
                let mut p = self.next[self.head() as usize];
                while p != self.tail() && ret.len() < n {
                    if lo <= p && p < hi {
                        ret.push(p);
                    }
                    p = self.next[p as usize];
                }
                if ret.len() < n {
                    return Err(KvError::Value(format!(
                        "Insufficient free pages in requested DP shard range: range=[{lo}, {hi}) requested={n} found={}",
                        ret.len()
                    )));
                }
                for &p in &ret {
                    self.unlink(p);
                }
            }
        }
        for &p in &ret {
            if self.caching {
                self.evict(p);
            }
            if self.ref_cnt[p as usize] != 0 {
                return Err(KvError::Runtime(format!("Expected ref_cnt to be 0 for free page {p}")));
            }
            self.ref_cnt[p as usize] = 1;
        }
        Ok(ret)
    }

    /// Gives each prefix-hit page one more reference, taking pages with no reference off the free list.
    pub fn touch(&mut self, groups: &[Vec<u32>]) {
        for pages in groups {
            for &p in pages {
                if p == NULL_PAGE {
                    continue;
                }
                if self.ref_cnt[p as usize] == 0 && self.is_free(p) {
                    self.unlink(p);
                }
                self.ref_cnt[p as usize] += 1;
            }
        }
    }

    /// Drops one reference per entry; pages that reach zero go to the back of the free list in entry order.
    /// Fails, with nothing changed, on a reference-count underflow.
    pub fn release<I>(&mut self, pages: I) -> Result<(), KvError>
    where
        I: Iterator<Item = u32> + Clone,
    {
        let mut freed = Vec::new();
        for (done, p) in pages.clone().enumerate() {
            if p != NULL_PAGE {
                if self.ref_cnt[p as usize] == 0 {
                    for q in pages.clone().take(done) {
                        if q != NULL_PAGE {
                            self.ref_cnt[q as usize] += 1;
                        }
                    }
                    return Err(KvError::Runtime(format!("ref_cnt underflow on page {p}")));
                }
                self.ref_cnt[p as usize] -= 1;
                if self.ref_cnt[p as usize] == 0 {
                    freed.push(p);
                }
            }
        }
        for p in freed {
            self.append(p);
        }
        Ok(())
    }

    /// Publishes page `p` under `(key, group)`. Fails when the page is already published.
    pub fn publish(&mut self, p: u32, key: PageKey, group: u32) -> Result<(), KvError> {
        if self.key[p as usize].is_some() {
            return Err(KvError::Runtime(format!(
                "Expected page_hash to be None for new full page {p}"
            )));
        }
        self.key[p as usize] = Some((key, group));
        self.map.entry((key, group)).or_default().push(p);
        Ok(())
    }

    /// Forgets every published page; refused (`false`) while any page besides the null page is in use.
    pub fn reset(&mut self) -> bool {
        if self.num_pages - self.num_free != 1 {
            return false;
        }
        self.map.clear();
        self.key.iter_mut().for_each(|k| *k = None);
        true
    }

    /// Checks the pool's own invariants: the free list holds exactly the pages with no reference (the null page
    /// never), its length is `num_free`, and every prefix entry points to pages whose key is the entry's key.
    pub fn check(&self) -> Result<(), String> {
        let order = self.free_order();
        if order.len() != self.num_free {
            return Err(format!(
                "free list holds {} pages, num_free {}",
                order.len(),
                self.num_free
            ));
        }
        let mut on_list = vec![false; self.num_pages];
        for &p in &order {
            if p == NULL_PAGE {
                return Err("the null page is on the free list".into());
            }
            on_list[p as usize] = true;
        }
        for (p, (&listed, &refs)) in on_list.iter().zip(&self.ref_cnt).enumerate().skip(1) {
            if listed != (refs == 0) {
                return Err(format!("page {p}: ref_cnt {refs} but on free list {listed}"));
            }
        }
        for (k, pages) in &self.map {
            if pages.is_empty() {
                return Err("empty prefix entry".into());
            }
            for &p in pages {
                if self.key[p as usize] != Some(*k) {
                    return Err(format!("prefix entry points to page {p} with another key"));
                }
            }
        }
        if self.key[NULL_PAGE as usize].is_some() {
            return Err("the null page is published".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_order_eviction_and_reset() {
        let mut pool = PagePool::new(6, true).unwrap();
        assert_eq!(pool.free_order(), vec![1, 2, 3, 4, 5]);
        let a = pool.get_new(3, None).unwrap();
        assert_eq!(a, vec![1, 2, 3]);
        pool.publish(1, 11, 0).unwrap();
        pool.release(a.iter().rev().copied()).unwrap();
        assert_eq!(pool.free_order(), vec![4, 5, 3, 2, 1]);
        let mut out = Vec::new();
        assert!(pool.get_cached(11, &[0], None, &mut out));
        assert_eq!(out, vec![1]);
        pool.touch(&[vec![1]]);
        assert_eq!(pool.num_free(), 4);
        assert!(!pool.reset());
        pool.release([1].into_iter()).unwrap();
        assert!(pool.reset());
        assert!(!pool.get_cached(11, &[0], None, &mut out));
        assert!(pool.release([1].into_iter()).is_err());
        pool.check().unwrap();
    }

    #[test]
    fn shard_range_restricts_allocation() {
        let mut pool = PagePool::new(8, false).unwrap();
        assert_eq!(pool.shard_range(Some(1), Some(2)), Some((4, 8)));
        assert_eq!(pool.shard_range(Some(0), Some(2)), Some((1, 4)));
        assert_eq!(pool.shard_range(Some(0), Some(3)), None);
        assert_eq!(pool.get_new(2, Some((4, 8))).unwrap(), vec![4, 5]);
        assert!(pool.get_new(3, Some((4, 8))).is_err());
        assert_eq!(pool.num_free(), 5);
        pool.check().unwrap();
    }
}
