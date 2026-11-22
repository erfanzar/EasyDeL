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

//! One cache group's page rules: full attention, sliding window and chunked local attention.
//!
//! Each kind defines its prefix-hit search, which pages it drops behind the attention window (they become the null
//! page in the request's list), how many free pages a request still draws, and its common-prefix count.

use super::hash::PageKey;
use super::pool::{PagePool, NULL_PAGE};
use super::KvError;

/// Pages per cache group of one request (or of one prefix hit).
pub type GroupPages = Vec<Vec<u32>>;

/// The attention kind of a cache group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Every token attends to every earlier token; every page is kept.
    Full,
    /// Each token attends to the last `window` tokens.
    Sliding(i64),
    /// Tokens attend within aligned chunks of `chunk` tokens.
    Chunked(i64),
}

/// One cache group.
#[derive(Clone, Copy, Debug)]
pub struct Group {
    /// The attention kind.
    pub kind: Kind,
    /// Tokens per page.
    pub page_size: i64,
    /// Groups with equal cache specs share a type index.
    pub type_index: i64,
}

/// Ceiling division for a positive divisor (any sign of `a`).
pub fn cdiv(a: i64, b: i64) -> i64 {
    (a + b - 1).div_euclid(b)
}

/// Longest run of published pages from page 0, at most `max_len / page_size` pages, in every group of `gids`.
/// With `eagle`, the last hit page is dropped (its hidden states must be recomputed).
pub fn find_full(
    pool: &PagePool,
    keys: &[PageKey],
    max_len: i64,
    page_size: i64,
    gids: &[u32],
    range: Option<(u32, u32)>,
    eagle: bool,
) -> GroupPages {
    let mut out: GroupPages = vec![Vec::new(); gids.len()];
    let max_pages = max_len.div_euclid(page_size).max(0) as usize;
    let mut buf = Vec::with_capacity(gids.len());
    for &key in keys.iter().take(max_pages) {
        if !pool.get_cached(key, gids, range, &mut buf) {
            break;
        }
        for (list, &p) in out.iter_mut().zip(&buf) {
            list.push(p);
        }
    }
    if eagle && out.first().is_some_and(|l| !l.is_empty()) {
        out.iter_mut().for_each(|l| {
            l.pop();
        });
    }
    out
}

/// Sliding-window hit: scans backward for a run of published pages that covers the window; positions before the run
/// are null pages. Fails with an index error when fewer keys than `max_len / page_size` are known.
#[allow(clippy::too_many_arguments)]
pub fn find_sliding(
    pool: &PagePool,
    keys: &[PageKey],
    max_len: i64,
    page_size: i64,
    window: i64,
    gids: &[u32],
    range: Option<(u32, u32)>,
    eagle: bool,
) -> Result<GroupPages, KvError> {
    let mut need = cdiv(window - 1, page_size);
    if eagle {
        need += 1;
    }
    let max_pages = max_len.div_euclid(page_size).max(0) as usize;
    let mut out: GroupPages = vec![vec![NULL_PAGE; max_pages]; gids.len()];
    let mut run = 0usize;
    let mut matched = false;
    let mut buf = Vec::with_capacity(gids.len());
    for i in (0..max_pages).rev() {
        let Some(&key) = keys.get(i) else {
            return Err(KvError::Index("list index out of range".into()));
        };
        if pool.get_cached(key, gids, range, &mut buf) {
            for (list, &p) in out.iter_mut().zip(&buf) {
                list[i] = p;
            }
            run += 1;
            if run as i64 >= need {
                out.iter_mut().for_each(|l| l.truncate(i + run));
                matched = true;
                break;
            }
        } else {
            run = 0;
        }
    }
    if !matched {
        out.iter_mut().for_each(|l| l.truncate(run));
    }
    if eagle && out.first().is_some_and(|l| !l.is_empty()) {
        out.iter_mut().for_each(|l| {
            l.pop();
        });
    }
    Ok(out)
}

/// Chunked-local hit: pages before the chunk that holds the next token are null; the pages inside it are looked up.
/// A miss moves the endpoint back and the search restarts there; with `eagle` one page is reserved once.
#[allow(clippy::too_many_arguments)]
pub fn find_chunked(
    pool: &PagePool,
    keys: &[PageKey],
    max_len: i64,
    page_size: i64,
    chunk: i64,
    gids: &[u32],
    range: Option<(u32, u32)>,
    eagle: bool,
) -> GroupPages {
    let mut endpoint = max_len.div_euclid(page_size).max(0).min(keys.len() as i64) * page_size;
    let mut reserve = eagle;
    let mut buf = Vec::with_capacity(gids.len());
    loop {
        let start_page = (endpoint.div_euclid(chunk) * chunk).div_euclid(page_size) as usize;
        let mut out: GroupPages = vec![vec![NULL_PAGE; start_page]; gids.len()];
        for &key in &keys[start_page..endpoint.div_euclid(page_size) as usize] {
            if !pool.get_cached(key, gids, range, &mut buf) {
                break;
            }
            for (list, &p) in out.iter_mut().zip(&buf) {
                list.push(p);
            }
        }
        let hit = out.first().map_or(0, Vec::len) as i64 * page_size;
        if hit != endpoint {
            endpoint = hit;
            continue;
        }
        if reserve && endpoint != 0 {
            reserve = false;
            endpoint -= page_size;
            continue;
        }
        return out;
    }
}

impl Group {
    /// This group kind's hit search over the groups `gids` (which share this group's spec).
    pub fn find(
        &self,
        pool: &PagePool,
        keys: &[PageKey],
        max_len: i64,
        gids: &[u32],
        range: Option<(u32, u32)>,
        eagle: bool,
    ) -> Result<GroupPages, KvError> {
        let ps = self.page_size;
        match self.kind {
            Kind::Full => Ok(find_full(pool, keys, max_len, ps, gids, range, eagle)),
            Kind::Sliding(w) => find_sliding(pool, keys, max_len, ps, w, gids, range, eagle),
            Kind::Chunked(c) => Ok(find_chunked(pool, keys, max_len, ps, c, gids, range, eagle)),
        }
    }

    /// Replaces the pages that fell behind the window (after `computed` tokens) by the null page and releases them,
    /// newest first. `cached` is the request's published-page count in this group.
    pub fn remove_skipped(
        &self,
        pool: &mut PagePool,
        pages: &mut [u32],
        cached: Option<i64>,
        computed: i64,
    ) -> Result<(), KvError> {
        let ps = self.page_size;
        let first_useful = match self.kind {
            Kind::Full => return Ok(()),
            Kind::Sliding(w) => (computed - w + 1).div_euclid(ps),
            Kind::Chunked(c) => {
                let mut first = (computed.div_euclid(c) * c).div_euclid(ps);
                let nc = cached.unwrap_or(0);
                if nc > 0 {
                    first = first.min(nc - 1);
                }
                first
            }
        };
        if first_useful <= 0 {
            return Ok(());
        }
        if first_useful as usize > pages.len() {
            return Err(KvError::Index("list index out of range".into()));
        }
        let mut removed = Vec::new();
        for i in (0..first_useful as usize).rev() {
            if pages[i] == NULL_PAGE {
                break;
            }
            removed.push(pages[i]);
            pages[i] = NULL_PAGE;
        }
        pool.release(removed.iter().copied())
    }

    /// Free pages this group still draws before the request holds `num_tokens` slots (read-only). Windowed groups
    /// count at most the pages held at once while the rest arrives in steps of at most `max_step` slots.
    pub fn needed(&self, pool: &PagePool, held: &[u32], hits: &[u32], num_tokens: i64, max_step: Option<i64>) -> i64 {
        let ps = self.page_size;
        let new = (cdiv(num_tokens, ps) - hits.len() as i64 - held.len() as i64).max(0);
        let evictable = hits.iter().filter(|&&p| p != NULL_PAGE && pool.ref_cnt(p) == 0).count() as i64;
        let base = new + evictable;
        let window = match self.kind {
            Kind::Full => return base,
            Kind::Sliding(w) | Kind::Chunked(w) => w,
        };
        let Some(step) = max_step else {
            return base;
        };
        let alone = held
            .iter()
            .rev()
            .take_while(|&&p| p != NULL_PAGE)
            .filter(|&&p| pool.ref_cnt(p) == 1)
            .count() as i64;
        let at_once = cdiv(step + window + ps - 2, ps) + 1;
        base.min((at_once - alone).max(0))
    }

    /// Leading pages every one of `num_scheduled` requests shares (full attention only; 0 otherwise).
    pub fn common_prefix(&self, pool: &PagePool, pages: &[u32], num_scheduled: i64) -> i64 {
        match self.kind {
            Kind::Full => pages
                .iter()
                .take_while(|&&p| i64::from(pool.ref_cnt(p)) >= num_scheduled)
                .count() as i64,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdiv_matches_python() {
        assert_eq!(cdiv(7, 4), 2);
        assert_eq!(cdiv(8, 4), 2);
        assert_eq!(cdiv(0, 4), 0);
        assert_eq!(cdiv(-1, 4), 0);
        assert_eq!(cdiv(-5, 4), -1);
    }

    #[test]
    fn sliding_hit_covers_the_window_with_null_prefix() {
        let mut pool = PagePool::new(10, true).unwrap();
        let pages = pool.get_new(4, None).unwrap();
        let keys: Vec<PageKey> = (1..=4).collect();
        for (&p, &k) in pages.iter().zip(&keys) {
            pool.publish(p, k, 0).unwrap();
        }
        let g = Group {
            kind: Kind::Sliding(8),
            page_size: 4,
            type_index: 0,
        };
        let hit = g.find(&pool, &keys, 15, &[0], None, false).unwrap();
        assert_eq!(hit[0], vec![NULL_PAGE, pages[1], pages[2]]);
        assert!(g.find(&pool, &keys[..2], 15, &[0], None, false).is_err());
        let full = find_full(&pool, &keys, 15, 4, &[0], None, true);
        assert_eq!(full[0], vec![pages[0], pages[1]]);
    }
}
