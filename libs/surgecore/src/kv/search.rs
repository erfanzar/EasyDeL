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

//! The prefix-hit search over every cache group: no cache, one group, the two-type hybrid and the grouped fixed point.

use super::group::{find_full, Group, GroupPages, Kind};
use super::hash::PageKey;
use super::pool::PagePool;
use super::KvError;

/// How the groups' hits combine into one hit.
#[derive(Clone, Debug)]
pub enum HitSearch {
    /// Prefix caching is off: never a hit.
    NoCache,
    /// One cache group.
    Unitary,
    /// Full-attention groups of one type plus groups of one other type, not interleaved.
    Hybrid {
        /// The full-attention group ids.
        full: Vec<u32>,
        /// The other group ids.
        other: Vec<u32>,
        /// Whether the full-attention groups come first.
        full_first: bool,
    },
    /// Any other layout sharing one page size: every group is searched until all agree on the hit length.
    Grouped,
}

impl HitSearch {
    /// Picks the search for `groups` the way the cache coordinator factory does.
    pub fn select(groups: &[Group], caching: bool) -> Result<Self, KvError> {
        if !caching {
            return Ok(Self::NoCache);
        }
        if groups.len() == 1 {
            return Ok(Self::Unitary);
        }
        let full: Vec<u32> = (0..groups.len() as u32)
            .filter(|&i| groups[i as usize].kind == Kind::Full)
            .collect();
        let other: Vec<u32> = (0..groups.len() as u32)
            .filter(|&i| groups[i as usize].kind != Kind::Full)
            .collect();
        let distinct = |ids: &[u32]| {
            let mut t: Vec<i64> = ids.iter().map(|&i| groups[i as usize].type_index).collect();
            t.sort_unstable();
            t.dedup();
            t.len()
        };
        if distinct(&full) == 1 && distinct(&other) == 1 {
            let (fmax, fmin) = (*full.iter().max().unwrap(), *full.iter().min().unwrap());
            let (omax, omin) = (*other.iter().max().unwrap(), *other.iter().min().unwrap());
            if fmax < omin || omax < fmin {
                let fps = groups[full[0] as usize].page_size;
                let ops = groups[other[0] as usize].page_size;
                // Page keys are chained at one page size and every group indexes them by its own page index, so a
                // group with another page size would look up the wrong keys.
                if ops != fps {
                    return Err(KvError::Value(format!(
                        "Hybrid prefix caching requires one token-page size for page hashes; got {fps} for full \
                         attention and {ops} for the other layers."
                    )));
                }
                return Ok(Self::Hybrid {
                    full,
                    other,
                    full_first: fmax < omin,
                });
            }
        }
        let page_size = groups[0].page_size;
        if groups.iter().any(|g| g.page_size != page_size) {
            return Err(KvError::Value(
                "Grouped prefix caching requires a common token-page size for page hashes.".into(),
            ));
        }
        Ok(Self::Grouped)
    }

    /// The longest common prefix hit of `keys`, at most `max_len` tokens: per-group pages and the hit token count.
    pub fn find(
        &self,
        groups: &[Group],
        pool: &PagePool,
        keys: &[PageKey],
        max_len: i64,
        range: Option<(u32, u32)>,
        eagle: bool,
    ) -> Result<(GroupPages, i64), KvError> {
        match self {
            Self::NoCache => Ok((vec![Vec::new(); groups.len()], 0)),
            Self::Unitary => {
                let hit = groups[0].find(pool, keys, max_len, &[0], range, eagle)?;
                let len = hit[0].len() as i64 * groups[0].page_size;
                Ok((hit, len))
            }
            Self::Hybrid {
                full,
                other,
                full_first,
            } => {
                let fps = groups[full[0] as usize].page_size;
                let mut hit_full = find_full(pool, keys, max_len, fps, full, range, eagle);
                let len = hit_full[0].len() as i64 * fps;
                let og = groups[other[0] as usize];
                let hit_other = og.find(pool, keys, len, other, range, false)?;
                let len = hit_other[0].len() as i64 * og.page_size;
                if len % fps != 0 {
                    return Err(KvError::Runtime(format!(
                        "hit_length ({len}) must be divisible by full_attention_page_size ({fps})"
                    )));
                }
                hit_full.iter_mut().for_each(|l| l.truncate((len / fps) as usize));
                let out = if *full_first {
                    hit_full.into_iter().chain(hit_other).collect()
                } else {
                    hit_other.into_iter().chain(hit_full).collect()
                };
                Ok((out, len))
            }
            Self::Grouped => {
                let ps = groups[0].page_size;
                let mut len = max_len.div_euclid(ps).min(keys.len() as i64) * ps;
                let mut reserve = eagle;
                loop {
                    let mut hits = Vec::with_capacity(groups.len());
                    for (gid, g) in groups.iter().enumerate() {
                        let mut h = g.find(pool, keys, len, &[gid as u32], range, false)?;
                        hits.push(h.swap_remove(0));
                    }
                    let common = hits.iter().map(Vec::len).min().unwrap_or(0) as i64 * ps;
                    if common == len {
                        if reserve && len != 0 {
                            reserve = false;
                            len -= ps;
                            continue;
                        }
                        return Ok((hits, len));
                    }
                    len = common;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(kind: Kind, type_index: i64) -> Group {
        Group {
            kind,
            page_size: 16,
            type_index,
        }
    }

    /// A shorter hit in one group moves the endpoint back; chunk groups whose earlier answer was all null pages
    /// must be searched again at the new endpoint (64 -> 48 -> 32), never truncated.
    #[test]
    fn grouped_search_rechecks_chunk_boundary_null_pages() {
        let groups = vec![g(Kind::Chunked(64), 0), g(Kind::Chunked(128), 1), g(Kind::Full, 2)];
        assert!(matches!(HitSearch::select(&groups, true).unwrap(), HitSearch::Grouped));
        let mut pool = PagePool::new(32, true).unwrap();
        let keys: Vec<PageKey> = (1..=4).collect();
        let mut owned = Vec::new();
        for (gid, depth) in [(0u32, 2usize), (1, 3), (2, 4)] {
            let pages = pool.get_new(4, None).unwrap();
            for (&p, &k) in pages.iter().zip(&keys).take(depth) {
                pool.publish(p, k, gid).unwrap();
            }
            owned.push(pages);
        }
        let search = HitSearch::select(&groups, true).unwrap();
        let (hit, len) = search.find(&groups, &pool, &keys, 64, None, false).unwrap();
        assert_eq!(len, 32);
        for (h, pages) in hit.iter().zip(&owned) {
            assert_eq!(h, &pages[..2].to_vec());
        }
    }

    #[test]
    fn hybrid_and_grouped_searches_need_one_page_size() {
        let full = Group {
            kind: Kind::Full,
            page_size: 16,
            type_index: 0,
        };
        let sw = |ps| Group {
            kind: Kind::Sliding(64),
            page_size: ps,
            type_index: 1,
        };
        assert!(matches!(
            HitSearch::select(&[full, sw(16)], true).unwrap(),
            HitSearch::Hybrid { full_first: true, .. }
        ));
        assert!(HitSearch::select(&[full, sw(32)], true).is_err());
        assert!(HitSearch::select(&[full, sw(24)], true).is_err());
        assert!(matches!(
            HitSearch::select(&[full, sw(24)], false).unwrap(),
            HitSearch::NoCache
        ));
    }

    #[test]
    fn hybrid_full_sliding_eagle_reserves_only_one_page() {
        use super::super::pool::NULL_PAGE;

        let full = Group {
            kind: Kind::Full,
            page_size: 4,
            type_index: 0,
        };
        let sw4 = Group {
            kind: Kind::Sliding(4),
            page_size: 4,
            type_index: 1,
        };
        // 2 published pages (8 tokens) with window=4, page_size=4: EAGLE reserves 1 page -> 4 tokens (1 page), not 0.
        for groups in [vec![full, sw4], vec![sw4, full]] {
            let search = HitSearch::select(&groups, true).unwrap();
            assert!(matches!(search, HitSearch::Hybrid { .. }));
            let mut pool = PagePool::new(16, true).unwrap();
            let keys: Vec<PageKey> = (1..=2).collect();
            let mut owned = Vec::new();
            for gid in 0..2u32 {
                let pages = pool.get_new(2, None).unwrap();
                for (&p, &k) in pages.iter().zip(&keys) {
                    pool.publish(p, k, gid).unwrap();
                }
                owned.push(pages);
            }
            let (hit, len) = search.find(&groups, &pool, &keys, 8, None, true).unwrap();
            assert_eq!(len, 4);
            assert_eq!(hit[0], vec![owned[0][0]]);
            assert_eq!(hit[1], vec![owned[1][0]]);
        }

        // 4 published pages (16 tokens) with window=8, page_size=4: EAGLE reserves 1 page -> 12 tokens (3 pages),
        // with the sliding group dropping page 0 behind the window (NULL_PAGE).
        let sw8 = Group {
            kind: Kind::Sliding(8),
            page_size: 4,
            type_index: 1,
        };
        let groups = vec![full, sw8];
        let search = HitSearch::select(&groups, true).unwrap();
        let mut pool = PagePool::new(32, true).unwrap();
        let keys: Vec<PageKey> = (10..=13).collect();
        let full_pages = pool.get_new(4, None).unwrap();
        let sw_pages = pool.get_new(4, None).unwrap();
        for (&p, &k) in full_pages.iter().zip(&keys) {
            pool.publish(p, k, 0).unwrap();
        }
        for (&p, &k) in sw_pages.iter().zip(&keys) {
            pool.publish(p, k, 1).unwrap();
        }
        let (hit, len) = search.find(&groups, &pool, &keys, 16, None, true).unwrap();
        assert_eq!(len, 12);
        assert_eq!(hit[0], full_pages[..3].to_vec());
        assert_eq!(hit[1], vec![NULL_PAGE, sw_pages[1], sw_pages[2]]);

        // If sliding only retains pages 2 and 3 (pages 0 and 1 missing), dropping page 3 for EAGLE leaves page 1
        // uncovered for window=8 at len=12, so the hit must be 0.
        let mut pool2 = PagePool::new(32, true).unwrap();
        let full_pages2 = pool2.get_new(4, None).unwrap();
        let sw_pages2 = pool2.get_new(2, None).unwrap();
        for (&p, &k) in full_pages2.iter().zip(&keys) {
            pool2.publish(p, k, 0).unwrap();
        }
        pool2.publish(sw_pages2[0], keys[2], 1).unwrap();
        pool2.publish(sw_pages2[1], keys[3], 1).unwrap();
        let (hit2, len2) = search.find(&groups, &pool2, &keys, 16, None, true).unwrap();
        assert_eq!(len2, 0);
        assert!(hit2[0].is_empty() && hit2[1].is_empty());
    }

    #[test]
    fn hybrid_full_chunked_eagle_reserves_once_at_chunk_boundary() {
        use super::super::pool::NULL_PAGE;

        let full = Group {
            kind: Kind::Full,
            page_size: 4,
            type_index: 0,
        };
        let ch = Group {
            kind: Kind::Chunked(8),
            page_size: 4,
            type_index: 1,
        };
        let keys: Vec<PageKey> = (20..=22).collect();

        // Full has 3 cached pages (12 tokens); chunked has no cached pages in chunk 0 or page 2 (chunk 0 was dropped
        // behind chunk boundary 8). EAGLE drops 1 page from full (12 -> 8) and chunked hits chunk boundary 8 with
        // NULL pages without subtracting another page into the uncached previous chunk.
        for (groups, full_idx, ch_idx) in [(vec![full, ch], 0usize, 1usize), (vec![ch, full], 1usize, 0usize)] {
            let search = HitSearch::select(&groups, true).unwrap();
            assert!(matches!(search, HitSearch::Hybrid { .. }));
            let mut pool = PagePool::new(16, true).unwrap();
            let full_pages = pool.get_new(3, None).unwrap();
            for (&p, &k) in full_pages.iter().zip(&keys) {
                pool.publish(p, k, full_idx as u32).unwrap();
            }
            let (hit, len) = search.find(&groups, &pool, &keys, 12, None, true).unwrap();
            assert_eq!(len, 8);
            assert_eq!(hit[full_idx], full_pages[..2].to_vec());
            assert_eq!(hit[ch_idx], vec![NULL_PAGE, NULL_PAGE]);
        }
    }
}
