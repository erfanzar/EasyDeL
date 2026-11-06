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
                if ops % fps != 0 {
                    return Err(KvError::Value(
                        "CacheCoordinator assumes the page_size of full attention layers is divisible by other layers now."
                            .into(),
                    ));
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
                let other_eagle = eagle && !matches!(og.kind, Kind::Chunked(_));
                let hit_other = og.find(pool, keys, len, other, range, other_eagle)?;
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
    fn hybrid_needs_divisible_page_sizes_and_grouped_a_common_one() {
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
            HitSearch::select(&[full, sw(32)], true).unwrap(),
            HitSearch::Hybrid { full_first: true, .. }
        ));
        assert!(HitSearch::select(&[full, sw(24)], true).is_err());
        assert!(matches!(
            HitSearch::select(&[full, sw(24)], false).unwrap(),
            HitSearch::NoCache
        ));
    }
}
