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

//! Chained 128-bit page keys for the prefix cache.
//!
//! A full page's key is `h(parent key, page tokens, extra key)`, where the extra key identifies multimodal content
//! (0 for text). The two 64-bit halves come from two independently keyed std `RandomState` hashers, random per core,
//! so key values never mean anything outside the core: every decision depends only on key equality.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};

/// A page key: identifies a full page together with the whole prefix before it.
pub type PageKey = u128;

/// The parent key of a request's first page.
pub const ROOT: PageKey = 0;

/// Computes page keys; holds the two random hasher keys of one core.
pub struct KeyHasher {
    a: RandomState,
    b: RandomState,
}

impl Default for KeyHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyHasher {
    /// A hasher with fresh random keys.
    pub fn new() -> Self {
        Self {
            a: RandomState::new(),
            b: RandomState::new(),
        }
    }

    /// The key of one page given its parent's key, its tokens and the request's extra key.
    pub fn page_key(&self, parent: PageKey, tokens: &[i32], extra: u128) -> PageKey {
        let mut ha = self.a.build_hasher();
        let mut hb = self.b.build_hasher();
        for h in [&mut ha, &mut hb] {
            h.write_u128(parent);
            h.write_u128(extra);
            h.write_usize(tokens.len());
            for &t in tokens {
                h.write_i32(t);
            }
        }
        let key = (u128::from(ha.finish()) << 64) | u128::from(hb.finish());
        if key == ROOT {
            1
        } else {
            key
        }
    }

    /// The chained keys of every full page of `tokens`.
    pub fn chain(&self, tokens: &[i32], page_size: usize, extra: u128) -> Vec<PageKey> {
        if page_size == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(tokens.len() / page_size);
        let mut parent = ROOT;
        for page in tokens.chunks_exact(page_size) {
            parent = self.page_key(parent, page, extra);
            out.push(parent);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_depends_on_prefix_tokens_and_extra() {
        let h = KeyHasher::new();
        let a = h.chain(&[1, 2, 3, 4, 5], 2, 0);
        assert_eq!(a.len(), 2);
        assert_eq!(a, h.chain(&[1, 2, 3, 4], 2, 0));
        assert_ne!(a[1], h.chain(&[9, 2, 3, 4], 2, 0)[1]);
        assert_ne!(a[0], h.chain(&[1, 2], 2, 7)[0]);
        assert_eq!(a[1], h.page_key(a[0], &[3, 4], 0));
    }
}
