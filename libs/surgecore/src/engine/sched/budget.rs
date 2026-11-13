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

//! The KV-aware token budget (on only with a token safety margin) and image-span chunk alignment.

/// The per-step token budget bounded by the free KV pages minus a safety margin per running request.
#[derive(Clone, Copy, Debug)]
pub struct TokenBudget {
    /// The configured per-step token budget.
    pub max_batch: i64,
    /// Tokens per page.
    pub page_size: i64,
    /// Tokens reserved per running request.
    pub margin: i64,
    /// Tokens left in this cycle.
    pub remaining: i64,
}

impl TokenBudget {
    /// A budget of `max_batch` tokens with `margin` tokens reserved per running request.
    pub fn new(max_batch: i64, page_size: i64, margin: i64) -> Self {
        Self {
            max_batch,
            page_size,
            margin,
            remaining: max_batch,
        }
    }

    /// Starts a cycle: `min(max_batch, max(0, free_pages * page_size - running * margin, demand))`.
    pub fn begin_cycle(&mut self, free_pages: i64, running: i64, demand: i64) -> i64 {
        let capacity = (free_pages * self.page_size - running * self.margin).max(0).max(demand);
        self.remaining = self.max_batch.min(capacity);
        self.remaining
    }

    /// Takes up to `tokens` from the budget; returns what was granted.
    pub fn consume(&mut self, tokens: i64) -> i64 {
        if tokens <= 0 || self.remaining <= 0 {
            return 0;
        }
        let grant = tokens.min(self.remaining);
        self.remaining -= grant;
        grant
    }

    /// Gives back tokens taken earlier in the cycle.
    pub fn release(&mut self, tokens: i64) {
        if tokens > 0 {
            self.remaining += tokens;
        }
    }
}

/// Shrinks a prefill chunk `[computed, computed + n)` so it does not end strictly inside an image span: to the span's
/// start, or, when the chunk starts at the span's start, to the whole span. `0` means the span does not fit `budget`.
pub fn align_prefill_chunk(spans: &[(i64, i64)], computed: i64, n: i64, budget: i64) -> i64 {
    let end = computed + n;
    for &(start, stop) in spans {
        if start < end && end < stop {
            if start > computed {
                return start - computed;
            }
            let needed = stop - computed;
            return if needed <= budget { needed } else { 0 };
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_cycle() {
        let mut b = TokenBudget::new(32, 4, 8);
        assert_eq!(b.begin_cycle(5, 2, 0), 4);
        assert_eq!(b.begin_cycle(0, 2, 7), 7);
        assert_eq!(b.consume(10), 7);
        assert_eq!(b.consume(1), 0);
        b.release(3);
        assert_eq!(b.remaining, 3);
        assert_eq!(b.begin_cycle(100, 0, 0), 32);
    }

    #[test]
    fn span_alignment() {
        let spans = [(4, 10), (12, 14)];
        assert_eq!(align_prefill_chunk(&spans, 0, 6, 32), 4);
        assert_eq!(align_prefill_chunk(&spans, 4, 2, 32), 6);
        assert_eq!(align_prefill_chunk(&spans, 4, 2, 5), 0);
        assert_eq!(align_prefill_chunk(&spans, 0, 10, 32), 10);
    }
}
