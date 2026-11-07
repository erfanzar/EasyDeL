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

//! Slot bookkeeping for recurrent-state snapshots taken at prefix-cache page boundaries.
//!
//! A slot is pending from the step that asks the runner to write it until that step's output is seen, and ready
//! afterwards. Only ready slots are found by `lookup`. When every slot is taken, the least recently used ready slot
//! that is not excluded is recycled; pending slots are never recycled.

use super::KvError;

/// A snapshot key: the page key of the boundary page (any 128-bit value).
pub type SnapKey = u128;

/// `(ready in LRU order, pending sorted by slot, free stack)` of a [`SnapshotIndex`].
pub type SnapState = (Vec<(SnapKey, i64)>, Vec<(i64, SnapKey)>, Vec<i64>);

/// The snapshot index of one core (one data-parallel rank's slot block).
pub struct SnapshotIndex {
    num_slots: i64,
    offset: i64,
    /// Ready snapshots, least recently used first.
    ready: Vec<(SnapKey, i64)>,
    /// Pending reservations in reservation order.
    pending: Vec<(i64, SnapKey)>,
    /// Free slots; the last one is taken first.
    free: Vec<i64>,
}

impl SnapshotIndex {
    /// `num_slots` slots numbered from `offset`.
    pub fn new(num_slots: i64, offset: i64) -> Result<Self, KvError> {
        if num_slots <= 0 {
            return Err(KvError::Value(format!("num_slots must be positive, got {num_slots}")));
        }
        if offset < 0 {
            return Err(KvError::Value(format!("slot_offset must be >= 0, got {offset}")));
        }
        let mut index = Self {
            num_slots,
            offset,
            ready: Vec::new(),
            pending: Vec::new(),
            free: Vec::new(),
        };
        index.reset();
        Ok(index)
    }

    /// Slot count.
    pub fn num_slots(&self) -> i64 {
        self.num_slots
    }

    /// First slot.
    pub fn offset(&self) -> i64 {
        self.offset
    }

    /// Forgets every snapshot.
    pub fn reset(&mut self) {
        self.ready.clear();
        self.pending.clear();
        self.free = (self.offset..self.offset + self.num_slots).rev().collect();
    }

    /// Whether `key` is ready or about to be written.
    pub fn contains(&self, key: SnapKey) -> bool {
        self.is_ready(key) || self.pending.iter().any(|&(_, k)| k == key)
    }

    /// Whether a written snapshot of `key` exists.
    pub fn is_ready(&self, key: SnapKey) -> bool {
        self.ready.iter().any(|&(k, _)| k == key)
    }

    /// The ready slot holding `key`, marked most recently used.
    pub fn lookup(&mut self, key: SnapKey) -> Option<i64> {
        let pos = self.ready.iter().position(|&(k, _)| k == key)?;
        let entry = self.ready.remove(pos);
        self.ready.push(entry);
        Some(entry.1)
    }

    /// Claims a slot for a new snapshot of `key`; `None` when `key` exists or no slot can be freed.
    pub fn reserve(&mut self, key: SnapKey, exclude: &[i64]) -> Option<i64> {
        if self.contains(key) {
            return None;
        }
        let slot = match self.free.pop() {
            Some(slot) => slot,
            None => {
                let pos = self.ready.iter().position(|(_, s)| !exclude.contains(s))?;
                self.ready.remove(pos).1
            }
        };
        self.pending.push((slot, key));
        Some(slot)
    }

    fn take_pending(&mut self, slot: i64) -> Option<SnapKey> {
        let pos = self.pending.iter().position(|&(s, _)| s == slot)?;
        Some(self.pending.remove(pos).1)
    }

    /// Drops a reservation whose write will not happen (no-op for a slot that is not pending).
    pub fn cancel(&mut self, slot: i64) {
        if self.take_pending(slot).is_some() {
            self.free.push(slot);
        }
    }

    /// Marks a reserved slot as written (no-op for a slot that is not pending).
    pub fn commit(&mut self, slot: i64) {
        if let Some(key) = self.take_pending(slot) {
            self.ready.push((key, slot));
        }
    }

    /// `(ready in LRU order, pending sorted by slot, free stack)`, for tests.
    pub fn state(&self) -> SnapState {
        let mut pending = self.pending.clone();
        pending.sort_unstable();
        (self.ready.clone(), pending, self.free.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_commit_recycles_lru_ready_slot_not_excluded() {
        let mut idx = SnapshotIndex::new(2, 4).unwrap();
        let a = idx.reserve(1, &[]).unwrap();
        assert_eq!(a, 4);
        assert_eq!(idx.reserve(1, &[]), None);
        idx.commit(a);
        let b = idx.reserve(2, &[]).unwrap();
        idx.commit(b);
        assert_eq!(idx.lookup(1), Some(4));
        assert_eq!(idx.reserve(3, &[5]), Some(4));
        assert!(!idx.is_ready(1));
        assert_eq!(idx.reserve(9, &[5]), None);
        idx.cancel(4);
        assert_eq!(idx.state().2, vec![4]);
        assert!(SnapshotIndex::new(0, 0).is_err());
    }
}
