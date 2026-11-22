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

/// The key of a pending slot whose key is set only at commit ([`SnapshotIndex::commit_keyed`]). Never a page key: page
/// keys are never 0.
const UNKEYED: SnapKey = 0;

/// Cumulative counters of a [`SnapshotIndex`] (kept across [`SnapshotIndex::reset`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapStats {
    /// Ready snapshots handed out by [`SnapshotIndex::lookup`] (prefix-hit restores).
    pub restores: u64,
    /// Reservations committed as ready.
    pub saves: u64,
    /// Of `saves`, those committed with a key set at commit (decode snapshots).
    pub keyed_saves: u64,
    /// Ready snapshots recycled to make room for a new reservation.
    pub evictions: u64,
    /// Reservations dropped without being written (or committed under a key another slot already holds).
    pub cancels: u64,
}

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
    stats: SnapStats,
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
            stats: SnapStats::default(),
        };
        index.reset();
        Ok(index)
    }

    /// The cumulative counters.
    pub fn stats(&self) -> SnapStats {
        self.stats
    }

    /// `(ready, pending, free)` slot counts.
    pub fn occupancy(&self) -> (usize, usize, usize) {
        (self.ready.len(), self.pending.len(), self.free.len())
    }

    /// Slot count.
    pub fn num_slots(&self) -> i64 {
        self.num_slots
    }

    /// The key a pending reservation of `slot` was made for, or `None` when `slot` is not pending.
    pub fn pending_key(&self, slot: i64) -> Option<SnapKey> {
        self.pending.iter().find(|&&(s, _)| s == slot).map(|&(_, k)| k)
    }

    /// First slot.
    pub fn offset(&self) -> i64 {
        self.offset
    }

    /// Forgets every snapshot (the counters are kept).
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
        self.stats.restores += 1;
        Some(entry.1)
    }

    /// A free slot, else the least recently used ready slot not in `exclude` (counted as an eviction).
    fn take_slot(&mut self, exclude: &[i64]) -> Option<i64> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        let pos = self.ready.iter().position(|(_, s)| !exclude.contains(s))?;
        self.stats.evictions += 1;
        Some(self.ready.remove(pos).1)
    }

    /// Claims a slot for a new snapshot of `key`; `None` when `key` exists or no slot can be freed.
    pub fn reserve(&mut self, key: SnapKey, exclude: &[i64]) -> Option<i64> {
        if self.contains(key) {
            return None;
        }
        let slot = self.take_slot(exclude)?;
        self.pending.push((slot, key));
        Some(slot)
    }

    /// Claims a slot whose key is set only when it is committed ([`commit_keyed`](Self::commit_keyed)); `None` when no
    /// slot can be freed. The slot stays pending, and may be written any number of times, until then.
    pub fn reserve_unkeyed(&mut self, exclude: &[i64]) -> Option<i64> {
        let slot = self.take_slot(exclude)?;
        self.pending.push((slot, UNKEYED));
        Some(slot)
    }

    /// Number of slots pending without a key.
    pub fn num_unkeyed(&self) -> usize {
        self.pending.iter().filter(|&&(_, k)| k == UNKEYED).count()
    }

    /// Whether `slot` is pending without a key (a [`reserve_unkeyed`](Self::reserve_unkeyed) slot).
    pub fn is_unkeyed(&self, slot: i64) -> bool {
        self.pending_key(slot) == Some(UNKEYED)
    }

    fn take_pending(&mut self, slot: i64) -> Option<SnapKey> {
        let pos = self.pending.iter().position(|&(s, _)| s == slot)?;
        Some(self.pending.remove(pos).1)
    }

    /// Drops a reservation whose write will not happen (no-op for a slot that is not pending).
    pub fn cancel(&mut self, slot: i64) {
        if self.take_pending(slot).is_some() {
            self.stats.cancels += 1;
            self.free.push(slot);
        }
    }

    /// Marks a reserved slot as written (no-op for a slot that is not pending).
    pub fn commit(&mut self, slot: i64) {
        if let Some(key) = self.take_pending(slot) {
            self.stats.saves += 1;
            self.ready.push((key, slot));
        }
    }

    /// Marks an unkeyed slot as written under `key`; when `key` is already ready or pending elsewhere, the slot is
    /// dropped instead. No-op for a slot that is not pending unkeyed. Returns whether `key` became ready here.
    pub fn commit_keyed(&mut self, slot: i64, key: SnapKey) -> bool {
        if !self.is_unkeyed(slot) || key == UNKEYED {
            return false;
        }
        self.take_pending(slot);
        if self.contains(key) {
            self.stats.cancels += 1;
            self.free.push(slot);
            return false;
        }
        self.stats.saves += 1;
        self.stats.keyed_saves += 1;
        self.ready.push((key, slot));
        true
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

    #[test]
    fn unkeyed_slots_take_their_key_at_commit_and_never_shadow_an_existing_one() {
        let mut idx = SnapshotIndex::new(3, 0).unwrap();
        let a = idx.reserve_unkeyed(&[]).unwrap();
        assert!(idx.is_unkeyed(a));
        assert!(idx.lookup(UNKEYED).is_none());
        // An unkeyed slot is never found by lookup and is not recycled while pending.
        let b = idx.reserve(7, &[]).unwrap();
        idx.commit(b);
        let c = idx.reserve(8, &[]).unwrap();
        idx.commit(c);
        assert_eq!(idx.occupancy(), (2, 1, 0));
        // Committing under a fresh key makes it ready.
        assert!(idx.commit_keyed(a, 9));
        assert!(idx.is_ready(9));
        assert!(!idx.is_unkeyed(a));
        // Committing again is a no-op (no longer pending).
        assert!(!idx.commit_keyed(a, 10));
        // A key that is already ready drops the unkeyed slot back to the free list.
        let d = idx.reserve_unkeyed(&[]).unwrap(); // recycles the LRU ready slot (key 7)
        assert!(!idx.is_ready(7));
        assert!(!idx.commit_keyed(d, 8));
        assert_eq!(idx.occupancy(), (2, 0, 1));
        // Keyed commit refuses a keyed (normal) pending slot.
        let e = idx.reserve(11, &[]).unwrap();
        assert!(!idx.commit_keyed(e, 12));
        assert!(!idx.is_ready(12));
        idx.commit(e);
        assert!(idx.is_ready(11));
    }

    #[test]
    fn counters_track_restores_saves_evictions_and_cancels() {
        let mut idx = SnapshotIndex::new(1, 0).unwrap();
        let a = idx.reserve(1, &[]).unwrap();
        idx.commit(a);
        assert_eq!(idx.lookup(1), Some(a));
        let b = idx.reserve(2, &[]).unwrap(); // evicts key 1
        idx.cancel(b);
        let c = idx.reserve_unkeyed(&[]).unwrap();
        assert!(idx.commit_keyed(c, 3));
        idx.reset();
        assert_eq!(
            idx.stats(),
            SnapStats {
                restores: 1,
                saves: 2,
                keyed_saves: 1,
                evictions: 1,
                cancels: 1,
            }
        );
        assert_eq!(idx.occupancy(), (0, 0, 1));
    }
}
