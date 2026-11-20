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

//! Waiting queues: FCFS (a deque) and priority (ordered by `(priority, arrival, insertion sequence)`).
//!
//! The priority queue keeps its entries sorted by that key. The key is unique (the sequence number breaks every
//! tie), so popping the minimum and iterating in key order give exactly what a binary heap of the same keys gives.

use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::engine::request::Handle;

/// The scheduling policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// First come, first served; the last running request is preempted first.
    Fcfs,
    /// Lowest `(priority, arrival)` first; the highest is preempted first.
    Priority,
}

impl Policy {
    /// The policy of a numeric code: 0 FCFS, 1 priority.
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(Self::Fcfs),
            1 => Some(Self::Priority),
            _ => None,
        }
    }
}

type Entry = (i64, f64, u64, Handle);

fn cmp_entry(a: &Entry, b: &Entry) -> Ordering {
    a.0.cmp(&b.0)
        .then(a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .then(a.2.cmp(&b.2))
}

/// A waiting queue of request handles.
#[derive(Clone, Debug)]
pub enum Queue {
    /// FCFS order.
    Fcfs(VecDeque<Handle>),
    /// Priority order: sorted entries and the next insertion sequence number.
    Priority(Vec<Entry>, u64),
}

impl Queue {
    /// An empty queue for `policy`.
    pub fn new(policy: Policy) -> Self {
        match policy {
            Policy::Fcfs => Self::Fcfs(VecDeque::new()),
            Policy::Priority => Self::Priority(Vec::new(), 0),
        }
    }

    /// Queued requests.
    pub fn len(&self) -> usize {
        match self {
            Self::Fcfs(q) => q.len(),
            Self::Priority(v, _) => v.len(),
        }
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends a request (`key` is its `(priority, arrival)`; FCFS ignores it).
    pub fn push(&mut self, h: Handle, key: (i64, f64)) {
        match self {
            Self::Fcfs(q) => q.push_back(h),
            Self::Priority(v, seq) => {
                let e = (key.0, key.1, *seq, h);
                *seq += 1;
                let at = v.partition_point(|x| cmp_entry(x, &e) == Ordering::Less);
                v.insert(at, e);
            }
        }
    }

    /// Puts a request back at the front (FCFS) or re-inserts it (priority).
    pub fn prepend(&mut self, h: Handle, key: (i64, f64)) {
        match self {
            Self::Fcfs(q) => q.push_front(h),
            Self::Priority(..) => self.push(h, key),
        }
    }

    /// Puts every request of `other` back: in its order at the front (FCFS), or re-inserted in its order (priority).
    pub fn prepend_all(&mut self, other: Queue) {
        match other {
            Self::Fcfs(o) => {
                for h in o.into_iter().rev() {
                    self.prepend(h, (0, 0.0));
                }
            }
            Self::Priority(o, _) => {
                for (p, a, _, h) in o {
                    self.push(h, (p, a));
                }
            }
        }
    }

    /// The head.
    pub fn peek(&self) -> Option<Handle> {
        match self {
            Self::Fcfs(q) => q.front().copied(),
            Self::Priority(v, _) => v.first().map(|e| e.3),
        }
    }

    /// Removes and returns the head.
    pub fn pop(&mut self) -> Option<Handle> {
        match self {
            Self::Fcfs(q) => q.pop_front(),
            Self::Priority(v, _) => (!v.is_empty()).then(|| v.remove(0).3),
        }
    }

    /// Removes every listed request, keeping the order of the rest.
    pub fn remove(&mut self, handles: &[Handle]) {
        match self {
            Self::Fcfs(q) => q.retain(|h| !handles.contains(h)),
            Self::Priority(v, _) => v.retain(|e| !handles.contains(&e.3)),
        }
    }

    /// The handles in queue order.
    pub fn iter(&self) -> Box<dyn Iterator<Item = Handle> + '_> {
        match self {
            Self::Fcfs(q) => Box::new(q.iter().copied()),
            Self::Priority(v, _) => Box::new(v.iter().map(|e| e.3)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcfs_prepend_keeps_order() {
        let mut q = Queue::new(Policy::Fcfs);
        q.push(1, (0, 0.0));
        q.push(2, (0, 0.0));
        let mut skipped = Queue::new(Policy::Fcfs);
        skipped.push(3, (0, 0.0));
        skipped.push(4, (0, 0.0));
        q.prepend_all(skipped);
        q.prepend(5, (0, 0.0));
        assert_eq!(q.iter().collect::<Vec<_>>(), vec![5, 3, 4, 1, 2]);
        q.remove(&[3, 1]);
        assert_eq!(q.pop(), Some(5));
        assert_eq!(q.iter().collect::<Vec<_>>(), vec![4, 2]);
    }

    #[test]
    fn priority_orders_by_priority_arrival_sequence() {
        let mut q = Queue::new(Policy::Priority);
        q.push(1, (1, 0.0));
        q.push(2, (0, 5.0));
        q.push(3, (0, 5.0));
        q.push(4, (0, 1.0));
        assert_eq!(q.iter().collect::<Vec<_>>(), vec![4, 2, 3, 1]);
        assert_eq!(q.pop(), Some(4));
        q.prepend(4, (0, 1.0));
        assert_eq!(q.peek(), Some(4));
    }
}
