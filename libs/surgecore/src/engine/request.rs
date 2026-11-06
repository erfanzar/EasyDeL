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

//! The request table: every live request (waiting, running or preempted) behind a generational handle.
//!
//! A [`Request`] holds the scheduler's state of one request: its tokens (prompt then output), the computed-token
//! and in-flight counters, its stop and length parameters, its sampling parameters, its image spans and its KV slot.
//! [`Request::check_stop`] is eSurge's per-token stop check.

/// A request handle: `(generation << 32) | index`. A finished request's handle never becomes live again.
pub type Handle = u64;

/// Admission flag: the request uses structured output (a grammar mask).
pub const FLAG_GRAMMAR: u32 = 1;
/// Admission flag: the request never takes a prefix-cache hit (prompt logprobs were requested).
pub const FLAG_NO_PREFIX_HIT: u32 = 2;
/// Admission flag: the request carries vision inputs.
pub const FLAG_VISION: u32 = 4;
/// Admission flag: the request may be given speculative drafts.
pub const FLAG_SPEC_ELIGIBLE: u32 = 8;

/// A request's scheduling state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Queued, never scheduled.
    Waiting,
    /// Holds a running slot.
    Running,
    /// Preempted back to the queue; every token is prompt again.
    Preempted,
    /// Finished by EOS, a stop token or a stop string.
    Stopped,
    /// Finished by `max_tokens` or `max_model_len`.
    Length,
    /// Aborted.
    Aborted,
}

impl Status {
    /// Whether the status is a finished one.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Stopped | Self::Length | Self::Aborted)
    }

    /// The event finish code: 0 running, 1 stop, 2 length, 3 abort.
    pub fn finish_code(self) -> i8 {
        match self {
            Self::Stopped => 1,
            Self::Length => 2,
            Self::Aborted => 3,
            _ => 0,
        }
    }
}

/// Why an external `finish` ends requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop string matched (code 1).
    Stop,
    /// A length limit (code 2).
    Length,
    /// An abort (code 3).
    Abort,
}

impl FinishReason {
    /// The reason of an event finish code (1, 2 or 3).
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            1 => Some(Self::Stop),
            2 => Some(Self::Length),
            3 => Some(Self::Abort),
            _ => None,
        }
    }

    /// The finished status the reason sets.
    pub fn status(self) -> Status {
        match self {
            Self::Stop => Status::Stopped,
            Self::Length => Status::Length,
            Self::Abort => Status::Aborted,
        }
    }
}

/// Sampling parameters, kept for the row columns the step-input builder reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    /// Temperature (0 is greedy).
    pub temperature: f32,
    /// Nucleus mass.
    pub top_p: f32,
    /// Top-k (0 is off).
    pub top_k: i32,
    /// Min-p.
    pub min_p: f32,
    /// Frequency penalty.
    pub frequency_penalty: f32,
    /// Presence penalty.
    pub presence_penalty: f32,
    /// Repetition penalty (1 is off).
    pub repetition_penalty: f32,
    /// Seed (-1 is none).
    pub seed: i64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: -1,
        }
    }
}

/// One request to admit.
#[derive(Clone, Copy, Debug)]
pub struct NewRequest<'a> {
    /// Prompt tokens (image placeholders already expanded).
    pub prompt: &'a [i32],
    /// Output-token limit.
    pub max_tokens: i64,
    /// Minimum output tokens (stored; not enforced, as today).
    pub min_tokens: i64,
    /// EOS token, if any.
    pub eos: Option<i32>,
    /// Whether EOS (also as a stop id) is ignored.
    pub ignore_eos: bool,
    /// Priority: lower is served first, higher is preempted first.
    pub priority: i64,
    /// Arrival time (the priority policy's tie-break).
    pub arrival: f64,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Stop token ids.
    pub stop_ids: &'a [i32],
    /// Half-open, sorted image spans `(start, stop)` of prompt positions.
    pub spans: &'a [(i64, i64)],
    /// The data-parallel rank to place it on; `None` lets the core route it.
    pub dp_rank: Option<usize>,
    /// The request's 128-bit multimodal page-key identity (0 for text).
    pub hash_extra: u128,
    /// Sampling parameters.
    pub sampling: Sampling,
}

/// The scheduler state of one live request.
#[derive(Clone, Debug)]
pub struct Request {
    /// The request's handle.
    pub handle: Handle,
    /// Its data-parallel rank.
    pub rank: usize,
    /// Its status.
    pub status: Status,
    /// Prompt tokens followed by output tokens.
    pub tokens: Vec<i32>,
    /// Prompt length.
    pub num_prompt: i64,
    /// Tokens whose KV is computed or scheduled (advanced at schedule, as today).
    pub computed: i64,
    /// Async in-flight output placeholders.
    pub placeholders: i64,
    /// Output-token limit.
    pub max_tokens: i64,
    /// Minimum output tokens (stored only).
    pub min_tokens: i64,
    /// EOS token.
    pub eos: Option<i32>,
    /// Ignore EOS.
    pub ignore_eos: bool,
    /// Stop token ids.
    pub stop_ids: Vec<i32>,
    /// Priority (lower first).
    pub priority: i64,
    /// Arrival time.
    pub arrival: f64,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Image spans.
    pub spans: Vec<(i64, i64)>,
    /// Multimodal page-key identity.
    pub hash_extra: u128,
    /// Sampling parameters.
    pub sampling: Sampling,
    /// The request's slot in its rank's KV core, once opened.
    pub kv_slot: Option<usize>,
    /// The recurrent-snapshot save boundary and key the request still has to reach.
    pub save_target: Option<(i64, u128)>,
    /// Prefix-hit tokens of the first admission (-1 before it).
    pub cached_tokens: i64,
    /// The stop token that finished the request, if one did.
    pub stop_reason: Option<i32>,
    /// Speculative drafts the next step may verify (the runner's last proposal, trimmed to what was scheduled;
    /// `-1` placeholders under async scheduling).
    pub drafts: Vec<i32>,
}

impl Request {
    /// Prompt plus output tokens.
    pub fn num_tokens(&self) -> i64 {
        self.tokens.len() as i64
    }

    /// Output tokens.
    pub fn num_output(&self) -> i64 {
        self.num_tokens() - self.num_prompt
    }

    /// Whether prefix-cache hits are disabled for the request.
    pub fn no_prefix_hit(&self) -> bool {
        self.flags & FLAG_NO_PREFIX_HIT != 0
    }

    /// The priority queue key `(priority, arrival)`.
    pub fn queue_key(&self) -> (i64, f64) {
        (self.priority, self.arrival)
    }

    /// The stop check after one appended output token: length (`max_model_len`, then `max_tokens`), then EOS unless
    /// ignored, then stop ids (an ignored EOS among the stop ids does not stop). Sets the finished status and, for a
    /// stop id, the stop reason; returns whether the request stopped.
    pub fn check_stop(&mut self, max_model_len: i64) -> bool {
        if self.num_tokens() >= max_model_len || self.num_output() >= self.max_tokens {
            self.status = Status::Length;
            return true;
        }
        let Some(&last) = self.tokens.last() else {
            return false;
        };
        if !self.ignore_eos && self.eos == Some(last) {
            self.status = Status::Stopped;
            return true;
        }
        let ignored_eos = self.ignore_eos && self.eos == Some(last);
        if !ignored_eos && self.stop_ids.contains(&last) {
            self.status = Status::Stopped;
            self.stop_reason = Some(last);
            return true;
        }
        false
    }
}

/// Every live request, addressed by generational handles.
#[derive(Default)]
pub struct RequestTable {
    slots: Vec<Option<Request>>,
    gens: Vec<u32>,
    free: Vec<u32>,
    live: usize,
}

fn split(handle: Handle) -> (usize, u32) {
    ((handle & 0xFFFF_FFFF) as usize, (handle >> 32) as u32)
}

impl RequestTable {
    /// Number of live requests.
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether no request is live.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// The handle the next [`insert`](Self::insert) returns.
    pub fn next_handle(&self) -> Handle {
        match self.free.last() {
            Some(&i) => (u64::from(self.gens[i as usize]) << 32) | u64::from(i),
            None => (1u64 << 32) | self.slots.len() as u64,
        }
    }

    /// Inserts the request built from its handle.
    pub fn insert(&mut self, make: impl FnOnce(Handle) -> Request) -> Handle {
        let handle = self.next_handle();
        let (index, _) = split(handle);
        if index == self.slots.len() {
            self.slots.push(None);
            self.gens.push(1);
        } else {
            self.free.pop();
        }
        self.slots[index] = Some(make(handle));
        self.live += 1;
        handle
    }

    /// The live request of `handle`.
    pub fn get(&self, handle: Handle) -> Option<&Request> {
        let (index, gen) = split(handle);
        if self.gens.get(index) != Some(&gen) {
            return None;
        }
        self.slots[index].as_ref()
    }

    /// The live request of `handle`, mutably.
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut Request> {
        let (index, gen) = split(handle);
        if self.gens.get(index) != Some(&gen) {
            return None;
        }
        self.slots[index].as_mut()
    }

    /// Removes the request; its handle is never live again.
    pub fn remove(&mut self, handle: Handle) -> Option<Request> {
        let (index, gen) = split(handle);
        if self.gens.get(index) != Some(&gen) {
            return None;
        }
        let request = self.slots[index].take()?;
        self.gens[index] = self.gens[index].wrapping_add(1).max(1);
        self.free.push(index as u32);
        self.live -= 1;
        Some(request)
    }

    /// Live requests in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &Request> {
        self.slots.iter().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(handle: Handle) -> Request {
        Request {
            handle,
            rank: 0,
            status: Status::Waiting,
            tokens: vec![3, 4],
            num_prompt: 2,
            computed: 0,
            placeholders: 0,
            max_tokens: 3,
            min_tokens: 0,
            eos: Some(2),
            ignore_eos: false,
            stop_ids: vec![5, 2],
            priority: 0,
            arrival: 0.0,
            flags: 0,
            spans: Vec::new(),
            hash_extra: 0,
            sampling: Sampling::default(),
            kv_slot: None,
            save_target: None,
            cached_tokens: -1,
            stop_reason: None,
            drafts: Vec::new(),
        }
    }

    #[test]
    fn handles_are_generational() {
        let mut table = RequestTable::default();
        let a = table.insert(req);
        assert!(table.remove(a).is_some());
        let b = table.insert(req);
        assert_ne!(a, b);
        assert!(table.get(a).is_none());
        assert_eq!(table.get(b).unwrap().handle, b);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn stop_check_order() {
        let mut r = req(1);
        r.tokens.push(2);
        assert!(r.check_stop(100));
        assert_eq!((r.status, r.stop_reason), (Status::Stopped, None));
        let mut r = req(1);
        r.ignore_eos = true;
        r.tokens.push(2);
        assert!(!r.check_stop(100), "an ignored EOS among the stop ids does not stop");
        r.tokens.push(5);
        assert!(r.check_stop(100));
        assert_eq!((r.status, r.stop_reason), (Status::Stopped, Some(5)));
        let mut r = req(1);
        r.tokens.extend([7, 7, 7]);
        assert!(r.check_stop(100));
        assert_eq!(r.status, Status::Length);
        let mut r = req(1);
        r.tokens.push(2);
        assert!(r.check_stop(3), "length is checked before EOS");
        assert_eq!(r.status, Status::Length);
    }
}
