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

//! Step records: a [`Plan`] as a flat byte string, and back.
//!
//! Integers are zigzag LEB128 varints, so a decode step of a few dozen rows takes a few hundred bytes. The record
//! carries every plan field; applying it on a follower core (which replays the leader's decisions instead of
//! scheduling) is a later step.

use crate::engine::plan::Plan;
use crate::engine::CoreError;

/// Record format version (the first varint).
pub const RECORD_VERSION: i64 = 2;

/// Appends `v` as a zigzag LEB128 varint.
pub fn put_varint(out: &mut Vec<u8>, v: i64) {
    let mut z = ((v << 1) ^ (v >> 63)) as u64;
    loop {
        let b = (z & 0x7F) as u8;
        z >>= 7;
        if z == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// A reader of zigzag LEB128 varints.
pub struct VarintReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> VarintReader<'a> {
    /// A reader at the start of `bytes`.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Whether every byte was read.
    pub fn done(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// The next varint.
    pub fn next_i64(&mut self) -> Result<i64, CoreError> {
        let mut z = 0u64;
        let mut shift = 0u32;
        loop {
            let &b = self
                .bytes
                .get(self.pos)
                .ok_or_else(|| CoreError::Value("truncated varint stream".into()))?;
            self.pos += 1;
            if shift >= 64 {
                return Err(CoreError::Value("varint too long".into()));
            }
            z |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok((z >> 1) as i64 ^ -((z & 1) as i64));
            }
            shift += 7;
        }
    }

    /// The next varint as a count.
    pub fn next_len(&mut self) -> Result<usize, CoreError> {
        let v = self.next_i64()?;
        usize::try_from(v).map_err(|_| CoreError::Value(format!("negative length {v}")))
    }

    /// `n` varints.
    pub fn next_vec(&mut self, n: usize) -> Result<Vec<i64>, CoreError> {
        (0..n).map(|_| self.next_i64()).collect()
    }
}

fn put_list<T: Copy + Into<i64>>(out: &mut Vec<u8>, values: &[T]) {
    put_varint(out, values.len() as i64);
    for &v in values {
        put_varint(out, v.into());
    }
}

fn get_list(r: &mut VarintReader<'_>) -> Result<Vec<i64>, CoreError> {
    let n = r.next_len()?;
    r.next_vec(n)
}

/// Encodes a plan.
pub fn encode(plan: &Plan) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + plan.len() * 8 + plan.new_pages.len() * 2);
    put_varint(&mut out, RECORD_VERSION);
    put_varint(&mut out, plan.id as i64);
    put_varint(&mut out, i64::from(plan.flags));
    put_list(&mut out, &plan.handles.iter().map(|&h| h as i64).collect::<Vec<_>>());
    put_list(&mut out, &plan.scheduled);
    put_list(&mut out, &plan.computed);
    put_list(&mut out, &plan.kinds);
    put_list(&mut out, &plan.emits);
    put_list(&mut out, &plan.new_pages);
    put_list(&mut out, &plan.new_page_offsets);
    put_list(&mut out, &plan.rank_offsets);
    put_list(&mut out, &plan.preempted.iter().map(|&h| h as i64).collect::<Vec<_>>());
    put_list(&mut out, &plan.finished.iter().map(|&h| h as i64).collect::<Vec<_>>());
    for pairs in [&plan.snapshot_restores, &plan.snapshot_saves] {
        put_list(&mut out, &pairs.iter().flatten().copied().collect::<Vec<_>>());
    }
    put_list(&mut out, &plan.stats);
    put_list(&mut out, &plan.draft_tokens);
    put_list(&mut out, &plan.draft_offsets);
    put_list(&mut out, &plan.new_tokens);
    put_list(&mut out, &plan.new_token_offsets);
    put_list(&mut out, &plan.common_prefix_pages);
    put_varint(&mut out, plan.req_bucket);
    out
}

fn narrow<T: TryFrom<i64>>(values: Vec<i64>) -> Result<Vec<T>, CoreError> {
    values
        .into_iter()
        .map(|v| T::try_from(v).map_err(|_| CoreError::Value(format!("record value {v} out of range"))))
        .collect()
}

fn pairs(values: Vec<i64>) -> Result<Vec<[i32; 2]>, CoreError> {
    if values.len() % 2 != 0 {
        return Err(CoreError::Value("odd snapshot pair list".into()));
    }
    let flat: Vec<i32> = narrow(values)?;
    Ok(flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
}

/// Decodes a plan encoded by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<Plan, CoreError> {
    let mut r = VarintReader::new(bytes);
    let version = r.next_i64()?;
    if version != RECORD_VERSION {
        return Err(CoreError::Value(format!("unknown step record version {version}")));
    }
    let id = r.next_i64()? as u64;
    let flags = u32::try_from(r.next_i64()?).map_err(|_| CoreError::Value("bad plan flags".into()))?;
    let plan = Plan {
        id,
        flags,
        handles: get_list(&mut r)?.into_iter().map(|v| v as u64).collect(),
        scheduled: narrow(get_list(&mut r)?)?,
        computed: narrow(get_list(&mut r)?)?,
        kinds: narrow(get_list(&mut r)?)?,
        emits: narrow(get_list(&mut r)?)?,
        new_pages: narrow(get_list(&mut r)?)?,
        new_page_offsets: narrow(get_list(&mut r)?)?,
        rank_offsets: narrow(get_list(&mut r)?)?,
        preempted: get_list(&mut r)?.into_iter().map(|v| v as u64).collect(),
        finished: get_list(&mut r)?.into_iter().map(|v| v as u64).collect(),
        snapshot_restores: pairs(get_list(&mut r)?)?,
        snapshot_saves: pairs(get_list(&mut r)?)?,
        stats: get_list(&mut r)?,
        draft_tokens: narrow(get_list(&mut r)?)?,
        draft_offsets: narrow(get_list(&mut r)?)?,
        new_tokens: narrow(get_list(&mut r)?)?,
        new_token_offsets: narrow(get_list(&mut r)?)?,
        common_prefix_pages: narrow(get_list(&mut r)?)?,
        req_bucket: r.next_i64()?,
    };
    if !r.done() {
        return Err(CoreError::Value("trailing bytes after a step record".into()));
    }
    let n = plan.handles.len();
    if [
        plan.scheduled.len(),
        plan.computed.len(),
        plan.kinds.len(),
        plan.emits.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(CoreError::Value("step record columns differ in length".into()));
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip() {
        let values = [0i64, 1, -1, 63, -64, 64, 300, -300, i64::MAX, i64::MIN];
        let mut out = Vec::new();
        for &v in &values {
            put_varint(&mut out, v);
        }
        let mut r = VarintReader::new(&out);
        for &v in &values {
            assert_eq!(r.next_i64().unwrap(), v);
        }
        assert!(r.done());
    }

    #[test]
    fn plan_round_trips() {
        let plan = Plan {
            id: 7,
            flags: 9,
            handles: vec![(1 << 32) | 3, (2 << 32) | 1],
            scheduled: vec![1, 17],
            computed: vec![40, 0],
            kinds: vec![0, 1],
            emits: vec![1, 0],
            new_pages: vec![5, 9, 10, 0],
            new_page_offsets: vec![0, 1, 4],
            rank_offsets: vec![0, 2],
            preempted: vec![(1 << 32) | 8],
            finished: vec![],
            snapshot_restores: vec![[1, 3]],
            snapshot_saves: vec![[0, 2], [1, 4]],
            stats: vec![18, 2, 0, 30, 32, 14, 1, 18, 18, 2, 0, 30],
            draft_tokens: vec![44, 45],
            draft_offsets: vec![0, 2, 2],
            new_tokens: vec![33],
            new_token_offsets: vec![0, 1, 1],
            common_prefix_pages: vec![1],
            req_bucket: 4,
        };
        let bytes = encode(&plan);
        assert_eq!(decode(&bytes).unwrap(), plan);
        assert!(decode(&bytes[..bytes.len() - 1]).is_err());
    }
}
