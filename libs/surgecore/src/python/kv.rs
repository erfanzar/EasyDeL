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

//! The KV core's bindings in the `surgecore._native` extension module: `KvCore` and `SnapshotIndex`.
//!
//! Requests are addressed by the slot `KvCore.open` returns; eSurge's `CacheManager` keeps the request-id-to-slot
//! map. Page lists cross as lists of ints, page keys as 128-bit ints.

use pyo3::exceptions::{PyAssertionError, PyIndexError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::kv::group::{Group, Kind};
use crate::kv::snapshot::SnapshotIndex as Index;
use crate::kv::{KvConfig, KvCore as Core, KvError, Tokens};

fn kv_error(err: KvError) -> PyErr {
    match err {
        KvError::Value(m) => PyValueError::new_err(m),
        KvError::Index(m) => PyIndexError::new_err(m),
        KvError::Runtime(m) => PyRuntimeError::new_err(m),
        KvError::Assertion(m) => PyAssertionError::new_err(m),
        KvError::Key(m) => PyKeyError::new_err(m),
    }
}

fn tokens(values: &Option<Vec<i32>>, offset: usize) -> Option<Tokens<'_>> {
    values.as_deref().map(|tokens| Tokens { tokens, offset })
}

/// One KV page pool with its prefix cache and the page state of every request that touched it.
#[pyclass(module = "surgecore._native", name = "KvCore")]
pub struct KvCore {
    core: Core,
}

#[pymethods]
impl KvCore {
    /// `groups` holds one `(kind, page_size, window_or_chunk, type_index)` per cache group: kind 0 full attention,
    /// 1 sliding window, 2 chunked local attention; groups with equal cache specs share a type index.
    #[new]
    #[pyo3(signature = (*, num_pages, groups, max_model_len, caching, eagle))]
    fn new(
        num_pages: i64,
        groups: Vec<(i64, i64, i64, i64)>,
        max_model_len: i64,
        caching: bool,
        eagle: bool,
    ) -> PyResult<Self> {
        if num_pages <= 0 {
            return Err(PyValueError::new_err("num_pages must be a positive integer"));
        }
        let mut parsed = Vec::with_capacity(groups.len());
        for (kind, page_size, window, type_index) in groups {
            let kind = match kind {
                0 => Kind::Full,
                1 => Kind::Sliding(window),
                2 => Kind::Chunked(window),
                other => return Err(PyValueError::new_err(format!("unknown cache group kind {other}"))),
            };
            parsed.push(Group {
                kind,
                page_size,
                type_index,
            });
        }
        let core = Core::new(KvConfig {
            num_pages: num_pages as usize,
            groups: parsed,
            max_model_len,
            caching,
            eagle,
        })
        .map_err(kv_error)?;
        Ok(Self { core })
    }

    /// A fresh request slot; `extra` is the request's 128-bit multimodal identity (0 for text).
    fn open(&mut self, extra: u128) -> usize {
        self.core.open(extra)
    }

    /// Number of known page keys of the request in `slot`.
    fn num_keys(&self, slot: usize) -> PyResult<usize> {
        self.core.num_keys(slot).map_err(kv_error)
    }

    /// The request's page keys, computed from `tokens` (all its tokens) when none are known.
    fn keys(&mut self, slot: usize, tokens_: Option<Vec<i32>>) -> PyResult<Vec<u128>> {
        Ok(self
            .core
            .ensure_keys(slot, tokens(&tokens_, 0))
            .map_err(kv_error)?
            .to_vec())
    }

    /// The longest prefix hit `(pages per group, hit tokens)`; computes and keeps the page keys first when needed.
    #[pyo3(signature = (slot, tokens_, no_hit, max_len, hint=None, dp=None))]
    fn lookup(
        &mut self,
        slot: usize,
        tokens_: Option<Vec<i32>>,
        no_hit: bool,
        max_len: i64,
        hint: Option<i64>,
        dp: Option<i64>,
    ) -> PyResult<(Vec<Vec<u32>>, i64)> {
        self.core
            .lookup(slot, tokens(&tokens_, 0), no_hit, max_len, hint, dp)
            .map_err(kv_error)
    }

    /// Hit tokens of a lookup that keeps nothing (the slot's known keys, else keys of `tokens` with `extra`).
    fn probe(&self, slot: Option<usize>, tokens_: Option<Vec<i32>>, extra: u128, max_len: i64) -> PyResult<i64> {
        self.core
            .probe(slot, tokens(&tokens_, 0), extra, max_len)
            .map_err(kv_error)
    }

    /// Transactional allocation; the new pages per group, or `None` with nothing changed.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (slot, computed, n_new, n_hit, hits, lookahead, hint=None, dp=None))]
    fn allocate(
        &mut self,
        slot: usize,
        computed: i64,
        n_new: i64,
        n_hit: i64,
        hits: Option<Vec<Vec<u32>>>,
        lookahead: i64,
        hint: Option<i64>,
        dp: Option<i64>,
    ) -> PyResult<Option<Vec<Vec<u32>>>> {
        self.core
            .allocate(slot, computed, n_new, n_hit, hits.as_ref(), lookahead, hint, dp)
            .map_err(kv_error)
    }

    /// Free pages the request still draws before it holds `num_tokens` slots (read-only).
    #[pyo3(signature = (slot, num_tokens, hits=None, max_step=None))]
    fn needed(
        &self,
        slot: Option<usize>,
        num_tokens: i64,
        hits: Option<Vec<Vec<u32>>>,
        max_step: Option<i64>,
    ) -> PyResult<i64> {
        self.core
            .needed(slot, num_tokens, hits.as_ref(), max_step)
            .map_err(kv_error)
    }

    /// Frees every page of the request; returns whether its slot was closed.
    fn free(&mut self, slot: usize) -> PyResult<bool> {
        self.core.free(slot).map_err(kv_error)
    }

    /// Forgets the request's page keys; returns whether its slot was closed.
    fn forget_keys(&mut self, slot: usize) -> PyResult<bool> {
        self.core.forget_keys(slot).map_err(kv_error)
    }

    /// Publishes the pages made full by the first `n` computed tokens. Without `tokens`, when page keys must be
    /// computed, nothing changes and the token range `(lo, hi)` to pass (as `tokens`, with `offset=lo`) is returned.
    #[pyo3(signature = (slot, n, tokens_=None, offset=0))]
    fn publish(
        &mut self,
        slot: usize,
        n: i64,
        tokens_: Option<Vec<i32>>,
        offset: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        if !self.core.caching() {
            return Ok(None);
        }
        if tokens_.is_none() {
            if let Some(range) = self.core.publish_tokens_needed(slot, n).map_err(kv_error)? {
                return Ok(Some(range));
            }
        }
        self.core.publish(slot, n, tokens(&tokens_, offset)).map_err(kv_error)?;
        Ok(None)
    }

    /// Page ids per group of the request.
    fn page_ids(&self, slot: usize) -> PyResult<Vec<Vec<u32>>> {
        Ok(self.core.page_ids(slot).map_err(kv_error)?.to_vec())
    }

    /// Leading pages per group that at least `num_scheduled` requests share.
    fn common_prefix(&self, slot: Option<usize>, num_scheduled: i64) -> Vec<i64> {
        self.core.common_prefix(slot, num_scheduled)
    }

    /// Releases the request's unpublished pages past those `keep_tokens` slots need; returns how many.
    fn release_tail(&mut self, slot: usize, keep_tokens: i64) -> PyResult<usize> {
        self.core.release_tail(slot, keep_tokens).map_err(kv_error)
    }

    /// Forgets every published page; `False` while any page is in use.
    fn reset(&mut self) -> bool {
        self.core.reset()
    }

    /// Pages in the pool, the null page included.
    #[getter]
    fn num_pages(&self) -> usize {
        self.core.pool().num_pages()
    }

    /// Pages on the free list.
    #[getter]
    fn num_free(&self) -> usize {
        self.core.pool().num_free()
    }

    /// `1 - free / num_pages`.
    #[getter]
    fn usage(&self) -> f64 {
        self.core.pool().usage()
    }

    /// Distinct published `(page key, group)` entries.
    #[getter]
    fn num_cached(&self) -> usize {
        self.core.pool().num_cached()
    }

    /// Whether prefix caching is on.
    #[getter]
    fn caching(&self) -> bool {
        self.core.caching()
    }

    /// Live request slots.
    #[getter]
    fn num_open(&self) -> usize {
        self.core.num_open()
    }

    /// The free list, evicted first to last.
    fn free_order(&self) -> Vec<u32> {
        self.core.pool().free_order()
    }

    /// Reference counts of pages 1.. (the null page is left out).
    fn refs(&self) -> Vec<u32> {
        self.core.pool().refs()
    }

    /// Checks the pool and page-list invariants; raises `RuntimeError` on a violation.
    fn check(&self) -> PyResult<()> {
        self.core.check().map_err(PyRuntimeError::new_err)
    }
}

/// Slot bookkeeping for recurrent-state snapshots at prefix-cache page boundaries.
#[pyclass(module = "surgecore._native", name = "SnapshotIndex")]
pub struct SnapshotIndex {
    idx: Index,
}

#[pymethods]
impl SnapshotIndex {
    #[new]
    #[pyo3(signature = (num_slots, slot_offset=0))]
    fn new(num_slots: i64, slot_offset: i64) -> PyResult<Self> {
        Ok(Self {
            idx: Index::new(num_slots, slot_offset).map_err(kv_error)?,
        })
    }

    /// Slot count.
    #[getter]
    fn num_slots(&self) -> i64 {
        self.idx.num_slots()
    }

    /// First slot.
    #[getter]
    fn slot_offset(&self) -> i64 {
        self.idx.offset()
    }

    /// Forgets every snapshot.
    fn reset(&mut self) {
        self.idx.reset();
    }

    /// Whether `key` is ready or about to be written.
    fn contains(&self, key: u128) -> bool {
        self.idx.contains(key)
    }

    /// Whether a written snapshot of `key` exists.
    fn is_ready(&self, key: u128) -> bool {
        self.idx.is_ready(key)
    }

    /// The ready slot holding `key`, marked most recently used.
    fn lookup(&mut self, key: u128) -> Option<i64> {
        self.idx.lookup(key)
    }

    /// Claims a slot for a new snapshot of `key`; ready slots in `exclude` are never recycled.
    #[pyo3(signature = (key, exclude=None))]
    fn reserve(&mut self, key: u128, exclude: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i64>> {
        let mut ex = Vec::new();
        if let Some(items) = exclude {
            for item in items.try_iter()? {
                ex.push(item?.extract::<i64>()?);
            }
        }
        Ok(self.idx.reserve(key, &ex))
    }

    /// Drops a reservation whose write will not happen.
    fn cancel(&mut self, slot: i64) {
        self.idx.cancel(slot);
    }

    /// Marks a reserved slot as written.
    fn commit(&mut self, slot: i64) {
        self.idx.commit(slot);
    }

    /// `(ready in LRU order, pending sorted by slot, free stack)`.
    #[allow(clippy::type_complexity)]
    fn state(&self) -> (Vec<(u128, i64)>, Vec<(i64, u128)>, Vec<i64>) {
        self.idx.state()
    }
}

/// Adds the KV core's classes to the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KvCore>()?;
    m.add_class::<SnapshotIndex>()?;
    Ok(())
}
