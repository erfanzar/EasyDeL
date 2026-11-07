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

//! The engine core's bindings in the `surgecore._native` extension module: `EngineCore`, `Plan` and `Events`.
//!
//! Every call copies its numpy inputs into Rust vectors, releases the GIL, then takes the core's mutex, so a thread
//! blocked on the mutex never holds the GIL. `EngineCore` is a frozen class: a second thread waits on the mutex
//! instead of failing a borrow check. An internal check failure raises `InvariantError` and poisons the core (every
//! later call raises it again); `shutdown` closes it (later calls raise `CoreClosed`).

use std::sync::Mutex;

use numpy::{ndarray::Dimension, PyUntypedArrayMethods};
use numpy::{AllowTypeChange, Element, PyArray1, PyArrayLike1, PyArrayLike2, PyArrayMethods, PyReadonlyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::engine::plan::{
    Events, Plan, PLAN_ASYNC_SAFE, PLAN_HAS_PREFILL, PLAN_HAS_VERIFY, PLAN_PAGES_CHANGED, PLAN_STRUCTURED_PENDING,
};
use crate::engine::record;
use crate::engine::request::{FinishReason, NewRequest, Sampling};
use crate::engine::sched::commit::Output;
use crate::engine::sched::Policy;
use crate::engine::{CoreError, EngineConfig, EngineCore as Core};
use crate::kv::group::{Group, Kind};

pyo3::create_exception!(
    surgecore._native,
    InvariantError,
    PyRuntimeError,
    "An internal check of the engine core failed; the core is no longer usable."
);
pyo3::create_exception!(
    surgecore._native,
    CoreClosed,
    PyRuntimeError,
    "The engine core was shut down."
);

enum Fail {
    Value(String),
    Invariant(String),
    Closed,
}

fn to_py(f: Fail) -> PyErr {
    match f {
        Fail::Value(m) => PyValueError::new_err(m),
        Fail::Invariant(m) => InvariantError::new_err(m),
        Fail::Closed => CoreClosed::new_err("the engine core is shut down"),
    }
}

fn values<T: Element + Copy, D: Dimension>(array: &PyReadonlyArray<'_, T, D>) -> Vec<T> {
    array.as_array().iter().copied().collect()
}

fn opt1<T: Element + Copy>(array: &Option<PyArrayLike1<'_, T, AllowTypeChange>>) -> Option<Vec<T>> {
    array.as_ref().map(|a| values(a))
}

fn column<T: Copy>(name: &str, v: Option<Vec<T>>, n: usize, default: T) -> Result<Vec<T>, Fail> {
    match v {
        None => Ok(vec![default; n]),
        Some(v) if v.len() == n => Ok(v),
        Some(v) => Err(Fail::Value(format!("{name} has {} values for {n} requests", v.len()))),
    }
}

fn offsets(name: &str, v: &[i32], n: usize, total: usize) -> Result<(), Fail> {
    if v.len() != n + 1 || v[0] != 0 || v[n] as usize != total || v.windows(2).any(|w| w[1] < w[0]) {
        return Err(Fail::Value(format!(
            "{name} must hold {} non-decreasing offsets from 0 to {total}",
            n + 1
        )));
    }
    Ok(())
}

/// `(rank, status, num_prompt, num_tokens, computed, placeholders, cached_tokens)` of one request.
type RequestInfo = (i64, i64, i64, i64, i64, i64, i64);

struct State {
    core: Option<Core>,
    poisoned: Option<String>,
}

/// eSurge's host state machine: the request table, every data-parallel rank's queues, budgets, KV pages, prefix
/// cache and recurrent snapshots, and the scheduler over them.
#[pyclass(module = "surgecore._native", name = "EngineCore", frozen)]
pub struct EngineCore {
    state: Mutex<State>,
}

impl EngineCore {
    fn with<T: Send>(&self, py: Python<'_>, f: impl FnOnce(&mut Core) -> Result<T, CoreError> + Send) -> PyResult<T> {
        py.detach(|| {
            let mut st = self
                .state
                .lock()
                .map_err(|_| Fail::Invariant("the engine core failed inside a call and is poisoned".into()))?;
            if let Some(m) = &st.poisoned {
                return Err(Fail::Invariant(m.clone()));
            }
            let core = st.core.as_mut().ok_or(Fail::Closed)?;
            match f(core) {
                Ok(v) => Ok(v),
                Err(CoreError::Value(m)) => Err(Fail::Value(m)),
                Err(CoreError::Invariant(m)) => {
                    st.poisoned = Some(m.clone());
                    Err(Fail::Invariant(m))
                }
            }
        })
        .map_err(to_py)
    }
}

fn groups_of(rows: Vec<i32>, cols: usize) -> Result<Vec<Group>, Fail> {
    if cols != 3 && cols != 4 {
        return Err(Fail::Value(format!(
            "groups must be int32[G, 3] or int32[G, 4], got {cols} columns"
        )));
    }
    let mut out = Vec::new();
    let mut types: Vec<(i32, i32, i32)> = Vec::new();
    for d in rows.chunks_exact(cols) {
        let kind = match d[0] {
            0 => Kind::Full,
            1 => Kind::Sliding(i64::from(d[2])),
            2 => Kind::Chunked(i64::from(d[2])),
            k => return Err(Fail::Value(format!("unknown cache-group kind {k}"))),
        };
        let type_index = if cols == 4 {
            i64::from(d[3])
        } else {
            let t = (d[0], d[1], d[2]);
            match types.iter().position(|&x| x == t) {
                Some(i) => i as i64,
                None => {
                    types.push(t);
                    types.len() as i64 - 1
                }
            }
        };
        out.push(Group {
            kind,
            page_size: i64::from(d[1]),
            type_index,
        });
    }
    Ok(out)
}

#[pymethods]
impl EngineCore {
    /// `groups` holds one `(kind, page_size, window_or_chunk[, type_index])` row per cache group: kind 0 full
    /// attention, 1 sliding window, 2 chunked local attention. Without the type column, equal rows share a type.
    #[new]
    #[pyo3(signature = (
        *, max_rows, max_model_len, num_pages, dp_size, groups, token_budget, long_prefill_threshold=0,
        chunked_prefill=false, token_safety_margin=-1, policy=0, prefix_caching=true, snapshot_slots=0,
        async_scheduling=true, req_buckets=None, num_draft_tokens=0, use_eagle=false, include_finished_set=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        max_rows: usize,
        max_model_len: i64,
        num_pages: usize,
        dp_size: usize,
        groups: PyArrayLike2<'_, i32, AllowTypeChange>,
        token_budget: i64,
        long_prefill_threshold: i64,
        chunked_prefill: bool,
        token_safety_margin: i64,
        policy: i64,
        prefix_caching: bool,
        snapshot_slots: i64,
        async_scheduling: bool,
        req_buckets: Option<PyArrayLike1<'_, i64, AllowTypeChange>>,
        num_draft_tokens: i64,
        use_eagle: bool,
        include_finished_set: bool,
    ) -> PyResult<Self> {
        let cols = groups.shape()[1];
        let groups = groups_of(values(&groups), cols).map_err(to_py)?;
        let policy =
            Policy::from_code(policy).ok_or_else(|| PyValueError::new_err(format!("unknown policy {policy}")))?;
        let cfg = EngineConfig {
            max_rows,
            max_model_len,
            num_pages,
            dp_size,
            groups,
            token_budget,
            long_prefill_threshold,
            chunked_prefill,
            token_safety_margin: (token_safety_margin >= 0).then_some(token_safety_margin),
            policy,
            prefix_caching,
            snapshot_slots,
            async_scheduling,
            req_buckets: opt1(&req_buckets).unwrap_or_default(),
            num_spec_tokens: num_draft_tokens,
            use_eagle,
            include_finished_set,
        };
        let core = Core::new(&cfg).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            state: Mutex::new(State {
                core: Some(core),
                poisoned: None,
            }),
        })
    }

    /// Admits requests (atomic: a bad request raises `ValueError` and nothing is admitted); returns their handles.
    #[pyo3(signature = (
        *, prompt_tokens, prompt_offsets, max_tokens, min_tokens=None, eos_token=None, priority=None, arrival=None,
        ignore_eos=None, flags=None, stop_tokens=None, stop_offsets=None, temperature=None, top_p=None, min_p=None,
        frequency_penalty=None, presence_penalty=None, repetition_penalty=None, top_k=None, dp_rank=None, seed=None,
        hash_extra=None, image_spans=None, span_offsets=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_requests<'py>(
        &self,
        py: Python<'py>,
        prompt_tokens: PyArrayLike1<'py, i32, AllowTypeChange>,
        prompt_offsets: PyArrayLike1<'py, i32, AllowTypeChange>,
        max_tokens: PyArrayLike1<'py, i64, AllowTypeChange>,
        min_tokens: Option<PyArrayLike1<'py, i64, AllowTypeChange>>,
        eos_token: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        priority: Option<PyArrayLike1<'py, i64, AllowTypeChange>>,
        arrival: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
        ignore_eos: Option<PyArrayLike1<'py, bool, AllowTypeChange>>,
        flags: Option<PyArrayLike1<'py, u32, AllowTypeChange>>,
        stop_tokens: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        stop_offsets: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        temperature: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        top_p: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        min_p: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        frequency_penalty: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        presence_penalty: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        repetition_penalty: Option<PyArrayLike1<'py, f32, AllowTypeChange>>,
        top_k: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        dp_rank: Option<PyArrayLike1<'py, i64, AllowTypeChange>>,
        seed: Option<PyArrayLike1<'py, i64, AllowTypeChange>>,
        hash_extra: Option<PyArrayLike2<'py, u64, AllowTypeChange>>,
        image_spans: Option<PyArrayLike2<'py, i64, AllowTypeChange>>,
        span_offsets: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let tokens = values(&prompt_tokens);
        let poff = values(&prompt_offsets);
        let n = poff.len().saturating_sub(1);
        let build = || -> Result<_, Fail> {
            offsets("prompt_offsets", &poff, n, tokens.len())?;
            let max_tokens = column("max_tokens", Some(values(&max_tokens)), n, 0)?;
            let min_tokens = column("min_tokens", opt1(&min_tokens), n, 0)?;
            let eos = column("eos_token", opt1(&eos_token), n, -1)?;
            let priority = column("priority", opt1(&priority), n, 0)?;
            let arrival = column("arrival", opt1(&arrival), n, 0.0)?;
            let ignore_eos = column("ignore_eos", opt1(&ignore_eos), n, false)?;
            let flags = column("flags", opt1(&flags), n, 0)?;
            let stops = opt1(&stop_tokens).unwrap_or_default();
            let soff = column("stop_offsets", opt1(&stop_offsets), n + 1, 0)?;
            offsets("stop_offsets", &soff, n, stops.len())?;
            let d = Sampling::default();
            let temperature = column("temperature", opt1(&temperature), n, d.temperature)?;
            let top_p = column("top_p", opt1(&top_p), n, d.top_p)?;
            let min_p = column("min_p", opt1(&min_p), n, d.min_p)?;
            let freq = column("frequency_penalty", opt1(&frequency_penalty), n, d.frequency_penalty)?;
            let pres = column("presence_penalty", opt1(&presence_penalty), n, d.presence_penalty)?;
            let rep = column("repetition_penalty", opt1(&repetition_penalty), n, d.repetition_penalty)?;
            let top_k = column("top_k", opt1(&top_k), n, d.top_k)?;
            let dp_rank = column("dp_rank", opt1(&dp_rank), n, -1)?;
            let seed = column("seed", opt1(&seed), n, -1)?;
            let extra: Vec<u128> = match &hash_extra {
                None => vec![0; n],
                Some(a) => {
                    let v = values(a);
                    if a.shape() != [n, 2] {
                        return Err(Fail::Value(format!("hash_extra must be uint64[{n}, 2]")));
                    }
                    v.chunks_exact(2)
                        .map(|c| (u128::from(c[0]) << 64) | u128::from(c[1]))
                        .collect()
                }
            };
            let spans: Vec<(i64, i64)> = match &image_spans {
                None => Vec::new(),
                Some(a) => {
                    if a.shape()[1] != 2 {
                        return Err(Fail::Value("image_spans must be int64[I, 2]".into()));
                    }
                    values(a).chunks_exact(2).map(|c| (c[0], c[1])).collect()
                }
            };
            let spoff = column("span_offsets", opt1(&span_offsets), n + 1, 0)?;
            offsets("span_offsets", &spoff, n, spans.len())?;
            Ok((
                max_tokens,
                min_tokens,
                eos,
                priority,
                arrival,
                ignore_eos,
                flags,
                stops,
                soff,
                temperature,
                top_p,
                min_p,
                freq,
                pres,
                rep,
                top_k,
                dp_rank,
                seed,
                extra,
                spans,
                spoff,
            ))
        };
        let (
            max_tokens,
            min_tokens,
            eos,
            priority,
            arrival,
            ignore_eos,
            flags,
            stops,
            soff,
            temperature,
            top_p,
            min_p,
            freq,
            pres,
            rep,
            top_k,
            dp_rank,
            seed,
            extra,
            spans,
            spoff,
        ) = build().map_err(to_py)?;
        let handles = self.with(py, |core| {
            let reqs: Vec<NewRequest<'_>> = (0..n)
                .map(|i| NewRequest {
                    prompt: &tokens[poff[i] as usize..poff[i + 1] as usize],
                    max_tokens: max_tokens[i],
                    min_tokens: min_tokens[i],
                    eos: (eos[i] >= 0).then_some(eos[i]),
                    ignore_eos: ignore_eos[i],
                    priority: priority[i],
                    arrival: arrival[i],
                    flags: flags[i],
                    stop_ids: &stops[soff[i] as usize..soff[i + 1] as usize],
                    spans: &spans[spoff[i] as usize..spoff[i + 1] as usize],
                    dp_rank: (dp_rank[i] >= 0).then_some(dp_rank[i] as usize),
                    hash_extra: extra[i],
                    sampling: Sampling {
                        temperature: temperature[i],
                        top_p: top_p[i],
                        top_k: top_k[i],
                        min_p: min_p[i],
                        frequency_penalty: freq[i],
                        presence_penalty: pres[i],
                        repetition_penalty: rep[i],
                        seed: seed[i],
                    },
                })
                .collect();
            core.add(&reqs)
        })?;
        Ok(PyArray1::from_vec(py, handles.into_iter().map(|h| h as i64).collect()))
    }

    /// Schedules one step over every rank.
    fn schedule(&self, py: Python<'_>) -> PyResult<PyPlan> {
        let inner = self.with(py, |core| core.schedule())?;
        Ok(PyPlan { inner })
    }

    /// Commits a plan's runner output, per plan entry in plan order; plans with entries commit in dispatch order.
    /// `sampled_tokens` with `sampled_offsets` hold each entry's sampled list (accepted drafts then the bonus
    /// token); `present` (0 absent, 1 present, 2 listed without a sampled row), `accepted` (-1 absent) and
    /// `new_drafts` with `new_draft_offsets` (None: the runner returned no drafts) are optional.
    #[pyo3(signature = (
        plan, *, sampled_tokens=None, sampled_offsets=None, accepted=None, new_drafts=None, new_draft_offsets=None,
        present=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &self,
        py: Python<'_>,
        plan: &Bound<'_, PyPlan>,
        sampled_tokens: Option<PyArrayLike1<'_, i32, AllowTypeChange>>,
        sampled_offsets: Option<PyArrayLike1<'_, i32, AllowTypeChange>>,
        accepted: Option<PyArrayLike1<'_, i32, AllowTypeChange>>,
        new_drafts: Option<PyArrayLike1<'_, i32, AllowTypeChange>>,
        new_draft_offsets: Option<PyArrayLike1<'_, i32, AllowTypeChange>>,
        present: Option<PyArrayLike1<'_, u8, AllowTypeChange>>,
    ) -> PyResult<PyEvents> {
        let p: &Plan = &plan.get().inner;
        let (flat, offs) = match (opt1(&sampled_tokens), opt1(&sampled_offsets)) {
            (Some(t), Some(o)) => (t, o),
            (None, None) => (Vec::new(), vec![0; p.len() + 1]),
            _ => return Err(PyValueError::new_err("sampled_tokens and sampled_offsets go together")),
        };
        let present = opt1(&present);
        let accepted = opt1(&accepted);
        let drafts = match (opt1(&new_drafts), opt1(&new_draft_offsets)) {
            (Some(d), Some(o)) => Some((d, o)),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("new_drafts and new_draft_offsets go together")),
        };
        let inner = self.with(py, |core| {
            let out = Output {
                sampled: &flat,
                sampled_offsets: &offs,
                present: present.as_deref(),
                accepted: accepted.as_deref(),
                drafts: drafts.as_ref().map(|(d, o)| (d.as_slice(), o.as_slice())),
            };
            core.commit(p, &out)
        })?;
        Ok(PyEvents { inner })
    }

    /// Finishes live requests (`reason` 1 stop, 2 length, 3 abort); returns the handles that were live.
    fn finish<'py>(
        &self,
        py: Python<'py>,
        handles: PyArrayLike1<'py, i64, AllowTypeChange>,
        reason: i64,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let reason =
            FinishReason::from_code(reason).ok_or_else(|| PyValueError::new_err(format!("unknown reason {reason}")))?;
        let hs: Vec<u64> = values(&handles).into_iter().map(|h| h as u64).collect();
        let live = self.with(py, |core| core.finish(&hs, reason))?;
        Ok(PyArray1::from_vec(py, live.into_iter().map(|h| h as i64).collect()))
    }

    /// Forgets every published page and snapshot; `False` while pages are in use.
    fn reset_prefix_cache(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |core| Ok(core.reset_prefix_cache()))
    }

    /// Running, waiting, free pages and pending prefill tokens, then the same four per rank.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let c = self.with(py, |core| Ok(core.counts()))?;
        Ok(PyArray1::from_vec(py, c))
    }

    /// The plan as a step record (bytes).
    fn record<'py>(&self, py: Python<'py>, plan: &Bound<'py, PyPlan>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &record::encode(&plan.get().inner))
    }

    /// A hash of tokens, counters and global page ids (never page keys).
    fn digest(&self, py: Python<'_>) -> PyResult<u64> {
        self.with(py, |core| Ok(core.digest()))
    }

    /// Checks the invariants; raises `InvariantError` (and poisons the core) when one fails.
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |core| core.check())
    }

    /// Plans scheduled with entries and not yet committed.
    fn num_inflight(&self, py: Python<'_>) -> PyResult<usize> {
        self.with(py, |core| Ok(core.num_inflight()))
    }

    /// The data-parallel rank of a live request, or `None`.
    fn rank_of(&self, py: Python<'_>, handle: i64) -> PyResult<Option<usize>> {
        self.with(py, |core| Ok(core.rank_of(handle as u64)))
    }

    /// `(rank, status, num_prompt, num_tokens, computed, placeholders, cached_tokens)` of a live request (status 0
    /// waiting, 1 running, 2 preempted), or `None`.
    fn request_info(&self, py: Python<'_>, handle: i64) -> PyResult<Option<RequestInfo>> {
        self.with(py, |core| {
            Ok(core
                .request_info(handle as u64)
                .map(|v| (v[0], v[1], v[2], v[3], v[4], v[5], v[6])))
        })
    }

    /// Turns on the load-aware verification budget of synchronous speculative steps: `buckets` (ascending) are the
    /// compiled token buckets of the model step and `step_ms` their measured times. `prior` starts every draft
    /// position's acceptance estimate, `alpha` weighs each observation, `correction` each measured step time.
    #[pyo3(signature = (buckets, step_ms, *, prior=0.7, alpha=0.2, correction=0.1))]
    fn set_verify_budget(
        &self,
        py: Python<'_>,
        buckets: PyArrayLike1<'_, i64, AllowTypeChange>,
        step_ms: PyArrayLike1<'_, f64, AllowTypeChange>,
        prior: f64,
        alpha: f64,
        correction: f64,
    ) -> PyResult<()> {
        let (b, t) = (values(&buckets), values(&step_ms));
        self.with(py, |core| core.set_verify_budget(&b, &t, prior, alpha, correction))
    }

    /// The verification budget's global acceptance prior per draft position and its step-time correction (ms), or
    /// `None` without a budget.
    fn verify_state(&self, py: Python<'_>) -> PyResult<Option<(Vec<f64>, f64)>> {
        self.with(py, |core| {
            Ok(core.verify_budget().map(|v| (v.prior().to_vec(), v.offset())))
        })
    }

    /// The speculative drafts a live request carries into its next step, or `None`.
    fn drafts_of<'py>(&self, py: Python<'py>, handle: i64) -> PyResult<Option<Bound<'py, PyArray1<i32>>>> {
        let d = self.with(py, |core| Ok(core.drafts_of(handle as u64).map(<[i32]>::to_vec)))?;
        Ok(d.map(|v| PyArray1::from_vec(py, v)))
    }

    /// Closes the core; later calls raise `CoreClosed`.
    fn shutdown(&self, py: Python<'_>) {
        py.detach(|| {
            if let Ok(mut st) = self.state.lock() {
                st.core = None;
            }
        });
    }
}

fn arr<'py, T: Element + Copy>(py: Python<'py>, v: &[T]) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_slice(py, v)
}

fn handles<'py>(py: Python<'py>, v: &[u64]) -> Bound<'py, PyArray1<i64>> {
    PyArray1::from_vec(py, v.iter().map(|&h| h as i64).collect())
}

fn pairs<'py>(py: Python<'py>, v: &[[i32; 2]]) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
    PyArray1::from_vec(py, v.iter().flatten().copied().collect()).reshape([v.len(), 2])
}

/// One step's scheduling decisions, in decision order (rank-major); arrays are copies.
#[pyclass(module = "surgecore._native", name = "Plan", frozen)]
pub struct PyPlan {
    inner: Plan,
}

#[pymethods]
impl PyPlan {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn flags(&self) -> u32 {
        self.inner.flags
    }
    #[getter]
    fn handles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        handles(py, &self.inner.handles)
    }
    #[getter]
    fn scheduled<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.scheduled)
    }
    #[getter]
    fn computed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.computed)
    }
    #[getter]
    fn kinds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        arr(py, &self.inner.kinds)
    }
    #[getter]
    fn emits<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        arr(py, &self.inner.emits)
    }
    #[getter]
    fn new_pages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.new_pages)
    }
    #[getter]
    fn new_page_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.new_page_offsets)
    }
    #[getter]
    fn rank_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.rank_offsets)
    }
    #[getter]
    fn preempted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        handles(py, &self.inner.preempted)
    }
    #[getter]
    fn finished<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        handles(py, &self.inner.finished)
    }
    #[getter]
    fn snapshot_restores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
        pairs(py, &self.inner.snapshot_restores)
    }
    #[getter]
    fn snapshot_saves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
        pairs(py, &self.inner.snapshot_saves)
    }
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        arr(py, &self.inner.stats)
    }
    #[getter]
    fn draft_tokens<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.draft_tokens)
    }
    #[getter]
    fn draft_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.draft_offsets)
    }
    #[getter]
    fn new_tokens<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.new_tokens)
    }
    #[getter]
    fn new_token_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.new_token_offsets)
    }
    #[getter]
    fn common_prefix_pages<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.common_prefix_pages)
    }
    #[getter]
    fn req_bucket(&self) -> i64 {
        self.inner.req_bucket
    }
    #[getter]
    fn num_draft_tokens_next(&self) -> i64 {
        self.inner.num_draft_tokens_next
    }
    #[getter]
    fn spec_slots<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.spec_slots)
    }
}

/// A commit's outcome in plan order: one entry per request that received a token.
#[pyclass(module = "surgecore._native", name = "Events", frozen)]
pub struct PyEvents {
    inner: Events,
}

#[pymethods]
impl PyEvents {
    #[getter]
    fn n(&self) -> usize {
        self.inner.len()
    }
    #[getter]
    fn handles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        handles(py, &self.inner.handles)
    }
    #[getter]
    fn token_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.token_offsets)
    }
    #[getter]
    fn tokens<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.tokens)
    }
    #[getter]
    fn logprob_starts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.logprob_starts)
    }
    #[getter]
    fn finish<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        arr(py, &self.inner.finish)
    }
    #[getter]
    fn stop_token<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.stop_token)
    }
    #[getter]
    fn cached_tokens<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        arr(py, &self.inner.cached_tokens)
    }
    #[getter]
    fn finished_all<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        handles(py, &self.inner.finished_all)
    }
}

/// Adds the engine core's classes and exceptions to the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EngineCore>()?;
    m.add_class::<PyPlan>()?;
    m.add_class::<PyEvents>()?;
    m.add("InvariantError", m.py().get_type::<InvariantError>())?;
    m.add("CoreClosed", m.py().get_type::<CoreClosed>())?;
    m.add("PLAN_ASYNC_SAFE", PLAN_ASYNC_SAFE)?;
    m.add("PLAN_HAS_PREFILL", PLAN_HAS_PREFILL)?;
    m.add("PLAN_HAS_VERIFY", PLAN_HAS_VERIFY)?;
    m.add("PLAN_PAGES_CHANGED", PLAN_PAGES_CHANGED)?;
    m.add("PLAN_STRUCTURED_PENDING", PLAN_STRUCTURED_PENDING)?;
    Ok(())
}
