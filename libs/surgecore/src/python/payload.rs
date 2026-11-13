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

//! The payload builder's bindings in the `surgecore._native` extension module, over numpy buffers.
//!
//! [`HostPayloadBuilder`] holds one batch preparer's static layout and its preallocated staging arrays (registered
//! once, at construction). Each `build` call borrows the step's input arrays and the staging arrays through numpy's
//! borrow checker, releases the GIL, and runs [`crate::payload::build`] over plain slices: no Python object is created
//! or touched per row or per token. The staging arrays belong to the caller's scheduler thread; nothing else may write
//! them while a build runs without the GIL.

use std::borrow::Cow;

use numpy::ndarray::Dimension;
use numpy::{
    AllowTypeChange, Element, PyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods, PyReadonlyArray,
    PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::payload::{self, BuildError, Config, Inputs, Mat, MatMut, Outputs, Scratch};

fn value_error(err: BuildError) -> PyErr {
    PyValueError::new_err(err.0)
}

/// The array's values in row-major order: borrowed when it is C-contiguous, copied otherwise.
fn contiguous<'a, T: Element + Copy, D: Dimension>(array: &'a PyReadonlyArray<'_, T, D>) -> Cow<'a, [T]> {
    if array.is_c_contiguous() {
        if let Ok(values) = array.as_slice() {
            return Cow::Borrowed(values);
        }
    }
    Cow::Owned(array.as_array().iter().copied().collect())
}

fn dims2<T: Element, D: Dimension>(array: &PyReadonlyArray<'_, T, D>) -> (usize, usize) {
    let shape = array.shape();
    (shape[0], shape[1])
}

fn mat_mut<'a, T: Element + Copy>(guard: &'a mut PyReadwriteArray2<'_, T>) -> PyResult<MatMut<'a, T>> {
    let (rows, cols) = (guard.shape()[0], guard.shape()[1]);
    MatMut::new(guard.as_slice_mut()?, rows, cols).map_err(value_error)
}

/// A writeable, C-contiguous staging array from the `buffers` mapping.
fn staging<'py, T: Element, D: Dimension>(buffers: &Bound<'py, PyDict>, name: &str) -> PyResult<Py<PyArray<T, D>>> {
    let Some(value) = buffers.get_item(name)? else {
        return Err(PyValueError::new_err(format!(
            "HostPayloadBuilder: missing staging buffer {name:?}"
        )));
    };
    let array = value.cast_into::<PyArray<T, D>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "HostPayloadBuilder: staging buffer {name:?} must be a numpy array of dtype {} with {} dimension(s)",
            std::any::type_name::<T>(),
            D::NDIM.map_or_else(|| "any".to_string(), |n| n.to_string()),
        ))
    })?;
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "HostPayloadBuilder: staging buffer {name:?} must be C-contiguous"
        )));
    }
    // A trial borrow rejects read-only arrays (and arrays already borrowed elsewhere) at construction.
    drop(array.try_readwrite()?);
    Ok(array.unbind())
}

struct Staging {
    scheduled: Py<PyArray1<i32>>,
    query_start_loc: Py<PyArray1<i32>>,
    input_ids: Py<PyArray1<i32>>,
    positions: Py<PyArray1<i32>>,
    seq_lens: Py<PyArray1<i32>>,
    logits_indices: Py<PyArray1<i32>>,
    pages_tables: Py<PyArray2<i32>>,
    packed_qsl_seqlens: Py<PyArray2<i32>>,
    packed_i32_padded: Py<PyArray2<i32>>,
    packed_f32_padded: Py<PyArray2<f32>>,
    packed_misc_i32: Py<PyArray1<i32>>,
    spec_recurrent_commit: Py<PyArray2<i32>>,
    request_distribution: Py<PyArray1<i32>>,
    dp_query_start_loc: Py<PyArray2<i32>>,
    dp_request_distribution: Py<PyArray2<i32>>,
    dp_context_lens: Py<PyArray2<i32>>,
    dp_recurrent_state_indices: Py<PyArray2<i32>>,
    dp_model_row_for_request: Py<PyArray1<i32>>,
    num_kv_update: Py<PyArray1<i32>>,
    slot_mapping: Option<Py<PyArray2<i32>>>,
}

/// Fills one batch preparer's staging arrays with the step-input payload of a forward window.
#[pyclass(module = "surgecore._native")]
pub struct HostPayloadBuilder {
    cfg: Config,
    staging: Staging,
    scratch: Scratch,
}

#[pymethods]
impl HostPayloadBuilder {
    #[new]
    #[pyo3(signature = (
        *,
        buffers,
        max_num_reqs,
        num_reqs_max_model_len,
        data_parallel_size,
        use_slot_mapping,
        use_request_distribution,
        plain_recurrent_slots,
        recurrent_slot_group_size,
        enable_spec_recurrent_commit,
        enable_dp_local_page_path,
        page_size,
        max_num_pages_per_req,
        slices_per_page,
        max_padded_slices,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        buffers: &Bound<'_, PyDict>,
        max_num_reqs: usize,
        num_reqs_max_model_len: usize,
        data_parallel_size: usize,
        use_slot_mapping: bool,
        use_request_distribution: bool,
        plain_recurrent_slots: bool,
        recurrent_slot_group_size: usize,
        enable_spec_recurrent_commit: bool,
        enable_dp_local_page_path: bool,
        page_size: i64,
        max_num_pages_per_req: usize,
        slices_per_page: i64,
        max_padded_slices: usize,
    ) -> PyResult<Self> {
        let cfg = Config {
            max_num_reqs,
            num_reqs_max_model_len,
            data_parallel_size: data_parallel_size.max(1),
            use_slot_mapping,
            use_request_distribution,
            plain_recurrent_slots,
            recurrent_slot_group_size: recurrent_slot_group_size.max(1),
            enable_spec_recurrent_commit,
            enable_dp_local_page_path,
            page_size,
            max_num_pages_per_req,
            slices_per_page,
            max_padded_slices,
        };
        let slot_mapping = if use_slot_mapping {
            Some(staging(buffers, "slot_mapping")?)
        } else {
            None
        };
        let staging = Staging {
            scheduled: staging(buffers, "scheduled")?,
            query_start_loc: staging(buffers, "query_start_loc")?,
            input_ids: staging(buffers, "input_ids")?,
            positions: staging(buffers, "positions")?,
            seq_lens: staging(buffers, "seq_lens")?,
            logits_indices: staging(buffers, "logits_indices")?,
            pages_tables: staging(buffers, "pages_tables")?,
            packed_qsl_seqlens: staging(buffers, "packed_qsl_seqlens")?,
            packed_i32_padded: staging(buffers, "packed_i32_padded")?,
            packed_f32_padded: staging(buffers, "packed_f32_padded")?,
            packed_misc_i32: staging(buffers, "packed_misc_i32")?,
            spec_recurrent_commit: staging(buffers, "spec_recurrent_commit")?,
            request_distribution: staging(buffers, "request_distribution")?,
            dp_query_start_loc: staging(buffers, "dp_query_start_loc")?,
            dp_request_distribution: staging(buffers, "dp_request_distribution")?,
            dp_context_lens: staging(buffers, "dp_context_lens")?,
            dp_recurrent_state_indices: staging(buffers, "dp_recurrent_state_indices")?,
            dp_model_row_for_request: staging(buffers, "dp_model_row_for_request")?,
            num_kv_update: staging(buffers, "num_kv_update")?,
            slot_mapping,
        };
        Ok(Self {
            cfg,
            staging,
            scratch: Scratch::default(),
        })
    }

    /// Fill the staging arrays for one window; returns
    /// `(num_requests, padded_num_reqs, rows_to_copy, reuse_pages_tables, total_pages)`.
    #[pyo3(signature = (
        num_tokens_static,
        scheduled_full,
        active_mask_full,
        token_ids,
        num_computed_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalties,
        presence_penalties,
        repetition_penalties,
        page_table,
        padded_num_reqs_in,
        window_row_indices=None,
        recurrent_slot_indices=None,
        spec_recurrent_commit=None,
        page_table_version=None,
        cached_page_table_version=None,
        cached_rows_to_copy=None,
        num_pages=0,
        metadata_data_parallel_size=1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn build<'py>(
        &mut self,
        py: Python<'py>,
        num_tokens_static: i64,
        scheduled_full: PyArrayLike1<'py, i32, AllowTypeChange>,
        active_mask_full: PyArrayLike1<'py, bool, AllowTypeChange>,
        token_ids: PyArrayLike2<'py, i32, AllowTypeChange>,
        num_computed_tokens: PyArrayLike1<'py, i32, AllowTypeChange>,
        temperature: PyArrayLike1<'py, f32, AllowTypeChange>,
        top_p: PyArrayLike1<'py, f32, AllowTypeChange>,
        top_k: PyArrayLike1<'py, i32, AllowTypeChange>,
        min_p: PyArrayLike1<'py, f32, AllowTypeChange>,
        frequency_penalties: PyArrayLike1<'py, f32, AllowTypeChange>,
        presence_penalties: PyArrayLike1<'py, f32, AllowTypeChange>,
        repetition_penalties: PyArrayLike1<'py, f32, AllowTypeChange>,
        page_table: PyArrayLike2<'py, i32, AllowTypeChange>,
        padded_num_reqs_in: i64,
        window_row_indices: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        recurrent_slot_indices: Option<PyArrayLike1<'py, i32, AllowTypeChange>>,
        spec_recurrent_commit: Option<PyArrayLike2<'py, i32, AllowTypeChange>>,
        page_table_version: Option<i64>,
        cached_page_table_version: Option<i64>,
        cached_rows_to_copy: Option<i64>,
        num_pages: i64,
        metadata_data_parallel_size: i64,
    ) -> PyResult<(usize, usize, usize, bool, i64)> {
        // Inputs: read once, borrowed in place when C-contiguous.
        let sched = contiguous(&scheduled_full);
        let mask = contiguous(&active_mask_full);
        let tokens = contiguous(&token_ids);
        let (token_rows, token_cols) = dims2(&token_ids);
        let computed = contiguous(&num_computed_tokens);
        let temperature = contiguous(&temperature);
        let top_p = contiguous(&top_p);
        let top_k = contiguous(&top_k);
        let min_p = contiguous(&min_p);
        let frequency = contiguous(&frequency_penalties);
        let presence = contiguous(&presence_penalties);
        let repetition = contiguous(&repetition_penalties);
        let pages = contiguous(&page_table);
        let (page_rows, page_cols) = dims2(&page_table);
        let window = window_row_indices.as_ref().map(|rows| contiguous(rows));
        let slots = recurrent_slot_indices.as_ref().map(|rows| contiguous(rows));
        let spec = spec_recurrent_commit
            .as_ref()
            .map(|commit| (contiguous(commit), dims2(commit)));
        let spec = match &spec {
            Some((values, (rows, cols))) => Some(Mat::new(values, *rows, *cols).map_err(value_error)?),
            None => None,
        };
        let inputs = Inputs {
            num_tokens_static,
            scheduled_full: &sched,
            active_mask_full: &mask,
            token_ids: Mat::new(&tokens, token_rows, token_cols).map_err(value_error)?,
            num_computed: &computed,
            temperature: &temperature,
            top_p: &top_p,
            top_k: &top_k,
            min_p: &min_p,
            frequency_penalties: &frequency,
            presence_penalties: &presence,
            repetition_penalties: &repetition,
            page_table: Mat::new(&pages, page_rows, page_cols).map_err(value_error)?,
            padded_num_reqs_in,
            window_row_indices: window.as_deref(),
            recurrent_slot_indices: slots.as_deref(),
            spec_recurrent_commit: spec,
            page_table_version,
            cached_page_table_version,
            cached_rows_to_copy,
            num_pages,
            metadata_data_parallel_size,
        };

        // Staging arrays: exclusive borrows for the duration of the build.
        let stage = &self.staging;
        let mut scheduled = stage.scheduled.bind(py).try_readwrite()?;
        let mut query_start_loc = stage.query_start_loc.bind(py).try_readwrite()?;
        let mut input_ids = stage.input_ids.bind(py).try_readwrite()?;
        let mut positions = stage.positions.bind(py).try_readwrite()?;
        let mut seq_lens = stage.seq_lens.bind(py).try_readwrite()?;
        let mut logits_indices = stage.logits_indices.bind(py).try_readwrite()?;
        let mut pages_tables = stage.pages_tables.bind(py).try_readwrite()?;
        let mut packed_qsl_seqlens = stage.packed_qsl_seqlens.bind(py).try_readwrite()?;
        let mut packed_i32_padded = stage.packed_i32_padded.bind(py).try_readwrite()?;
        let mut packed_f32_padded = stage.packed_f32_padded.bind(py).try_readwrite()?;
        let mut packed_misc_i32 = stage.packed_misc_i32.bind(py).try_readwrite()?;
        let mut spec_commit = stage.spec_recurrent_commit.bind(py).try_readwrite()?;
        let mut request_distribution = stage.request_distribution.bind(py).try_readwrite()?;
        let mut dp_query_start_loc = stage.dp_query_start_loc.bind(py).try_readwrite()?;
        let mut dp_request_distribution = stage.dp_request_distribution.bind(py).try_readwrite()?;
        let mut dp_context_lens = stage.dp_context_lens.bind(py).try_readwrite()?;
        let mut dp_recurrent_state_indices = stage.dp_recurrent_state_indices.bind(py).try_readwrite()?;
        let mut dp_model_row_for_request = stage.dp_model_row_for_request.bind(py).try_readwrite()?;
        let mut num_kv_update = stage.num_kv_update.bind(py).try_readwrite()?;
        let mut slot_mapping = match &stage.slot_mapping {
            Some(array) => Some(array.bind(py).try_readwrite()?),
            None => None,
        };
        let mut outputs = Outputs {
            scheduled: scheduled.as_slice_mut()?,
            query_start_loc: query_start_loc.as_slice_mut()?,
            input_ids: input_ids.as_slice_mut()?,
            positions: positions.as_slice_mut()?,
            seq_lens: seq_lens.as_slice_mut()?,
            logits_indices: logits_indices.as_slice_mut()?,
            pages_tables: mat_mut(&mut pages_tables)?,
            packed_qsl_seqlens: mat_mut(&mut packed_qsl_seqlens)?,
            packed_i32_padded: mat_mut(&mut packed_i32_padded)?,
            packed_f32_padded: mat_mut(&mut packed_f32_padded)?,
            packed_misc_i32: packed_misc_i32.as_slice_mut()?,
            spec_recurrent_commit: mat_mut(&mut spec_commit)?,
            request_distribution: request_distribution.as_slice_mut()?,
            dp_query_start_loc: mat_mut(&mut dp_query_start_loc)?,
            dp_request_distribution: mat_mut(&mut dp_request_distribution)?,
            dp_context_lens: mat_mut(&mut dp_context_lens)?,
            dp_recurrent_state_indices: mat_mut(&mut dp_recurrent_state_indices)?,
            dp_model_row_for_request: dp_model_row_for_request.as_slice_mut()?,
            num_kv_update: num_kv_update.as_slice_mut()?,
            slot_mapping: match slot_mapping.as_mut() {
                Some(guard) => Some(mat_mut(guard)?),
                None => None,
            },
        };

        let cfg = &self.cfg;
        let scratch = &mut self.scratch;
        let built = py
            .detach(|| payload::build(cfg, &inputs, &mut outputs, scratch))
            .map_err(value_error)?;
        Ok((
            built.num_requests,
            built.padded_num_reqs,
            built.rows_to_copy,
            built.reuse_pages_tables,
            built.total_pages,
        ))
    }
}

/// `(decode_end, prefill_end, num_requests)` of eSurge's positional request-distribution contract.
#[pyfunction]
fn request_distribution_prefix(
    scheduled: PyArrayLike1<'_, i64, AllowTypeChange>,
    num_computed_tokens: PyArrayLike1<'_, i64, AllowTypeChange>,
    num_requests: i64,
) -> PyResult<(i64, i64, i64)> {
    if num_requests <= 0 {
        return Ok((0, 0, 0));
    }
    let n = num_requests as usize;
    let sched = contiguous(&scheduled);
    let computed = contiguous(&num_computed_tokens);
    if sched.len() < n || computed.len() < n {
        return Err(PyValueError::new_err(format!(
            "request_distribution_prefix needs {n} rows, got scheduled={} num_computed_tokens={}",
            sched.len(),
            computed.len()
        )));
    }
    let (decode_end, prefill_end) = payload::distribution_prefix_by(n, |i| (sched[i], computed[i]));
    Ok((decode_end as i64, prefill_end as i64, num_requests))
}

/// Adds the payload builder and the request-distribution binding to the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HostPayloadBuilder>()?;
    m.add_function(wrap_pyfunction!(request_distribution_prefix, m)?)?;
    Ok(())
}
