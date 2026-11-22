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

//! The step-input payload of one eSurge forward window.
//!
//! This is eSurge's only implementation of the per-step host payload computation. Given the scheduler's per-row
//! token counts and the runner's row-backed request state, [`build`] fills the preallocated buffers that the batch
//! preparer transfers to the device:
//!
//! * the packed token stream (`input_ids`, `positions`), in request order or rank-major under data parallelism;
//! * query start locations, sequence lengths and logits indices;
//! * the page-table rows of the window, and either the v2 KV slot mapping or the v3 request distribution;
//! * the packed sampling vectors, the rank-major DP metadata and the recurrent state slots.
//!
//! It works on borrowed slices only, allocates nothing beyond the reusable [`Scratch`], never calls into Python and
//! reports every inconsistent input as a [`BuildError`] instead of panicking.

use std::fmt;

/// Page-table padding value (eSurge's `PAGE_TABLE_PADDING_VAL`).
pub const PAGE_TABLE_PADDING_VAL: i32 = 0;
/// Slot-mapping padding value (eSurge's `SLOT_MAPPING_PADDING_VAL`).
pub const SLOT_MAPPING_PADDING_VAL: i32 = 0;

/// An input the payload cannot be built from; the message is the one eSurge reports to its caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildError(pub String);

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BuildError {}

/// Result of the payload computation.
pub type Result<T> = std::result::Result<T, BuildError>;

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(BuildError(format!($($arg)*)))
    };
}

/// Read-only row-major (C order) matrix over a borrowed slice.
#[derive(Debug, Clone, Copy)]
pub struct Mat<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> Mat<'a, T> {
    /// Wrap `data` as a `rows x cols` matrix.
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            bail!(
                "a {rows}x{cols} matrix needs {} values, got {}",
                rows.saturating_mul(cols),
                data.len()
            );
        }
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row `r` (callers check `r < rows`).
    pub fn row(&self, r: usize) -> &'a [T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// Mutable row-major (C order) matrix over a borrowed slice.
#[derive(Debug)]
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: Copy> MatMut<'a, T> {
    /// Wrap `data` as a `rows x cols` matrix.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            bail!(
                "a {rows}x{cols} matrix needs {} values, got {}",
                rows.saturating_mul(cols),
                data.len()
            );
        }
        Ok(Self { data, rows, cols })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row `r` (callers check `r < rows`).
    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Mutable row `r` (callers check `r < rows`).
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Set every element to `value`.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

/// Static layout of one batch preparer: fixed for the lifetime of its buffers.
#[derive(Debug, Clone)]
pub struct Config {
    /// Request rows of a scheduler window (`max_num_reqs`): the length of every per-request buffer.
    pub max_num_reqs: usize,
    /// Rows of one forward (`min(metadata.get_max_num_seqs(), max_num_reqs)`): page-table rows and DP row capacity.
    pub num_reqs_max_model_len: usize,
    /// Data-parallel size of the cache layout; rank-major packing when greater than one.
    pub data_parallel_size: usize,
    /// Ragged-page attention v2: build the KV slot mapping.
    pub use_slot_mapping: bool,
    /// Ragged-page attention v3 (and MLA): build the positional request distribution.
    pub use_request_distribution: bool,
    /// Ship per-row physical recurrent state slots outside rank-major DP.
    pub plain_recurrent_slots: bool,
    /// Recurrent slots per request, `W` (at least 1); a rank's slot block holds `rows_per_rank * W` slots.
    pub recurrent_slot_group_size: usize,
    /// Copy the speculative recurrent commit metadata into its payload slot.
    pub enable_spec_recurrent_commit: bool,
    /// Check that active rows only reference page ids local to their DP shard.
    pub enable_dp_local_page_path: bool,
    /// Tokens per KV page.
    pub page_size: i64,
    /// Row stride of the page table in the v2 slot-mapping gather (`metadata.max_num_pages_per_req`).
    pub max_num_pages_per_req: usize,
    /// v2 slices per KV-cache update page (slot counts are padded to a multiple of it).
    pub slices_per_page: i64,
    /// v2 slot-mapping capacity (columns of the slot-mapping buffer that may be written).
    pub max_padded_slices: usize,
}

/// One step's inputs: the scheduler window and the runner's row state, borrowed from numpy arrays.
#[derive(Debug, Clone, Copy)]
pub struct Inputs<'a> {
    /// Token bucket of the step (length of the shipped `input_ids` / `positions`).
    pub num_tokens_static: i64,
    /// Scheduled tokens per window row.
    pub scheduled_full: &'a [i32],
    /// Live rows of the window.
    pub active_mask_full: &'a [bool],
    /// Token history per window row, `[rows, max_model_len]`.
    pub token_ids: Mat<'a, i32>,
    /// Tokens already computed per window row.
    pub num_computed: &'a [i32],
    /// Sampling temperature per row.
    pub temperature: &'a [f32],
    /// Top-p per row.
    pub top_p: &'a [f32],
    /// Top-k per row.
    pub top_k: &'a [i32],
    /// Min-p per row.
    pub min_p: &'a [f32],
    /// Frequency penalty per row.
    pub frequency_penalties: &'a [f32],
    /// Presence penalty per row.
    pub presence_penalties: &'a [f32],
    /// Repetition penalty per row.
    pub repetition_penalties: &'a [f32],
    /// Page ids per window row, `[rows, pages]`.
    pub page_table: Mat<'a, i32>,
    /// Requested padded request count (compile bucket).
    pub padded_num_reqs_in: i64,
    /// Global row of each window row (rank-major DP, DP-local page checks).
    pub window_row_indices: Option<&'a [i32]>,
    /// Physical recurrent state slot of each window row.
    pub recurrent_slot_indices: Option<&'a [i32]>,
    /// Speculative recurrent commit metadata, `[2, rows]`.
    pub spec_recurrent_commit: Option<Mat<'a, i32>>,
    /// Page-table version of this window, `None` when unversioned.
    pub page_table_version: Option<i64>,
    /// Version of the device page table the caller holds, `None` when it holds none.
    pub cached_page_table_version: Option<i64>,
    /// Rows the cached device page table was built with.
    pub cached_rows_to_copy: Option<i64>,
    /// Pages of the KV cache (DP-local page checks).
    pub num_pages: i64,
    /// Data-parallel size of the cache metadata (DP-local page checks).
    pub metadata_data_parallel_size: i64,
}

/// The preallocated buffers one step fills (the batch preparer's staging arrays).
#[derive(Debug)]
pub struct Outputs<'a> {
    /// `[max_num_reqs]` scheduled tokens (prefix of live rows, zero elsewhere).
    pub scheduled: &'a mut [i32],
    /// `[max_num_reqs + 1]` query start locations.
    pub query_start_loc: &'a mut [i32],
    /// `[max_num_tokens]` token stream; the first `num_tokens_static` values are written.
    pub input_ids: &'a mut [i32],
    /// `[max_num_tokens]` positions, aligned with `input_ids`.
    pub positions: &'a mut [i32],
    /// `[max_num_reqs]` sequence lengths after this step.
    pub seq_lens: &'a mut [i32],
    /// `[max_num_reqs]` index of each row's last token in the stream.
    pub logits_indices: &'a mut [i32],
    /// `[num_reqs_max_model_len, pages]` page table of the forward.
    pub pages_tables: MatMut<'a, i32>,
    /// `[2, max_num_reqs + 1]` query start locations and sequence lengths.
    pub packed_qsl_seqlens: MatMut<'a, i32>,
    /// `[3, max_num_reqs]` scheduled, logits indices, top-k.
    pub packed_i32_padded: MatMut<'a, i32>,
    /// `[6, max_num_reqs]` temperature, top-p, min-p, frequency, presence and repetition penalties.
    pub packed_f32_padded: MatMut<'a, f32>,
    /// `[5]` num_requests, padded_num_reqs, request distribution.
    pub packed_misc_i32: &'a mut [i32],
    /// `[2, max_num_reqs]` speculative recurrent commit metadata.
    pub spec_recurrent_commit: MatMut<'a, i32>,
    /// `[3]` request distribution of the window.
    pub request_distribution: &'a mut [i32],
    /// `[dp, rows_per_rank + 1]` rank-local query start locations.
    pub dp_query_start_loc: MatMut<'a, i32>,
    /// `[dp, 3]` rank-local request distributions.
    pub dp_request_distribution: MatMut<'a, i32>,
    /// `[dp, rows_per_rank]` rank-local context lengths.
    pub dp_context_lens: MatMut<'a, i32>,
    /// `[dp, rows_per_rank]` rank-local recurrent state slots (`[1, rows]` outside DP).
    pub dp_recurrent_state_indices: MatMut<'a, i32>,
    /// `[max_num_reqs]` model row of each request under rank-major DP (-1 when unscheduled).
    pub dp_model_row_for_request: &'a mut [i32],
    /// `[1]` KV-cache update slices (pages touched).
    pub num_kv_update: &'a mut [i32],
    /// `[3, max_padded_slices]` v2 KV slot mapping.
    pub slot_mapping: Option<MatMut<'a, i32>>,
}

/// What one build decided, for the caller's bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Built {
    /// Highest live row plus one.
    pub num_requests: usize,
    /// Padded request count of the compile bucket.
    pub padded_num_reqs: usize,
    /// Page-table rows of the forward.
    pub rows_to_copy: usize,
    /// The caller's cached device page table is still valid; the host page table was left untouched.
    pub reuse_pages_tables: bool,
    /// KV pages touched by the step (v2 slot mapping), 0 otherwise.
    pub total_pages: i64,
}

/// Reusable per-builder scratch (no allocation in steady state).
#[derive(Debug, Default)]
pub struct Scratch {
    rank_used_tokens: Vec<i64>,
    rank_live_rows: Vec<usize>,
    first_page: Vec<i64>,
    page_lens: Vec<i64>,
    page_cum: Vec<i64>,
    token_offsets: Vec<i64>,
}

fn head<'s, T>(values: &'s [T], n: usize, name: &str) -> Result<&'s [T]> {
    match values.get(..n) {
        Some(prefix) => Ok(prefix),
        None => bail!("{name} holds {} values but {n} are needed", values.len()),
    }
}

/// Padded request count: the requested bucket, at least the live rows, at most `max_num_reqs`.
pub fn padded_num_reqs(num_requests: usize, max_num_reqs: usize, padded_num_reqs_in: i64) -> usize {
    let max_reqs = (max_num_reqs as i64).max(1);
    let live = (num_requests as i64).max(1);
    padded_num_reqs_in.max(live).min(max_reqs) as usize
}

/// `(decode_end, prefill_end)` of eSurge's positional request-distribution contract over `n` rows.
///
/// `row(i)` gives `(scheduled, computed)` of row `i`. Rows `[0, decode_end)` are the longest leading run of decodes
/// (one scheduled token after at least one computed token), rows `[decode_end, prefill_end)` the following run of
/// active non-decode rows; everything after belongs to the order-agnostic mixed range.
pub fn distribution_prefix_by<F: Fn(usize) -> (i64, i64)>(n: usize, row: F) -> (usize, usize) {
    let kind = |i: usize| {
        let (scheduled, computed) = row(i);
        if scheduled <= 0 {
            0
        } else if scheduled == 1 && computed > 0 {
            1
        } else {
            2
        }
    };
    let mut decode_end = 0;
    while decode_end < n && kind(decode_end) == 1 {
        decode_end += 1;
    }
    let mut prefill_end = decode_end;
    while prefill_end < n && kind(prefill_end) == 2 {
        prefill_end += 1;
    }
    (decode_end, prefill_end)
}

/// [`distribution_prefix_by`] over the first `min(len)` rows of two count vectors.
pub fn request_distribution_prefix(scheduled: &[i64], computed: &[i64]) -> (usize, usize) {
    let n = scheduled.len().min(computed.len());
    distribution_prefix_by(n, |i| (scheduled[i], computed[i]))
}

/// Copy one page-table row with numpy assignment semantics (`dst[:] = src[:len(dst)]`, a one-page row broadcasts).
fn copy_page_row(dst: &mut [i32], src: &[i32]) -> Result<()> {
    let used = src.len().min(dst.len());
    if used == dst.len() {
        dst.copy_from_slice(&src[..used]);
    } else if used == 1 {
        dst.fill(src[0]);
    } else {
        bail!(
            "could not broadcast input array from shape ({used},) into shape ({},)",
            dst.len()
        );
    }
    Ok(())
}

/// Write `n` tokens of `row` starting at `start` into the stream at `off`, with their positions.
fn write_run(ids: &mut [i32], pos: &mut [i32], row: &[i32], off: i64, start: i64, n: i64) -> Result<()> {
    let (Ok(off), Ok(begin), Ok(count)) = (usize::try_from(off), usize::try_from(start), usize::try_from(n)) else {
        bail!("negative token offset, start or count (offset={off} start={start} count={n})");
    };
    let (Some(dst), Some(src)) = (ids.get_mut(off..off + count), row.get(begin..begin + count)) else {
        bail!(
            "could not pack {count} tokens at stream offset {off} from position {begin}: the token bucket holds {} and \
             the row {}",
            ids.len(),
            row.len()
        );
    };
    dst.copy_from_slice(src);
    for (j, p) in pos[off..off + count].iter_mut().enumerate() {
        *p = (start + j as i64) as i32;
    }
    Ok(())
}

/// Reject rows whose pages lie outside their DP shard (when the DP-local page path is on and pages split evenly).
fn check_dp_local_pages(cfg: &Config, inp: &Inputs<'_>, sched: &[i32], computed: &[i32], use_dp: bool) -> Result<()> {
    if !cfg.enable_dp_local_page_path {
        return Ok(());
    }
    let dp = inp.metadata_data_parallel_size.max(1);
    if dp <= 1 || sched.is_empty() {
        return Ok(());
    }
    let total_rows = cfg.num_reqs_max_model_len as i64;
    if total_rows % dp != 0 {
        bail!(
            "DP-local page-table invariant requires rows divisible by data-parallel size: rows={total_rows}, \
             dp_size={dp}."
        );
    }
    if inp.num_pages <= 0 || inp.num_pages % dp != 0 {
        // Pages that do not split evenly across shards have no DP-local partition to check against.
        return Ok(());
    }
    let pages_per_shard = inp.num_pages / dp;
    let rows_per_shard = total_rows / dp;
    let page_size = cfg.page_size.max(1);
    let max_pages = inp.page_table.cols() as i64;
    let window_rows = if use_dp { inp.window_row_indices } else { None };
    for (req, (&scheduled, &done)) in sched.iter().zip(computed).enumerate() {
        let seq_len = i64::from(done) + i64::from(scheduled);
        if seq_len <= 0 {
            continue;
        }
        let global_row = match window_rows {
            Some(rows) => i64::from(*head(rows, req + 1, "window_row_indices")?.last().unwrap_or(&0)),
            None => req as i64,
        };
        let shard = (global_row.max(0) / rows_per_shard).min(dp - 1);
        let lo = if shard == 0 { 1 } else { shard * pages_per_shard };
        let hi = shard * pages_per_shard + pages_per_shard;
        let page_cnt = ((seq_len + page_size - 1) / page_size).min(max_pages) as usize;
        if req >= inp.page_table.rows() {
            bail!(
                "page_table holds {} rows but row {req} is active",
                inp.page_table.rows()
            );
        }
        for &page in &inp.page_table.row(req)[..page_cnt] {
            if page == PAGE_TABLE_PADDING_VAL {
                continue;
            }
            let page = i64::from(page);
            if page < lo || page >= hi {
                bail!(
                    "Non-DP-local page IDs detected for active request row. req_idx={req} global_row={global_row} \
                     req_shard={shard} valid_range=[{lo}, {hi}) sample_bad_page={page}. Ensure scheduler/allocator \
                     produce per-DP-local block IDs (or run with data-parallel page sharding disabled)."
                );
            }
        }
    }
    Ok(())
}

/// Where each request's new tokens start in the packed stream (the v2 slot-mapping sources).
#[derive(Debug, Clone, Copy)]
enum TokenOffsets<'a> {
    /// Contiguous request-order packing: the exclusive prefix sum of the scheduled counts.
    Packed,
    /// Rank-major packing: `logits_index - scheduled + 1` of each request.
    FromLogits(&'a [i32]),
}

/// Fill the v2 KV slot mapping (one column per KV page slice) and return the number of pages touched.
fn slot_mapping_v2(
    cfg: &Config,
    sched: &[i32],
    computed: &[i32],
    page_table: &Mat<'_, i32>,
    offsets: TokenOffsets<'_>,
    slot_mapping: &mut MatMut<'_, i32>,
    scratch: &mut Scratch,
) -> Result<i64> {
    slot_mapping.fill(SLOT_MAPPING_PADDING_VAL);
    let n = sched.len();
    if n == 0 || sched.iter().all(|&s| s == 0) {
        return Ok(0);
    }
    let page_size = cfg.page_size;
    if page_size <= 0 {
        bail!("the v2 slot mapping needs a positive page_size, got {page_size}");
    }
    let slices_per_page = if cfg.slices_per_page == 0 {
        1
    } else {
        cfg.slices_per_page
    };
    if slices_per_page < 0 {
        bail!("the v2 slot mapping needs a positive slices_per_page, got {slices_per_page}");
    }
    let Scratch {
        first_page,
        page_lens,
        page_cum,
        token_offsets,
        ..
    } = scratch;
    first_page.clear();
    page_lens.clear();
    for (&scheduled, &done) in sched.iter().zip(computed) {
        let start = i64::from(done);
        let end = start + i64::from(scheduled);
        let lps = start.div_euclid(page_size);
        let lpe = (end.max(1) - 1).div_euclid(page_size);
        first_page.push(lps);
        page_lens.push(if scheduled > 0 { lpe - lps + 1 } else { 0 });
    }
    if page_lens.iter().all(|&len| len == 0) {
        return Ok(0);
    }
    page_cum.clear();
    let mut total = 0i64;
    for &len in page_lens.iter() {
        total += len;
        page_cum.push(total);
    }
    let total_pages = total;
    let max_padded = (cfg.max_padded_slices.max(1) as i64).min(slot_mapping.cols() as i64);
    let padded_slices =
        (((total_pages + slices_per_page - 1).div_euclid(slices_per_page)) * slices_per_page).min(max_padded);
    let active = total_pages.min(padded_slices).max(0) as usize;
    if active == 0 {
        return Ok(total_pages);
    }
    // The gather reads the window's `[:n, :max_num_pages_per_req]` block flattened row-major, with a row stride of
    // `max_num_pages_per_req` and clipped flat indices.
    let gather_rows = n.min(page_table.rows());
    let gather_cols = cfg.max_num_pages_per_req.min(page_table.cols());
    let gather_size = (gather_rows * gather_cols) as i64;
    if gather_size == 0 {
        bail!("the v2 slot mapping needs a non-empty page table window");
    }
    let stride = cfg.max_num_pages_per_req as i64;
    token_offsets.clear();
    let mut packed = 0i64;
    for (req, &scheduled) in sched.iter().enumerate() {
        token_offsets.push(match offsets {
            TokenOffsets::Packed => packed,
            TokenOffsets::FromLogits(logits) => i64::from(logits[req]) - i64::from(scheduled) + 1,
        });
        packed += i64::from(scheduled);
    }
    let mut req = 0usize;
    let mut prev_cum = 0i64;
    for slice in 0..active {
        let position = slice as i64;
        while page_cum[req] <= position {
            prev_cum = page_cum[req];
            req += 1;
        }
        let local_off = position - prev_cum;
        let start = i64::from(computed[req]);
        let end = start + i64::from(sched[req]);
        let flat = (req as i64 * stride + first_page[req] + local_off).clamp(0, gather_size - 1);
        let page_number =
            i64::from(page_table.row((flat / gather_cols as i64) as usize)[(flat % gather_cols as i64) as usize]);
        let kv_start = if local_off == 0 { start.rem_euclid(page_size) } else { 0 };
        let kv_end = if local_off == page_lens[req] - 1 {
            (end.max(1) - 1).rem_euclid(page_size) + 1
        } else {
            page_size
        };
        let slice_len = (kv_end - kv_start).max(0);
        let kv_cache_start = kv_start + page_number * page_size;
        let new_kv_start = token_offsets[req] + (first_page[req] + local_off) * page_size + kv_start - start;
        slot_mapping.row_mut(0)[slice] = kv_cache_start as i32;
        slot_mapping.row_mut(1)[slice] = new_kv_start as i32;
        slot_mapping.row_mut(2)[slice] = slice_len as i32;
    }
    Ok(total_pages)
}

/// Fill every staging buffer of one step from the scheduler window and the runner's row state.
pub fn build(cfg: &Config, inp: &Inputs<'_>, out: &mut Outputs<'_>, scratch: &mut Scratch) -> Result<Built> {
    let Outputs {
        scheduled,
        query_start_loc: qsl,
        input_ids,
        positions,
        seq_lens,
        logits_indices,
        pages_tables,
        packed_qsl_seqlens,
        packed_i32_padded,
        packed_f32_padded,
        packed_misc_i32,
        spec_recurrent_commit,
        request_distribution,
        dp_query_start_loc,
        dp_request_distribution,
        dp_context_lens,
        dp_recurrent_state_indices,
        dp_model_row_for_request,
        num_kv_update,
        slot_mapping,
    } = out;
    let max_num_reqs = cfg.max_num_reqs;
    let window_rows = cfg.num_reqs_max_model_len;
    if scheduled.len() != max_num_reqs
        || qsl.len() != max_num_reqs + 1
        || seq_lens.len() != max_num_reqs
        || logits_indices.len() != max_num_reqs
        || dp_model_row_for_request.len() != max_num_reqs
        || positions.len() != input_ids.len()
        || packed_qsl_seqlens.rows() != 2
        || packed_qsl_seqlens.cols() != max_num_reqs + 1
        || packed_i32_padded.rows() != 3
        || packed_i32_padded.cols() != max_num_reqs
        || packed_f32_padded.rows() != 6
        || packed_f32_padded.cols() != max_num_reqs
        || packed_misc_i32.len() != 5
        || request_distribution.len() != 3
        || num_kv_update.is_empty()
        || pages_tables.rows() != window_rows
    {
        bail!("staging buffers do not match the builder layout");
    }

    let mask = &inp.active_mask_full[..inp.active_mask_full.len().min(max_num_reqs)];
    let num_requests = mask.iter().rposition(|&live| live).map_or(0, |last| last + 1);
    let padded = padded_num_reqs(num_requests, max_num_reqs, inp.padded_num_reqs_in);
    let nts = inp.num_tokens_static;
    if nts < 0 {
        bail!("num_tokens_static must be non-negative, got {nts}");
    }

    scheduled.fill(0);
    if num_requests > 0 {
        scheduled[..num_requests].copy_from_slice(head(inp.scheduled_full, num_requests, "scheduled_full")?);
    }
    let sched_all: &[i32] = &scheduled[..];
    let sched = &sched_all[..num_requests];
    let use_dp = cfg.data_parallel_size > 1;

    qsl.fill(0);
    if num_requests > 0 {
        let mut acc = 0i64;
        for (i, &s) in sched.iter().enumerate() {
            acc += i64::from(s);
            qsl[i + 1] = acc as i32;
        }
        let last = qsl[num_requests];
        qsl[num_requests + 1..].fill(last);
    }
    let actual_num_tokens = if num_requests > 0 {
        i64::from(qsl[num_requests])
    } else {
        0
    };
    if actual_num_tokens > nts {
        bail!("Scheduled {actual_num_tokens} tokens but `num_tokens_static`={nts}; select a larger token bucket.");
    }

    let tok_len = usize::try_from(nts).unwrap_or(usize::MAX).min(input_ids.len());
    let ids = &mut input_ids[..tok_len];
    let pos = &mut positions[..tok_len];
    ids.fill(0);
    pos.fill(0);
    logits_indices.fill(0);
    dp_query_start_loc.fill(0);
    dp_request_distribution.fill(0);
    dp_context_lens.fill(0);
    dp_recurrent_state_indices.fill(0);
    dp_model_row_for_request.fill(-1);

    let computed = if num_requests > 0 {
        head(inp.num_computed, num_requests, "num_computed_tokens")?
    } else {
        &[]
    };
    let width = inp.token_ids.cols() as i64;
    let token_rows = inp.token_ids.rows();

    if use_dp {
        let dp = cfg.data_parallel_size;
        if nts % dp as i64 != 0 {
            bail!(
                "rank-major SPMD DP requires token buckets divisible by DP size: num_tokens_static={nts}, \
                 dp_size={dp}."
            );
        }
        if window_rows % dp != 0 {
            bail!("rank-major SPMD DP requires request rows divisible by DP size: rows={window_rows}, dp_size={dp}.");
        }
        let tokens_per_rank = nts / dp as i64;
        let rows_per_rank = window_rows / dp;
        let slots_per_rank = (rows_per_rank * cfg.recurrent_slot_group_size.max(1)) as i64;
        if dp_query_start_loc.rows() != dp
            || dp_query_start_loc.cols() != rows_per_rank + 1
            || dp_request_distribution.rows() != dp
            || dp_request_distribution.cols() != 3
            || dp_context_lens.rows() != dp
            || dp_context_lens.cols() < rows_per_rank
            || dp_recurrent_state_indices.rows() != dp
            || dp_recurrent_state_indices.cols() < rows_per_rank
        {
            bail!("rank-major DP buffers do not match the builder layout");
        }
        let global_rows = match inp.window_row_indices {
            Some(rows) => Some(head(rows, num_requests, "window_row_indices")?),
            None => None,
        };
        let slots = match inp.recurrent_slot_indices {
            Some(rows) => Some(head(rows, num_requests, "recurrent_slot_indices")?),
            None => None,
        };
        scratch.rank_used_tokens.clear();
        scratch.rank_used_tokens.resize(dp, 0);
        scratch.rank_live_rows.clear();
        scratch.rank_live_rows.resize(dp, 0);
        for (req, &scheduled_tokens) in sched.iter().enumerate() {
            let n = i64::from(scheduled_tokens);
            if n <= 0 {
                continue;
            }
            let global_row = global_rows.map_or(req as i64, |rows| i64::from(rows[req]));
            let rank = ((global_row.max(0) / rows_per_rank as i64) as usize).min(dp - 1);
            let local_row = scratch.rank_live_rows[rank];
            if local_row >= rows_per_rank {
                bail!(
                    "rank-major SPMD DP rank-local request budget exceeded: rank={rank} rows_per_rank={rows_per_rank} \
                     global_row={global_row}."
                );
            }
            let start = i64::from(computed[req]);
            let end = start + n;
            if end > width {
                bail!("Request {req} scheduled [{start}:{end}] exceeds token_ids width {width}.");
            }
            let rank_used = scratch.rank_used_tokens[rank];
            if rank_used + n > tokens_per_rank {
                bail!(
                    "rank-major SPMD DP rank-local token budget exceeded: rank={rank} scheduled={} \
                     tokens_per_rank={tokens_per_rank}.",
                    rank_used + n
                );
            }
            if req >= token_rows {
                bail!("token_ids holds {token_rows} rows but row {req} is scheduled");
            }
            let off = rank as i64 * tokens_per_rank + rank_used;
            write_run(ids, pos, inp.token_ids.row(req), off, start, n)?;
            scratch.rank_used_tokens[rank] = rank_used + n;
            scratch.rank_live_rows[rank] = local_row + 1;
            dp_query_start_loc.row_mut(rank)[local_row + 1] = (rank_used + n) as i32;
            dp_model_row_for_request[req] = (rank * rows_per_rank + local_row) as i32;
            dp_context_lens.row_mut(rank)[local_row] = end as i32;
            let state_slot = slots.map_or(global_row, |rows| i64::from(rows[req]));
            let state_rank = ((state_slot.max(0) / slots_per_rank) as usize).min(dp - 1);
            if state_rank != rank {
                bail!(
                    "rank-major SPMD DP recurrent state slot must belong to the same DP rank as the request: \
                     req_idx={req} rank={rank} state_slot={state_slot} state_rank={state_rank}."
                );
            }
            dp_recurrent_state_indices.row_mut(rank)[local_row] = (state_slot - rank as i64 * slots_per_rank) as i32;
            logits_indices[req] = (off + n - 1) as i32;
        }
        for rank in 0..dp {
            let live = scratch.rank_live_rows[rank];
            let used = scratch.rank_used_tokens[rank] as i32;
            dp_query_start_loc.row_mut(rank)[live + 1..=rows_per_rank].fill(used);
            let starts = dp_query_start_loc.row(rank);
            let context = dp_context_lens.row(rank);
            let (decode_end, prefill_end) = distribution_prefix_by(live, |i| {
                let len = i64::from(starts[i + 1]) - i64::from(starts[i]);
                (len, i64::from(context[i]) - len)
            });
            let distribution = dp_request_distribution.row_mut(rank);
            distribution[0] = decode_end as i32;
            distribution[1] = prefill_end as i32;
            distribution[2] = live as i32;
        }
    } else {
        let mut off = 0i64;
        for (req, &scheduled_tokens) in sched.iter().enumerate() {
            let n = i64::from(scheduled_tokens);
            if n <= 0 {
                continue;
            }
            let start = i64::from(computed[req]);
            let end = start + n;
            if end > width {
                bail!("Request {req} scheduled [{start}:{end}] exceeds token_ids width {width}.");
            }
            if req >= token_rows {
                bail!("token_ids holds {token_rows} rows but row {req} is scheduled");
            }
            write_run(ids, pos, inp.token_ids.row(req), off, start, n)?;
            off += n;
        }
        if cfg.plain_recurrent_slots {
            let fill_rows = num_requests.min(dp_recurrent_state_indices.cols());
            let row0 = dp_recurrent_state_indices.row_mut(0);
            match inp.recurrent_slot_indices {
                Some(slots) => row0[..fill_rows].copy_from_slice(head(slots, fill_rows, "recurrent_slot_indices")?),
                None => {
                    for (i, slot) in row0[..fill_rows].iter_mut().enumerate() {
                        *slot = i as i32;
                    }
                }
            }
        }
    }

    seq_lens.fill(0);
    if num_requests > 0 {
        for ((len, &s), &done) in seq_lens.iter_mut().zip(sched).zip(computed) {
            // Budget-skipped rows keep a zero length: a context with no query tokens can hang the attention kernels.
            *len = if s == 0 { 0 } else { done.wrapping_add(s) };
        }
        check_dp_local_pages(cfg, inp, sched, computed, use_dp)?;
    }

    if num_requests > 0 && !use_dp {
        for (index, &end) in logits_indices.iter_mut().zip(&qsl[1..=num_requests]) {
            *index = end.wrapping_sub(1);
        }
    }

    let rows_to_copy = if use_dp {
        window_rows
    } else {
        num_requests.min(window_rows).min(inp.page_table.rows())
    };
    let reuse_pages_tables = !use_dp
        && inp.page_table_version.is_some()
        && inp.cached_page_table_version == inp.page_table_version
        && inp.cached_rows_to_copy == Some(rows_to_copy as i64);
    if !reuse_pages_tables {
        // Rank-major rows only when a row is live; an empty DP window copies the whole window like the plain path.
        if use_dp && num_requests > 0 {
            pages_tables.fill(PAGE_TABLE_PADDING_VAL);
            for req in 0..num_requests {
                let model_row = dp_model_row_for_request[req];
                if model_row < 0 {
                    continue;
                }
                let model_row = model_row as usize;
                if model_row >= pages_tables.rows() {
                    bail!(
                        "rank-major SPMD DP model row exceeds page-table capacity: model_row={model_row}, \
                         capacity={}.",
                        pages_tables.rows()
                    );
                }
                if req >= inp.page_table.rows() {
                    bail!(
                        "page_table holds {} rows but row {req} is scheduled",
                        inp.page_table.rows()
                    );
                }
                copy_page_row(pages_tables.row_mut(model_row), inp.page_table.row(req))?;
            }
        } else {
            // numpy assignment `pages_tables[:rows_to_copy, :] = page_table[:rows_to_copy, :cols]`: each source
            // dimension must equal the destination's or be 1 (broadcast). Under DP with no live row, rows_to_copy is
            // the whole window, which may exceed the page table's rows.
            let cols = pages_tables.cols();
            let src_rows = rows_to_copy.min(inp.page_table.rows());
            let src_cols = cols.min(inp.page_table.cols());
            let rows_ok = src_rows == rows_to_copy || src_rows == 1;
            let cols_ok = src_cols == cols || src_cols == 1;
            if rows_to_copy > 0 && !(rows_ok && cols_ok) {
                bail!(
                    "could not broadcast input array from shape ({src_rows},{src_cols}) into shape \
                     ({rows_to_copy},{cols})"
                );
            }
            for row in 0..rows_to_copy {
                let src = if src_rows == rows_to_copy { row } else { 0 };
                copy_page_row(pages_tables.row_mut(row), inp.page_table.row(src))?;
            }
            for row in rows_to_copy..pages_tables.rows() {
                pages_tables.row_mut(row).fill(PAGE_TABLE_PADDING_VAL);
            }
        }
    }

    request_distribution.fill(0);
    num_kv_update.fill(0);
    if cfg.use_request_distribution {
        let (decode_end, prefill_end) =
            distribution_prefix_by(num_requests, |i| (i64::from(sched[i]), i64::from(computed[i])));
        request_distribution[0] = decode_end as i32;
        request_distribution[1] = prefill_end as i32;
        request_distribution[2] = num_requests as i32;
    }
    let mut total_pages = 0;
    if cfg.use_slot_mapping {
        let Some(slots) = slot_mapping.as_mut() else {
            bail!("the v2 slot mapping needs a slot_mapping buffer");
        };
        let offsets = if use_dp {
            TokenOffsets::FromLogits(&logits_indices[..num_requests])
        } else {
            TokenOffsets::Packed
        };
        total_pages = slot_mapping_v2(cfg, sched, computed, &inp.page_table, offsets, slots, scratch)?;
        num_kv_update[0] = total_pages as i32;
    }

    packed_qsl_seqlens.row_mut(0).copy_from_slice(&qsl[..]);
    let lens_row = packed_qsl_seqlens.row_mut(1);
    lens_row[..max_num_reqs].copy_from_slice(&seq_lens[..]);
    lens_row[max_num_reqs..].fill(0);

    packed_i32_padded.fill(0);
    packed_i32_padded.row_mut(0)[..padded].copy_from_slice(&sched_all[..padded]);
    packed_i32_padded.row_mut(1)[..padded].copy_from_slice(&logits_indices[..padded]);
    packed_i32_padded.row_mut(2)[..padded].copy_from_slice(head(inp.top_k, padded, "top_k")?);
    packed_f32_padded.fill(0.0);
    let sampling = [
        (inp.temperature, "temperature"),
        (inp.top_p, "top_p"),
        (inp.min_p, "min_p"),
        (inp.frequency_penalties, "frequency_penalties"),
        (inp.presence_penalties, "presence_penalties"),
        (inp.repetition_penalties, "repetition_penalties"),
    ];
    for (row, (values, name)) in sampling.into_iter().enumerate() {
        packed_f32_padded.row_mut(row)[..padded].copy_from_slice(head(values, padded, name)?);
    }
    packed_misc_i32.fill(0);
    packed_misc_i32[0] = num_requests as i32;
    packed_misc_i32[1] = padded as i32;
    packed_misc_i32[2..5].copy_from_slice(&request_distribution[..3]);

    if cfg.enable_spec_recurrent_commit {
        spec_recurrent_commit.fill(0);
        if let Some(commit) = &inp.spec_recurrent_commit {
            let rows = spec_recurrent_commit.rows().min(commit.rows());
            let cols = spec_recurrent_commit.cols().min(commit.cols());
            for row in 0..rows {
                spec_recurrent_commit.row_mut(row)[..cols].copy_from_slice(&commit.row(row)[..cols]);
            }
        }
    }

    Ok(Built {
        num_requests,
        padded_num_reqs: padded,
        rows_to_copy,
        reuse_pages_tables,
        total_pages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Owned staging buffers shaped like the batch preparer's.
    struct Buffers {
        scheduled: Vec<i32>,
        qsl: Vec<i32>,
        input_ids: Vec<i32>,
        positions: Vec<i32>,
        seq_lens: Vec<i32>,
        logits: Vec<i32>,
        pages: Vec<i32>,
        packed_qsl: Vec<i32>,
        packed_i32: Vec<i32>,
        packed_f32: Vec<f32>,
        misc: Vec<i32>,
        spec: Vec<i32>,
        dist: Vec<i32>,
        dp_qsl: Vec<i32>,
        dp_dist: Vec<i32>,
        dp_ctx: Vec<i32>,
        dp_slots: Vec<i32>,
        dp_rows: Vec<i32>,
        kv: Vec<i32>,
        slots: Vec<i32>,
        dp: usize,
        dp_q_cols: usize,
        dp_cols: usize,
        page_cols: usize,
        window_rows: usize,
        max_slices: usize,
    }

    impl Buffers {
        fn new(cfg: &Config, max_num_tokens: usize, page_cols: usize) -> Self {
            let m = cfg.max_num_reqs;
            let r = cfg.num_reqs_max_model_len;
            let dp = cfg.data_parallel_size.max(1);
            let dp_q_cols = if r % dp == 0 { r / dp + 1 } else { 1 };
            let dp_cols = (dp_q_cols - 1).max(1);
            Self {
                scheduled: vec![0; m],
                qsl: vec![0; m + 1],
                input_ids: vec![0; max_num_tokens],
                positions: vec![0; max_num_tokens],
                seq_lens: vec![0; m],
                logits: vec![0; m],
                pages: vec![PAGE_TABLE_PADDING_VAL; r * page_cols],
                packed_qsl: vec![0; 2 * (m + 1)],
                packed_i32: vec![0; 3 * m],
                packed_f32: vec![0.0; 6 * m],
                misc: vec![0; 5],
                spec: vec![0; 2 * m],
                dist: vec![0; 3],
                dp_qsl: vec![0; dp * dp_q_cols],
                dp_dist: vec![0; dp * 3],
                dp_ctx: vec![0; dp * dp_cols],
                dp_slots: vec![0; dp * dp_cols],
                dp_rows: vec![-1; m],
                kv: vec![0; 1],
                slots: vec![SLOT_MAPPING_PADDING_VAL; 3 * cfg.max_padded_slices.max(1)],
                dp,
                dp_q_cols,
                dp_cols,
                page_cols,
                window_rows: r,
                max_slices: cfg.max_padded_slices.max(1),
            }
        }

        fn outputs(&mut self, cfg: &Config) -> Outputs<'_> {
            let m = cfg.max_num_reqs;
            Outputs {
                scheduled: &mut self.scheduled,
                query_start_loc: &mut self.qsl,
                input_ids: &mut self.input_ids,
                positions: &mut self.positions,
                seq_lens: &mut self.seq_lens,
                logits_indices: &mut self.logits,
                pages_tables: MatMut::new(&mut self.pages, self.window_rows, self.page_cols).unwrap(),
                packed_qsl_seqlens: MatMut::new(&mut self.packed_qsl, 2, m + 1).unwrap(),
                packed_i32_padded: MatMut::new(&mut self.packed_i32, 3, m).unwrap(),
                packed_f32_padded: MatMut::new(&mut self.packed_f32, 6, m).unwrap(),
                packed_misc_i32: &mut self.misc,
                spec_recurrent_commit: MatMut::new(&mut self.spec, 2, m).unwrap(),
                request_distribution: &mut self.dist,
                dp_query_start_loc: MatMut::new(&mut self.dp_qsl, self.dp, self.dp_q_cols).unwrap(),
                dp_request_distribution: MatMut::new(&mut self.dp_dist, self.dp, 3).unwrap(),
                dp_context_lens: MatMut::new(&mut self.dp_ctx, self.dp, self.dp_cols).unwrap(),
                dp_recurrent_state_indices: MatMut::new(&mut self.dp_slots, self.dp, self.dp_cols).unwrap(),
                dp_model_row_for_request: &mut self.dp_rows,
                num_kv_update: &mut self.kv,
                slot_mapping: if cfg.use_slot_mapping {
                    Some(MatMut::new(&mut self.slots, 3, self.max_slices).unwrap())
                } else {
                    None
                },
            }
        }
    }

    fn config(max_num_reqs: usize, dp: usize, v2: bool) -> Config {
        Config {
            max_num_reqs,
            num_reqs_max_model_len: max_num_reqs,
            data_parallel_size: dp,
            use_slot_mapping: v2,
            use_request_distribution: !v2,
            plain_recurrent_slots: false,
            recurrent_slot_group_size: 1,
            enable_spec_recurrent_commit: false,
            enable_dp_local_page_path: true,
            page_size: 4,
            max_num_pages_per_req: 4,
            slices_per_page: 2,
            max_padded_slices: 8,
        }
    }

    struct Step {
        scheduled: Vec<i32>,
        active: Vec<bool>,
        tokens: Vec<i32>,
        token_cols: usize,
        computed: Vec<i32>,
        floats: Vec<f32>,
        top_k: Vec<i32>,
        pages: Vec<i32>,
        page_cols: usize,
    }

    impl Step {
        fn new(scheduled: &[i32], computed: &[i32], token_cols: usize, pages: &[i32], page_cols: usize) -> Self {
            let rows = scheduled.len();
            Self {
                scheduled: scheduled.to_vec(),
                active: scheduled.iter().map(|_| true).collect(),
                tokens: (0..(rows * token_cols) as i32).collect(),
                token_cols,
                computed: computed.to_vec(),
                floats: (0..rows).map(|i| i as f32 * 0.25).collect(),
                top_k: (0..rows as i32).collect(),
                pages: pages.to_vec(),
                page_cols,
            }
        }

        fn inputs(&self, num_tokens_static: i64, padded: i64) -> Inputs<'_> {
            let rows = self.scheduled.len();
            Inputs {
                num_tokens_static,
                scheduled_full: &self.scheduled,
                active_mask_full: &self.active,
                token_ids: Mat::new(&self.tokens, rows, self.token_cols).unwrap(),
                num_computed: &self.computed,
                temperature: &self.floats,
                top_p: &self.floats,
                top_k: &self.top_k,
                min_p: &self.floats,
                frequency_penalties: &self.floats,
                presence_penalties: &self.floats,
                repetition_penalties: &self.floats,
                page_table: Mat::new(&self.pages, rows, self.page_cols).unwrap(),
                padded_num_reqs_in: padded,
                window_row_indices: None,
                recurrent_slot_indices: None,
                spec_recurrent_commit: None,
                page_table_version: None,
                cached_page_table_version: None,
                cached_rows_to_copy: None,
                num_pages: 0,
                metadata_data_parallel_size: 1,
            }
        }
    }

    #[test]
    fn distribution_prefix_follows_the_positional_contract() {
        assert_eq!(request_distribution_prefix(&[1; 8], &[100; 8]), (8, 8));
        assert_eq!(request_distribution_prefix(&[4, 4, 4], &[0, 0, 0]), (0, 3));
        assert_eq!(request_distribution_prefix(&[1, 1, 5, 5], &[9, 9, 0, 0]), (2, 4));
        // A prefill inside the decode run ends the decode prefix; the decode after it is mixed.
        assert_eq!(request_distribution_prefix(&[1, 3, 1], &[7, 0, 7]), (1, 2));
        // A one-token first prefill (nothing computed yet) is not a decode.
        assert_eq!(request_distribution_prefix(&[1, 1], &[0, 5]), (0, 1));
        // An unscheduled row ends both runs.
        assert_eq!(request_distribution_prefix(&[1, 0, 1], &[3, 3, 3]), (1, 1));
        assert_eq!(request_distribution_prefix(&[], &[]), (0, 0));
    }

    #[test]
    fn request_order_packing() {
        let cfg = config(4, 1, false);
        let mut buffers = Buffers::new(&cfg, 16, 2);
        let mut step = Step::new(&[2, 1, 0, 3], &[0, 5, 2, 1], 8, &[1, 2, 3, 0, 5, 0, 6, 7], 2);
        step.active = vec![true, true, false, true];
        let built = build(
            &cfg,
            &step.inputs(8, 4),
            &mut buffers.outputs(&cfg),
            &mut Scratch::default(),
        )
        .unwrap();
        assert_eq!(
            built,
            Built {
                num_requests: 4,
                padded_num_reqs: 4,
                rows_to_copy: 4,
                reuse_pages_tables: false,
                total_pages: 0
            }
        );
        assert_eq!(buffers.qsl, [0, 2, 3, 3, 6]);
        assert_eq!(&buffers.input_ids[..8], [0, 1, 13, 25, 26, 27, 0, 0]);
        assert_eq!(&buffers.positions[..8], [0, 1, 5, 1, 2, 3, 0, 0]);
        assert_eq!(buffers.seq_lens, [2, 6, 0, 4]);
        // The unscheduled row points at the previous row's last token.
        assert_eq!(buffers.logits, [1, 2, 2, 5]);
        assert_eq!(buffers.dist, [0, 1, 4]);
        assert_eq!(buffers.misc, [4, 4, 0, 1, 4]);
        assert_eq!(buffers.pages, [1, 2, 3, 0, 5, 0, 6, 7]);
        assert_eq!(&buffers.packed_qsl[5..], [2, 6, 0, 4, 0]);
        assert_eq!(buffers.dp_rows, [-1; 4]);
    }

    #[test]
    fn rank_major_packing() {
        let mut cfg = config(4, 2, false);
        cfg.enable_dp_local_page_path = false;
        let mut buffers = Buffers::new(&cfg, 8, 2);
        let step = Step::new(&[2, 1, 3, 1], &[0, 5, 0, 7], 16, &[0; 8], 2);
        build(
            &cfg,
            &step.inputs(8, 4),
            &mut buffers.outputs(&cfg),
            &mut Scratch::default(),
        )
        .unwrap();
        assert_eq!(buffers.input_ids, [0, 1, 21, 0, 32, 33, 34, 55]);
        assert_eq!(buffers.positions, [0, 1, 5, 0, 0, 1, 2, 7]);
        assert_eq!(buffers.dp_qsl, [0, 2, 3, 0, 3, 4]);
        assert_eq!(buffers.dp_ctx, [2, 6, 3, 8]);
        assert_eq!(buffers.dp_rows, [0, 1, 2, 3]);
        assert_eq!(buffers.dp_slots, [0, 1, 0, 1]);
        assert_eq!(buffers.dp_dist, [0, 1, 2, 0, 1, 2]);
        assert_eq!(buffers.logits, [1, 2, 6, 7]);
    }

    #[test]
    fn v2_slot_mapping_slices_pages() {
        let cfg = config(2, 1, true);
        let mut buffers = Buffers::new(&cfg, 16, 4);
        let step = Step::new(&[7, 0], &[2, 0], 16, &[10, 11, 12, 13, 0, 0, 0, 0], 4);
        let built = build(
            &cfg,
            &step.inputs(16, 2),
            &mut buffers.outputs(&cfg),
            &mut Scratch::default(),
        )
        .unwrap();
        assert_eq!(built.total_pages, 3);
        assert_eq!(&buffers.slots[..8], [42, 44, 48, 0, 0, 0, 0, 0]);
        assert_eq!(&buffers.slots[8..16], [0, 2, 6, 0, 0, 0, 0, 0]);
        assert_eq!(&buffers.slots[16..], [2, 4, 1, 0, 0, 0, 0, 0]);
        assert_eq!(buffers.kv, [3]);
        assert_eq!(buffers.dist, [0, 0, 0]);
    }

    #[test]
    fn errors_carry_esurge_messages() {
        let cfg = config(2, 1, false);
        let mut buffers = Buffers::new(&cfg, 8, 2);
        let step = Step::new(&[3, 3], &[0, 0], 8, &[1, 2, 3, 4], 2);
        let err = build(
            &cfg,
            &step.inputs(4, 2),
            &mut buffers.outputs(&cfg),
            &mut Scratch::default(),
        )
        .unwrap_err();
        assert_eq!(
            err.0,
            "Scheduled 6 tokens but `num_tokens_static`=4; select a larger token bucket."
        );
        let wide = Step::new(&[3, 3], &[0, 6], 8, &[1, 2, 3, 4], 2);
        let err = build(
            &cfg,
            &wide.inputs(8, 2),
            &mut buffers.outputs(&cfg),
            &mut Scratch::default(),
        )
        .unwrap_err();
        assert_eq!(err.0, "Request 1 scheduled [6:9] exceeds token_ids width 8.");

        let dp_cfg = config(4, 2, false);
        let mut dp_buffers = Buffers::new(&dp_cfg, 8, 2);
        let mut dp_step = Step::new(&[1, 0, 0, 0], &[0, 0, 0, 0], 16, &[4, 0, 0, 0, 0, 0, 0, 0], 2);
        dp_step.active = vec![true, false, false, false];
        let mut inputs = dp_step.inputs(8, 4);
        inputs.num_pages = 8;
        inputs.metadata_data_parallel_size = 2;
        let err = build(
            &dp_cfg,
            &inputs,
            &mut dp_buffers.outputs(&dp_cfg),
            &mut Scratch::default(),
        )
        .unwrap_err();
        assert!(err
            .0
            .starts_with("Non-DP-local page IDs detected for active request row. req_idx=0 global_row=0"));
        assert!(err.0.contains("valid_range=[1, 4) sample_bad_page=4."));
    }

    #[test]
    fn page_table_reuse_and_broadcast() {
        let cfg = config(3, 1, false);
        let mut buffers = Buffers::new(&cfg, 8, 2);
        let step = Step::new(&[1, 1, 0], &[3, 3, 0], 8, &[7, 8, 9], 1);
        let mut inputs = step.inputs(8, 3);
        inputs.active_mask_full = &[true, true, false];
        let built = build(&cfg, &inputs, &mut buffers.outputs(&cfg), &mut Scratch::default()).unwrap();
        assert_eq!(built.rows_to_copy, 2);
        // One-page rows broadcast across the page columns; rows past the copy are padding.
        assert_eq!(buffers.pages, [7, 7, 8, 8, 0, 0]);

        buffers.pages.fill(-5);
        inputs.page_table_version = Some(9);
        inputs.cached_page_table_version = Some(9);
        inputs.cached_rows_to_copy = Some(2);
        let built = build(&cfg, &inputs, &mut buffers.outputs(&cfg), &mut Scratch::default()).unwrap();
        assert!(built.reuse_pages_tables);
        assert_eq!(buffers.pages, [-5; 6]);
        inputs.cached_rows_to_copy = Some(3);
        let built = build(&cfg, &inputs, &mut buffers.outputs(&cfg), &mut Scratch::default()).unwrap();
        assert!(!built.reuse_pages_tables);
        assert_eq!(buffers.pages, [7, 7, 8, 8, 0, 0]);
    }

    #[test]
    fn plain_recurrent_slots_and_empty_windows() {
        let mut cfg = config(3, 1, false);
        cfg.plain_recurrent_slots = true;
        let mut buffers = Buffers::new(&cfg, 4, 1);
        let step = Step::new(&[1, 1, 1], &[2, 2, 2], 4, &[1, 2, 3], 1);
        let mut inputs = step.inputs(4, 1);
        let slots = [5, 2, 9];
        inputs.recurrent_slot_indices = Some(&slots);
        build(&cfg, &inputs, &mut buffers.outputs(&cfg), &mut Scratch::default()).unwrap();
        assert_eq!(buffers.dp_slots, [5, 2, 9]);

        let idle = [false, false, false];
        inputs.active_mask_full = &idle;
        let built = build(&cfg, &inputs, &mut buffers.outputs(&cfg), &mut Scratch::default()).unwrap();
        assert_eq!(
            (built.num_requests, built.padded_num_reqs, built.rows_to_copy),
            (0, 1, 0)
        );
        assert_eq!(buffers.qsl, [0; 4]);
        assert_eq!(buffers.misc, [0, 1, 0, 0, 0]);
        assert_eq!(buffers.pages, [0; 3]);
        assert_eq!(buffers.dp_slots, [0; 3]);
    }
}
