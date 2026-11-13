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

//! In-process streaming detokenization over a HuggingFace tokenizer.
//!
//! The tokenizer remains a Python object because HuggingFace owns the vocabulary-specific decoding rules. Everything
//! around it lives here: request token history, incremental context, incomplete UTF-8 buffering, bounded FIFO state,
//! final suffix reconciliation and batching. Normal results contain only new text; a full snapshot crosses the Python
//! boundary on a terminal call or when the caller's text offset shows that it needs to resynchronize.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, Debug)]
pub struct DetokenizerRequest {
    request_id: String,
    tokens_delta: Vec<i64>,
    token_offset: usize,
    finished: bool,
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
    prompt_context: Option<Vec<i64>>,
    text_offset: Option<usize>,
}

impl DetokenizerRequest {
    fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        fn item<'py>(dict: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
            dict.get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("detokenizer request is missing {name:?}")))
        }
        fn optional<'py>(dict: &Bound<'py, PyDict>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
            dict.get_item(name)
        }
        Ok(Self {
            request_id: item(dict, "request_id")?.extract()?,
            tokens_delta: optional(dict, "tokens_delta")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or_default(),
            token_offset: optional(dict, "token_offset")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0),
            finished: optional(dict, "finished")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false),
            skip_special_tokens: optional(dict, "skip_special_tokens")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(false),
            spaces_between_special_tokens: optional(dict, "spaces_between_special_tokens")?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(true),
            prompt_context: optional(dict, "prompt_context")?
                .filter(|value| !value.is_none())
                .map(|v| v.extract())
                .transpose()?,
            text_offset: optional(dict, "text_offset")?
                .filter(|value| !value.is_none())
                .map(|v| v.extract())
                .transpose()?,
        })
    }
}

/// One request's detokenization result. `accumulated_text` is `None` on the bounded streaming path.
#[pyclass(module = "surgecore._native")]
#[derive(Clone, Debug)]
pub struct DetokenizerResult {
    #[pyo3(get)]
    pub accumulated_text: Option<String>,
    #[pyo3(get)]
    pub delta_text: String,
    #[pyo3(get)]
    pub last_decoded_index: usize,
    #[pyo3(get)]
    pub finished: bool,
    #[pyo3(get)]
    pub detoktook: f64,
}

#[pymethods]
impl DetokenizerResult {
    #[new]
    #[pyo3(signature = (accumulated_text, delta_text, last_decoded_index, finished, detoktook=0.0))]
    fn new(
        accumulated_text: Option<String>,
        delta_text: String,
        last_decoded_index: usize,
        finished: bool,
        detoktook: f64,
    ) -> Self {
        Self {
            accumulated_text,
            delta_text,
            last_decoded_index,
            finished,
            detoktook,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DetokenizerResult(accumulated_text={:?}, delta_text={:?}, last_decoded_index={}, finished={}, detoktook={})",
            self.accumulated_text, self.delta_text, self.last_decoded_index, self.finished, self.detoktook
        )
    }
}

#[derive(Default)]
struct State {
    order: u64,
    last_index: usize,
    tokens: Vec<i64>,
    text: String,
    emitted_chars: usize,
    buffered: Vec<i64>,
    prompt_context: Option<Vec<i64>>,
}

fn decode_with_fallback(
    py: Python<'_>,
    tokenizer: &Py<PyAny>,
    tokens: &[i64],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> PyResult<String> {
    let token_list = PyList::new(py, tokens)?;
    let full_kwargs = PyDict::new(py);
    full_kwargs.set_item("skip_special_tokens", skip_special_tokens)?;
    full_kwargs.set_item("spaces_between_special_tokens", spaces_between_special_tokens)?;
    full_kwargs.set_item("clean_up_tokenization_spaces", false)?;
    let tokenizer = tokenizer.bind(py);
    match tokenizer.call_method("decode", (&token_list,), Some(&full_kwargs)) {
        Ok(value) => return value.extract(),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => {}
        Err(error) => return Err(error),
    }

    let cleanup_kwargs = PyDict::new(py);
    cleanup_kwargs.set_item("skip_special_tokens", skip_special_tokens)?;
    cleanup_kwargs.set_item("clean_up_tokenization_spaces", false)?;
    match tokenizer.call_method("decode", (&token_list,), Some(&cleanup_kwargs)) {
        Ok(value) => return value.extract(),
        Err(error) if error.is_instance_of::<PyTypeError>(py) => {}
        Err(error) => return Err(error),
    }

    let basic_kwargs = PyDict::new(py);
    basic_kwargs.set_item("skip_special_tokens", skip_special_tokens)?;
    tokenizer
        .call_method("decode", (&token_list,), Some(&basic_kwargs))?
        .extract()
}

fn prefix_bytes(text: &str, chars: usize) -> usize {
    if chars == 0 {
        return 0;
    }
    text.char_indices().nth(chars).map_or(text.len(), |(index, _)| index)
}

fn suffix_start_bytes(text: &str, chars: usize) -> usize {
    let total = text.chars().count();
    if chars >= total {
        return 0;
    }
    prefix_bytes(text, total - chars)
}

/// The append-only delta from `previous` to `current`, with the former worker's overlap recovery on a rewrite.
fn suffix_delta(current: &str, previous: &str) -> String {
    if current.is_empty() {
        return String::new();
    }
    if previous.is_empty() {
        return current.to_string();
    }
    if let Some(suffix) = current.strip_prefix(previous) {
        return suffix.to_string();
    }

    let max_overlap = previous.chars().count().min(current.chars().count());
    for overlap in (1..=max_overlap).rev() {
        let previous_start = suffix_start_bytes(previous, overlap);
        let current_end = prefix_bytes(current, overlap);
        if previous[previous_start..] == current[..current_end] {
            return current[current_end..].to_string();
        }
    }
    String::new()
}

/// A batch of stateful incremental decoders sharing one Python tokenizer object.
#[pyclass(module = "surgecore._native")]
pub struct Detokenizer {
    tokenizer: Py<PyAny>,
    context_window: usize,
    max_states: usize,
    states: HashMap<String, State>,
    insertion_order: BTreeMap<u64, String>,
    next_order: u64,
    closed: bool,
}

impl Detokenizer {
    fn decode_one(&mut self, py: Python<'_>, request: DetokenizerRequest) -> PyResult<DetokenizerResult> {
        let started = Instant::now();
        let DetokenizerRequest {
            request_id,
            tokens_delta,
            token_offset,
            finished,
            skip_special_tokens,
            spaces_between_special_tokens,
            prompt_context,
            text_offset,
        } = request;

        if token_offset == 0 || !self.states.contains_key(&request_id) {
            if token_offset != 0 {
                return Err(PyValueError::new_err(format!(
                    "Detokenizer delta stream lost synchronization: request_id={request_id:?} offset={token_offset} stored=0"
                )));
            }
            if let Some(previous) = self.states.remove(&request_id) {
                self.insertion_order.remove(&previous.order);
            }
            let order = self.next_order;
            self.next_order = self.next_order.wrapping_add(1);
            self.states.insert(
                request_id.clone(),
                State {
                    order,
                    ..State::default()
                },
            );
            self.insertion_order.insert(order, request_id.clone());
        }

        let state = self.states.get_mut(&request_id).expect("state inserted above");
        if token_offset != state.tokens.len() {
            return Err(PyValueError::new_err(format!(
                "Detokenizer delta stream lost synchronization: request_id={request_id:?} offset={token_offset} stored={}",
                state.tokens.len()
            )));
        }
        state.tokens.extend(tokens_delta);
        if state.prompt_context.is_none() {
            state.prompt_context = prompt_context.filter(|context| !context.is_empty());
        }

        let previous_emitted_chars = state.emitted_chars;
        let last_index = state.last_index.min(state.tokens.len());
        let new_tokens = state.tokens[last_index..].to_vec();
        let buffered = std::mem::take(&mut state.buffered);
        let emitted_index = last_index.saturating_sub(buffered.len());
        let context = if self.context_window == 0 {
            Vec::new()
        } else if emitted_index == 0 && state.prompt_context.is_some() {
            let prompt = state.prompt_context.as_ref().expect("checked above");
            prompt[prompt.len().saturating_sub(self.context_window)..].to_vec()
        } else {
            state.tokens[emitted_index.saturating_sub(self.context_window)..emitted_index].to_vec()
        };

        let mut candidate = Vec::with_capacity(context.len() + buffered.len() + new_tokens.len());
        candidate.extend_from_slice(&context);
        candidate.extend_from_slice(&buffered);
        candidate.extend_from_slice(&new_tokens);

        let mut delta = String::new();
        if !candidate.is_empty() {
            let mut decoded_candidate = decode_with_fallback(
                py,
                &self.tokenizer,
                &candidate,
                skip_special_tokens,
                spaces_between_special_tokens,
            )?;
            let mut decoded_context = if context.is_empty() {
                String::new()
            } else {
                decode_with_fallback(
                    py,
                    &self.tokenizer,
                    &context,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                )?
            };
            if !decoded_context.is_empty() && !decoded_candidate.starts_with(&decoded_context) {
                decoded_context.clear();
                let mut without_context = Vec::with_capacity(buffered.len() + new_tokens.len());
                without_context.extend_from_slice(&buffered);
                without_context.extend_from_slice(&new_tokens);
                decoded_candidate = decode_with_fallback(
                    py,
                    &self.tokenizer,
                    &without_context,
                    skip_special_tokens,
                    spaces_between_special_tokens,
                )?;
            }
            delta = decoded_candidate[decoded_context.len()..].to_string();
            if delta.ends_with('\u{fffd}') {
                state.buffered = buffered;
                state.buffered.extend_from_slice(&new_tokens);
                delta.clear();
            }
        }
        state.last_index = state.tokens.len();

        let last_decoded_index = state.last_index;
        let accumulated_text;
        if finished {
            let full = decode_with_fallback(
                py,
                &self.tokenizer,
                &state.tokens,
                skip_special_tokens,
                spaces_between_special_tokens,
            )?;
            delta = suffix_delta(&full, &state.text);
            accumulated_text = Some(full);
        } else {
            state.text.push_str(&delta);
            state.emitted_chars += delta.chars().count();
            accumulated_text = if text_offset == Some(previous_emitted_chars) {
                None
            } else {
                Some(state.text.clone())
            };
        }

        if finished {
            if let Some(state) = self.states.remove(&request_id) {
                self.insertion_order.remove(&state.order);
            }
        }
        while self.states.len() > self.max_states {
            let Some((order, oldest_id)) = self.insertion_order.pop_first() else {
                break;
            };
            if self.states.get(&oldest_id).is_some_and(|state| state.order == order) {
                self.states.remove(&oldest_id);
            }
        }

        debug_assert_eq!(self.states.len(), self.insertion_order.len());
        Ok(DetokenizerResult {
            accumulated_text,
            delta_text: delta,
            last_decoded_index,
            finished,
            detoktook: started.elapsed().as_secs_f64(),
        })
    }
}

#[pymethods]
impl Detokenizer {
    #[new]
    #[pyo3(signature = (tokenizer, *, context_window=4, max_states=65536))]
    fn new(tokenizer: Py<PyAny>, context_window: usize, max_states: usize) -> Self {
        Self {
            tokenizer,
            context_window,
            max_states,
            states: HashMap::new(),
            insertion_order: BTreeMap::new(),
            next_order: 0,
            closed: false,
        }
    }

    /// Decode a list of request dictionaries in order, returning results in the same order.
    fn decode_batch(&mut self, py: Python<'_>, requests: &Bound<'_, PyList>) -> PyResult<Vec<DetokenizerResult>> {
        if self.closed {
            return Err(PyRuntimeError::new_err("Detokenizer is shut down"));
        }
        let requests = requests
            .iter()
            .map(|request| {
                let dict = request.cast::<PyDict>()?;
                DetokenizerRequest::from_dict(dict)
            })
            .collect::<PyResult<Vec<_>>>()?;
        requests
            .into_iter()
            .map(|request| self.decode_one(py, request))
            .collect()
    }

    /// Drop one request's state. Unknown ids are ignored.
    fn reset(&mut self, request_id: &str) {
        if let Some(state) = self.states.remove(request_id) {
            self.insertion_order.remove(&state.order);
        }
    }

    /// Drop every streaming state.
    fn drain(&mut self) {
        self.states.clear();
        self.insertion_order.clear();
    }

    /// Release all request state. Later decode calls fail; the Python tokenizer is released with this object.
    fn shutdown(&mut self) {
        self.drain();
        self.closed = true;
    }

    #[getter]
    fn num_states(&self) -> usize {
        self.states.len()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Detokenizer>()?;
    m.add_class::<DetokenizerResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::suffix_delta;

    #[test]
    fn suffix_delta_matches_streaming_contract() {
        assert_eq!(suffix_delta("hello world", "hello "), "world");
        assert_eq!(suffix_delta("world!", "hello world"), "!");
        assert_eq!(suffix_delta("fresh text", "old output"), "");
        assert_eq!(suffix_delta("hé🙂!", "hé🙂"), "!");
        assert_eq!(suffix_delta("🙂!", "hé🙂"), "!");
    }
}
