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

//! surgecore: EasyDeL's native host library for the eSurge inference engine.
//!
//! The pure-Rust modules own eSurge's host computation: [`payload`] builds a forward window's step inputs, [`kv`]
//! holds the KV core and [`engine`] the scheduler core. The `python` feature (on by default) adds the PyO3 extension
//! module `surgecore._native`, with one binding file per module, which binds that logic over preallocated numpy
//! buffers and releases the GIL while it runs. The `detok` binding owns in-process batched streaming detokenization
//! around a Python HuggingFace tokenizer, returning only deltas on normal steps.

pub mod engine;
pub mod kv;
pub mod payload;

#[cfg(feature = "python")]
mod python;
