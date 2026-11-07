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

//! The `surgecore._native` extension module.
//!
//! Each Rust module of the crate has one binding file here whose `register` function adds that module's classes and
//! functions: `payload` (the step-input builder), `kv` (the KV core) and `engine` (the scheduler core). The crate-root
//! modules hold the logic and stay free of PyO3, so `cargo test --no-default-features` tests them without Python.

use pyo3::prelude::*;

mod engine;
mod kv;
mod payload;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    payload::register(m)?;
    kv::register(m)?;
    engine::register(m)?;
    Ok(())
}
