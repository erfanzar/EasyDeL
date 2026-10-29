# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen4-Exp (Qwen3.8-Flash-Next): hybrid GatedDeltaNet/QSA MoE VLM."""

from .modeling_qwen4_exp import (
    Qwen4ExpAttention,
    Qwen4ExpCache,
    Qwen4ExpCausalLMOutputWithPast,
    Qwen4ExpDecoderLayer,
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpLinearView,
    Qwen4ExpMLP,
    Qwen4ExpMLPStack,
    Qwen4ExpModel,
    Qwen4ExpMTPHead,
    Qwen4ExpMTPLayer,
    Qwen4ExpMTPOutput,
    Qwen4ExpPLELayer,
    Qwen4ExpQSAView,
    Qwen4ExpRMSNorm,
    Qwen4ExpSparseMoeBlock,
    Qwen4ExpTextModel,
    Qwen4ExpTextModelOutputWithPast,
    Qwen4ExpVisionTransformer,
)
from .qwen4_exp_configuration import Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig

__all__ = [
    "Qwen4ExpAttention",
    "Qwen4ExpCache",
    "Qwen4ExpCausalLMOutputWithPast",
    "Qwen4ExpConfig",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpGatedDeltaNet",
    "Qwen4ExpLinearView",
    "Qwen4ExpMLP",
    "Qwen4ExpMLPStack",
    "Qwen4ExpMTPHead",
    "Qwen4ExpMTPLayer",
    "Qwen4ExpMTPOutput",
    "Qwen4ExpModel",
    "Qwen4ExpPLELayer",
    "Qwen4ExpQSAView",
    "Qwen4ExpRMSNorm",
    "Qwen4ExpSparseMoeBlock",
    "Qwen4ExpTextConfig",
    "Qwen4ExpTextModel",
    "Qwen4ExpTextModelOutputWithPast",
    "Qwen4ExpVisionConfig",
    "Qwen4ExpVisionTransformer",
]
