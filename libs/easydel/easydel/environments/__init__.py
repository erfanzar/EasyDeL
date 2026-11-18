# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Reinforcement learning environments, Kiri/sandbox runtimes, and async rollout collectors."""

from ._base import (
    SandboxBashTool,
    SandboxEditFileTool,
    SandboxEnvironment,
    SandboxPythonTool,
    SandboxReadFileTool,
    SandboxWriteFileTool,
    build_environment,
    list_environments,
    make_environment_factory,
    register_environment,
)
from .async_collector import (
    AsyncEnvironmentRolloutCollector,
    EpisodeRecord,
    GeneratedTurn,
    GroupRolloutBatch,
)
from .auto import AutoEnvironment, make_environment_reward_fn
from .code_judge import CodeJudgeEnvironment, extract_code_block
from .harnesses import (
    GatewayTurnOutput,
    HarnessEnvironment,
    HarnessPreset,
    HarnessTITOGateway,
    get_harness_preset,
    list_harnesses,
    register_harness,
)
from .kiri_installer import install_kiri
from .python_tir import StatefulPythonEnvironment
from .runtime import (
    ContainerConfig,
    ContainerSandboxSession,
    EngineSocketClient,
    ExecResult,
    KiriRuntime,
    LocalSandboxSession,
    SandboxInfraError,
    SandboxPool,
    SandboxSession,
    discover_engine_endpoint,
)
from .swe_repo import SWERepoEnvironment
from .synthetic import DifficultyTracker, SyntheticTaskStream, TaskSanityReport, TaskVerificationGate
from .synthetic_tool import SyntheticToolEnvironment
from .terminal import TerminalEnvironment

__all__ = (
    "AsyncEnvironmentRolloutCollector",
    "AutoEnvironment",
    "CodeJudgeEnvironment",
    "ContainerConfig",
    "ContainerSandboxSession",
    "DifficultyTracker",
    "EngineSocketClient",
    "EpisodeRecord",
    "ExecResult",
    "GatewayTurnOutput",
    "GeneratedTurn",
    "GroupRolloutBatch",
    "HarnessEnvironment",
    "HarnessPreset",
    "HarnessTITOGateway",
    "KiriRuntime",
    "LocalSandboxSession",
    "SWERepoEnvironment",
    "SandboxBashTool",
    "SandboxEditFileTool",
    "SandboxEnvironment",
    "SandboxInfraError",
    "SandboxPool",
    "SandboxPythonTool",
    "SandboxReadFileTool",
    "SandboxSession",
    "SandboxWriteFileTool",
    "StatefulPythonEnvironment",
    "SyntheticTaskStream",
    "SyntheticToolEnvironment",
    "TaskSanityReport",
    "TaskVerificationGate",
    "TerminalEnvironment",
    "build_environment",
    "discover_engine_endpoint",
    "extract_code_block",
    "get_harness_preset",
    "install_kiri",
    "list_environments",
    "list_harnesses",
    "make_environment_factory",
    "make_environment_reward_fn",
    "register_environment",
    "register_harness",
)
