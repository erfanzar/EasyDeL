# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Job status types for distributed execution tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JobInfo:
    """
    Metadata describing a TPU/GPU/CPU job managed via Ray.

    Attributes:
        name (str): A human-readable identifier for the job.
        state (str): The current state of the job (e.g., "pending", "running", "succeeded", "failed").
        kind (str): The type or classification of the job (e.g., "training", "inference").
    """

    name: str
    state: str
    kind: str


@dataclass
class JobStatus:
    """
    Base class representing the final status of a job after a Ray call.

    This class wraps job metadata and serves as a common interface for
    distinguishing between successful and failed executions.

    Attributes:
        info (JobInfo): Metadata about the job.
    """

    info: JobInfo


@dataclass
class JobSucceeded(JobStatus):
    """
    Indicates that the job completed successfully and returned a result.

    Attributes:
        result (object): The output produced by the job.
    """

    result: object


@dataclass
class JobPreempted(JobStatus):
    """
    Indicates that the job was interrupted or preempted, likely by external factors
    such as TPU quota eviction or infrastructure scaling events.

    Attributes:
        error (Exception): The exception raised due to preemption.
    """

    error: Exception


@dataclass
class JobFailed(JobStatus):
    """
    Indicates that the job ran to completion but failed due to an expected runtime issue.

    This could include errors such as invalid input, failed assertions, or handled exceptions.

    Attributes:
        error (Exception): The exception describing why the job failed.
    """

    error: Exception


@dataclass
class JobError(JobStatus):
    """
    Indicates that the job encountered an internal or unexpected error.

    This is typically reserved for unexpected exceptions, infrastructure issues,
    or serialization problems in the Ray runtime.

    Attributes:
        error (Exception): The exception or error message from the failure.
    """

    error: Exception
