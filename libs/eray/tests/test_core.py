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

"""Tests for eray.core — job status, exceptions, sentinels, cluster info."""

import dataclasses

import pytest

from eray.core.cluster import HostInfo, MultisliceInfo, SliceInfo
from eray.core.exceptions import ExceptionInfo
from eray.core.sentinels import DONE, DoneSentinel, RefBox
from eray.core.status import (
    JobError,
    JobFailed,
    JobInfo,
    JobPreempted,
    JobStatus,
    JobSucceeded,
)


class TestJobStatus:
    def test_job_info_construction(self):
        info = JobInfo(name="train", state="running", kind="training")
        assert info.name == "train"
        assert info.state == "running"
        assert info.kind == "training"

    def test_job_succeeded(self):
        info = JobInfo(name="j", state="done", kind="train")
        s = JobSucceeded(info, result={"loss": 0.1})
        assert s.result == {"loss": 0.1}
        assert s.info is info
        assert isinstance(s, JobStatus)

    def test_job_failed(self):
        info = JobInfo(name="j", state="failed", kind="train")
        err = ValueError("bad input")
        f = JobFailed(info, error=err)
        assert isinstance(f.error, ValueError)
        assert isinstance(f, JobStatus)

    def test_job_preempted(self):
        info = JobInfo(name="j", state="preempted", kind="train")
        p = JobPreempted(info, error=RuntimeError("node died"))
        assert isinstance(p.error, RuntimeError)
        assert isinstance(p, JobStatus)

    def test_job_error(self):
        info = JobInfo(name="j", state="error", kind="train")
        e = JobError(info, error=RuntimeError("oops"))
        assert isinstance(e.error, RuntimeError)

    def test_all_are_dataclasses(self):
        for cls in [JobInfo, JobStatus, JobSucceeded, JobFailed, JobPreempted, JobError]:
            assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"


class TestExceptionInfo:
    def test_ser_exc_info_from_context(self):
        try:
            raise ValueError("test error")
        except Exception:
            ei = ExceptionInfo.ser_exc_info()
        assert ei.ex is not None
        assert "test error" in str(ei.ex)

    def test_reraise(self):
        try:
            raise TypeError("type error")
        except Exception:
            ei = ExceptionInfo.ser_exc_info()
        with pytest.raises(TypeError, match="type error"):
            ei.reraise()

    def test_restore(self):
        try:
            raise KeyError("missing")
        except Exception:
            ei = ExceptionInfo.ser_exc_info()
        exc_type, exc_val, exc_tb = ei.restore()
        assert exc_type is KeyError
        assert isinstance(exc_val, KeyError)
        assert exc_tb is not None

    def test_ser_exc_info_with_explicit_exception(self):
        try:
            raise IndexError("out of range")
        except IndexError as e:
            ei = ExceptionInfo.ser_exc_info(e)
        assert isinstance(ei.ex, IndexError)

    def test_exception_info_is_dataclass(self):
        assert dataclasses.is_dataclass(ExceptionInfo)


class TestSentinels:
    def test_done_is_done_sentinel(self):
        assert isinstance(DONE, DoneSentinel)

    def test_done_is_not_none(self):
        assert DONE is not None

    def test_refbox(self):
        assert dataclasses.is_dataclass(RefBox)


class TestClusterInfo:
    def test_multislice_info(self):
        ms = MultisliceInfo(coordinator_ip="10.0.0.1", slice_id=0, num_slices=4)
        assert ms.coordinator_ip == "10.0.0.1"
        assert ms.slice_id == 0
        assert ms.num_slices == 4
        assert ms.port == 8081  # default

    def test_slice_info(self):
        si = SliceInfo(
            slice_name="slice-0",
            num_hosts=8,
            ip_address="10.0.1.10",
            num_accelerators_per_host=4,
        )
        assert si.num_hosts == 8
        assert si.num_accelerators_per_host == 4

    def test_host_info_frozen(self):
        hi = HostInfo(host_id=0, slice_name="s0", num_devices=4, healthy=True, failed=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            hi.host_id = 999

    def test_host_info_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(HostInfo)
        params = getattr(HostInfo, "__dataclass_params__", None)
        assert params is not None and params.frozen

    def test_cluster_types_are_dataclasses(self):
        for cls in [MultisliceInfo, SliceInfo, HostInfo]:
            assert dataclasses.is_dataclass(cls)
