# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""JAX profiler for performance analysis and autotuning.

This module provides a comprehensive profiler for JAX operations that captures
execution traces, parses profile data, and provides detailed timing analysis.
It is designed to work with JAX's built-in profiling infrastructure and
supports advanced features for autotuning hyperparameter optimization.

Key Features:
    - Profile capture using JAX's native profiling system
    - Nested event time accounting with interval merging
    - Regex-based event filtering for focused analysis
    - Device-specific profiling across GPU/TPU/CPU platforms
    - Statistical aggregation with outlier removal

Classes:
    Profiler: Main profiler class for capturing and analyzing JAX traces
    ProfilingError: Exception raised when profiling operations fail

The profiler handles:
    - GPU and TPU timing formats (nanoseconds vs picoseconds)
    - Function identification by ID patterns for autotuning
    - Child event detection for accurate nested timing
    - Graceful fallback when TensorFlow hooks are unavailable

Example:
    >>> profiler = Profiler(prefix_filter='jit_', min_duration_ns=1000)
    >>> timings = profiler.profile_time_by_function_id(
    ...     closure, platform='gpu', total_calls_number=5
    ... )
"""

from __future__ import annotations

import importlib
import os
import re
import tempfile
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import jax
import numpy as np
from jax.profiler import ProfileData


class ProfilingError(Exception):
    """Exception raised when profiling operations fail."""

    pass


class Profiler:
    """JAX profile capture and parsing with nested-event accounting.

    A comprehensive profiler for JAX operations that captures execution traces,
    parses profile data, and provides detailed timing analysis with support for
    nested event filtering and device-specific profiling.

    This profiler is designed to work with JAX's built-in profiling infrastructure
    and provides advanced features like regex-based event filtering, minimum duration
    thresholds, nested event time accounting, and graceful handling of missing
    TensorFlow profiler hooks.

    The profiler detects TensorFlow Python profiler hook availability and gracefully
    skips profiling (allowing fallback to Python timing) if the hooks are not available.
    This makes it robust for deployment in environments where TensorFlow may not be
    fully configured.

    Attributes:
        prefix_filter: String prefix to filter events by name
        event_filter_regex: Optional regex pattern for event filtering
        min_duration_ns: Minimum event duration in nanoseconds to include
        max_events_per_profile: Maximum number of events to process per profile
        verbose: Enable verbose logging output
        require_tf: Whether to require TensorFlow profiler hooks
    """

    def __init__(
        self,
        *,
        prefix_filter: str = "jit_",
        event_filter_regex: str | None = None,
        min_duration_ns: float = 1000.0,
        max_events_per_profile: int | None = 10000,
        verbose: bool = False,
        require_tf: bool = False,
        silence_tf_cpp_logs: bool = True,
    ):
        """Initialize the JAX profiler with filtering and processing options.

        Args:
            prefix_filter: String prefix to filter events by name (default: "jit_")
            event_filter_regex: Optional regex pattern for advanced event filtering
            min_duration_ns: Minimum event duration in nanoseconds to include in results
            max_events_per_profile: Maximum number of events to process per profile to
                                  prevent memory issues with large traces
            verbose: Enable verbose logging for debugging profiling operations
            require_tf: If True, require TensorFlow profiler hooks to be available
            silence_tf_cpp_logs: If True, set TF_CPP_MIN_LOG_LEVEL=3 to reduce TF noise
        """
        self.prefix_filter = prefix_filter
        self.event_filter_regex = event_filter_regex
        self.min_duration_ns = min_duration_ns
        self.max_events_per_profile = max_events_per_profile
        self.verbose = verbose

        self._pattern = re.compile(event_filter_regex) if event_filter_regex is not None else None

        self.require_tf = require_tf
        self._tf_avail_cache: bool | None = None
        if silence_tf_cpp_logs and "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    def _tf_python_profiler_available(self) -> bool:
        """Check once whether tensorflow.python.profiler.trace can be imported.

        Caches the result to avoid repeated import attempts. This is used to
        gracefully handle cases where TensorFlow is not available or the
        profiler hooks are not properly installed.

        Returns:
            True if TensorFlow profiler hooks are available, False otherwise
        """
        if self._tf_avail_cache is not None:
            return self._tf_avail_cache
        try:
            importlib.import_module("tensorflow.python.profiler.trace")
            self._tf_avail_cache = True
        except Exception:
            self._tf_avail_cache = False
        return self._tf_avail_cache

    @staticmethod
    def parse_profile_from_bytes(profile_bytes: bytes):  # type: ignore
        """Parse JAX profile data from serialized bytes.

        Converts raw profile bytes (typically from .xplane.pb files) into
        a structured ProfileData object that can be analyzed for performance metrics.

        Args:
            profile_bytes: Raw profile data as bytes from JAX profiler output

        Returns:
            ProfileData object containing parsed profiling information

        Raises:
            ProfilingError: If profile data cannot be parsed or is corrupted
        """
        try:
            return ProfileData.from_serialized_xspace(profile_bytes)
        except Exception as e:
            raise ProfilingError(f"Failed to parse profile data: {e}") from e

    @staticmethod
    def find_device_plane_ids(p: Any, device_str: str) -> list[int]:
        """Find plane IDs corresponding to a specific device in profile data.

        Searches through the profile's execution planes to find those matching
        the specified device string. Planes represent different execution contexts
        (e.g., GPU, TPU, CPU) in the profiling data.

        Args:
            p: Profile data object containing execution planes
            device_str: Device identifier to search for (case-insensitive)

        Returns:
            List of plane indices that match the device string

        Raises:
            ProfilingError: If no planes found for device or invalid profile structure
        """
        try:
            plane_ids = [i for i, plane in enumerate(p.planes) if device_str.lower() in plane.name.lower()]
            if not plane_ids:
                available_devices = [plane.name for plane in p.planes]
                raise ProfilingError(
                    f"No planes found for device '{device_str}'. Available devices: {available_devices}"
                )
            return plane_ids
        except AttributeError as e:
            raise ProfilingError(f"Invalid profile structure: {e}") from e

    @staticmethod
    def _get_stat_value(stat, metadata):
        """Extract the actual value from a profile statistic object.

        Profile statistics can store values in different formats (double, int64,
        uint64, ref, bytes, str). This method attempts to extract the actual
        value regardless of the storage format.

        Args:
            stat: Profile statistic object with various value fields
            metadata: Optional metadata mapping for reference values

        Returns:
            The extracted value, or None if no valid value found
        """
        try:
            if getattr(stat, "ref_value", 0) != 0:
                return metadata[stat.ref_value].name if metadata is not None else stat.ref_value
            for key in ("double", "int64", "uint64", "ref"):
                v = getattr(stat, f"{key}_value", 0)
                if v != 0:
                    return v
            for key in ("bytes", "str"):
                v = getattr(stat, f"{key}_value", b"" if key == "bytes" else "")
                if v:
                    return v
        except Exception:
            pass
        return None

    @classmethod
    def _parse_stats(cls, stats, stat_metadata):
        """Parse all statistics from a profile event into a dictionary.

        Converts the raw statistics data from profile events into a more
        accessible dictionary format, using metadata to resolve statistic names.

        Args:
            stats: Collection of statistic objects from a profile event
            stat_metadata: Optional metadata mapping for statistic names

        Returns:
            Dictionary mapping statistic names to their values
        """
        stats_list = list(stats)
        if stat_metadata is not None:
            return {stat_metadata[s.metadata_id].name: cls._get_stat_value(s, stat_metadata) for s in stats_list}
        return {getattr(s, "metadata_id", i): cls._get_stat_value(s, None) for i, s in enumerate(stats_list)}

    @classmethod
    def _parse_event(cls, event, event_metadata, stat_metadata, line_name: str = ""):
        """Parse a single profile event into a structured dictionary.

        Extracts timing information, statistics, and metadata from a raw profile
        event object, creating a standardized representation for analysis.
        Handles both GPU (uses start_ns/duration_ns) and TPU (uses offset_ps/duration_ps) timing formats.

        Args:
            event: Raw profile event object
            event_metadata: Optional metadata mapping for event names
            stat_metadata: Optional metadata mapping for statistic names
            line_name: Name of the execution line this event belongs to

        Returns:
            Dictionary containing parsed event data with timing and statistics
        """
        if event_metadata is not None:
            name = event_metadata[event.metadata_id].name
        else:
            name = event.name

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*event_stats.*__module__ attribute",
                category=DeprecationWarning,
            )
            stats = cls._parse_stats(event.stats, stat_metadata)

        name = stats.get("hlo_module", name)
        program_id = stats.get("program_id", stats.get("run_id"))
        scope_range_id = stats.get("scope_range_id", "None")
        key = f"{name}({program_id}-{scope_range_id})"

        if hasattr(event, "duration_ps"):
            start_ps = int(event.offset_ps)
            end_ps = start_ps + int(event.duration_ps)
            dur_ps = int(event.duration_ps)
        else:
            start_ps = int(event.start_ns * 1000)
            end_ps = start_ps + int(event.duration_ns * 1000)
            dur_ps = int(event.duration_ns * 1000)

        stats["start_ps"] = start_ps
        stats["end_ps"] = end_ps
        stats["duration_ps"] = dur_ps
        return dict(unified_name=key, fusion=name, line_name=line_name, **stats)

    @staticmethod
    def _find_children(
        own_name: str,
        start_ps: int,
        end_ps: int,
        events_sorted: list[dict[str, Any]],
        starts_sorted: np.ndarray,
    ):
        """Find all child events fully contained within a parent event's timespan.

        Uses binary search on precomputed sorted start times to efficiently
        locate all events that occur entirely within the specified time range,
        excluding the parent event itself. This is crucial for accurate nested
        event timing analysis.

        Args:
            own_name: Name of the parent event to exclude from results
            start_ps: Start time in picoseconds of the parent event
            end_ps: End time in picoseconds of the parent event
            events_sorted: List of all events sorted by start time
            starts_sorted: Precomputed numpy array of start times for binary search optimization

        Returns:
            List of child events that occur entirely within the time range
        """
        idx = int(np.searchsorted(starts_sorted, start_ps, side="left"))
        children = []
        for ev in events_sorted[idx:]:
            s = ev["start_ps"]
            if s > end_ps:
                break
            if ev["unified_name"] == own_name:
                continue
            if s >= start_ps and ev["end_ps"] <= end_ps:
                children.append(ev)
        return children

    @staticmethod
    def _sum_events(events):
        """Calculate total time covered by a collection of events using interval merging.

        Handles overlapping events by merging their time intervals to avoid
        double-counting execution time. This is essential for accurate nested
        event timing analysis where child events may overlap.

        Uses an efficient interval merging algorithm that sorts intervals by
        start time and then merges overlapping or adjacent intervals.

        Args:
            events: Collection of events with start_ps and end_ps timing data

        Returns:
            Total time in picoseconds covered by all events (with overlaps merged)
        """
        if not events:
            return 0
        intervals = sorted((int(e["start_ps"]), int(e["end_ps"])) for e in events)
        total = 0
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s > cur_e:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        total += cur_e - cur_s
        return total

    def get_events_from_plane(self, p: Any, plane_idx: int) -> dict[str, float]:
        """Extract and process events from a specific execution plane.

        Processes all events from the specified plane, applying filtering criteria
        and calculating accurate timing with nested event accounting. This is the
        main method for extracting performance metrics from profile data.

        Args:
            p: Profile data object containing execution planes
            plane_idx: Index of the specific plane to process

        Returns:
            Dictionary mapping event names to execution times in seconds

        Raises:
            ProfilingError: If plane index is invalid or event processing fails
        """
        try:
            planes = list(p.planes)
            if plane_idx >= len(planes):
                raise ProfilingError(f"Plane index {plane_idx} out of range (0-{len(planes) - 1})")

            plane = planes[plane_idx]
            event_metadata = getattr(plane, "event_metadata", None)
            stat_metadata = getattr(plane, "stat_metadata", None)

            min_duration_ps = int(self.min_duration_ns * 1000)
            all_events: list[dict[str, Any]] = []
            processed = 0

            for line in plane.lines:
                for event in line.events:
                    if self.max_events_per_profile is not None and processed >= self.max_events_per_profile:
                        break
                    ev = self._parse_event(event, event_metadata, stat_metadata, line_name=line.name)
                    if ev["duration_ps"] < min_duration_ps:
                        continue
                    all_events.append(ev)
                    processed += 1
                if self.max_events_per_profile is not None and processed >= self.max_events_per_profile:
                    break

            if not all_events:
                return {}

            events_sorted = sorted(all_events, key=lambda x: x["start_ps"])
            starts_sorted = np.fromiter((e["start_ps"] for e in events_sorted), dtype=np.int64, count=len(events_sorted))

            timed_events: dict[str, float] = {}
            for ev in events_sorted:
                name = ev["unified_name"]
                if self.prefix_filter and not name.startswith(self.prefix_filter):
                    continue

                children = self._find_children(name, ev["start_ps"], ev["end_ps"], events_sorted, starts_sorted)
                if self._pattern is not None:
                    children = [ch for ch in children if self._pattern.search(ch["unified_name"]) is not None]
                    children_duration_ps = self._sum_events(children)
                    duration_seconds = children_duration_ps / 1e12
                else:
                    duration_seconds = (ev["end_ps"] - ev["start_ps"]) / 1e12

                if duration_seconds >= 0:
                    timed_events[name] = float(duration_seconds)

            return timed_events

        except re.error as e:
            raise ProfilingError(f"Invalid regex pattern '{self.event_filter_regex}': {e}") from e
        except Exception as e:
            raise ProfilingError(f"Failed to extract events from plane {plane_idx}: {e}") from e

    def profile_time_by_function_id(
        self,
        timing_closure: Callable[[], None],
        platform: str,
        total_calls_number: int,
    ) -> dict[int, tuple[float, float]]:
        """Profile function execution times across multiple iterations with statistical analysis.

        Executes the provided closure multiple times under JAX profiler tracing,
        extracting timing data for functions identified by ID patterns. Provides
        statistical aggregation with outlier removal for reliable timing measurements.

        This method is specifically designed for autotuning scenarios where multiple
        function variants (identified by numeric IDs) need to be compared. It requires
        TensorFlow profiler hooks to be available and will raise ProfilingError if
        they are not found, allowing fallback to Python-level timing.

        Args:
            timing_closure: Function to execute and profile (should call the functions to time)
            platform: Target platform string (e.g., 'gpu', 'tpu', 'cpu') for device selection
            total_calls_number: Number of profiling iterations to perform for statistical accuracy

        Returns:
            Dictionary mapping function IDs to (mean_time, std_time) tuples in seconds.
            Function IDs are extracted from event names matching 'jit_autotune_fn_{id}' pattern.
            Statistical outliers are removed if more than 2 measurements are available.

        Raises:
            ProfilingError: If TensorFlow profiler hooks are not available or profiling fails
            RuntimeError: If no profile data is generated during execution
        """

        if self.require_tf and not self._tf_python_profiler_available():
            raise ProfilingError("TensorFlow Python profiler hooks are not available")
        if not self._tf_python_profiler_available():
            raise ProfilingError("Profiler not available (missing tensorflow.python.profiler.trace)")

        function_timings: dict[int, list[float]] = {}
        name_re = re.compile(r"^jit_autotune_fn_([0-9]+).*")

        for _ in range(total_calls_number):
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with tempfile.TemporaryDirectory(prefix=f"tuning_profile_{now}_") as tmpdir:
                with jax.profiler.trace(tmpdir):
                    timing_closure()

                profile_files = sorted(Path(tmpdir).glob("**/*.xplane.pb"), key=lambda f: f.stat().st_mtime)
                if not profile_files:
                    raise RuntimeError("No profile was created.")
                latest_profile = profile_files[-1]
                profile_proto = self.parse_profile_from_bytes(latest_profile.read_bytes())

                device_plane_id = self.find_device_plane_ids(profile_proto, platform)[0]
                profile_events = self.get_events_from_plane(profile_proto, device_plane_id)

                for k, dur in profile_events.items():
                    m = name_re.match(k)
                    if not m:
                        continue
                    key = int(m.group(1))
                    function_timings.setdefault(key, []).append(dur)

        agg: dict[int, tuple[float, float]] = {}
        for key, durations in function_timings.items():
            if len(durations) > 2:
                durations = sorted(durations)[1:-1]
            agg[key] = (float(np.mean(durations)), float(np.std(durations)))
        return agg
