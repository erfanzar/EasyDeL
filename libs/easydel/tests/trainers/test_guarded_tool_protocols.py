"""Guarded tools preserve callable and serialized structural protocols (no JAX)."""

from __future__ import annotations

import io
import json
import os
import socket
import urllib.request
import urllib.response

import pytest
from easydel.trainers._shared.environment import AgenticEnvironment, ResetResult, StepResult, ToolEnvWrapper
from easydel.trainers._shared.tools import (
    BashTool,
    FileReadTool,
    FunctionTool,
    JSONProcessorTool,
    NotepadTool,
    PythonCodeTool,
    UnitConverterTool,
    WebFetchTool,
    WikipediaTool,
    _SafeRedirectHandler,
    function_to_json,
)
from easydel.trainers.agentic_rollout.guards import RegexRuleFilter, RuleAction, TwoStageToolGuard
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import _CallableToolAdapter

pytestmark = pytest.mark.cpu_ok


class Environment(AgenticEnvironment):
    def reset(self, seed=None):
        return ResetResult("Use the echo tool")

    def step(self, action):
        return StepResult(action)


@pytest.mark.parametrize("arguments", ['{"text": "hello"}', '"hello"', "hello"])
@pytest.mark.parametrize("kind", ["function", "callable", "structural"])
@pytest.mark.parametrize("blocked", [False, True])
def test_guarded_wrapper_preserves_tool_argument_protocol(arguments, kind, blocked):
    executed = []

    def echo(text: str) -> str:
        executed.append(text)
        return f"echo={text}"

    if kind == "function":
        tool = FunctionTool(echo, tool_name="echo")
    elif kind == "callable":
        tool = _CallableToolAdapter(echo)
    else:

        class StructuralTool:
            name = "echo"

            @property
            def chat_schema(self):
                return {
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                }

            def execute(self, serialized):
                # Preserve bytes/formatting: external tools own this interface.
                assert serialized == arguments
                try:
                    value = json.loads(serialized)
                except json.JSONDecodeError:
                    value = serialized
                return echo(value["text"] if isinstance(value, dict) else value)

        tool = StructuralTool()

    guard = TwoStageToolGuard(RegexRuleFilter(["hello"], match_action=RuleAction.BLOCK)) if blocked else None
    wrapper = ToolEnvWrapper(Environment(), [tool], tool_guard=guard)
    wrapper.reset(seed=1)
    result = wrapper.step_with_tool_calls("", tool_calls=[("echo", arguments)])
    if blocked:
        assert executed == []
        assert "blocked by policy" in result.observation
        assert result.info["guard_events"]
    else:
        assert result.observation == "[echo]: echo=hello"
        assert executed == ["hello"]
    assert not result.terminated


def test_guarded_wrapper_keeps_duplicate_call_order_and_arguments():
    calls = []
    tool = FunctionTool(lambda text: calls.append(text) or text, tool_name="echo")
    wrapper = ToolEnvWrapper(Environment(), [tool])
    result = wrapper.step_with_tool_calls("", tool_calls=[("echo", "first"), ("echo", '"second"')])
    assert result.observation == "[echo]: first\n[echo]: second"
    assert calls == ["first", "second"]


def test_file_read_tool_requires_explicit_allowed_dirs():
    """FileReadTool must reject None, empty, or bare-string allowed_dirs."""
    with pytest.raises(ValueError, match="allowed_dirs"):
        FileReadTool()
    with pytest.raises(ValueError, match="allowed_dirs"):
        FileReadTool(allowed_dirs=None)
    with pytest.raises(ValueError, match="allowed_dirs"):
        FileReadTool(allowed_dirs=[])
    with pytest.raises(ValueError, match="allowed_dirs"):
        FileReadTool(allowed_dirs="/tmp")
    with pytest.raises(ValueError, match="allowed_dirs"):
        FileReadTool(allowed_dirs=["", "   "])


def test_file_read_tool_rejects_fifo_dir_null_bytes_and_handles_commonpath_error(tmp_path, monkeypatch):
    """FileReadTool must check S_ISREG without blocking on FIFOs and guard commonpath ValueError."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sub = allowed / "subdir"
    sub.mkdir()
    exact_file = allowed / "exact.txt"
    exact_file.write_text("abcde")
    long_file = allowed / "long.txt"
    long_file.write_text("abcdef")

    tool = FileReadTool(allowed_dirs=[str(allowed)], max_length=5)
    assert tool.call(str(exact_file)) == "abcde"
    assert tool.call(str(long_file)) == "abcde\n... (truncated)"
    assert "not a regular file" in tool.call(str(sub))
    assert tool.call("bad\x00path").startswith("Error:")

    if hasattr(os, "mkfifo"):
        fifo = allowed / "pipe"
        os.mkfifo(fifo)
        res = tool.call(str(fifo))
        assert "not a regular file" in res

    def raise_commonpath(_paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", raise_commonpath)
    assert "outside allowed directories" in tool.call(str(exact_file))


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/admin",
        "http://localhost/secret",
        "http://sub.localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://10.0.0.1/internal",
        "http://172.16.0.1/internal",
        "http://192.168.1.1/router",
        "http://100.100.100.200/latest/meta-data/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:169.254.169.254]/",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ],
)
def test_web_fetch_tool_blocks_ssrf_targets(blocked_url):
    tool = WebFetchTool()
    res = tool.call(blocked_url)
    assert res.startswith("Error:"), f"Expected SSRF block for {blocked_url}, got: {res!r}"


def test_web_fetch_tool_blocks_dns_resolving_to_private_ip(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))],
    )
    res = WebFetchTool().call("https://rebind.example.com/data")
    assert res.startswith("Error:")
    assert "non-public" in res


def test_web_fetch_tool_blocks_redirect_ssrf(monkeypatch):
    """HTTP redirects to loopback or cloud metadata must be blocked before following."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]
        if host == "public.example.com"
        else [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))],
    )

    # Exercise the real _SafeRedirectHandler through urllib's OpenerDirector.
    opened_urls: list[str] = []

    class _FakeHTTPHandler(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, req):
            opened_urls.append(req.full_url)
            if req.full_url == "https://public.example.com/start":
                from email.message import Message

                headers = Message()
                headers["Location"] = "http://169.254.169.254/latest/meta-data/"
                fp = io.BytesIO(b"redirecting")
                resp = urllib.response.addinfourl(fp, headers, req.full_url, code=302)
                resp.msg = "Found"
                return resp
            raise AssertionError(f"Should never follow redirect to {req.full_url}")

        http_open = https_open

    real_build_opener = urllib.request.build_opener
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *handlers: real_build_opener(_FakeHTTPHandler(), *handlers),
    )

    res = WebFetchTool().call("https://public.example.com/start")
    assert res.startswith("Error:")
    assert "Blocked" in res
    assert opened_urls == ["https://public.example.com/start"]

    # Also verify relative redirects to disallowed hosts are validated.
    handler = _SafeRedirectHandler()
    req = urllib.request.Request("https://public.example.com/start")
    with pytest.raises(ValueError, match="Blocked"):
        handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "http://127.0.0.1/secret")


def test_web_fetch_tool_caps_response_bytes(monkeypatch):
    """WebFetchTool must bound resp.read(max_bytes + 1) to prevent host OOM."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, **kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )
    read_sizes: list[int] = []

    class _BoundedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n: int = -1) -> bytes:
            read_sizes.append(n)
            return b"<html><body>" + (b"x" * 2048) + b"</body></html>" if n < 0 else (b"a" * n)

    class _FakeOpener:
        def open(self, req, timeout=None):
            del req, timeout
            return _BoundedResponse()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _FakeOpener())

    tool = WebFetchTool(max_length=100, max_bytes=256)
    res = tool.call("https://public.example.com/huge")
    assert read_sizes == [257]
    assert res.endswith("\n... (truncated)")
    assert len(res) <= 100 + len("\n... (truncated)")


def test_wikipedia_tool_escapes_slashes_and_caps_response_bytes(monkeypatch):
    """WikipediaTool must escape '/' in query and cap resp.read(max_bytes + 1)."""
    seen_urls: list[str] = []
    read_sizes: list[int] = []

    class _FakeResponse:
        def __init__(self, payload: bytes):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, n: int = -1) -> bytes:
            read_sizes.append(n)
            return self._payload if n < 0 else self._payload[:n]

    payload = json.dumps({"title": "AC/DC", "extract": "Australian rock band."}).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        del timeout
        seen_urls.append(req.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tool = WikipediaTool(max_length=1024, max_bytes=512)
    res = tool.call("AC/DC")
    assert seen_urls == ["https://en.wikipedia.org/api/rest_v1/page/summary/AC%2FDC"]
    assert read_sizes == [513]
    assert "AC/DC" in res and "Australian rock band." in res

    # Oversized Wikipedia response is rejected cleanly without unbounded read.
    big_payload = b"{" + b"x" * 1024 + b"}"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(big_payload))
    err = tool.call("Huge")
    assert "exceeded maximum size limit" in err

    with pytest.raises(ValueError, match="Invalid Wikipedia language"):
        WikipediaTool(language="en.evil.com#")


def test_function_to_json_handles_forward_refs_and_excludes_varargs_from_required():
    """function_to_json must survive NameError forward refs and omit *args/**kwargs from required."""

    def sample_tool(query: UnresolvedForwardRef, limit: int, *args: str, **kwargs: int) -> str:  # type: ignore[name-defined]  # noqa: F821
        """Run a search query.

        query: Search query.
        limit: Max results.
        """
        del query, limit, args, kwargs
        return ""

    schema = function_to_json(sample_tool)
    params = schema["function"]["parameters"]
    assert params["required"] == ["query", "limit"]
    assert "args" not in params["required"]
    assert "kwargs" not in params["required"]
    assert "args" not in params["properties"]
    assert "kwargs" not in params["properties"]
    assert params["properties"]["limit"]["type"] == "integer"
    assert params["properties"]["query"]["description"] == "Search query."


def test_python_and_bash_tools_expose_use_jail_and_emit_truncation_notice():
    """PythonCodeTool and BashTool validate use_jail and fire truncation notices via +1 capture."""
    with pytest.raises(ValueError, match="Invalid use_jail"):
        PythonCodeTool(use_jail="bogus")
    with pytest.raises(ValueError, match="Invalid use_jail"):
        BashTool(use_jail="bogus")

    py_tool = PythonCodeTool(timeout=5.0, max_output_length=15, use_jail="off")
    py_out = py_tool.call("print('a' * 40)")
    assert py_out.endswith("\n... (output truncated)")
    assert py_out.startswith("a" * 15)

    bash_tool = BashTool(timeout=5.0, max_output_length=15, use_jail="off")
    bash_out = bash_tool.call("printf 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'")
    assert bash_out.endswith("\n... (output truncated)")
    assert bash_out.startswith("b" * 15)


def test_json_notepad_and_unit_converter_input_guards():
    """JSONProcessorTool, NotepadTool, and UnitConverterTool return clean error strings on bad/huge inputs."""
    jp = JSONProcessorTool(max_input_length=64, max_output_length=64)
    assert jp.call("parse", "{" + '"a":1,' * 30 + '"z":2}').startswith("Error:")
    assert jp.call(123, "{}").startswith("Error:")  # type: ignore[arg-type]
    assert jp.call("query", '{"a": 1}', path=123).startswith("Error:")  # type: ignore[arg-type]
    assert jp.call("parse", "[" * 1500 + "]" * 1500).startswith("Error:")

    pad = NotepadTool(max_length=10)
    assert pad.call("write", "hello") == "Notes saved."
    assert pad.call("write", "x" * 20).startswith("Error:")
    assert pad.call("read") == "hello"  # rejected write must not clobber prior notes
    assert pad.call("append", "123456").startswith("Error:")
    assert pad.call("read") == "hello"  # rejected append must not mutate notes
    assert pad.call(123).startswith("Error:")  # type: ignore[arg-type]
    assert pad.call("write", 123).startswith("Error:")  # type: ignore[arg-type]

    uc = UnitConverterTool()
    assert uc.call(10, "km", "mi").startswith("10 km = ")
    assert uc.call(True, "km", "mi").startswith("Error:")  # type: ignore[arg-type]
    assert uc.call(float("nan"), "km", "mi").startswith("Error:")
    assert uc.call(float("inf"), "km", "mi").startswith("Error:")
    assert uc.call("not-a-number", "km", "mi").startswith("Error:")  # type: ignore[arg-type]
    assert uc.call("9" * 200, "km", "mi").startswith("Error:")  # type: ignore[arg-type]
    assert uc.call(10, 123, "mi").startswith("Error:")  # type: ignore[arg-type]
    assert uc.call(1e308, "m", "km").startswith("1e+308 m = ")
    assert uc.call(1e308, "km", "m").startswith("Error:")
