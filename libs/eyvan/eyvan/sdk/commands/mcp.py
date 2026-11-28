# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan mcp``: connect an MCP client (Xerxes, Claude Code, Codex, Cursor) to this plane.

``eyvan mcp config --client C`` prints that client's configuration entry for
the active plane with every secret redacted. ``eyvan mcp install --client C``
mints a fresh, named, revocable personal API token (team mode) and merges one
server entry into the client's own configuration file without dropping any
other content, written atomically and owner-only (0600); the token is revoked
again if the file cannot be written. No token is ever printed. ``--xerxes``
remains an alias of ``--client xerxes``.

Client files (``--path`` overrides):

* Xerxes: ``~/.xerxes/mcp.json`` (a server map or list; ``streamable_http`` entry).
* Claude Code: ``~/.claude.json``, user-scope ``mcpServers`` with a ``"type": "http"`` entry.
  Claude Code rewrites this file while running; ``install`` re-reads it just
  before the atomic replace and stops if it changed, but a rewrite in the
  instant between that check and the replace is not detectable. Quit Claude
  Code first (or re-run ``install`` if the entry is missing afterwards).
* Codex: ``$CODEX_HOME/config.toml`` (default ``~/.codex/config.toml``), table
  ``[mcp_servers.<name>]`` with ``url`` and ``http_headers``. Edited textually:
  only that table is replaced or appended, and the result must parse with
  :mod:`tomllib` and leave every other key unchanged, else nothing is written.
* Cursor: ``~/.cursor/mcp.json``, ``mcpServers`` with ``url`` and ``headers``.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import socket
import stat
import tomllib
import typing as tp
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import click

from eyvan._internal.workspace import create_private_directory
from eyvan.sdk.commands.state import CliSession
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER

MCP_ROUTE = "/api/v1/mcp"
DEFAULT_XERXES_MCP_PATH = Path("~/.xerxes/mcp.json")
REDACTED = "<redacted>"
_MAX_DOCUMENT_BYTES = 1_048_576
#: ``~/.claude.json`` also holds Claude Code's project history and grows large.
_MAX_CLAUDE_DOCUMENT_BYTES = 64 * 1_048_576
#: Server names Claude Code, Codex and Cursor accept as plain keys.
_PORTABLE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: Lifetime of a token minted by ``eyvan mcp install``; the same default as the
#: dashboard's Connect-an-agent panel. ``--no-expiry`` is the explicit opt-out.
DEFAULT_AGENT_TOKEN_DAYS = 90


def agent_token_name(client_key: str, host: str, now: datetime | None = None) -> str:
    """Return ``agent-<client>-<YYYYMMDD-HHMM>-<host>`` (local time, at most 64 characters).

    The prefix and stamp match the dashboard's agent tokens, so a token list shows
    which agent and when; the host says which machine's config holds it.
    """

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    return f"agent-{client_key}-{stamp}-{host}"[:64]


@dataclass(frozen=True, slots=True)
class McpClient:
    """One MCP client whose configuration file ``install`` can update."""

    key: str
    label: str
    file_format: str  # "json" | "toml"
    max_bytes: int

    def default_path(self) -> Path:
        if self.key == "claude-code":
            return Path("~/.claude.json").expanduser()
        if self.key == "codex":
            # Codex's own location contract: CODEX_HOME, else ~/.codex.
            home = os.environ.get("CODEX_HOME")
            return (Path(home).expanduser() if home else Path("~/.codex").expanduser()) / "config.toml"
        if self.key == "cursor":
            return Path("~/.cursor/mcp.json").expanduser()
        return DEFAULT_XERXES_MCP_PATH.expanduser()


CLIENTS: tp.Final[dict[str, McpClient]] = {
    "xerxes": McpClient("xerxes", "Xerxes", "json", _MAX_DOCUMENT_BYTES),
    "claude-code": McpClient("claude-code", "Claude Code", "json", _MAX_CLAUDE_DOCUMENT_BYTES),
    "codex": McpClient("codex", "Codex", "toml", _MAX_DOCUMENT_BYTES),
    "cursor": McpClient("cursor", "Cursor", "json", _MAX_DOCUMENT_BYTES),
}


def mcp_url(base_url: str) -> str:
    """The plane's MCP endpoint."""

    return base_url.rstrip("/") + MCP_ROUTE


def needs_private_network(url: str) -> bool:
    """Whether Xerxes would refuse ``url`` without ``allowPrivateNetwork``.

    Xerxes classifies only ``localhost`` and address literals (it never resolves
    names), so a hostname is reported as public here too.
    """

    host = (urlsplit(url).hostname or "").strip("[]").rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def _team_mode(session: CliSession) -> bool:
    try:
        payload = session.client.request("GET", "/api/v1/team/status")
    except click.ClickException as exc:
        raise click.ClickException(f"could not ask {session.base_url} whether it runs team mode: {exc}") from exc
    return isinstance(payload, dict) and payload.get("team_mode") is True


def _scope_headers(session: CliSession) -> dict[str, str]:
    headers: dict[str, str] = {}
    if session.org_id:
        headers["X-Eyvan-Org"] = session.org_id
    if session.project_id:
        headers["X-Eyvan-Project"] = session.project_id
    return headers


def build_entry(session: CliSession, *, team_mode: bool, token: str | None = None) -> dict[str, tp.Any]:
    """The Xerxes server entry; :func:`client_entry` reshapes it for other clients.

    ``token`` is the bearer to embed on a team plane; ``None`` renders it
    redacted. On a development-auth plane the entry carries this CLI's identity
    headers, and this machine's local development credential only when
    ``token`` is not ``None`` (``install``); ``config`` passes ``None``.
    """

    url = mcp_url(session.base_url)
    headers: dict[str, str]
    identity = session.identity
    if team_mode or identity.token is not None:
        headers = {"Authorization": f"Bearer {token if token is not None else REDACTED}"}
        headers.update(_scope_headers(session))
    else:
        # A development-auth plane authenticates the same headers the CLI sends.
        user = identity.dev_user
        if not user:
            raise click.ClickException(
                f"{session.base_url} is not a team plane and this CLI has no development identity for it; "
                "sign in with `eyvan login` or pass --user"
            )
        headers = {"X-Eyvan-User": user, "X-Eyvan-Roles": ",".join(identity.dev_roles)}
        headers.update(_scope_headers(session))
        credential = identity.local_development_credential
        if credential:
            headers[LOCAL_DEVELOPMENT_CREDENTIAL_HEADER] = credential if token is not None else REDACTED
    return {
        "transport": "streamable_http",
        "url": url,
        "headers": headers,
        "allowPrivateNetwork": needs_private_network(url),
    }


def client_entry(client: str, entry: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """``entry`` (Xerxes shape) in ``client``'s own configuration shape."""

    headers = dict(entry.get("headers") or {})
    if client == "claude-code":
        return {"type": "http", "url": entry["url"], "headers": headers}
    if client == "cursor":
        return {"url": entry["url"], "headers": headers}
    if client == "codex":
        return {"url": entry["url"], "http_headers": headers}
    return entry


def redacted(entry: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """A copy of ``entry`` with the bearer and local credential values replaced."""

    result = dict(entry)
    for field in ("headers", "http_headers"):
        if field not in entry:
            continue
        headers = dict(entry.get(field) or {})
        for name in list(headers):
            if name.lower() == "authorization":
                headers[name] = f"Bearer {REDACTED}"
            elif name.lower() == LOCAL_DEVELOPMENT_CREDENTIAL_HEADER.lower():
                headers[name] = REDACTED
        result[field] = headers
    return result


# ------------------------------------------------------------------ reading


def _read_raw(path: Path, max_bytes: int = _MAX_DOCUMENT_BYTES) -> bytes | None:
    """The file's bytes, ``None`` when absent; refuses symlinks, non-files and oversize files."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise click.ClickException(f"could not open {path} (a symlink or unreadable file?): {exc}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise click.ClickException(f"{path} is not a regular file")
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise click.ClickException(f"{path} is larger than {max_bytes // 1_048_576} MiB; refusing to rewrite it")
    return raw


def _parse(client: McpClient, raw: bytes | None, path: Path) -> tp.Any:
    """The parsed document (``None`` when absent or blank); never quotes parser errors."""

    if raw is None or not raw.strip():
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise click.ClickException(f"{path} is not UTF-8 text; fix or move it first (nothing was changed)") from None
    if client.file_format == "toml":
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            # Never quote the parser error: the file holds credentials.
            raise click.ClickException(f"{path} is not valid TOML; fix or move it first (nothing was changed)") from None
    try:
        return json.loads(text)
    except ValueError:
        raise click.ClickException(f"{path} is not valid JSON; fix or move it first (nothing was changed)") from None


def _load_document(path: Path) -> tp.Any:
    """Read Xerxes's mcp.json, refusing anything it would not be safe to rewrite."""

    return _parse(CLIENTS["xerxes"], _read_raw(path), path)


def _servers(document: tp.Any) -> list[tp.Any] | None:
    if isinstance(document, list):
        return document
    if isinstance(document, dict) and isinstance(document.get("servers"), list):
        return tp.cast(list[tp.Any], document["servers"])
    return None


def has_entry(document: tp.Any, name: str) -> bool:
    """Whether ``document`` (any shape Xerxes accepts) already holds a server named ``name``."""

    servers = _servers(document)
    if servers is not None:
        return any(isinstance(item, dict) and str(item.get("name", "")).strip() == name for item in servers)
    return isinstance(document, dict) and name in document


def merged(document: tp.Any, name: str, entry: dict[str, tp.Any]) -> tp.Any:
    """``document`` with ``name`` set to ``entry``, keeping its shape and every other server."""

    if document is None:
        return {name: entry}
    servers = _servers(document)
    if servers is not None:
        kept = [item for item in servers if not (isinstance(item, dict) and str(item.get("name", "")).strip() == name)]
        kept.append({"name": name, **entry})
        return kept if isinstance(document, list) else {**document, "servers": kept}
    if isinstance(document, dict):
        return {**document, name: entry}
    raise click.ClickException("the existing mcp.json is neither a server map nor a list; nothing was changed")


def _server_map_key(client: McpClient) -> str:
    return "mcp_servers" if client.key == "codex" else "mcpServers"


def _check_shape(client: McpClient, document: tp.Any, path: Path) -> None:
    if document is None:
        return
    if client.key == "xerxes":
        if _servers(document) is None and not isinstance(document, dict):
            raise click.ClickException(f"{path} is neither a server map nor a list; nothing was changed")
        return
    key = _server_map_key(client)
    if not isinstance(document, dict):
        raise click.ClickException(f"{path} is not a JSON object; nothing was changed")
    if key in document and not isinstance(document[key], dict):
        raise click.ClickException(f"{path}: {key} is not a table/object; nothing was changed")


def has_client_entry(client: McpClient, document: tp.Any, name: str) -> bool:
    """Whether ``client``'s parsed configuration already holds a server named ``name``."""

    if client.key == "xerxes":
        return has_entry(document, name)
    servers = document.get(_server_map_key(client)) if isinstance(document, dict) else None
    return isinstance(servers, dict) and name in servers


# ------------------------------------------------------------------ Codex TOML

_TABLE_HEADER = re.compile(r"^\s*\[(\[?)([^\[\]]+)\](\]?)\s*(?:#.*)?$")
_KEY_PART = re.compile(r"""\s*(?:"((?:[^"\\]|\\.)*)"|'([^']*)'|([A-Za-z0-9_-]+))\s*(\.|$)""")
_TOML_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _dotted_key(text: str) -> list[str] | None:
    """The parts of a TOML dotted key, or ``None`` when it is not one this parser understands."""

    parts: list[str] = []
    position = 0
    while position < len(text):
        match = _KEY_PART.match(text, position)
        if match is None:
            return None
        basic, literal, bare, separator = match.groups()
        if basic is not None:
            try:
                parts.append(json.loads(f'"{basic}"'))
            except ValueError:
                return None
        else:
            parts.append(literal if literal is not None else bare)
        position = match.end()
        if separator == "":
            break
    return parts if position >= len(text) else None


def _toml_key(key: str) -> str:
    return key if _TOML_BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=True)


def _toml_string(value: str) -> str:
    # JSON string escapes (\" \\ \n \t \uXXXX ...) are valid TOML basic-string escapes.
    return json.dumps(value, ensure_ascii=True)


def codex_table(name: str, entry: dict[str, tp.Any]) -> str:
    """The ``[mcp_servers.<name>]`` table for ``entry`` (Codex shape)."""

    lines = [f"[mcp_servers.{_toml_key(name)}]", f"url = {_toml_string(entry['url'])}"]
    headers = entry.get("http_headers") or {}
    if headers:
        inner = ", ".join(f"{_toml_string(key)} = {_toml_string(value)}" for key, value in headers.items())
        lines.append(f"http_headers = {{ {inner} }}")
    return "\n".join(lines) + "\n"


def _without_codex_table(text: str, name: str) -> str:
    """``text`` without the ``[mcp_servers.<name>]`` table and its sub-tables."""

    kept: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        header = _TABLE_HEADER.match(line)
        if header is not None:
            parts = _dotted_key(header.group(2))
            skipping = parts is not None and parts[:2] == ["mcp_servers", name]
        if not skipping:
            kept.append(line)
    return "".join(kept)


def _others(document: dict[str, tp.Any], name: str) -> dict[str, tp.Any]:
    rest = copy.deepcopy(document)
    servers = rest.get("mcp_servers")
    if isinstance(servers, dict):
        servers.pop(name, None)
        if not servers:
            rest.pop("mcp_servers")
    return rest


def render_codex(text: str, name: str, entry: dict[str, tp.Any]) -> str:
    """``text`` with only ``[mcp_servers.<name>]`` replaced or appended, verified with :mod:`tomllib`."""

    before = tomllib.loads(text) if text.strip() else {}
    stripped = _without_codex_table(text, name)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    result = stripped + ("\n" if stripped.strip() else "") + codex_table(name, entry)
    refusal = (
        f"could not add [mcp_servers.{name}] without disturbing the rest of the file (is the server defined inline "
        "or with dotted keys?); edit it by hand. Nothing was changed"
    )
    try:
        after = tomllib.loads(result)
    except tomllib.TOMLDecodeError:
        raise click.ClickException(refusal) from None
    written = (after.get("mcp_servers") or {}).get(name)
    expected = {"url": entry["url"], **({"http_headers": entry["http_headers"]} if entry.get("http_headers") else {})}
    if written != expected or _others(after, name) != _others(before, name):
        raise click.ClickException(refusal)
    return result


# ------------------------------------------------------------------ rendering and writing


def render(client: McpClient, raw: bytes | None, document: tp.Any, name: str, entry: dict[str, tp.Any]) -> str:
    """The complete new file text for ``client`` with ``name`` set to ``entry``."""

    if client.key == "codex":
        return render_codex((raw or b"").decode("utf-8"), name, entry)
    if client.key == "xerxes":
        updated: tp.Any = merged(document, name, entry)
    else:
        base: dict[str, tp.Any] = document if isinstance(document, dict) else {}
        existing = base.get("mcpServers")
        servers: dict[str, tp.Any] = existing if isinstance(existing, dict) else {}
        updated = {**base, "mcpServers": {**servers, name: entry}}
    return json.dumps(updated, indent=2, ensure_ascii=False) + "\n"


def write_private_text(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text``, owner-only (0600)."""

    create_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_private(path: Path, document: tp.Any) -> None:
    """Atomically replace ``path`` with ``document`` as JSON, owner-only (0600)."""

    write_private_text(path, json.dumps(document, indent=2) + "\n")


def snippet(client: McpClient, name: str, entry: dict[str, tp.Any]) -> str:
    """Paste-ready configuration text for ``client``."""

    if client.key == "codex":
        return codex_table(name, entry).rstrip("\n")
    if client.key == "xerxes":
        return json.dumps({name: entry}, indent=2)
    return json.dumps({"mcpServers": {name: entry}}, indent=2)


# ------------------------------------------------------------------ commands


@click.group("mcp")
def mcp() -> None:
    """Use this plane as an MCP tool server (Xerxes, Claude Code, Codex, Cursor)."""


_NAME = click.option("--name", default="eyvan", show_default=True, help="Server name in the client's config.")
_CLIENT = click.option(
    "--client",
    "client_key",
    type=click.Choice(sorted(CLIENTS)),
    default=None,
    help="MCP client to configure.",
)
_XERXES = click.option("--xerxes", "xerxes", is_flag=True, help="Alias of --client xerxes.")


def _selected_client(client_key: str | None, xerxes: bool, *, required: bool) -> McpClient:
    if xerxes and client_key not in {None, "xerxes"}:
        raise click.UsageError("--xerxes is an alias of --client xerxes; pass only one of them")
    key = "xerxes" if xerxes else client_key
    if key is None:
        if required:
            raise click.UsageError("pass --client xerxes|claude-code|codex|cursor (or --xerxes)")
        key = "xerxes"
    return CLIENTS[key]


def _check_name(client: McpClient, name: str) -> None:
    if not name.strip() or len(name) > 128:
        raise click.BadParameter("must be 1-128 characters", param_hint="--name")
    if client.key != "xerxes" and not _PORTABLE_NAME.fullmatch(name):
        raise click.BadParameter(f"{client.label} server names must match [A-Za-z0-9_-]{{1,64}}", param_hint="--name")


@mcp.command("config")
@_CLIENT
@_XERXES
@_NAME
@click.option("--json", "as_json", is_flag=True, help="Print only the JSON entry.")
@click.pass_obj
def mcp_config(session: CliSession, client_key: str | None, xerxes: bool, name: str, as_json: bool) -> None:
    """Print a client's MCP entry for the active plane (secrets redacted; default client Xerxes)."""

    client = _selected_client(client_key, xerxes, required=False)
    _check_name(client, name)
    team_mode = _team_mode(session)
    entry = client_entry(client.key, redacted(build_entry(session, team_mode=team_mode, token=None)))
    if as_json:
        click.echo(json.dumps({name: entry}, indent=2))
        return
    click.echo(snippet(client, name, entry))
    if team_mode:
        click.echo(
            f"the bearer is redacted; `eyvan mcp install --client {client.key}` mints a personal token for it and "
            "writes this entry",
            err=True,
        )
    else:
        click.echo(
            "development-auth plane: no team token is needed; the entry carries this CLI's identity headers"
            + (" and this machine's local development credential (redacted here)" if REDACTED in str(entry) else ""),
            err=True,
        )


def _revoke(session: CliSession, minted: dict[str, tp.Any] | None, notes: list[str]) -> None:
    if minted is None or not minted.get("id"):
        return
    try:
        session.client.request("DELETE", f"/api/v1/team/tokens/{minted['id']}")
        notes.append(f"revoked {minted['id']} again because the install did not finish")
    except click.ClickException:
        notes.append(f"could not revoke {minted['id']}; revoke it with `eyvan tokens revoke {minted['id']}`")


@mcp.command("install")
@_CLIENT
@_XERXES
@click.option("--path", "path_text", type=click.Path(dir_okay=False), help="Config file to update (default per client).")
@_NAME
@click.option("--yes", is_flag=True, help="Replace an existing entry of the same name.")
@click.option(
    "--expires-in-days",
    type=click.IntRange(1, 3650),
    default=DEFAULT_AGENT_TOKEN_DAYS,
    show_default=True,
    help="Days until the minted token expires (the dashboard's Connect-an-agent default).",
)
@click.option("--no-expiry", is_flag=True, help="Mint a token that never expires (revoke it yourself).")
@click.option("--json", "as_json", is_flag=True, help="Report the result as JSON (secrets redacted).")
@click.pass_obj
def mcp_install(
    session: CliSession,
    client_key: str | None,
    xerxes: bool,
    path_text: str | None,
    name: str,
    yes: bool,
    expires_in_days: int,
    no_expiry: bool,
    as_json: bool,
) -> None:
    """Write this plane into an MCP client's config with a fresh personal token (team mode).

    \b
    Claude Code rewrites ~/.claude.json while it runs. This command re-reads
    the file just before replacing it and stops if it changed, but a rewrite
    landing in the instant between that check and the replace would be lost
    (or would drop this entry). Quit Claude Code first, or re-run the command
    if the entry is missing afterwards.
    """

    client = _selected_client(client_key, xerxes, required=True)
    _check_name(client, name)
    path = Path(path_text).expanduser() if path_text else client.default_path()
    raw = _read_raw(path, client.max_bytes)
    document = _parse(client, raw, path)
    _check_shape(client, document, path)
    if has_client_entry(client, document, name) and not yes:
        raise click.ClickException(f"{path} already has an MCP server named {name!r}; pass --yes to replace it")
    team_mode = _team_mode(session)
    if team_mode and session.identity.token is None:
        raise click.ClickException(f"{session.base_url} runs team mode; sign in first with `eyvan login`")
    if not team_mode and session.identity.token is not None:
        raise click.ClickException(
            f"{session.base_url} does not run team mode and this CLI authenticates with a bearer token; "
            "`eyvan mcp install` writes only revocable team tokens, so it will not copy that one. "
            "Use `eyvan mcp config` and add the credential yourself"
        )
    # Prove the edit is possible with a redacted entry before anything is minted.
    render(client, raw, document, name, client_entry(client.key, build_entry(session, team_mode=team_mode)))

    notes: list[str] = []
    cleanup: list[str] = []
    minted: dict[str, tp.Any] | None = None
    secret: str | None = None
    text = ""
    # From the mint until the file is written, ANY failure (including Ctrl-C or
    # an unexpected answer) revokes the minted token: a token that reached no
    # file is one nobody knows to revoke.
    try:
        if team_mode:
            token_name = agent_token_name(client.key, socket.gethostname())
            payload = session.client.request(
                "POST",
                "/api/v1/team/tokens",
                json={"name": token_name, "expires_in_days": None if no_expiry else expires_in_days},
            )
            raw_record = payload.get("record") if isinstance(payload, dict) else None
            record: dict[str, tp.Any] = raw_record if isinstance(raw_record, dict) else {}
            if record.get("id"):
                minted = {"id": record.get("id"), "name": record.get("name", token_name)}
            raw_token = payload.get("token") if isinstance(payload, dict) else None
            if not isinstance(raw_token, str):
                raise click.ClickException("the plane returned an unexpected answer when minting the token")
            if minted is None:
                minted = {"id": None, "name": token_name}
            secret = raw_token
            lifetime = "never expires" if no_expiry else f"expires in {expires_in_days} days"
            notes.append(
                f"minted personal token {minted['id']} ({minted['name']}, {lifetime}); revoke it with "
                f"`eyvan tokens revoke {minted['id']}`"
            )
        else:
            secret = session.identity.local_development_credential or ""
            notes.append("development-auth plane: no team token is needed and none was minted")
        entry = client_entry(client.key, build_entry(session, team_mode=team_mode, token=secret))
        text = render(client, raw, document, name, entry)
        if _read_raw(path, client.max_bytes) != raw:
            raise click.ClickException(f"{path} changed while this command ran; run it again")
        write_private_text(path, text)
    except BaseException as exc:
        _revoke(session, minted, cleanup)
        suffix = "".join(f"; {note}" for note in cleanup)
        if isinstance(exc, OSError):
            raise click.ClickException(f"could not write {path}: {exc}{suffix}") from exc
        if isinstance(exc, click.ClickException):
            raise click.ClickException(f"{exc.message}{suffix}") from exc
        if cleanup:
            click.echo("; ".join(cleanup), err=True)
        raise
    finally:
        secret = None
        text = ""
    if as_json:
        click.echo(
            json.dumps(
                {
                    "client": client.key,
                    "path": str(path),
                    "name": name,
                    "entry": redacted(entry),
                    "token": minted,
                    "notes": notes,
                }
            )
        )
        return
    lines = [f"wrote MCP server {name!r} -> {entry['url']} into {path} for {client.label} (owner-only)"]
    lines.extend(f"  {note}" for note in notes)
    if client.key == "xerxes" and entry.get("allowPrivateNetwork"):
        lines.append("  allowPrivateNetwork is set because the plane is on a loopback/private address")
    if client.key == "codex":
        lines.append("  Codex releases before streamable HTTP was on by default also need experimental_use_rmcp_client")
    lines.append(f"  restart {client.label} (or reload its MCP servers) to connect")
    click.echo("\n".join(lines))


def register_mcp_commands(group: click.Group) -> None:
    """Attach ``eyvan mcp`` to the CLI root."""

    group.add_command(mcp)


__all__ = ("CLIENTS", "register_mcp_commands")
