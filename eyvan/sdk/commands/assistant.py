# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan ask``, ``eyvan size`` and ``eyvan requests``: the fleet assistant from the CLI.

The plane runs the assistant with the caller's own permissions and a provider
key that never leaves the server. A drafted workload is shown, and submitted
only when the user confirms it (``--yes`` or the prompt), as an ordinary
submission made by this CLI with the user's own identity.
"""

from __future__ import annotations

import json
import sys
import typing as tp
import uuid
from pathlib import Path

import click

from eyvan._internal.json_types import is_json_object
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession


def _object(payload: object) -> dict[str, tp.Any]:
    """Narrow one decoded API answer to a JSON object, or fail with a readable error."""

    if not isinstance(payload, dict):
        raise click.ClickException("the plane returned an unexpected answer")
    return payload


def _objects(value: object) -> list[dict[str, tp.Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _scope(session: CliSession, org_id: str | None, project_id: str | None) -> dict[str, str]:
    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    return {"org_id": scope.org_id, "project_id": scope.project_id}


def _submit_draft(session: CliSession, draft: dict[str, tp.Any]) -> tp.Any:
    """Submit one confirmed draft as this CLI's own user."""

    return session.client.request(
        "POST",
        "/api/v1/workloads",
        json=draft,
        headers={"Idempotency-Key": f"assistant-draft-{uuid.uuid4().hex}"},
    )


def _argument_summary(arguments: object, limit: int = 80) -> str:
    if not isinstance(arguments, dict) or not arguments:
        return ""
    text = ", ".join(f"{key}={json.dumps(value, default=str)}" for key, value in arguments.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _stream_turn(session: CliSession, body: dict[str, tp.Any], *, live: bool) -> dict[str, tp.Any]:
    """Run one streamed turn; print steps and tool calls to stderr and the answer to stdout as it arrives."""

    done: dict[str, tp.Any] | None = None
    conversation_id: str | None = None
    text_in_step = False
    for raw in session.client.stream_lines("POST", "/api/v1/assistant/chat/stream", json=body):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "start":
            conversation_id = event.get("conversation_id")
            if live:
                click.secho(f"· {event.get('model')} thinking…", dim=True, err=True)
        elif kind == "step":
            if text_in_step and live:
                click.echo()
            text_in_step = False
            if live and int(event.get("step") or 1) > 1:
                click.secho(f"· step {event['step']}/{event.get('max_steps')}", dim=True, err=True)
        elif kind == "text":
            text_in_step = True
            if live:
                click.echo(str(event.get("delta", "")), nl=False)
        elif kind == "tool_call":
            if live:
                if text_in_step:
                    click.echo()
                    text_in_step = False
                summary = _argument_summary(event.get("arguments"))
                click.secho(f"→ {event.get('name')}({summary})", fg="cyan", err=True)
        elif kind == "tool_result":
            if live:
                mark, colour = ("✓", "green") if event.get("ok") else ("✗", "red")
                click.secho(f"  {mark} {event.get('name')}", fg=colour, err=True)
        elif kind == "error":
            if text_in_step and live:
                click.echo()
            raise click.ClickException(f"the assistant failed: {event.get('detail')}")
        elif kind == "done":
            done = {key: value for key, value in event.items() if key != "type"}
    if done is None:
        raise click.ClickException("the assistant stream ended without an answer")
    if live and text_in_step:
        click.echo()
    done["conversation_id"] = conversation_id
    done["_streamed_text"] = live and text_in_step
    return done


@click.command("ask")
@click.argument("question", nargs=-1, required=True)
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--provider", "provider", help="Provider name or id (see `eyvan assistant providers list`).")
@click.option("--model", "model", help="One of the chosen provider's models (default: its default model).")
@click.option("--chat", "conversation_id", help="Continue a saved chat (see `eyvan assistant chats list`).")
@click.option("--yes", is_flag=True, help="Submit drafted workloads without asking.")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw assistant turn as JSON.")
@click.option("--quiet", is_flag=True, help="Print only the answer, not the live steps and tool calls.")
@click.pass_obj
def ask(
    session: CliSession,
    question: tuple[str, ...],
    org_id: str | None,
    project_id: str | None,
    provider: str | None,
    model: str | None,
    conversation_id: str | None,
    yes: bool,
    as_json: bool,
    quiet: bool,
) -> None:
    """Ask the fleet assistant (sizing, which fleet, quota, why a job waits).

    Progress streams as it happens: each step, every tool the assistant runs and
    whether it worked, then the answer as it is written. The turn is saved as a
    chat you can continue with ``--chat``.
    """

    text = " ".join(question).strip()
    status = session.client.request("GET", "/api/v1/assistant/status")
    if not (isinstance(status, dict) and status.get("enabled")):
        raise click.ClickException(
            "this plane has no assistant provider configured; a fleet admin adds one in the dashboard "
            "(Settings → Assistant providers) or with `eyvan assistant providers add`"
        )
    body: dict[str, tp.Any] = {"message": text, **_scope(session, org_id, project_id)}
    if provider is not None:
        body["provider_id"] = _provider_id(session, provider)
    if model is not None:
        body["model"] = model
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    payload = _stream_turn(session, body, live=not (quiet or as_json))
    streamed = bool(payload.pop("_streamed_text", False))
    if as_json:
        JsonRenderer().emit(payload)
        return
    if quiet or not streamed:
        click.echo(str(payload.get("reply", "")).strip())
    if payload.get("conversation_id") and not quiet:
        click.secho(f"\n(chat {payload['conversation_id']}; continue with --chat)", dim=True, err=True)
    for request in _objects(payload.get("requests")):
        click.echo(f"\nfiled request {request['id']} ({request['kind']}): {request['summary']}")
    for plan in _objects(payload.get("plans")):
        raw_budget = plan.get("budget")
        budget: dict[str, tp.Any] = raw_budget if isinstance(raw_budget, dict) else {}
        click.echo(
            f"\nproposed research plan {plan.get('id')} ({plan.get('title')}): {plan.get('jobs_planned')} jobs, "
            f"budget {budget.get('max_jobs')} jobs / {budget.get('max_chip_hours')} chip-hours / "
            f"{budget.get('max_wall_seconds')} s"
        )
        click.echo(f"nothing runs until you approve it: eyvan research approve {plan.get('id')}")
    for draft in _objects(payload.get("drafts")):
        spec = draft.get("spec", {})
        click.echo(f"\ndraft workload in {draft['org_id']}/{draft['project_id']}:")
        click.echo(json.dumps(spec, indent=2, sort_keys=True))
        confirmed = yes or (sys.stdin.isatty() and click.confirm("Submit this workload?", default=False))
        if not confirmed:
            click.echo("not submitted")
            continue
        submitted = _submit_draft(session, draft)
        workload_id = submitted.get("id") if is_json_object(submitted) else None
        click.echo(f"submitted {workload_id}")


@click.command("size")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), help="HF config.json.")
@click.option("--parameters", type=float, help="Parameter count, e.g. 7e9, when no config is at hand.")
@click.option("--trainer", type=click.Choice(["pretrain", "sft", "dpo", "orpo", "grpo", "inference"]))
@click.option("--dtype", type=click.Choice(["bfloat16", "float16", "float32"]))
@click.option("--optimizer", type=click.Choice(["adamw", "adafactor", "lion", "sgd", "none"]))
@click.option("--seq", "sequence_length", type=click.IntRange(min=1))
@click.option("--batch", "global_batch_size", type=click.IntRange(min=1))
@click.option("--lora-rank", type=click.IntRange(min=1))
@click.option("--generation-tokens", type=click.IntRange(min=0))
@click.option("--num-generations", type=click.IntRange(min=1))
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def size(
    session: CliSession, config_path: str | None, parameters: float | None, as_json: bool, **options: tp.Any
) -> None:
    """Estimate TPU memory for a job and the smallest slice that fits (EasyDeL-style)."""

    body: dict[str, tp.Any] = {key: value for key, value in options.items() if value is not None}
    if config_path is not None:
        body["model_config"] = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if parameters is not None:
        body["parameters"] = parameters
    if "model_config" not in body and "parameters" not in body:
        raise click.UsageError("pass --config CONFIG.json or --parameters N")
    payload = _object(session.client.request("POST", "/api/v1/assistant/size", json=body))
    if as_json:
        JsonRenderer().emit(payload)
        return
    estimate = _object(payload.get("estimate"))
    click.echo(f"{payload['parameters']:,} parameters; total {estimate['total_gib']} GiB")
    for fit in _objects(payload.get("fits")):
        click.echo(f"  {fit['accelerator_type']:<14} {fit['per_chip_gib']} GiB/chip of {fit['hbm_per_chip_gib']}")
    click.echo(f"recommended: {payload.get('recommended') or 'nothing in the candidate list fits'}")


@click.group("requests")
def requests_group() -> None:
    """Capacity and approval requests to the fleet admins (team mode)."""


@requests_group.command("list")
@click.option("--state", type=click.Choice(["open", "approved", "declined", "withdrawn"]))
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def requests_list(session: CliSession, state: str | None, as_json: bool) -> None:
    """Your requests (fleet admins see everyone's)."""

    payload = _object(session.client.request("GET", "/api/v1/team/requests", params={"state": state} if state else None))
    rows = _objects(payload.get("requests"))
    if as_json:
        JsonRenderer().emit(payload)
        return
    lines = [f"{row['id']}  {row['state']:<9} {row['kind']:<9} {row['subject']:<16} {row['summary']}" for row in rows]
    click.echo("\n".join(lines) or "no requests")


@requests_group.command("file")
@click.argument("kind", type=click.Choice(["capacity", "approval"]))
@click.argument("summary")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.pass_obj
def requests_file(session: CliSession, kind: str, summary: str, org_id: str | None, project_id: str | None) -> None:
    """File a request to the fleet admins."""

    payload = _object(
        session.client.request(
            "POST",
            "/api/v1/team/requests",
            json={"kind": kind, "summary": summary, **_scope(session, org_id, project_id)},
        )
    )
    click.echo(f"filed {payload['id']}")


_RESOLVER_COMMANDS = {"approved": "approve", "declined": "decline", "withdrawn": "withdraw"}


def _resolver(resolution: str) -> click.Command:
    @click.command(_RESOLVER_COMMANDS[resolution])
    @click.argument("request_id")
    @click.option("--note", default="")
    @click.pass_obj
    def resolve(session: CliSession, request_id: str, note: str) -> None:
        session.client.request(
            "POST",
            f"/api/v1/team/requests/{request_id}/resolve",
            json={"resolution": resolution, "note": note},
        )
        click.echo(f"{request_id} {resolution}")

    resolve.help = {
        "approved": "Approve an open request (fleet admins).",
        "declined": "Decline an open request (fleet admins).",
        "withdrawn": "Withdraw your own open request.",
    }[resolution]
    return resolve


for _resolution in ("approved", "declined", "withdrawn"):
    requests_group.add_command(_resolver(_resolution))


def _providers(session: CliSession) -> dict[str, tp.Any]:
    return _object(session.client.request("GET", "/api/v1/assistant/providers"))


def _provider_id(session: CliSession, name_or_id: str) -> str:
    """Resolve a provider name (case-insensitive) or id to its id."""

    rows = _objects(_providers(session).get("providers"))
    for row in rows:
        if row.get("id") == name_or_id or str(row.get("name", "")).lower() == name_or_id.lower():
            return str(row["id"])
    names = ", ".join(str(row.get("name")) for row in rows) or "none"
    raise click.ClickException(f"no assistant provider named {name_or_id!r}; available: {names}")


def _models(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def _read_key(key_stdin: bool) -> str | None:
    """Read a provider key without putting it on the command line."""

    if key_stdin:
        return sys.stdin.read().strip() or None
    if sys.stdin.isatty():
        entered = click.prompt("API key (empty for none)", hide_input=True, default="", show_default=False)
        return str(entered).strip() or None
    return None


@click.group("assistant")
def assistant_group() -> None:
    """The fleet assistant's LLM providers."""


@assistant_group.group("providers")
def providers_group() -> None:
    """LLM providers fleet admins configure; everyone picks one in chat."""


@providers_group.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def providers_list(session: CliSession, as_json: bool) -> None:
    """List providers (fleet admins also see URLs, state and whether a key is set)."""

    payload = _providers(session)
    if as_json:
        JsonRenderer().emit(payload)
        return
    rows = _objects(payload.get("providers"))
    lines = []
    for row in rows:
        extra = ""
        if "base_url" in row:
            key = "key set" if row.get("has_api_key") else "no key"
            extra = f"  {row['base_url']}  {'enabled' if row.get('enabled') else 'disabled'}  {key}"
        models = ", ".join(str(item) for item in row.get("models") or [])
        lines.append(f"{row['id']}  {row['name']:<20} models: {models} (default {row.get('default_model')}){extra}")
    click.echo("\n".join(lines) or "no assistant providers configured")


@providers_group.command("presets")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def providers_presets(session: CliSession, as_json: bool) -> None:
    """List built-in providers you can add with just a key (`providers add --preset ID`)."""

    payload = _object(session.client.request("GET", "/api/v1/assistant/presets"))
    if as_json:
        JsonRenderer().emit(payload)
        return
    for preset in _objects(payload.get("presets")):
        needs = "sign-in or key" if preset.get("sign_in") else ("key" if preset.get("key_required") else "no key")
        url = preset.get("base_url") or "(you give the URL)"
        click.echo(f"{preset['id']:<16} {preset['label']:<34} {needs:<15} {url}")


@providers_group.command("models")
@click.option("--preset", "preset_id", help="Preset id (see `providers presets`).")
@click.option("--url", "base_url", help="Base URL, for custom/local presets.")
@click.option("--provider", "provider", help="An existing provider: use its stored URL and key.")
@click.option("--key-stdin", is_flag=True, help="Read the API key from stdin instead of a hidden prompt.")
@click.pass_obj
def providers_models(
    session: CliSession, preset_id: str | None, base_url: str | None, provider: str | None, key_stdin: bool
) -> None:
    """Ask a provider which models it serves (fleet admins)."""

    body: dict[str, tp.Any] = {}
    if provider is not None:
        body["provider_id"] = _provider_id(session, provider)
    else:
        if preset_id is None and base_url is None:
            raise click.UsageError("give --preset, --url or --provider")
        if preset_id is not None:
            body["preset"] = preset_id
        if base_url is not None:
            body["base_url"] = base_url
        key = _read_key(key_stdin)
        if key:
            body["api_key"] = key
    payload = _object(session.client.request("POST", "/api/v1/assistant/models/discover", json=body))
    if payload.get("detail"):
        click.echo(f"note: {payload['detail']} ({payload.get('source')} list)", err=True)
    click.echo("\n".join(str(item) for item in payload.get("models") or []) or "no models")


@providers_group.command("add")
@click.argument("name", required=False)
@click.option("--preset", "preset_id", help="Built-in provider (see `providers presets`); then only a key is needed.")
@click.option("--url", "base_url", help="OpenAI-compatible base URL (custom/local presets only).")
@click.option("--models", help="Comma-separated model names (default: the preset's suggestions).")
@click.option("--default-model")
@click.option("--max-steps", type=click.IntRange(1, 20), default=6, show_default=True)
@click.option("--key-stdin", is_flag=True, help="Read the API key from stdin instead of a hidden prompt.")
@click.option("--disabled", is_flag=True, help="Add it switched off.")
@click.pass_obj
def providers_add(
    session: CliSession,
    name: str | None,
    preset_id: str | None,
    base_url: str | None,
    models: str | None,
    default_model: str | None,
    max_steps: int,
    key_stdin: bool,
    disabled: bool,
) -> None:
    """Add a provider (fleet admins). The key is never shown again."""

    if preset_id is None and not (name and base_url and models):
        raise click.UsageError("give --preset ID, or NAME with --url and --models")
    body: dict[str, tp.Any] = {"max_steps": max_steps, "enabled": not disabled}
    for field_name, value in (("preset", preset_id), ("name", name), ("base_url", base_url)):
        if value:
            body[field_name] = value
    if models:
        body["models"] = _models(models)
    if default_model:
        body["default_model"] = default_model
    key = _read_key(key_stdin)
    if key:
        body["api_key"] = key
    created = _object(session.client.request("POST", "/api/v1/assistant/providers", json=body))
    click.echo(f"added {created['id']} ({created['name']}); key {'set' if created.get('has_api_key') else 'not set'}")


@providers_group.command("update")
@click.argument("provider")
@click.option("--name")
@click.option("--url", "base_url")
@click.option("--models", help="Comma-separated model names; replaces the list.")
@click.option("--default-model")
@click.option("--max-steps", type=click.IntRange(1, 20))
@click.option("--enable/--disable", "enabled", default=None)
@click.option("--set-key", is_flag=True, help="Enter a new API key (hidden prompt, or stdin with --key-stdin).")
@click.option("--key-stdin", is_flag=True)
@click.option("--clear-key", is_flag=True, help="Remove the stored key.")
@click.pass_obj
def providers_update(
    session: CliSession,
    provider: str,
    name: str | None,
    base_url: str | None,
    models: str | None,
    default_model: str | None,
    max_steps: int | None,
    enabled: bool | None,
    set_key: bool,
    key_stdin: bool,
    clear_key: bool,
) -> None:
    """Change a provider (fleet admins); the stored key is kept unless you set or clear it."""

    body: dict[str, tp.Any] = {}
    for field_name, value in (
        ("name", name),
        ("base_url", base_url),
        ("default_model", default_model),
        ("max_steps", max_steps),
        ("enabled", enabled),
    ):
        if value is not None:
            body[field_name] = value
    if models is not None:
        body["models"] = _models(models)
    if clear_key:
        body["clear_api_key"] = True
    elif set_key or key_stdin:
        key = _read_key(key_stdin)
        if not key:
            raise click.UsageError("no key was entered")
        body["api_key"] = key
    provider_id = _provider_id(session, provider)
    updated = _object(session.client.request("PUT", f"/api/v1/assistant/providers/{provider_id}", json=body))
    click.echo(f"updated {updated['id']} ({updated['name']})")


@providers_group.command("remove")
@click.argument("provider")
@click.pass_obj
def providers_remove(session: CliSession, provider: str) -> None:
    """Remove a provider (fleet admins)."""

    provider_id = _provider_id(session, provider)
    session.client.request("DELETE", f"/api/v1/assistant/providers/{provider_id}")
    click.echo(f"removed {provider_id}")


@assistant_group.group("chats")
def chats_group() -> None:
    """Your saved assistant chats (only you can see them)."""


@chats_group.command("list")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def chats_list(session: CliSession, as_json: bool) -> None:
    """List your saved chats, newest first."""

    payload = _object(session.client.request("GET", "/api/v1/assistant/conversations"))
    if as_json:
        JsonRenderer().emit(payload)
        return
    rows = _objects(payload.get("conversations"))
    for row in rows:
        click.echo(
            f"{row['id']}  {str(row.get('updated_at', ''))[:16]}  {row.get('message_count', 0):>3} msgs  {row['title']}"
        )
    if not rows:
        click.echo("no saved chats")


@chats_group.command("show")
@click.argument("conversation_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def chats_show(session: CliSession, conversation_id: str, as_json: bool) -> None:
    """Print one saved chat with the tools each answer used."""

    payload = _object(session.client.request("GET", f"/api/v1/assistant/conversations/{conversation_id}"))
    if as_json:
        JsonRenderer().emit(payload)
        return
    click.secho(str(payload.get("title", "")), bold=True)
    for message in _objects(payload.get("messages")):
        click.echo()
        if message.get("role") == "user":
            click.secho(f"you: {message.get('content', '')}", fg="cyan")
            continue
        for item in _objects(message.get("trace")):
            mark = "✓" if item.get("ok") else "✗"
            click.secho(f"  {mark} {item.get('name')}", dim=True)
        if message.get("error"):
            click.secho(f"assistant failed: {message['error']}", fg="red")
        else:
            click.echo(str(message.get("content", "")))


@chats_group.command("rename")
@click.argument("conversation_id")
@click.argument("title", nargs=-1, required=True)
@click.pass_obj
def chats_rename(session: CliSession, conversation_id: str, title: tuple[str, ...]) -> None:
    """Rename one of your chats."""

    renamed = _object(
        session.client.request(
            "PUT", f"/api/v1/assistant/conversations/{conversation_id}", json={"title": " ".join(title)}
        )
    )
    click.echo(f"renamed {renamed['id']}: {renamed['title']}")


@chats_group.command("delete")
@click.argument("conversation_id")
@click.pass_obj
def chats_delete(session: CliSession, conversation_id: str) -> None:
    """Delete one of your chats."""

    session.client.request("DELETE", f"/api/v1/assistant/conversations/{conversation_id}")
    click.echo(f"deleted {conversation_id}")


def register_assistant_commands(group: click.Group) -> None:
    """Attach ask/size/requests/assistant to the CLI root."""

    from eyvan.sdk.commands.assistant_claude_code import claude_code_group

    if "claude-code" not in assistant_group.commands:
        assistant_group.add_command(claude_code_group)
    for command in (ask, size, requests_group, assistant_group):
        group.add_command(command)


__all__ = ("register_assistant_commands",)
