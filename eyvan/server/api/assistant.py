# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The fleet assistant: a bounded tool-calling loop the plane runs for one user.

The plane, not the browser or CLI, talks to an OpenAI-compatible
chat-completions endpoint. Fleet admins configure those endpoints ("providers")
from the dashboard; every user picks one, and one of its models, per chat.
A provider key is write-only: it is sealed at rest, decrypted only for the
turn that uses it, sent only as the provider's ``Authorization`` header, and
scrubbed from anything echoed back.

Every tool is a typed call into ``ControlPlaneService`` made with the *asking
user's* principal, so the assistant can never read or do more than that user
could with their own token. Acting tools do not act on their own:

* ``draft_workload`` returns a validated submission the user must confirm;
  confirming is an ordinary ``POST /api/v1/workloads`` by the user's client.
* ``file_request`` files a capacity/approval request for the fleet admins,
  which carries no authority until an admin resolves it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing as tp

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eyvan import assistant_knowledge as easydel
from eyvan.config import EyvanSettings
from eyvan.domain.sizing import (
    DEFAULT_CANDIDATE_TYPES,
    SizingRequest,
    model_shape_from_config,
    shape_from_parameter_count,
    size_job,
)
from eyvan.records import (
    AssistantConversationRecord,
    AssistantMessageRecord,
    AssistantProviderRecord,
    CapacityClass,
    CommandExecution,
    Principal,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    TeamRequestRecord,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.api.assistant_presets import provider_creation_fields, register_assistant_preset_routes
from eyvan.server.api.assistant_stream import (
    AssistantProviderError,
    ChatModel,
    OpenAICompatibleChat,
    StreamEvent,
    collect_turn,
    run_turn_stream,
)
from eyvan.server.api.auth import authenticated_principal
from eyvan.server.api.research_routes import plan_document, plan_summary
from eyvan.server.api.team_routes import require_team_mode
from eyvan.server.service._control_plane.assistant_conversations import (
    clip_message,
    clip_trace,
    conversation_title,
    new_conversation_id,
    new_message_id,
)
from eyvan.server.service.errors import EyvanServiceError

if tp.TYPE_CHECKING:
    from eyvan.server.service import ControlPlaneService

logger = logging.getLogger(__name__)

_Principal = tp.Annotated[Principal, Depends(authenticated_principal)]
_TeamOn = tp.Annotated[None, Depends(require_team_mode)]

#: Bounds on what one chat turn may carry and produce.
MAX_CLIENT_MESSAGES = 40
MAX_MESSAGE_CHARS = 8000
MAX_TOOL_RESULT_CHARS = 12000
MAX_LISTED_WORKLOADS = 25
PROVIDER_TIMEOUT_S = 120.0

SYSTEM_PROMPT = """You are the Eyvan fleet assistant for one signed-in user of an Eyvan control plane.
Eyvan schedules jobs (mostly EasyDeL/JAX training on TPUs) onto fleets of capacity pools.
Use the tools to answer from real data; never invent fleets, quotas or numbers.
Tools run with this user's own permissions; if a tool returns an error, say what is not allowed.
To size a job, call size_job (prefer the model's HuggingFace config fields, else a parameter count),
then compare the recommendation with what list_capacity shows the user can actually use.
EasyDeL training: you know every EasyDeL trainer config through list_easydel_trainers and
describe_easydel_trainer; use them instead of memory. When the user asks to train (e.g. "DPO an 8B model"),
first describe that trainer, then ASK the user for each essential argument they have not given
(for DPO: beta, loss_type, reference model or reference_free, max_length, max_prompt_length; plus model,
dataset, learning_rate, total_batch_size, training length, save_directory/save_steps and sharding),
stating the default for each so they can accept it. Do not draft until they have answered.
Then validate_easydel_arguments, fix what it flags, and draft_workload with easydel_trainer,
trainer_arguments and easydel_launch; size_job and list_capacity decide tpu_chips and the accelerator.
Research mode: when the user wants several runs explored ("run these four jobs and auto-research on
them"), agree the jobs and ASK for a budget (max jobs, chip-hours, wall-clock time, rounds), then call
propose_research_plan. Say plainly that nothing runs until they approve it; after approval the plane
runs it as them and get_research_plan reports progress.
You cannot submit anything. draft_workload prepares a submission the user must confirm themselves.
If the user needs capacity or an approval they do not have, offer to file_request to the fleet admins;
only file one when the user asks for it. Be brief and concrete."""


def _scrub(text: str, secret: str | None) -> str:
    return text.replace(secret, "[redacted]") if secret else text


def _jsonable(value: tp.Any) -> tp.Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


# ------------------------------------------------------------------ tools


def _schema(properties: dict[str, tp.Any], required: tp.Sequence[str] = ()) -> dict[str, tp.Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_SPECS: tuple[dict[str, tp.Any], ...] = (
    {
        "name": "list_fleets",
        "description": "Fleets this user may submit to in the current org/project.",
        "parameters": _schema({}),
    },
    {
        "name": "list_capacity",
        "description": "Enabled capacity pools this user can submit to, optionally filtered by accelerator type.",
        "parameters": _schema({"accelerator_type": _STR, "capacity_class": _STR}),
    },
    {
        "name": "list_capacity_pools",
        "description": "Every capacity pool in detail (fleet admins only).",
        "parameters": _schema({"fleet_name": _STR}),
    },
    {
        "name": "get_quota",
        "description": "Quotas that apply to this user in the current org/project, and whether any binds.",
        "parameters": _schema({}),
    },
    {
        "name": "list_workloads",
        "description": "This project's most recent workloads, optionally filtered by state.",
        "parameters": _schema({"state": _STR}),
    },
    {
        "name": "explain_workload",
        "description": "Why one workload is or is not placeable right now.",
        "parameters": _schema({"workload_id": _STR}, ["workload_id"]),
    },
    {
        "name": "size_job",
        "description": (
            "Estimate TPU memory for a training/inference job and the smallest slice that fits. Pass "
            "model_config (HuggingFace config.json fields) or parameters (count)."
        ),
        "parameters": _schema(
            {
                "model_config": {"type": "object"},
                "parameters": {"type": "number"},
                "dtype": {"type": "string", "enum": ["bfloat16", "float16", "float32"]},
                "optimizer": {"type": "string", "enum": ["adamw", "adafactor", "lion", "sgd", "none"]},
                "trainer": {"type": "string", "enum": ["pretrain", "sft", "dpo", "orpo", "grpo", "inference"]},
                "sequence_length": _INT,
                "global_batch_size": _INT,
                "gradient_checkpointing": {"type": "boolean"},
                "lora_rank": _INT,
                "generation_tokens": _INT,
                "num_generations": _INT,
                "accelerator_types": {"type": "array", "items": _STR},
            }
        ),
    },
    {
        "name": "list_easydel_trainers",
        "description": "Every EasyDeL trainer (sft, dpo, grpo, ...) with a summary and the arguments to confirm.",
        "parameters": _schema({}),
    },
    {
        "name": "describe_easydel_trainer",
        "description": (
            "Arguments of one EasyDeL trainer config with types, defaults and docs, inherited "
            "TrainingArguments included. query keeps fields matching any of its space-separated terms."
        ),
        "parameters": _schema(
            {"trainer": _STR, "query": _STR, "include_inherited": {"type": "boolean"}},
            ["trainer"],
        ),
    },
    {
        "name": "validate_easydel_arguments",
        "description": (
            "Check trainer arguments against the EasyDeL config: unknown names (with suggestions), "
            "invalid choices, wrong types and the essential arguments still missing."
        ),
        "parameters": _schema({"trainer": _STR, "arguments": {"type": "object"}}, ["trainer", "arguments"]),
    },
    {
        "name": "draft_workload",
        "description": (
            "Prepare (NOT submit) a workload for the user to confirm. Either command (a shell command) or "
            "easydel_trainer + trainer_arguments + easydel_launch (renders a runnable EasyDeL eLarge launch). "
            "tpu_chips is chips per replica."
        ),
        "parameters": _schema(
            {
                "name": _STR,
                "command": _STR,
                "easydel_trainer": _STR,
                "trainer_arguments": {"type": "object"},
                "easydel_launch": {
                    "type": "object",
                    "properties": {
                        "model": _STR,
                        "dataset": _STR,
                        "dataset_split": _STR,
                        "dataset_type": {
                            "type": "string",
                            "enum": ["huggingface", "json", "parquet", "csv", "tsv", "arrow", "text"],
                        },
                        "dataset_subset": _STR,
                        "content_field": _STR,
                        "tokenizer": _STR,
                        "reference_model": _STR,
                        "reward_model": _STR,
                        "teacher_model": _STR,
                        "target_model": _STR,
                        "num_devices": _INT,
                        "dtype": _STR,
                        "sharding": {
                            "type": "object",
                            "properties": {
                                "axis_dims": {"type": "array", "items": _INT},
                                "axis_names": {"type": "array", "items": _STR},
                            },
                        },
                    },
                    "required": ["model", "dataset"],
                },
                "accelerator_type": _STR,
                "capacity_class": {"type": "string", "enum": [item.value for item in CapacityClass]},
                "tpu_chips": _INT,
                "replicas": _INT,
                "priority_band": {"type": "string", "enum": [item.value for item in PriorityBand]},
                "checkpointable": {"type": "boolean"},
                "max_preemption_retries": _INT,
                "dependencies": {
                    "type": "array",
                    "items": _STR,
                    "description": (
                        "pip requirements installed for this job (e.g. EasyDeL from git). EasyDeL launches add the "
                        "CPU-only torch wheel automatically unless torch is listed here."
                    ),
                },
            },
            ["name"],
        ),
    },
    {
        "name": "list_my_requests",
        "description": "Capacity/approval requests (yours; fleet admins see everyone's).",
        "parameters": _schema({"state": {"type": "string", "enum": ["open", "approved", "declined", "withdrawn"]}}),
    },
    {
        "name": "file_request",
        "description": "File a capacity or approval request to the fleet admins. Only when the user asks.",
        "parameters": _schema(
            {
                "kind": {"type": "string", "enum": ["capacity", "approval"]},
                "summary": _STR,
                "details": {"type": "object"},
            },
            ["kind", "summary"],
        ),
    },
    {
        "name": "propose_research_plan",
        "description": (
            "Propose (NOT start) a research plan: several jobs plus a budget the plane enforces. Each job is "
            "{name, spec} or the draft_workload fields (command, or easydel_trainer + trainer_arguments + "
            "easydel_launch, with accelerator_type/tpu_chips). Nothing runs until the user approves it; then the "
            "plane submits the jobs as the user, monitors them and, when iterate is true, asks for further rounds "
            "inside the budget. Ask the user for the budget first."
        ),
        "parameters": _schema(
            {
                "title": _STR,
                "goal": _STR,
                "jobs": {"type": "array", "items": {"type": "object"}},
                "budget": {
                    "type": "object",
                    "properties": {
                        "max_jobs": _INT,
                        "max_chip_hours": {"type": "number"},
                        "max_wall_seconds": _INT,
                        "max_rounds": _INT,
                    },
                    "required": ["max_jobs", "max_chip_hours", "max_wall_seconds"],
                },
                "iterate": {"type": "boolean"},
            },
            ["title", "jobs", "budget"],
        ),
    },
    {
        "name": "list_research_plans",
        "description": "Research plans (yours; fleet admins see everyone's) with state, budget and spend.",
        "parameters": _schema(
            {
                "state": {
                    "type": "string",
                    "enum": [
                        "proposed",
                        "approved",
                        "running",
                        "stopping",
                        "finished",
                        "stopped",
                        "failed",
                        "over_budget",
                    ],
                }
            }
        ),
    },
    {
        "name": "get_research_plan",
        "description": "One research plan: each job's workload state and failure, spend against budget, notes.",
        "parameters": _schema({"plan_id": _STR}, ["plan_id"]),
    },
)
TOOL_NAMES = frozenset(spec["name"] for spec in TOOL_SPECS)


class ToolOutcome(tp.NamedTuple):
    name: str
    ok: bool
    result: tp.Any


class AssistantTools:
    """Typed reads and drafts executed as one principal in one org/project scope."""

    def __init__(self, service: ControlPlaneService, principal: Principal, *, org_id: str, project_id: str) -> None:
        self.service = service
        self.principal = principal
        self.org_id = org_id
        self.project_id = project_id
        self.drafts: list[dict[str, tp.Any]] = []
        self.requests: list[TeamRequestRecord] = []
        #: Research plans proposed this turn (summaries); each still needs the user's approval.
        self.plans: list[dict[str, tp.Any]] = []
        self.team_mode = False
        #: The chat's own provider and model, used for a proposed plan's iteration rounds.
        self.provider_id: str | None = None
        self.model: str | None = None

    async def call(self, name: str, arguments: dict[str, tp.Any]) -> ToolOutcome:
        if name not in TOOL_NAMES:
            return ToolOutcome(name, False, {"error": f"unknown tool {name!r}"})
        try:
            result = await getattr(self, f"_tool_{name}")(**arguments)
        except EyvanServiceError as exc:
            return ToolOutcome(name, False, {"error": f"{type(exc).__name__}: {exc}"})
        except (TypeError, ValueError, ValidationError) as exc:
            return ToolOutcome(name, False, {"error": f"invalid arguments: {str(exc)[:500]}"})
        return ToolOutcome(name, True, _jsonable(result))

    async def _tool_list_fleets(self) -> tp.Any:
        return await self.service.list_submit_fleets(self.principal, org_id=self.org_id, project_id=self.project_id)

    async def _tool_list_capacity(
        self, accelerator_type: str | None = None, capacity_class: str | None = None
    ) -> tp.Any:
        pools = await self.service.list_capacity(
            self.principal,
            org_id=self.org_id,
            project_id=self.project_id,
            accelerator_type=accelerator_type,
            capacity_class=CapacityClass(capacity_class) if capacity_class else None,
        )
        return [
            {
                "fleet": pool.fleet_name,
                "pool": pool.node_type,
                "accelerator_type": pool.accelerator_type,
                "capacity_class": pool.capacity_class,
                "region": pool.region,
                "zone": pool.zone,
                "min_cells": pool.min_cells,
                "max_cells": pool.max_cells,
            }
            for pool in pools
        ]

    async def _tool_list_capacity_pools(self, fleet_name: str | None = None) -> tp.Any:
        return await self.service.list_capacity_pools(self.principal, fleet_name=fleet_name)

    async def _tool_get_quota(self) -> tp.Any:
        quota = await self.service.get_quota(
            self.principal, org_id=self.org_id, project_id=self.project_id, principal_id=self.principal.subject
        )
        # ``unlimited`` is a property, which model_dump drops; without it the
        # model would have to infer "no limit" from an empty-looking chain.
        document = {**quota.model_dump(mode="json"), "unlimited": quota.unlimited}
        # Cross-project caps apply in addition to the project quota chain.
        caps = await self.service.team_tpu_cap(self.principal)
        if caps is not None:
            document["team_resource_caps"] = caps
            document["team_tpu_cap"] = {
                key: caps[key] for key in ("tier", "max_tpu_units", "tpu_units_in_use_or_queued")
            }
        return document

    async def _tool_list_workloads(self, state: str | None = None) -> tp.Any:
        workloads = await self.service.list_workloads(
            self.principal,
            org_id=self.org_id,
            project_id=self.project_id,
            state=WorkloadState(state) if state else None,
        )
        recent = sorted(workloads, key=lambda item: item.created_at, reverse=True)[:MAX_LISTED_WORKLOADS]
        return {
            "total": len(workloads),
            "workloads": [
                {
                    "id": item.id,
                    "name": item.spec.name,
                    "state": item.state,
                    "accelerator_type": item.spec.accelerator_type,
                    "created_by": item.created_by,
                    "created_at": item.created_at.isoformat(),
                }
                for item in recent
            ],
        }

    async def _tool_explain_workload(self, workload_id: str) -> tp.Any:
        return await self.service.explain_workload(self.principal, workload_id)

    async def _tool_size_job(
        self,
        model_config: dict[str, tp.Any] | None = None,
        parameters: float | None = None,
        accelerator_types: list[str] | None = None,
        **options: tp.Any,
    ) -> tp.Any:
        return size_request(
            model_config=model_config, parameters=parameters, accelerator_types=accelerator_types, **options
        )

    async def _tool_list_easydel_trainers(self) -> tp.Any:
        return {"catalog": easydel.catalog_info(), "trainers": easydel.list_trainers()}

    async def _tool_describe_easydel_trainer(
        self, trainer: str, query: str | None = None, include_inherited: bool = True
    ) -> tp.Any:
        return easydel.describe_trainer(trainer, query=query, include_inherited=include_inherited)

    async def _tool_validate_easydel_arguments(self, trainer: str, arguments: dict[str, tp.Any]) -> tp.Any:
        return easydel.validate_trainer_arguments(trainer, arguments)

    def _render_draft(
        self,
        name: str,
        command: str | None = None,
        accelerator_type: str | None = None,
        capacity_class: str | None = None,
        tpu_chips: int | None = None,
        replicas: int = 1,
        priority_band: str | None = None,
        checkpointable: bool = False,
        max_preemption_retries: int | None = None,
        easydel_trainer: str | None = None,
        trainer_arguments: dict[str, tp.Any] | None = None,
        easydel_launch: dict[str, tp.Any] | None = None,
        dependencies: list[str] | None = None,
    ) -> tuple[dict[str, tp.Any], dict[str, tp.Any] | None]:
        """Validate one job and return ``(tool outcome, draft or None when refused)``; records nothing."""

        if dependencies is not None and (
            not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies)
        ):
            raise ValueError("dependencies must be a list of requirement strings")
        declared = [item.strip() for item in dependencies or () if item.strip()]
        launch: dict[str, tp.Any] | None = None
        if easydel_trainer is not None:
            if command:
                raise ValueError("pass either command or easydel_trainer, not both")
            options = dict(easydel_launch or {})
            if not options.get("model") or not options.get("dataset"):
                raise ValueError("easydel_launch needs model and dataset")
            launch = easydel.render_launch(easydel_trainer, trainer_arguments or {}, **options)
            if not launch["runnable"]:
                # Not drafted: a launch eLarge would refuse, or silently run with
                # dropped arguments, must go back to the user first.
                refusal = {
                    "drafted": False,
                    "blocking": launch["blocking"],
                    "missing_essential": launch["validation"]["missing_essential"],
                    "suggestions": launch["validation"]["suggestions"],
                }
                return refusal, None
            command = launch["script"].strip()
            # The launch's own requirements (CPU torch for the HF loader) unless
            # the user already pinned that project: their pin wins outright.
            pinned = {easydel.requirement_name(item) for item in declared}
            declared.extend(
                item for item in launch.get("dependencies", ()) if easydel.requirement_name(item) not in pinned
            )
        if not command:
            raise ValueError("pass command, or easydel_trainer with easydel_launch")
        fields: dict[str, tp.Any] = {
            "name": name,
            "execution": CommandExecution(command=command),
            "accelerator_type": accelerator_type,
            "checkpointable": checkpointable,
        }
        if capacity_class:
            fields["capacity_class"] = CapacityClass(capacity_class)
        if priority_band:
            fields["priority_band"] = PriorityBand(priority_band)
        if max_preemption_retries is not None:
            fields["max_preemption_retries"] = max_preemption_retries
        if tpu_chips:
            fields["gang"] = ResourceGang(replicas=replicas, per_replica=ResourceVector(tpu=tpu_chips))
        if declared:
            fields["dependencies"] = tuple(dict.fromkeys(declared))
        spec = WorkloadDefinition(**fields)
        document = spec.model_dump(mode="json", exclude_defaults=True)
        # The execution's ``kind`` is its union discriminator and also its
        # default, so exclude_defaults would drop it and make the draft
        # unsubmittable. Keep the execution block whole.
        document["execution"] = spec.execution.model_dump(mode="json")
        draft: dict[str, tp.Any] = {"org_id": self.org_id, "project_id": self.project_id, "spec": document}
        outcome: dict[str, tp.Any] = {
            "drafted": True,
            "submitted": False,
            "note": "the user must confirm this draft to submit it",
        }
        if launch is not None:
            draft["easydel"] = {
                "trainer": launch["trainer"],
                "config_yaml": launch["config_yaml"],
                "command": launch["command"],
                "assumptions": launch["assumptions"],
            }
            outcome["easydel"] = {
                "assumptions": launch["assumptions"],
                "missing_essential": launch["validation"]["missing_essential"],
                "config_yaml": launch["config_yaml"],
            }
        return outcome, draft

    async def _tool_draft_workload(self, **arguments: tp.Any) -> tp.Any:
        outcome, draft = self._render_draft(**arguments)
        if draft is not None:
            self.drafts.append(draft)
        return outcome

    async def _tool_propose_research_plan(
        self,
        title: str,
        jobs: list[tp.Any],
        budget: dict[str, tp.Any],
        goal: str = "",
        iterate: bool = False,
    ) -> tp.Any:
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("jobs must be a non-empty list")
        planned: list[dict[str, tp.Any]] = []
        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise ValueError(f"job {index} must be an object")
            if "spec" in job:
                spec = job["spec"]
                name = job.get("name") or (spec.get("name") if isinstance(spec, dict) else None)
                if set(job) - {"name", "spec"} or not name:
                    raise ValueError(f"job {index}: pass {{name, spec}} or the draft_workload fields, not both")
                planned.append({"name": str(name), "spec": spec})
                continue
            # The same validation a single draft gets, EasyDeL launches included.
            outcome, draft = self._render_draft(**job)
            if draft is None:
                return {"proposed": False, "job": index, **outcome}
            planned.append({"name": str(job["name"]), "spec": draft["spec"]})
        plan = await self.service.propose_research_plan(
            self.principal,
            title=title,
            goal=goal,
            org_id=self.org_id,
            project_id=self.project_id,
            jobs=planned,
            budget=budget,
            iterate=iterate,
            provider_id=self.provider_id if iterate else None,
            model=self.model if iterate else None,
        )
        self.plans.append(plan_summary(plan))
        return {
            "proposed": True,
            "approved": False,
            "plan_id": plan.id,
            "jobs": len(plan.jobs),
            "budget": plan.budget.model_dump(mode="json"),
            "note": "nothing runs until the user approves this plan (dashboard or `eyvan research approve`)",
        }

    async def _tool_list_research_plans(self, state: str | None = None) -> tp.Any:
        plans = await self.service.list_research_plans(self.principal, state=state)
        return [plan_summary(plan) for plan in plans]

    async def _tool_get_research_plan(self, plan_id: str) -> tp.Any:
        plan, workloads = await self.service.describe_research_plan(self.principal, plan_id)
        return plan_document(plan, workloads)

    async def _tool_list_my_requests(self, state: str | None = None) -> tp.Any:
        if not self.team_mode:
            return {"error": "requests need team mode on this plane"}
        return await self.service.list_team_requests(self.principal, state=state)

    async def _tool_file_request(self, kind: str, summary: str, details: dict[str, tp.Any] | None = None) -> tp.Any:
        if not self.team_mode:
            return {"error": "requests need team mode on this plane"}
        request = await self.service.file_team_request(
            self.principal,
            kind=kind,
            summary=summary,
            details=details,
            org_id=self.org_id,
            project_id=self.project_id,
        )
        self.requests.append(request)
        return {"filed": True, "id": request.id, "state": request.state}


def size_request(
    *,
    model_config: dict[str, tp.Any] | None = None,
    parameters: float | None = None,
    accelerator_types: list[str] | None = None,
    **options: tp.Any,
) -> dict[str, tp.Any]:
    """Run the sizing model from a config or a parameter count (shared by the route and the tool)."""

    if model_config:
        shape = model_shape_from_config(model_config)
    elif parameters:
        shape = shape_from_parameter_count(int(parameters))
    else:
        raise ValueError("pass model_config (HuggingFace config fields) or parameters")
    request = SizingRequest(shape=shape, **{key: value for key, value in options.items() if value is not None})
    return size_job(request, tuple(accelerator_types) if accelerator_types else DEFAULT_CANDIDATE_TYPES)


# ------------------------------------------------------------------- loop


#: Kept for callers that imported the one-shot provider; it now streams over SSE.
AssistantProvider = OpenAICompatibleChat


def _keep(request: Request, task: asyncio.Future[tp.Any]) -> asyncio.Future[tp.Any]:
    """Hold a background save until it finishes, so a stopped stream never drops it mid-write."""

    pending: set[asyncio.Future[tp.Any]] = request.app.state.__dict__.setdefault("assistant_pending_saves", set())
    pending.add(task)
    task.add_done_callback(pending.discard)
    return task


def _conversation_saves(request: Request) -> dict[str, asyncio.Future[tp.Any]]:
    """The newest unfinished save per conversation, so a follow-up turn reads after it lands."""

    return tp.cast(
        dict[str, asyncio.Future[tp.Any]], request.app.state.__dict__.setdefault("assistant_conversation_saves", {})
    )


def _track_save(request: Request, conversation_id: str, task: asyncio.Future[tp.Any]) -> None:
    saves = _conversation_saves(request)
    saves[conversation_id] = task

    def forget(done: asyncio.Future[tp.Any]) -> None:
        if saves.get(conversation_id) is done:
            del saves[conversation_id]

    task.add_done_callback(forget)


async def _await_pending_save(request: Request, conversation_id: str) -> None:
    pending = _conversation_saves(request).get(conversation_id)
    if pending is not None:
        # The previous turn's save; its own failure was already logged.
        await asyncio.wait({pending})


def system_prompt(tools: AssistantTools) -> str:
    return f"{SYSTEM_PROMPT}\nUser: {tools.principal.subject}. Scope: {tools.org_id}/{tools.project_id}."


async def run_chat(
    provider: ChatModel, tools: AssistantTools, history: list[dict[str, str]], *, max_steps: int
) -> dict[str, tp.Any]:
    """Run one whole turn and return the reply plus what tools did (the non-streaming answer)."""

    events = run_turn_stream(
        provider, tools, list(history), system=system_prompt(tools), tool_specs=list(TOOL_SPECS), max_steps=max_steps
    )
    return await collect_turn(events)


def chat_model_for(record: AssistantProviderRecord, model: str, key: str | None, request: Request) -> ChatModel:
    """The streaming chat model for one resolved provider: Claude Code, or an OpenAI-compatible endpoint."""

    if record.preset == "claude-code":
        from eyvan.server.api.assistant_claude_code import claude_code_chat

        return tp.cast(ChatModel, claude_code_chat(model, request))
    return OpenAICompatibleChat(
        base_url=record.base_url,
        model=model,
        api_key=key,
        # A test/embedding seam, not configuration: a fake provider transport.
        transport=getattr(request.app.state, "assistant_transport", None),
    )


# ----------------------------------------------------------------- routes


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(_Body):
    #: Only the visible conversation travels from clients; system and tool
    #: messages are the plane's own and cannot be injected.
    role: tp.Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatBody(_Body):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_CLIENT_MESSAGES)
    org_id: str = Field(default="default", min_length=1, max_length=128)
    project_id: str = Field(default="default", min_length=1, max_length=128)
    #: Which configured provider and model answer; omitted means the first
    #: enabled provider by name and its default model.
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)


class StreamChatBody(_Body):
    #: Only the new message travels; earlier turns come from the saved conversation.
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(default=None, max_length=32)
    org_id: str = Field(default="default", min_length=1, max_length=128)
    project_id: str = Field(default="default", min_length=1, max_length=128)
    provider_id: str | None = Field(default=None, max_length=32)
    model: str | None = Field(default=None, max_length=128)


class RenameConversationBody(_Body):
    title: str = Field(min_length=1, max_length=200)


def conversation_summary(record: AssistantConversationRecord) -> dict[str, tp.Any]:
    return {
        "id": record.id,
        "title": record.title,
        "provider_id": record.provider_id,
        "model": record.model,
        "org_id": record.org_id,
        "project_id": record.project_id,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
        "message_count": len(record.messages),
    }


class ProviderCreateBody(_Body):
    #: A built-in preset (``GET /api/v1/assistant/presets``) fills name, URL and models.
    preset: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    #: Write-only; never returned.
    api_key: str | None = Field(default=None, max_length=4096)
    models: list[str] | None = Field(default=None, min_length=1, max_length=32)
    default_model: str | None = Field(default=None, max_length=128)
    max_steps: int = Field(default=6, ge=1, le=20)
    enabled: bool = True


class ProviderUpdateBody(_Body):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    base_url: str | None = Field(default=None, min_length=1, max_length=512)
    api_key: str | None = Field(default=None, max_length=4096)
    clear_api_key: bool = False
    models: list[str] | None = Field(default=None, min_length=1, max_length=32)
    default_model: str | None = Field(default=None, max_length=128)
    max_steps: int | None = Field(default=None, ge=1, le=20)
    enabled: bool | None = None


def provider_document(provider: AssistantProviderRecord, *, admin: bool) -> dict[str, tp.Any]:
    """Public view of a provider: never the key or its ciphertext."""

    document: dict[str, tp.Any] = {
        "id": provider.id,
        "name": provider.name,
        "preset": provider.preset,
        "models": list(provider.models),
        "default_model": provider.default_model,
    }
    if admin:
        document.update(
            base_url=provider.base_url,
            max_steps=provider.max_steps,
            enabled=provider.enabled,
            has_api_key=provider.api_key_ciphertext is not None,
            created_by=provider.created_by,
            created_at=provider.created_at.isoformat(),
            updated_at=provider.updated_at.isoformat(),
        )
    return document


class SizeBody(_Body):
    model_config_fields: dict[str, tp.Any] | None = Field(default=None, alias="model_config")
    parameters: float | None = Field(default=None, gt=0)
    dtype: str | None = None
    optimizer: str | None = None
    trainer: str | None = None
    sequence_length: int | None = Field(default=None, ge=1)
    global_batch_size: int | None = Field(default=None, ge=1)
    gradient_checkpointing: bool | None = None
    lora_rank: int | None = Field(default=None, ge=1)
    generation_tokens: int | None = Field(default=None, ge=0)
    num_generations: int | None = Field(default=None, ge=1)
    accelerator_types: list[str] | None = None


class RequestBody(_Body):
    kind: tp.Literal["capacity", "approval"]
    summary: str = Field(min_length=1, max_length=2000)
    details: dict[str, tp.Any] = Field(default_factory=dict)
    org_id: str | None = Field(default=None, max_length=128)
    project_id: str | None = Field(default=None, max_length=128)


class ResolveBody(_Body):
    resolution: tp.Literal["approved", "declined", "withdrawn"]
    note: str = Field(default="", max_length=2000)


def register_assistant_routes(
    app: FastAPI,
    *,
    settings: EyvanSettings,
    service: ControlPlaneService,
) -> None:
    """Register assistant, provider, sizing and team-request routes.

    Providers live in the plane's own store and are managed by fleet admins
    over this API (the dashboard's Settings page), usually from a built-in
    preset plus a key. With none enabled, chat answers 404 and status reports
    disabled.
    """

    register_assistant_preset_routes(app, settings=settings, service=service)

    @app.get("/api/v1/assistant/status", tags=["assistant"])
    async def assistant_status(principal: _Principal) -> dict[str, tp.Any]:
        """Whether any provider is available to the caller, and its default model (never a key)."""

        providers = [item for item in await service.list_assistant_providers(principal) if item.enabled]
        first = providers[0] if providers else None
        return {
            "enabled": first is not None,
            "model": first.default_model if first is not None else None,
            "providers": len(providers),
        }

    @app.get("/api/v1/assistant/providers", tags=["assistant"])
    async def list_assistant_providers(principal: _Principal) -> dict[str, tp.Any]:
        """Configured providers: every field but the key for fleet admins, names and models for others."""

        admin = service.can_manage_assistant_providers(principal)
        providers = await service.list_assistant_providers(principal)
        return {"providers": [provider_document(item, admin=admin) for item in providers], "can_manage": admin}

    @app.post("/api/v1/assistant/providers", status_code=201, tags=["assistant"])
    async def create_assistant_provider(
        body: ProviderCreateBody, request: Request, principal: _Principal
    ) -> dict[str, tp.Any]:
        """Add a provider (fleet admins); a preset needs only a key. The key is sealed and never returned."""

        if not service.can_manage_assistant_providers(principal):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="only fleet admins configure assistant providers"
            )
        fields = await provider_creation_fields(
            service,
            principal,
            preset_id=body.preset,
            name=body.name,
            base_url=body.base_url,
            models=body.models,
            api_key=body.api_key,
            transport=getattr(request.app.state, "assistant_transport", None),
            claude_code_executable=getattr(request.app.state, "claude_code_executable", None),
        )
        created = await service.create_assistant_provider(
            principal,
            **fields,
            api_key=body.api_key,
            default_model=body.default_model,
            max_steps=body.max_steps,
            enabled=body.enabled,
        )
        return provider_document(created, admin=True)

    @app.put("/api/v1/assistant/providers/{provider_id}", tags=["assistant"])
    async def update_assistant_provider(
        provider_id: str, body: ProviderUpdateBody, principal: _Principal
    ) -> dict[str, tp.Any]:
        """Change a provider (fleet admins); an omitted ``api_key`` keeps the stored one."""

        updated = await service.update_assistant_provider(
            principal,
            provider_id,
            name=body.name,
            base_url=body.base_url,
            models=body.models,
            default_model=body.default_model,
            max_steps=body.max_steps,
            enabled=body.enabled,
            api_key=body.api_key,
            clear_api_key=body.clear_api_key,
        )
        return provider_document(updated, admin=True)

    @app.delete("/api/v1/assistant/providers/{provider_id}", tags=["assistant"])
    async def delete_assistant_provider(provider_id: str, principal: _Principal) -> dict[str, tp.Any]:
        """Remove a provider (fleet admins)."""

        await service.delete_assistant_provider(principal, provider_id)
        return {"deleted": provider_id}

    @app.post("/api/v1/assistant/chat", tags=["assistant"])
    async def assistant_chat(body: ChatBody, request: Request, principal: _Principal) -> dict[str, tp.Any]:
        """Run one assistant turn as the caller with the chosen provider; drafts are never submitted."""

        record, model, key = await service.resolve_assistant_provider(
            principal, provider_id=body.provider_id, model=body.model
        )
        chosen = chat_model_for(record, model, key, request)
        tools = AssistantTools(service, principal, org_id=body.org_id, project_id=body.project_id)
        tools.team_mode = settings.team_mode
        tools.provider_id, tools.model = record.id, model
        history = [message.model_dump() for message in body.messages]
        try:
            turn = await run_chat(chosen, tools, history, max_steps=record.max_steps)
        except AssistantProviderError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from None
        return {**turn, "provider_id": record.id, "model": model}

    @app.post("/api/v1/assistant/chat/stream", tags=["assistant"])
    async def assistant_chat_stream(body: StreamChatBody, request: Request, principal: _Principal) -> StreamingResponse:
        """Run one turn and stream every step, tool call and text delta as NDJSON; save it to the conversation.

        Refusals (unknown provider or conversation, unlisted model) are ordinary
        HTTP errors raised before the stream starts; a provider failure after it
        started arrives as a final ``error`` event and is saved on the turn.
        """

        record, model, key = await service.resolve_assistant_provider(
            principal, provider_id=body.provider_id, model=body.model
        )
        user_message = AssistantMessageRecord(id=new_message_id(), role="user", content=clip_message(body.message))
        prior: tp.Sequence[AssistantMessageRecord]
        if body.conversation_id is not None:
            # A snapshot read: refusing someone else's chat must not wait behind a commit.
            # Only this conversation's own unfinished save is waited for, so the
            # previous answer is in the history this turn sends.
            await _await_pending_save(request, body.conversation_id)
            existing = await service.get_assistant_conversation(principal, body.conversation_id)
            conversation_id, title, prior = existing.id, existing.title, existing.messages
            save_user = service.append_assistant_messages(principal, conversation_id, [user_message])
        else:
            conversation_id, prior = new_conversation_id(), ()
            title = conversation_title(body.message)
            # One commit: the conversation is created already holding this turn.
            save_user = service.start_assistant_conversation(
                principal,
                first_message=body.message,
                org_id=body.org_id,
                project_id=body.project_id,
                provider_id=record.id,
                model=model,
                messages=[user_message],
                conversation_id=conversation_id,
            )
        # The user's turn is written while the model already answers: on a large
        # local store a commit can wait seconds for the transaction lock, and
        # the first streamed byte must not wait with it.
        user_saved = _keep(request, asyncio.ensure_future(save_user))
        _track_save(request, conversation_id, user_saved)
        history = [
            {"role": item.role, "content": item.content} for item in prior if item.content.strip() and item.error is None
        ][-(MAX_CLIENT_MESSAGES - 1) :]
        history.append({"role": "user", "content": body.message})
        chosen = chat_model_for(record, model, key, request)
        tools = AssistantTools(service, principal, org_id=body.org_id, project_id=body.project_id)
        tools.team_mode = settings.team_mode
        tools.provider_id, tools.model = record.id, model

        async def events() -> tp.AsyncIterator[bytes]:
            def line(event: StreamEvent) -> bytes:
                return (json.dumps(event, default=str) + "\n").encode("utf-8")

            yield line(
                {
                    "type": "start",
                    "conversation_id": conversation_id,
                    "provider_id": record.id,
                    "model": model,
                    "title": title,
                    "user_message_id": user_message.id,
                }
            )
            trace: list[dict[str, tp.Any]] = []
            by_id: dict[str, dict[str, tp.Any]] = {}
            message_id = new_message_id()

            async def write_answer(answer: AssistantMessageRecord) -> None:
                # After the user's turn, so the saved order is always question then answer.
                try:
                    await user_saved
                except Exception:
                    logger.warning("assistant user turn was not saved", exc_info=True)
                    return
                await service.append_assistant_messages(
                    principal, conversation_id, [answer], provider_id=record.id, model=model
                )

            async def save_answer(answer: AssistantMessageRecord) -> None:
                task = _keep(request, asyncio.ensure_future(write_answer(answer)))
                _track_save(request, conversation_id, task)
                # Shielded: a client that disconnects right after ``done`` must not cancel the write.
                await asyncio.shield(task)

            try:
                async for event in run_turn_stream(
                    chosen,
                    tools,
                    history,
                    system=system_prompt(tools),
                    tool_specs=list(TOOL_SPECS),
                    max_steps=record.max_steps,
                ):
                    kind = event.get("type")
                    if kind == "tool_call":
                        item = {"type": "tool", "name": event["name"], "arguments": event.get("arguments"), "ok": None}
                        by_id[str(event.get("id"))] = item
                        trace.append(item)
                    elif kind == "tool_result":
                        item = by_id.get(str(event.get("id"))) or {"type": "tool", "name": event["name"]}
                        item.update(ok=event.get("ok"), summary=event.get("summary"))
                    elif kind == "done":
                        # The client sees ``done`` at once; the save happens after
                        # it, before the response body closes, so a client that
                        # reads to the end still finds the turn saved.
                        yield line({**event, "message_id": message_id})
                        await save_answer(
                            AssistantMessageRecord(
                                id=message_id,
                                role="assistant",
                                content=clip_message(str(event.get("reply") or "")),
                                provider_id=record.id,
                                model=model,
                                trace=clip_trace(trace),
                                drafts=tuple(event.get("drafts") or ())[:32],
                                requests=tuple(event.get("requests") or ())[:32],
                                plans=tuple(event.get("plans") or ())[:8],
                            )
                        )
                        continue
                    yield line(event)
            except AssistantProviderError as exc:
                yield line({"type": "error", "detail": str(exc), "message_id": message_id})
                await save_answer(
                    AssistantMessageRecord(
                        id=message_id,
                        role="assistant",
                        content="",
                        provider_id=record.id,
                        model=model,
                        trace=clip_trace(trace),
                        error=str(exc)[:2000],
                    )
                )

        # A client that stops reading (the dock's Stop button) cancels this
        # generator; the user turn is still saved (its task is kept on the app)
        # and no answer is recorded.
        return StreamingResponse(
            events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
        )

    @app.get("/api/v1/assistant/conversations", tags=["assistant"])
    async def list_assistant_conversations(principal: _Principal) -> dict[str, tp.Any]:
        """The caller's own saved chats, newest first (never anyone else's)."""

        records = await service.list_assistant_conversations(principal)
        return {"conversations": [conversation_summary(item) for item in records]}

    @app.get("/api/v1/assistant/conversations/{conversation_id}", tags=["assistant"])
    async def get_assistant_conversation(conversation_id: str, principal: _Principal) -> dict[str, tp.Any]:
        """One of the caller's chats with every message, its trace, drafts and requests."""

        record = await service.get_assistant_conversation(principal, conversation_id)
        return {
            **conversation_summary(record),
            "messages": [item.model_dump(mode="json", exclude_none=True) for item in record.messages],
        }

    @app.put("/api/v1/assistant/conversations/{conversation_id}", tags=["assistant"])
    async def rename_assistant_conversation(
        conversation_id: str, body: RenameConversationBody, principal: _Principal
    ) -> dict[str, tp.Any]:
        """Rename one of the caller's chats."""

        record = await service.rename_assistant_conversation(principal, conversation_id, title=body.title)
        return conversation_summary(record)

    @app.delete("/api/v1/assistant/conversations/{conversation_id}", tags=["assistant"])
    async def delete_assistant_conversation(conversation_id: str, principal: _Principal) -> dict[str, tp.Any]:
        """Delete one of the caller's chats."""

        await service.delete_assistant_conversation(principal, conversation_id)
        return {"deleted": conversation_id}

    @app.post("/api/v1/assistant/size", tags=["assistant"])
    async def assistant_size(body: SizeBody, principal: _Principal) -> dict[str, tp.Any]:
        """Estimate per-chip TPU memory and the smallest fitting slice (no provider involved)."""

        del principal
        options = body.model_dump(exclude={"model_config_fields", "parameters", "accelerator_types"})
        try:
            return size_request(
                model_config=body.model_config_fields,
                parameters=body.parameters,
                accelerator_types=body.accelerator_types,
                **options,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    @app.post("/api/v1/team/requests", status_code=201, tags=["team"])
    async def file_team_request(body: RequestBody, principal: _Principal, _on: _TeamOn) -> dict[str, tp.Any]:
        record = await service.file_team_request(
            principal,
            kind=body.kind,
            summary=body.summary,
            details=body.details,
            org_id=body.org_id,
            project_id=body.project_id,
        )
        return record.model_dump(mode="json")

    @app.get("/api/v1/team/requests", tags=["team"])
    async def list_team_requests(principal: _Principal, _on: _TeamOn, state: str | None = None) -> dict[str, tp.Any]:
        records = await service.list_team_requests(principal, state=state)
        return {"requests": [record.model_dump(mode="json") for record in records]}

    @app.post("/api/v1/team/requests/{request_id}/resolve", tags=["team"])
    async def resolve_team_request(
        request_id: str, body: ResolveBody, principal: _Principal, _on: _TeamOn
    ) -> dict[str, tp.Any]:
        record = await service.resolve_team_request(principal, request_id, resolution=body.resolution, note=body.note)
        return record.model_dump(mode="json")


__all__ = (
    "SYSTEM_PROMPT",
    "TOOL_SPECS",
    "AssistantProvider",
    "AssistantProviderError",
    "AssistantTools",
    "chat_model_for",
    "register_assistant_routes",
    "run_chat",
    "size_request",
    "system_prompt",
)
