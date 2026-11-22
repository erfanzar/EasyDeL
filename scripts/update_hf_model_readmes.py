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
"""Regenerate and optionally push Hugging Face model-card READMEs in bulk.

For each selected repo (one of: ``--repo-id`` flags, lines from a
``--repos-file``, or every model under ``--author``), the script:

1. Fetches the repo's ``config.json`` via the HF raw-file endpoint.
2. Fetches the existing ``README.md`` to extract the previous title /
   description / inferred source repo (so the regenerated card keeps the
   right "converted from <source>" attribution).
3. Calls ``easydel.utils.readme_generator.ReadmeGenerator`` to render a fresh
   README from a :class:`ModelInfo` populated from the config.
4. Optionally writes the new README to ``--output-dir`` for review and pushes
   the change back to the repo via a sparse ``git clone`` + ``git push``,
   authenticated using an in-memory ``GIT_ASKPASS`` script that reads
   ``HF_TOKEN`` from the environment.

When the regenerated README is byte-identical to the existing one, the repo
is skipped. ``--dry-run`` skips the push step entirely.

Side effects:
    - Issues authenticated HTTP requests against the HF Hub.
    - Spawns ``git clone`` / ``git commit`` / ``git push`` subprocesses
      against per-repo temporary directories.
    - May write per-repo previews under ``--output-dir``.
    - Sets HF git auth env vars (``HF_TOKEN``, ``GIT_ASKPASS``,
      ``GIT_LFS_SKIP_SMUDGE``, ``GIT_TERMINAL_PROMPT``) inside the per-repo
      subprocess environment only.

How to use

Regenerate Hugging Face model-card `README.md` files from each repo's `config.json`,
using `easydel/utils/readme_generator.py`, and optionally push the changes back to the Hub.

Update all repos under an org/user:

  python scripts/update_hf_model_readmes.py --author EasyDeL --token $HF_TOKEN

Update a curated list and preview without pushing:

  python scripts/update_hf_model_readmes.py \
    --repos-file models.txt \
    --dry-run \
    --output-dir /tmp/easydel-readmes
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HF_BASE_URL = "https://huggingface.co"
GENERIC_DESCRIPTION = "A model compatible with the EasyDeL JAX stack."
IMAGE_TEXT_TO_TEXT_MODEL_TYPES = frozenset(
    {
        "gemma3",
        "gemma4",
        "glm4v",
        "glm4v_moe",
        "glm46v",
        "idefics",
        "idefics2",
        "kimi_vl",
        "llava",
        "llama4",
        "mistral3",
        "paligemma",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen2vl",
        "qwen2_5vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
    }
)
H1_RE = re.compile(r"<h1\s+align=\"center\">\s*(.*?)\s*</h1>", flags=re.IGNORECASE | re.DOTALL)
CENTER_DIV_RE = re.compile(r"<div\s+align=\"center\">\s*(.*?)\s*</div>", flags=re.IGNORECASE | re.DOTALL)
SOURCE_PLAIN_RE = re.compile(r"converted from\s+(.+?)\.\s*$", flags=re.IGNORECASE)
SOURCE_LINK_RE = re.compile(
    r"converted from\s+\[(?P<label>[^\]]+)\]\((?P<url>[^)]+)\)\.?\s*$",
    flags=re.IGNORECASE,
)
LOCAL_PATH_RE = re.compile(r"(^|[\s`])(?:/|\./|\.\./|[A-Za-z]:[\\/])")


@dataclass
class ExistingCardContext:
    """Pieces of an existing model card preserved across regeneration.

    Attributes:
        title: Centered ``<h1>`` title extracted from the existing README,
            normalized of HTML tags / whitespace.
        description: First centered ``<div>`` paragraph after the title.
        source_repo: Inferred upstream HF repo id this checkpoint was
            converted from (parsed from the description or, as a fallback,
            from the title when it looks like ``owner/name``).
    """

    title: str | None = None
    description: str | None = None
    source_repo: str | None = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the README updater.

    Returns:
        argparse.ArgumentParser: Configured parser. See the help text for the
            full list of flags (``--repo-id`` / ``--repos-file`` / ``--author``
            select repos; ``--token`` / ``--revision`` configure HF access;
            ``--dry-run`` / ``--output-dir`` control preview vs push;
            ``--template-dir`` / ``--template-name`` override the Jinja
            template; ``--commit-message`` / ``--git-user-name`` /
            ``--git-user-email`` control the push commit).
    """
    parser = argparse.ArgumentParser(
        description="Regenerate and optionally push HF model-card README.md files from each repo's config.json."
    )
    parser.add_argument("--repo-id", action="append", default=[], help="Model repo id (repeatable).")
    parser.add_argument("--repos-file", help="Path to a file with one repo id per line.")
    parser.add_argument("--author", help="Update all model repos owned by this HF user/org.")
    parser.add_argument("--match", help="Only process repo ids containing this substring.")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token (defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision/branch to read from and push to.",
    )
    parser.add_argument(
        "--commit-message",
        default="Update README.md (EasyDeL auto-generated)",
        help="Commit message for pushes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate but do not push changes.")
    parser.add_argument("--output-dir", help="Optional directory to write generated READMEs for review.")
    parser.add_argument("--template-dir", help="Optional custom Jinja template directory.")
    parser.add_argument("--template-name", help="Optional template filename inside template_dir.")
    parser.add_argument(
        "--git-user-name",
        default="EasyDeL README Bot",
        help="Git author name for pushes.",
    )
    parser.add_argument(
        "--git-user-email",
        default="easydel-readme-bot@users.noreply.huggingface.co",
        help="Git author email for pushes.",
    )
    return parser


def _load_readme_generator():
    """Import ``easydel/utils/readme_generator.py`` from a relative file path.

    Avoids the regular package import so the script remains usable even when
    a partially-installed EasyDeL would otherwise fail at import time.

    Returns:
        ModuleType: The loaded ``readme_generator`` module, exposing
            ``ReadmeGenerator`` and ``ModelInfo``.

    Raises:
        RuntimeError: If the module spec cannot be created.
    """
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "easydel" / "utils" / "readme_generator.py"
    spec = importlib.util.spec_from_file_location("easydel_readme_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import readme generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_headers(token: str | None) -> dict[str, str]:
    """Build HTTP headers for an HF API or raw-file request.

    Args:
        token: Optional bearer token. When set it is attached as
            ``Authorization: Bearer <token>``.

    Returns:
        dict[str, str]: Header mapping with ``User-Agent``, ``Accept``, and
            (when provided) ``Authorization`` entries.
    """
    headers = {
        "User-Agent": "EasyDeL/update_hf_model_readmes.py",
        "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_bytes(url: str, *, token: str | None, required: bool = True) -> bytes | None:
    """GET ``url`` and return the raw response body.

    Args:
        url: Absolute HTTPS URL.
        token: Optional HF bearer token.
        required: When ``False`` a 404 response returns ``None`` instead of
            raising; any other failure still raises.

    Returns:
        bytes | None: Response body, or ``None`` if 404 and ``required`` is
            ``False``.

    Raises:
        RuntimeError: For HTTP errors other than the tolerated 404, and for
            URL-level errors (DNS, connection refused, ...).
    """
    request = urllib.request.Request(url, headers=_build_headers(token))
    try:
        with urllib.request.urlopen(request) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404 and not required:
            return None
        details = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"{url} -> HTTP {error.code}: {details or error.reason}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{url} -> {error.reason}") from error


def _fetch_text(url: str, *, token: str | None, required: bool = True) -> str | None:
    """GET ``url`` and decode the body as UTF-8 text.

    Args:
        url: Absolute HTTPS URL.
        token: Optional HF bearer token.
        required: See :func:`_fetch_bytes`.

    Returns:
        str | None: Decoded text or ``None`` for tolerated 404 responses.
    """
    payload = _fetch_bytes(url, token=token, required=required)
    if payload is None:
        return None
    return payload.decode("utf-8", "replace")


def _fetch_json(url: str, *, token: str | None, required: bool = True) -> Any:
    """GET ``url`` and parse the body as JSON.

    Args:
        url: Absolute HTTPS URL.
        token: Optional HF bearer token.
        required: See :func:`_fetch_bytes`.

    Returns:
        Any: Parsed JSON value, or ``None`` for tolerated 404 responses.
    """
    payload = _fetch_bytes(url, token=token, required=required)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _hf_api_models_url(author: str) -> str:
    """Build the HF API URL listing every model under ``author``.

    Args:
        author: HF user or org name.

    Returns:
        str: Absolute ``/api/models?author=...&limit=1000`` URL.
    """
    query = urllib.parse.urlencode({"author": author, "limit": 1000})
    return f"{HF_BASE_URL}/api/models?{query}"


def _hf_raw_file_url(repo_id: str, filename: str, revision: str) -> str:
    """Build the HF raw-file URL for ``filename`` in ``repo_id`` at ``revision``.

    Args:
        repo_id: HF repo id (``"owner/name"``).
        filename: File path within the repo.
        revision: Branch / tag / commit.

    Returns:
        str: Absolute URL safe for HTTP GET.
    """
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_filename = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"{HF_BASE_URL}/{quoted_repo}/raw/{quoted_revision}/{quoted_filename}"


def _read_text(path: str | os.PathLike[str]) -> str:
    """Read a file as UTF-8 text.

    Args:
        path: File path.

    Returns:
        str: File contents decoded as UTF-8.
    """
    return Path(path).read_text(encoding="utf-8")


def _collect_repo_ids(args: argparse.Namespace) -> list[str]:
    """Collate, dedup, and optionally filter the set of repos to update.

    Combines ``args.repo_id``, repos listed in ``args.repos_file`` (with ``#``
    comments stripped), and every model owned by ``args.author`` (queried via
    the HF API). Applies ``args.match`` substring filtering last.

    Args:
        args: Parsed CLI namespace.

    Returns:
        list[str]: Unique repo ids in encounter order.

    Raises:
        TypeError: If the HF models API response is not a list.
    """
    repo_ids: list[str] = list(args.repo_id or [])

    if args.repos_file:
        for line in _read_text(args.repos_file).splitlines():
            repo_id = line.strip()
            if not repo_id or repo_id.startswith("#"):
                continue
            repo_ids.append(repo_id)

    if args.author:
        models = _fetch_json(_hf_api_models_url(args.author), token=args.token)
        if not isinstance(models, list):
            raise TypeError(f"Expected a list from the HF models API, got {type(models).__name__}")
        for model in models:
            if not isinstance(model, dict):
                continue
            repo_id = model.get("id") or model.get("modelId")
            if repo_id:
                repo_ids.append(str(repo_id))

    if args.match:
        repo_ids = [repo_id for repo_id in repo_ids if args.match in repo_id]

    seen: set[str] = set()
    unique_repo_ids: list[str] = []
    for repo_id in repo_ids:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        unique_repo_ids.append(repo_id)
    return unique_repo_ids


def _infer_task_from_config(config: dict[str, Any]) -> str:
    """Guess an HF-style task slug for the model from its ``config.json``.

    Inspects ``architectures`` / ``model_type`` plus the nested ``text_config``
    + ``vision_config`` presence signal, and returns the README-friendly task
    slug (e.g. ``"image-text-to-text"``, ``"causal-language-model"``).

    Args:
        config: Parsed ``config.json`` dictionary.

    Returns:
        str: Task slug used by the README template; defaults to
            ``"causal-language-model"`` when no specific heuristic matches.
    """
    architectures = [str(arch) for arch in (config.get("architectures") or [])]
    joined = " ".join(architectures).lower()
    model_type = str(config.get("model_type") or "").lower()

    if model_type in {"clip", "siglip"}:
        return "zero-shot-image-classification"
    if "forspeechseq2seq" in joined or model_type in {
        "whisper",
        "speech_to_text",
        "speech-to-text",
    }:
        return "speech-sequence-to-sequence"
    if (
        "forimagetexttotext" in joined
        or "vision2seq" in joined
        or model_type in IMAGE_TEXT_TO_TEXT_MODEL_TYPES
        or (isinstance(config.get("text_config"), dict) and isinstance(config.get("vision_config"), dict))
    ):
        return "image-text-to-text"
    if bool(config.get("is_encoder_decoder")) or "forseq2seqlm" in joined:
        return "sequence-to-sequence"
    if "forzeroshotimageclassification" in joined or "zeroshotimageclassification" in joined:
        return "zero-shot-image-classification"
    if "forsequenceclassification" in joined or "sequenceclassification" in joined:
        return "sequence-classification"
    if "fordiffusionlm" in joined or model_type.endswith("diffusion"):
        return "diffusion-language-model"
    if "forcausallm" in joined or "causallm" in joined or "forconditionalgeneration" in joined:
        return "causal-language-model"
    return "causal-language-model"


def _get_attn_mechanism(config: dict[str, Any]) -> str:
    """Pick the most specific attention mechanism string from the config.

    Searches the top-level config and any nested ``text_config`` /
    ``vision_config`` subsections for ``attn_mechanism``,
    ``attn_mechanism_str``, or ``attention_mechanism`` keys, keeps the order
    they appear, de-duplicates case-insensitively, and drops ``None`` / null
    placeholders.

    Args:
        config: Parsed ``config.json`` dictionary.

    Returns:
        str: First non-null attention mechanism string found, or ``"auto"``
            when none is set.
    """
    candidates: list[str] = []
    config_sections = [config]
    for nested_key in ("text_config", "vision_config"):
        nested_value = config.get(nested_key)
        if isinstance(nested_value, dict):
            config_sections.append(nested_value)

    for section in config_sections:
        for key in ("attn_mechanism", "attn_mechanism_str", "attention_mechanism"):
            value = section.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                candidates.append(text)

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        lowered = candidate.lower()
        if lowered in {"none", "null"} or lowered in seen:
            continue
        seen.add(lowered)
        unique_candidates.append(candidate)

    return unique_candidates[0] if unique_candidates else "auto"


def _strip_html_tags(text: str) -> str:
    """Strip HTML-style ``<...>`` tags from ``text``.

    Args:
        text: Input string.

    Returns:
        str: ``text`` with everything matching ``<[^>]+>`` removed.
    """
    return re.sub(r"<[^>]+>", "", text)


def _normalize_inline_text(text: str) -> str:
    """Strip HTML, decode entities, and collapse whitespace.

    Args:
        text: Input HTML / Markdown fragment.

    Returns:
        str: Whitespace-collapsed plain text with HTML entities unescaped.
    """
    cleaned = html.unescape(_strip_html_tags(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_existing_card_context(readme: str | None, repo_id: str) -> ExistingCardContext:
    """Recover title, short description, and inferred source repo from an existing card.

    Looks for the first centered ``<h1>`` as the title, then walks the
    centered ``<div>`` blocks for the first short description after the
    title. Parses the source repo out of "converted from ..." descriptions
    (link or plain forms); as a last resort falls back to the title when it
    looks like an ``owner/name`` HF repo id that isn't the current repo.

    Args:
        readme: Existing README text, possibly ``None``.
        repo_id: The current repo id (used to suppress trivial self-matches
            when guessing the source repo).

    Returns:
        ExistingCardContext: Title / description / source-repo extracted from
            the README, with ``None`` for any field that could not be found.
    """
    if not readme:
        return ExistingCardContext()

    title = None
    description = None
    source_repo = None

    title_match = H1_RE.search(readme)
    if title_match:
        title = _normalize_inline_text(title_match.group(1))

    div_matches = CENTER_DIV_RE.findall(readme)
    if div_matches:
        # The first centered div after the H1 is the short description in current cards.
        for match in div_matches:
            normalized = _normalize_inline_text(match)
            if not normalized or normalized == title:
                continue
            description = normalized
            break

    if description:
        link_match = SOURCE_LINK_RE.search(description)
        if link_match:
            source_repo = link_match.group("label").strip()
        else:
            plain_match = SOURCE_PLAIN_RE.search(description)
            if plain_match:
                source_repo = plain_match.group(1).strip()

    repo_name = repo_id.split("/")[-1]
    if (
        not source_repo
        and title
        and "/" in title
        and title not in {repo_id, repo_name}
        and not title.startswith("EasyDeL/")
    ):
        source_repo = title

    return ExistingCardContext(title=title, description=description, source_repo=source_repo)


def _looks_like_local_path(value: str | None) -> bool:
    """Return ``True`` when ``value`` looks like a local filesystem path.

    Used to suppress propagating leftover local conversion paths into a fresh
    public-facing README description.

    Args:
        value: Candidate string.

    Returns:
        bool: ``True`` if the value contains a leading ``/``, ``./``, ``../``,
            or a Windows-style drive letter.
    """
    if not value:
        return False
    return bool(LOCAL_PATH_RE.search(value))


def _build_description(source_repo: str | None, existing_description: str | None) -> str | None:
    """Build the centered short description used in the regenerated card.

    Prefers a fresh "converted from <source_repo>" sentence when a source
    repo is known. Falls back to the existing description only when it isn't
    the generic placeholder and doesn't look like a local path leak.

    Args:
        source_repo: Inferred upstream repo id (or ``None``).
        existing_description: Description from the existing README (or ``None``).

    Returns:
        str | None: Description text for the new README, or ``None`` to let
            the template fall back to its default.
    """
    if source_repo:
        source_repo = source_repo.strip()
        source_url = f"{HF_BASE_URL}/{source_repo}"
        return f'EasyDeL checkpoint converted from <a href="{source_url}">{source_repo}</a>.'
    if (
        existing_description
        and existing_description != GENERIC_DESCRIPTION
        and not _looks_like_local_path(existing_description)
    ):
        return existing_description
    return None


def _generate_readme(
    *,
    repo_id: str,
    config: dict[str, Any],
    generator: Any,
    model_info_cls: Any,
    existing_context: ExistingCardContext,
) -> str:
    """Render a new README via the EasyDeL readme generator.

    Args:
        repo_id: Target HF repo id.
        config: Parsed ``config.json`` for the repo.
        generator: Loaded :class:`ReadmeGenerator` instance.
        model_info_cls: ``ModelInfo`` dataclass exposed by the generator module.
        existing_context: Pieces preserved from the existing card.

    Returns:
        str: Rendered README markdown.
    """
    architectures = [str(arch) for arch in (config.get("architectures") or [])]
    model_type = config.get("model_type")
    model_name = repo_id.split("/")[-1]
    model_class = architectures[0] if architectures else str(model_type or "EasyDeLModel")

    model_info = model_info_cls(
        name=model_name,
        type=model_class,
        repo_id=repo_id,
        description=_build_description(existing_context.source_repo, existing_context.description),
        model_type=str(model_type) if model_type else None,
        model_task=_infer_task_from_config(config),
        attn_mechanism=_get_attn_mechanism(config),
    )
    return generator.generate_readme(model_info)


def _write_generated_readme(output_dir: Path, repo_id: str, readme: str) -> None:
    """Persist the generated README to ``output_dir`` for review.

    The output filename is the repo id with ``/`` replaced by ``__`` so all
    files coexist in a single directory.

    Args:
        output_dir: Directory to write into; created if needed.
        repo_id: HF repo id.
        readme: README text to write.

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo_id.replace("/", "__")
    (output_dir / f"{safe_name}.README.md").write_text(readme, encoding="utf-8")


def _run_git(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a ``git`` subprocess with ``check=True``.

    Args:
        argv: Full ``git`` command including the leading ``"git"``.
        cwd: Working directory.
        env: Environment overrides (typically the result of
            :func:`_make_git_env`).
        capture_output: When ``True`` capture stdout/stderr as text.

    Returns:
        subprocess.CompletedProcess[str]: The completed process.

    Raises:
        subprocess.CalledProcessError: If git exits non-zero.
    """
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _make_git_env(token: str | None, askpass_path: Path | None) -> dict[str, str]:
    """Build the environment dict used for the per-repo git subprocesses.

    Disables interactive git prompts, skips LFS smudging, and wires the
    ``GIT_ASKPASS`` helper to the script that returns ``HF_TOKEN`` when a
    token is available.

    Args:
        token: HF access token, or ``None`` to skip auth env wiring.
        askpass_path: Path to the askpass helper script created by
            :func:`_write_askpass_script`. Required when ``token`` is set.

    Returns:
        dict[str, str]: Environment mapping suitable for ``subprocess.run``.
    """
    env = os.environ.copy()
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if token and askpass_path is not None:
        env["HF_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass_path)
    return env


def _write_askpass_script(path: Path) -> None:
    """Write a minimal ``GIT_ASKPASS`` helper that returns ``HF_TOKEN``.

    The helper responds to git's "username" prompt with the literal
    ``__token__`` value (HF treats this as "the token is the password") and
    to the "password" prompt with the ``HF_TOKEN`` environment variable.

    Args:
        path: Destination path for the script (will be made executable).

    Returns:
        None.
    """
    path.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  *sername*) printf "%s\\n" "__token__" ;;\n'
        '  *assword*) printf "%s\\n" "$HF_TOKEN" ;;\n'
        '  *) printf "\\n" ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _upload_readme_via_sparse_git(
    *,
    repo_id: str,
    revision: str,
    readme: str,
    token: str | None,
    commit_message: str,
    git_user_name: str,
    git_user_email: str,
) -> None:
    """Push a regenerated ``README.md`` back to the repo via sparse git clone.

    Uses a depth-1, blob-filtered, sparse-checkout clone that only materializes
    ``README.md`` so even huge repos clone in seconds. When the new README is
    byte-identical to the existing one the function returns without committing.

    Args:
        repo_id: HF repo id.
        revision: Branch/tag to push back to.
        readme: README text to commit.
        token: HF access token used through :func:`_write_askpass_script`.
        commit_message: Message for the commit.
        git_user_name: Author name set on the local clone.
        git_user_email: Author email set on the local clone.

    Returns:
        None.

    Raises:
        RuntimeError: If ``git diff --cached --quiet`` fails for reasons
            other than "no changes" (exit code > 1).
        subprocess.CalledProcessError: For any failing git invocation.
    """
    with tempfile.TemporaryDirectory(prefix="easydel-hf-readme-") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        askpass_path = tmpdir / "askpass.sh"
        if token:
            _write_askpass_script(askpass_path)
        else:
            askpass_path = None

        env = _make_git_env(token, askpass_path)
        repo_dir = tmpdir / "repo"
        remote_url = f"{HF_BASE_URL}/{repo_id}"

        clone_command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
        ]
        if revision:
            clone_command.extend(["--branch", revision])
        clone_command.extend([remote_url, str(repo_dir)])
        _run_git(clone_command, env=env)
        _run_git(["git", "sparse-checkout", "set", "README.md"], cwd=repo_dir, env=env)
        _run_git(["git", "config", "user.name", git_user_name], cwd=repo_dir, env=env)
        _run_git(["git", "config", "user.email", git_user_email], cwd=repo_dir, env=env)

        readme_path = repo_dir / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        _run_git(["git", "add", "README.md"], cwd=repo_dir, env=env)

        diff_check = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir, env=env)
        if diff_check.returncode == 0:
            return
        if diff_check.returncode > 1:
            raise RuntimeError(
                f"`git diff --cached --quiet` failed for {repo_id} with exit code {diff_check.returncode}"
            )

        _run_git(["git", "commit", "-m", commit_message], cwd=repo_dir, env=env)
        _run_git(["git", "push", "origin", f"HEAD:{revision}"], cwd=repo_dir, env=env)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the README updater.

    Resolves the set of repos to update, loads the README generator, and for
    each repo fetches the config + existing README, builds a fresh README,
    optionally writes a preview, and optionally pushes the change. Per-repo
    failures are logged but do not halt the loop.

    Args:
        argv: Optional list of CLI tokens. ``None`` reads from ``sys.argv``.

    Returns:
        int: ``0`` if every repo succeeded, ``1`` if any failed.

    Raises:
        SystemExit: If no repos were selected.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_ids = _collect_repo_ids(args)
    if not repo_ids:
        raise SystemExit("No repos selected. Use --repo-id, --repos-file, or --author.")

    readme_generator_module = _load_readme_generator()
    generator = readme_generator_module.ReadmeGenerator(
        template_dir=args.template_dir,
        template_name=args.template_name,
    )
    model_info_cls = readme_generator_module.ModelInfo

    output_dir = Path(args.output_dir) if args.output_dir else None
    updated = 0
    skipped = 0
    failed = 0
    total = len(repo_ids)

    for index, repo_id in enumerate(repo_ids, start=1):
        prefix = f"[{index}/{total}]"
        try:
            config = _fetch_json(
                _hf_raw_file_url(repo_id, "config.json", args.revision),
                token=args.token,
            )
            if not isinstance(config, dict):
                raise TypeError(f"Expected config.json for {repo_id} to be a JSON object, got {type(config).__name__}")

            existing_readme = _fetch_text(
                _hf_raw_file_url(repo_id, "README.md", args.revision),
                token=args.token,
                required=False,
            )
            existing_context = _extract_existing_card_context(existing_readme, repo_id)
            readme = _generate_readme(
                repo_id=repo_id,
                config=config,
                generator=generator,
                model_info_cls=model_info_cls,
                existing_context=existing_context,
            )

            if output_dir is not None:
                _write_generated_readme(output_dir, repo_id, readme)

            if existing_readme is not None and existing_readme.strip() == readme.strip():
                print(f"{prefix} [skip] {repo_id} (no changes)")
                skipped += 1
                continue

            if args.dry_run:
                print(f"{prefix} [dry-run] {repo_id} (generated)")
                updated += 1
                continue

            _upload_readme_via_sparse_git(
                repo_id=repo_id,
                revision=args.revision,
                readme=readme,
                token=args.token,
                commit_message=args.commit_message,
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
            )
            print(f"{prefix} [ok] {repo_id}")
            updated += 1
        except Exception as error:
            failed += 1
            print(f"{prefix} [error] {repo_id}: {error}", file=sys.stderr)

    print(f"done: updated={updated} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
