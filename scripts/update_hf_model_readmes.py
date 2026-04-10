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
"""
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
    title: str | None = None
    description: str | None = None
    source_repo: str | None = None


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--revision", default="main", help="Repo revision/branch to read from and push to.")
    parser.add_argument(
        "--commit-message",
        default="Update README.md (EasyDeL auto-generated)",
        help="Commit message for pushes.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate but do not push changes.")
    parser.add_argument("--output-dir", help="Optional directory to write generated READMEs for review.")
    parser.add_argument("--template-dir", help="Optional custom Jinja template directory.")
    parser.add_argument("--template-name", help="Optional template filename inside template_dir.")
    parser.add_argument("--git-user-name", default="EasyDeL README Bot", help="Git author name for pushes.")
    parser.add_argument(
        "--git-user-email",
        default="easydel-readme-bot@users.noreply.huggingface.co",
        help="Git author email for pushes.",
    )
    return parser


def _load_readme_generator():
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
    headers = {
        "User-Agent": "EasyDeL/update_hf_model_readmes.py",
        "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_bytes(url: str, *, token: str | None, required: bool = True) -> bytes | None:
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
    payload = _fetch_bytes(url, token=token, required=required)
    if payload is None:
        return None
    return payload.decode("utf-8", "replace")


def _fetch_json(url: str, *, token: str | None, required: bool = True) -> Any:
    payload = _fetch_bytes(url, token=token, required=required)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _hf_api_models_url(author: str) -> str:
    query = urllib.parse.urlencode({"author": author, "limit": 1000})
    return f"{HF_BASE_URL}/api/models?{query}"


def _hf_raw_file_url(repo_id: str, filename: str, revision: str) -> str:
    quoted_repo = "/".join(urllib.parse.quote(part, safe="") for part in repo_id.split("/"))
    quoted_filename = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    quoted_revision = urllib.parse.quote(revision, safe="")
    return f"{HF_BASE_URL}/{quoted_repo}/raw/{quoted_revision}/{quoted_filename}"


def _read_text(path: str | os.PathLike[str]) -> str:
    return Path(path).read_text(encoding="utf-8")


def _collect_repo_ids(args: argparse.Namespace) -> list[str]:
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
    architectures = [str(arch) for arch in (config.get("architectures") or [])]
    joined = " ".join(architectures).lower()
    model_type = str(config.get("model_type") or "").lower()

    if model_type in {"clip", "siglip"}:
        return "zero-shot-image-classification"
    if "forspeechseq2seq" in joined or model_type in {"whisper", "speech_to_text", "speech-to-text"}:
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
    return re.sub(r"<[^>]+>", "", text)


def _normalize_inline_text(text: str) -> str:
    cleaned = html.unescape(_strip_html_tags(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_existing_card_context(readme: str | None, repo_id: str) -> ExistingCardContext:
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
    if not value:
        return False
    return bool(LOCAL_PATH_RE.search(value))


def _build_description(source_repo: str | None, existing_description: str | None) -> str | None:
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
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _make_git_env(token: str | None, askpass_path: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GIT_LFS_SKIP_SMUDGE", "1")
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    if token and askpass_path is not None:
        env["HF_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass_path)
    return env


def _write_askpass_script(path: Path) -> None:
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

        clone_command = ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse"]
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
