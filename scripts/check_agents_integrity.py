#!/usr/bin/env python3
"""Integrity checker for .agents/**, AGENTS.md, and WORKSPACE.md.

Validates:
  (a) Every .agents/skills/*/SKILL.md has valid YAML frontmatter (name matches dir,
      non-empty description, no harness-specific allowed-tools) and is listed in
      .agents/skills/README.md with its full, untruncated description; no stray skill dirs.
  (b) Every .agents/agents/*.md has valid YAML frontmatter (name matches filename stem,
      non-empty description, no harness-specific allowed-tools) and is listed in AGENTS.md's
      Available agents table (with no extra stale entries).
  (c) Every backtick repo path (libs/..., scripts/..., .agents/...) across .agents/**,
      AGENTS.md, and WORKSPACE.md exists on disk (expanding brace groups and skipping
      template placeholders containing <...>, *, ..., Ellipsis, $, or single-key {var}).
  (d) Zero banned placeholder/myth strings across .agents/**, AGENTS.md, and WORKSPACE.md.
  (e) Zero exact-duplicate files under .agents/.
  (f) Balanced backticks on every non-fenced line (and closed code fences) in .agents/**,
      plus valid YAML in .agents/repo-map.yaml.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / ".agents"
SKILLS_DIR = AGENTS_DIR / "skills"
AGENTS_ROSTER_DIR = AGENTS_DIR / "agents"
SKILLS_README = SKILLS_DIR / "README.md"
REPO_MAP = AGENTS_DIR / "repo-map.yaml"
AGENTS_MD = REPO_ROOT / "AGENTS.md"
WORKSPACE_MD = REPO_ROOT / "WORKSPACE.md"

BANNED_STRINGS: tuple[str, ...] = (
    "an eSurge benchmark harness",
    "a checkpoint verification harness",
    "a sharded-Llama benchmark harness",
    "an SFT PP/TP comparison harness",
    "external optimizer package",
    "add-easydel-operation",
    "eformer-checkpoint-sharding",
    "spectrax-core",
    "spectrax-nn",
    "spectrax-transforms",
    "caching/_specs.py",
    "test/quantization/",
    "PageTable.commit()",
    "EasyDeLQuantizationConfig",
    "SpeculativeMTPDriver",
    "total_batch_size % num_generations",
)

# Match repo-root-relative path tokens inside inline backticks.
REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_./\-])((?:libs|scripts|\.agents)/[A-Za-z0-9_.\-/{},*<>:$~+@]+)"
)
# Detect single-variable placeholder like {family} or {i} (no comma inside braces).
SINGLE_BRACE_VAR_RE = re.compile(r"\{[^{},]+\}")
# Detect file extension followed by :line, :line-line, :Symbol, or ::nodeid.
COLON_SUFFIX_RE = re.compile(
    r"(\.(?:py|pyi|rs|md|yaml|yml|toml|sh|json|rst|txt|cu|cuh|h|hpp|cc|cpp|lock|npz|db|cfg|ini|cmake))(?::.*)?$"
)


def parse_frontmatter(path: Path) -> tuple[dict | None, str | None]:
    """Parse YAML frontmatter delimited by leading '---' lines."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return None, "missing leading '---' YAML frontmatter"
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None, "unclosed YAML frontmatter"
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        return None, f"invalid YAML frontmatter: {exc}"
    if not isinstance(data, dict):
        return None, "YAML frontmatter must be a mapping"
    return data, None


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def expand_braces(pattern: str) -> list[str]:
    """Recursively expand shell-style {a,b} brace groups (supports nesting)."""
    # Find the innermost { ... } that contains a comma.
    depth = 0
    start = -1
    for i, ch in enumerate(pattern):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                inner = pattern[start + 1 : i]
                # Split on top-level commas inside this brace group.
                choices: list[str] = []
                buf: list[str] = []
                inner_depth = 0
                for c in inner:
                    if c == "{":
                        inner_depth += 1
                        buf.append(c)
                    elif c == "}":
                        inner_depth -= 1
                        buf.append(c)
                    elif c == "," and inner_depth == 0:
                        choices.append("".join(buf))
                        buf = []
                    else:
                        buf.append(c)
                choices.append("".join(buf))
                if len(choices) <= 1:
                    return [pattern]
                prefix = pattern[:start]
                suffix = pattern[i + 1 :]
                expanded: list[str] = []
                for choice in choices:
                    expanded.extend(expand_braces(prefix + choice + suffix))
                return expanded
    return [pattern]


def clean_path_token(raw: str) -> str:
    """Strip trailing punctuation and line/symbol suffixes from a path token."""
    token = raw.rstrip(".,;)]'\"")
    while token.endswith("}") and token.count("}") > token.count("{"):
        token = token[:-1]
    # Strip markdown anchor fragments if present.
    if "#" in token:
        token = token.split("#", 1)[0]
    # Strip :line, :line-line, ::test_name, :Symbol after known file extensions.
    m = COLON_SUFFIX_RE.search(token)
    if m:
        token = token[: m.start() + len(m.group(1))]
    else:
        # Handle directory:line or stray trailing colon.
        token = token.rstrip(":")
    return token


def is_template_path(token: str) -> bool:
    """Return True if token contains template wildcards/placeholders."""
    if any(marker in token for marker in ("<", ">", "*", "...", "Ellipsis", "$")):
        return True
    if SINGLE_BRACE_VAR_RE.search(token):
        return True
    return False


def iter_doc_files() -> list[Path]:
    files = sorted(p for p in AGENTS_DIR.rglob("*") if p.is_file())
    for top in (AGENTS_MD, WORKSPACE_MD):
        if top.is_file():
            files.append(top)
    return files


def check_skills(errors: list[str]) -> None:
    if not SKILLS_README.is_file():
        errors.append("[skills] Missing .agents/skills/README.md")
        return

    readme_text = SKILLS_README.read_text(encoding="utf-8")
    readme_norm = normalize_ws(readme_text)

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())
    seen_skills: set[str] = set()

    for sdir in skill_dirs:
        skill_md = sdir / "SKILL.md"
        if not skill_md.is_file():
            errors.append(f"[skills] Directory .agents/skills/{sdir.name} is missing SKILL.md")
            continue
        data, err = parse_frontmatter(skill_md)
        if err or data is None:
            errors.append(f"[skills] .agents/skills/{sdir.name}/SKILL.md: {err}")
            continue
        name = data.get("name")
        desc = data.get("description")
        if name != sdir.name:
            errors.append(
                f"[skills] .agents/skills/{sdir.name}/SKILL.md: frontmatter name {name!r} != dir name {sdir.name!r}"
            )
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"[skills] .agents/skills/{sdir.name}/SKILL.md: missing or empty description")
            continue
        if "allowed-tools" in data:
            errors.append(
                f"[skills] .agents/skills/{sdir.name}/SKILL.md: contains Claude-specific 'allowed-tools' key"
            )
        seen_skills.add(sdir.name)

        expected_link = f"[{sdir.name}]({sdir.name}/SKILL.md)"
        if expected_link not in readme_text:
            errors.append(f"[skills] .agents/skills/README.md missing clickable link {expected_link}")
        if normalize_ws(desc) not in readme_norm:
            errors.append(
                f"[skills] .agents/skills/README.md description for '{sdir.name}' is missing, drifted, or truncated"
            )

    # Check README has no stale links to deleted skills.
    for m in re.finditer(r"\[([a-z0-9-]+)\]\(([a-z0-9-]+)/SKILL\.md\)", readme_text):
        label, target = m.group(1), m.group(2)
        if label != target or target not in seen_skills:
            errors.append(
                f"[skills] .agents/skills/README.md references unknown or mismatched skill [{label}]({target}/SKILL.md)"
            )


def check_agents(errors: list[str]) -> None:
    if not AGENTS_MD.is_file():
        errors.append("[agents] Missing AGENTS.md")
        return

    agents_md_text = AGENTS_MD.read_text(encoding="utf-8")
    # Extract agent names from the "### Available agents" table in AGENTS.md.
    table_section = ""
    m = re.search(r"### Available agents[^\n]*\n(.*?)(?:\n### |\Z)", agents_md_text, re.S)
    if not m:
        errors.append("[agents] AGENTS.md missing '### Available agents' section")
    else:
        table_section = m.group(1)
    listed_in_agents_md = set(re.findall(r"^\|\s*`([a-z0-9-]+)`\s*\|", table_section, re.M))

    actual_agents: set[str] = set()
    for md_path in sorted(AGENTS_ROSTER_DIR.glob("*.md")):
        stem = md_path.stem
        actual_agents.add(stem)
        data, err = parse_frontmatter(md_path)
        if err or data is None:
            errors.append(f"[agents] .agents/agents/{md_path.name}: {err}")
            continue
        name = data.get("name")
        desc = data.get("description")
        if name != stem:
            errors.append(
                f"[agents] .agents/agents/{md_path.name}: frontmatter name {name!r} != filename stem {stem!r}"
            )
        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"[agents] .agents/agents/{md_path.name}: missing or empty description")
        if "allowed-tools" in data:
            errors.append(
                f"[agents] .agents/agents/{md_path.name}: contains Claude-specific 'allowed-tools' key"
            )
        if stem not in listed_in_agents_md:
            errors.append(f"[agents] Agent '{stem}' (.agents/agents/{md_path.name}) is not listed in AGENTS.md table")

    for extra in sorted(listed_in_agents_md - actual_agents):
        errors.append(f"[agents] AGENTS.md lists unknown agent '{extra}' with no .agents/agents/{extra}.md file")


def check_repo_map_yaml(errors: list[str]) -> None:
    if not REPO_MAP.is_file():
        errors.append("[yaml] Missing .agents/repo-map.yaml")
        return
    try:
        data = yaml.safe_load(REPO_MAP.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        errors.append(f"[yaml] .agents/repo-map.yaml failed to parse: {exc}")
        return
    if not isinstance(data, dict):
        errors.append("[yaml] .agents/repo-map.yaml must parse to a YAML mapping")
        return
    # Verify every ownership_routing key and package path exists on disk.
    routing = data.get("ownership_routing", {})
    if isinstance(routing, dict):
        for route_path in routing:
            if not (REPO_ROOT / route_path).exists():
                errors.append(f"[repo-map] ownership_routing path does not exist: {route_path}")
    packages = data.get("packages", {})
    if isinstance(packages, dict):
        for pkg_name, pkg in packages.items():
            if isinstance(pkg, dict) and isinstance(pkg.get("path"), str):
                p = pkg["path"]
                if not (REPO_ROOT / p).exists():
                    errors.append(f"[repo-map] packages.{pkg_name}.path does not exist: {p}")


def check_paths_and_backticks_and_banned(errors: list[str]) -> tuple[int, int]:
    link_re = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    paths_checked = 0
    links_checked = 0

    for path in iter_doc_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        under_agents = path.is_relative_to(AGENTS_DIR)

        # (d) Banned placeholder / myth strings across all lines.
        for lineno, line in enumerate(lines, start=1):
            for banned in BANNED_STRINGS:
                if banned in line:
                    errors.append(f"[banned] {rel}:{lineno}: contains banned string {banned!r}")

        # Track fenced code blocks for backtick balance and inline backtick scanning.
        in_fence = False
        fence_char = ""
        fence_len = 0
        fence_start_line = 0

        for lineno, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
            if fence_match:
                marker = fence_match.group(1)
                if not in_fence:
                    in_fence = True
                    fence_char = marker[0]
                    fence_len = len(marker)
                    fence_start_line = lineno
                    continue
                if marker[0] == fence_char and len(marker) >= fence_len and stripped[len(marker) :].strip() == "":
                    in_fence = False
                    continue

            if in_fence:
                continue

            # (f) Balanced backticks on every non-fenced line under .agents/**.
            if under_agents:
                if line.count("`") % 2 != 0:
                    errors.append(f"[backticks] {rel}:{lineno}: unbalanced backticks on line")

            # Validate relative markdown links outside code fences.
            for lm in link_re.finditer(line):
                target = lm.group(1).split("#", 1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                if is_template_path(target):
                    continue
                links_checked += 1
                if not (path.parent / target).resolve().exists():
                    errors.append(f"[link] {rel}:{lineno}: broken markdown link target: {lm.group(1)}")

            # (c) Extract inline backtick spans and validate any repo-root-relative paths.
            # Support both single (`...`) and double (``...``) inline code spans cleanly.
            idx = 0
            n = len(line)
            while idx < n:
                if line[idx] == "`":
                    run_start = idx
                    while idx < n and line[idx] == "`":
                        idx += 1
                    tick_len = idx - run_start
                    delim = "`" * tick_len
                    end = line.find(delim, idx)
                    if end == -1:
                        break
                    span = line[idx:end]
                    idx = end + tick_len
                    for pm in REPO_PATH_RE.finditer(span):
                        raw_tok = pm.group(1)
                        cleaned = clean_path_token(raw_tok)
                        if not cleaned or is_template_path(cleaned):
                            continue
                        for candidate in expand_braces(cleaned):
                            candidate = candidate.rstrip("/")
                            if not candidate or is_template_path(candidate):
                                continue
                            paths_checked += 1
                            if not (REPO_ROOT / candidate).exists():
                                errors.append(
                                    f"[path] {rel}:{lineno}: backtick path does not exist on disk: {candidate}"
                                )
                else:
                    idx += 1

        if under_agents and in_fence:
            errors.append(f"[backticks] {rel}:{fence_start_line}: unclosed code fence")

    return paths_checked, links_checked


def check_duplicates(errors: list[str]) -> None:
    by_hash: dict[str, list[str]] = {}
    for path in sorted(p for p in AGENTS_DIR.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(REPO_ROOT).as_posix()
        by_hash.setdefault(digest, []).append(rel)
    for digest, paths in sorted(by_hash.items()):
        if len(paths) > 1:
            errors.append(f"[duplicate] Exact duplicate files under .agents/ ({digest[:12]}): {', '.join(paths)}")


def main() -> int:
    errors: list[str] = []
    check_skills(errors)
    check_agents(errors)
    check_repo_map_yaml(errors)
    paths_checked, links_checked = check_paths_and_backticks_and_banned(errors)
    check_duplicates(errors)

    if errors:
        print(f"FAILED: {len(errors)} integrity issue(s) found:\n", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    skill_count = len(list(SKILLS_DIR.glob("*/SKILL.md")))
    agent_count = len(list(AGENTS_ROSTER_DIR.glob("*.md")))
    doc_count = len(iter_doc_files())
    print(
        f"OK: {skill_count} skills, {agent_count} agents, {doc_count} docs, "
        f"{paths_checked} backtick paths, {links_checked} links checked — 0 issues."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
