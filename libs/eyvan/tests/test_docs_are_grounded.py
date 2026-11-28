# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Documentation that names a command, flag or setting that exists.

Docs rot in one specific way: a command is renamed, the code and its tests move
together, and the prose keeps confidently describing the old name. Nothing
fails, because prose has no compiler. The reader finds out.

That is worse than no documentation. Absent docs make someone read the code;
wrong docs make them trust a sentence and lose an afternoon -- and they lose it
believing the tool is broken rather than the page.

So every ``eyvan …`` invocation in ``docs/`` is resolved against the real click
command tree, and every ``EYVAN_*`` variable against the real settings loader.
This does not check that the prose is *true*, which no test can. It checks the
part that can be checked, which is exactly the part that silently drifts.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re

import click
import pytest

from eyvan.sdk import cli

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

#: Invocations that are illustrative rather than literal.
_PLACEHOLDER = re.compile(r"[<{]")


def _pages() -> list[pathlib.Path]:
    """Return every hand-written page, including nested design records."""

    return sorted(DOCS.rglob("*.md"))


#: An invocation, not a mention: ``eyvan`` opening a fenced line or a backtick
#: span (behind an optional ``$ `` prompt), or launched via ``uv run``.
#: Anywhere else on the line it is an argument to another program -- in
#: ``ruff check eyvan tests`` the words name paths, and reading them as an
#: invocation demanded a command named ``eyvan tests``.
_LAUNCH = re.compile(r"(?:^\s*(?:\$ )?|`\$? ?|uv run )eyvan((?: [a-z][a-z0-9-]*)+)")


def _command_paths(text: str) -> set[tuple[str, ...]]:
    """Every ``eyvan a b c`` invocation, as the command path it names."""

    found: set[tuple[str, ...]] = set()
    for line in text.splitlines():
        # `from eyvan import EyvanClient` is Python, not an invocation. Reading it
        # as one made the checker demand a CLI command named `import`, which would
        # push a correct example out of the documentation to satisfy a test.
        if re.match(r"\s*(from eyvan\b|import eyvan\b)", line):
            continue
        for match in _LAUNCH.finditer(line):
            words = tuple(match.group(1).split())
            if words:
                found.add(words)
    return found


def _names_a_real_command(path: tuple[str, ...]) -> bool:
    """Whether a documented invocation names a command that exists.

    Word by word, and the distinction is exact: a word that follows a *group* has
    to be one of its subcommands, while words that follow a *leaf* are its
    arguments and are not this test's business.

    An earlier version accepted any resolvable prefix, to let arguments through.
    It let a renamed subcommand through as well -- ``eyvan capacity warmness``
    passed because ``capacity`` resolved -- which is precisely the rot this file
    exists to catch, and a mutation found it.
    """

    node: click.Command = cli.main
    for word in path:
        if not isinstance(node, click.Group):
            return True  # a leaf was reached; the rest are arguments
        child = node.commands.get(word)
        if child is None:
            return False
        node = child
    return True


def _documented_options(text: str) -> set[tuple[tuple[str, ...], str]]:
    """Every ``eyvan a b --flag`` pairing, as (command path, flag).

    Walks each invocation through the command tree rather than requiring the
    flag to follow the command words directly, because the common runbook shape
    puts an argument in between: ``eyvan workload get <id> --json``. Matching
    only adjacent flags silently skipped exactly those lines.

    Long flags only. A short flag is one character and collides constantly with
    ordinary prose, and the failure this catches -- copying an invocation out of
    a runbook mid-run -- is a long-flag failure in practice.
    """

    found: set[tuple[tuple[str, ...], str]] = set()
    for line in text.splitlines():
        for match in re.finditer(r"`?\beyvan((?: [^\s`]+)+)", line):
            path: list[str] = []
            flags: list[str] = []
            node: click.Command = cli.main
            for token in match.group(1).split():
                # A redirect or pipe ends the invocation; `<id>` does not, and an
                # earlier pattern that excluded ">" wholesale skipped every line
                # shaped `eyvan workload get <id> --json` -- which is most of them.
                # `--` ends Eyvan's own options: everything after it belongs to
                # the process being exec'd, so `exec -- py-spy --pid` was being
                # read as two eyvan flags.
                if token in {"--", ">", ">>", "|", "&&", ";", "#"}:
                    break
                if token.startswith("--"):
                    flags.append(token.split("=")[0])
                    continue
                if flags:
                    continue  # an option's value, not a subcommand
                if isinstance(node, click.Group) and token in node.commands:
                    path.append(token)
                    node = node.commands[token]
            if path:
                found.update((tuple(path), flag) for flag in flags)
    return found


def _resolve(path: tuple[str, ...]) -> click.Command | None:
    """Return the command a path names, or ``None`` if it names none."""

    node: click.Command = cli.main
    for word in path:
        if not isinstance(node, click.Group):
            return node
        child = node.commands.get(word)
        if child is None:
            return None
        node = child
    return node


def _accepts(command: click.Command, flag: str) -> bool:
    """Whether ``command`` -- or the root, for global flags -- declares ``flag``."""

    for source in (command, cli.main):
        for parameter in source.params:
            if flag in getattr(parameter, "opts", ()) or flag in getattr(parameter, "secondary_opts", ()):
                return True
    return False


@pytest.mark.parametrize("page", _pages(), ids=lambda path: path.name)
def test_every_documented_flag_exists(page: pathlib.Path) -> None:
    """A runbook flag that does not exist fails in the middle of a paid run.

    The sibling test above checks command *paths*, and passed while
    ``docs/proofs.md`` told an operator to run ``eyvan cells --json`` five
    different ways. ``cells`` resolves, so the path was fine; the flag has never
    existed, and those commands emit JSON with no flag at all.

    That is worse than an ordinary documentation error. These pages are followed
    while a TPU is billing, by someone who cannot tell a typo in the runbook from
    a regression in the tool.
    """

    unknown: list[str] = []
    for path, flag in sorted(_documented_options(page.read_text(encoding="utf-8"))):
        command = _resolve(path)
        if command is not None and not _accepts(command, flag):
            unknown.append(f"eyvan {' '.join(path)} {flag}")

    assert not unknown, f"{page.name} documents flags that do not exist: {', '.join(unknown)}"


def test_the_docs_directory_exists_and_is_not_empty() -> None:
    """The README was 49 KB of reference prose and there was no ``docs/``.

    A single page that long is one nobody reads to the end, so the tasks that
    matter -- install, operate, secure, debug -- were each buried in it.
    """

    assert _pages(), "docs/ is empty"
    assert {page.name for page in _pages()} >= {
        "install.md",
        "operate.md",
        "security.md",
        "troubleshoot.md",
    }


def test_design_records_are_reachable_from_an_operator_landing_page() -> None:
    """A nested record with no inbound link exists only for repository archaeologists."""

    landing_pages = "\n".join((DOCS / name).read_text(encoding="utf-8") for name in ("operate.md", "security.md"))
    design_records = sorted((DOCS / "design").glob("*.md"))

    assert design_records, "docs/design/ is empty"
    for record in design_records:
        target = record.relative_to(DOCS).as_posix()
        assert f"]({target})" in landing_pages, f"{target} is not linked from operate.md or security.md"


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_every_documented_command_exists(page: pathlib.Path) -> None:
    """A renamed command leaves the prose describing the old name.

    Nothing fails, because prose has no compiler -- so this is the compiler.
    """

    unknown: list[str] = []
    for path in sorted(_command_paths(page.read_text(encoding="utf-8"))):
        if not _names_a_real_command(path):
            unknown.append("eyvan " + " ".join(path))

    assert not unknown, f"{page.name} names commands that do not exist: {', '.join(unknown)}"


def _known_variables() -> set[str]:
    """Every ``EYVAN_*`` name the repository actually spells.

    Not just the settings loader. Two distinct families exist and both are
    documented: variables an *operator* sets to configure a control plane, and
    variables the control plane *sets for a job* -- ``EYVAN_SOURCE_JOB_ID`` and
    its siblings, which a diagnostic reads inside an exec'd process. Checking
    only the loader would have rejected the second family as unknown, which is
    how a guard that is too narrow trains people to weaken it.

    The suite owns a third family of its own -- ``EYVAN_REQUIRE_POSTGRES`` turns
    a skip into a failure -- and a doc telling a developer to set one of those
    is exactly as wrong when it drifts, so the tests tree counts too.
    """

    root = pathlib.Path(__file__).resolve().parent.parent
    found: set[str] = set()
    for tree in (root / "eyvan", root / "tests"):
        for path in tree.rglob("*.py"):
            found |= set(re.findall(r"\bEYVAN_[A-Z0-9_]+\b", path.read_text(encoding="utf-8")))
    return found


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_every_documented_variable_exists(page: pathlib.Path) -> None:
    """An environment variable that was renamed is worse than one never documented.

    The reader sets it, nothing complains, and the behaviour they were promised
    silently does not happen.
    """

    documented = set(re.findall(r"\bEYVAN_[A-Z0-9_]+\b", page.read_text(encoding="utf-8")))
    known = _known_variables()
    # A name ending in an underscore is a family glob -- ``EYVAN_AWS_*`` stands
    # for every variable carrying that prefix, and the glob exists when at least
    # one member does. Forbidding the form would push prose toward enumerations
    # that rot faster than the family they list.
    unknown = sorted(
        name
        for name in documented
        if name not in known and not (name.endswith("_") and any(variable.startswith(name) for variable in known))
    )

    assert not unknown, f"{page.name} names variables the package does not spell: {', '.join(unknown)}"


def test_the_settings_family_is_still_checked_against_the_loader() -> None:
    """Widening the guard must not stop it catching a renamed *setting*.

    A variable the package merely mentions is not necessarily one the loader
    reads, and the operator-facing family is the one whose drift is expensive.
    """

    import inspect

    from eyvan.config import EyvanSettings

    loader = set(re.findall(r'"(EYVAN_[A-Z0-9_]+)"', inspect.getsource(EyvanSettings)))
    install = set(re.findall(r"\bEYVAN_[A-Z0-9_]+\b", (DOCS / "install.md").read_text(encoding="utf-8")))

    assert install
    assert install <= loader, (
        f"install.md names settings the loader does not read: {', '.join(sorted(install - loader))}"
    )


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_no_page_links_to_a_file_that_is_not_there(page: pathlib.Path) -> None:
    """A broken relative link is the cheapest possible rot and the most common."""

    broken = [
        target
        for target in re.findall(r"\]\((?!https?:)([^)#]+)", page.read_text(encoding="utf-8"))
        if not (page.parent / target).exists()
    ]

    assert not broken, f"{page.name} links to missing files: {', '.join(broken)}"


def test_the_readme_points_at_the_split() -> None:
    """Otherwise the pages exist and nobody arriving at the repository finds them."""

    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")

    assert "docs/install.md" in readme
    assert "docs/troubleshoot.md" in readme


def test_the_install_page_still_states_the_real_number_of_prerequisites() -> None:
    """Fifteen separately-configured things, said out loud.

    A quickstart that works locally and strands a reader at step nine is the
    documentation failure this page exists to avoid, and the honest count is
    what prevents it.
    """

    install = (DOCS / "install.md").read_text(encoding="utf-8")
    rows = [line for line in install.splitlines() if re.match(r"^\| \d+ \|", line)]

    assert len(rows) == 15
    assert "fifteen" in install.lower()


def test_the_proofs_page_states_a_criterion_and_a_shape_for_each_run() -> None:
    """A proof without a pass criterion is a plan to look at something.

    The page exists so approving a spend is a decision about a known quantity,
    and a row that says what it costs without saying what would satisfy it is
    the opposite of that.
    """

    proofs = (DOCS / "proofs.md").read_text(encoding="utf-8")
    sections = [block for block in proofs.split("\n## ") if block.startswith("P")]

    assert len(sections) >= 8
    for section in sections:
        title = section.splitlines()[0]
        assert "**Passes when**" in section, f"{title} states no pass criterion"
        assert "**Shape**" in section, f"{title} states no shape"


def test_the_proofs_page_does_not_promote_local_or_historical_runs_to_hardware_proof() -> None:
    """Completed host checks and old receipts must remain scoped to their candidates."""

    # Whitespace-normalized: the prose wraps, and a substring check against
    # wrapped text asserts the line width rather than the sentence.
    proofs = " ".join((DOCS / "proofs.md").read_text(encoding="utf-8").split()).lower()

    assert "none of that is evidence about a tpu" in proofs
    assert "no historical run qualifies the current candidate" in proofs
    assert "hardware runs do not substitute for cpu-establishable software gates" in proofs
    # No row may quietly acquire a result without an exact-candidate receipt.
    for claimed in ("verified on hardware", "confirmed on hardware", "passed on v5p", "proven on v5p"):
        assert claimed not in proofs


def _python_blocks(text: str) -> list[str]:
    """Every fenced ```python block in one page."""

    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_every_documented_keyword_argument_exists(page: pathlib.Path) -> None:
    """A Python example may not invent a keyword the constructor does not take.

    The command and flag checks above cover the CLI, and the Python examples had
    nothing at all -- so a guide could document
    ``TpuConfig(accelerator=..., slices=...)`` against a class whose fields are
    ``type`` and ``pods``, and every test would still pass. It did.
    """

    import dataclasses

    import eyvan

    def accepted(target: object) -> set[str] | None:
        """Return the keywords a constructor takes, or None if it is not ours.

        Both shapes are asked for on purpose. The first version of this check
        looked only for ``model_fields``, so every dataclass config -- which is
        all of them -- fell through the "not an Eyvan model" branch and was
        skipped. It passed against the very example that motivated it.
        """

        fields = getattr(target, "model_fields", None)
        if fields is not None:
            return set(fields)
        if dataclasses.is_dataclass(target):
            return {field.name for field in dataclasses.fields(target)}
        return None

    unknown: list[str] = []
    for block in _python_blocks(page.read_text(encoding="utf-8")):
        for name, arguments in re.findall(r"\b([A-Z][A-Za-z0-9_]*)\(([^)]*)\)", block):
            fields = accepted(getattr(eyvan, name, None))
            if fields is None:
                continue  # not an exported Eyvan config; the other checks own it
            for keyword in re.findall(r"(\w+)\s*=", arguments):
                if keyword not in fields:
                    unknown.append(f"{name}(...{keyword}=...)")

    assert not unknown, f"{page.name} passes keywords that do not exist: {', '.join(sorted(set(unknown)))}"


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_every_documented_invocation_supplies_its_required_arguments(page: pathlib.Path) -> None:
    """A documented command that cannot run is worse than an undocumented one.

    ``eyvan status`` takes a required FLEET_OR_CONFIG, and the guide showed it
    bare -- so the first thing a reader copied exited with a usage error. The
    command existed and every flag existed, which is exactly why the two checks
    above passed it.
    """

    missing: list[str] = []
    for line in page.read_text(encoding="utf-8").splitlines():
        if re.match(r"\s*(from eyvan\b|import eyvan\b)", line):
            continue
        match = re.search(r"\beyvan((?: [a-z][a-z0-9-]*)+)(.*)$", line)
        if match is None:
            continue
        node: click.Command = cli.main
        words = match.group(1).split()
        consumed = 0
        for word in words:
            if not isinstance(node, click.Group):
                break
            child = node.commands.get(word)
            if child is None:
                node = cli.main  # not a real path; the command check owns that
                break
            node = child
            consumed += 1
        if node is cli.main or isinstance(node, click.Group):
            continue
        required = [p for p in node.params if isinstance(p, click.Argument) and p.required]
        if not required:
            continue
        # Everything after the command path, minus flags and their values, is
        # what the reader would actually pass positionally.
        tail = " ".join(words[consumed:]) + " " + match.group(2)
        tail = tail.split("#")[0]
        positional = [
            token
            for token in tail.split()
            if not token.startswith("-") and token not in {"\\", "|", "&&"} and "=" not in token
        ]
        if len(positional) < len(required):
            missing.append(f"eyvan {' '.join(words)} (needs {', '.join(p.name for p in required)})")

    assert not missing, f"{page.name} documents invocations that cannot run: {'; '.join(sorted(set(missing)))}"


_CONSOLE_SCRIPTS = ("eyvan-migrate", "eyvan-server", "eyvan-agent", "eyvan-native-worker")


@pytest.mark.parametrize("page", _pages(), ids=lambda page: page.name)
def test_every_documented_console_script_invocation_parses(page: pathlib.Path) -> None:
    """The four argparse binaries were documented and checked by nothing.

    Every check above walks the click CLI, so `eyvan-migrate upgrade head` -- an
    Alembic habit, and not this tool's grammar -- would sail through all of them.
    Its parser takes an optional revision and no subcommand, and online upgrades
    are fixed to the packaged head regardless.
    """

    import contextlib
    import io
    import shlex

    from eyvan.server.persistence import migrate

    parsers = {"eyvan-migrate": migrate._parser}

    unparsed: list[str] = []
    for line in page.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("$").strip()
        for script in _CONSOLE_SCRIPTS:
            marker = f"{script} "
            if marker not in stripped or script not in parsers:
                continue
            tail = stripped.split(marker, 1)[1].split("#")[0].split("|")[0].split(">")[0]
            # Prose quotes a command inline as `eyvan-migrate --check`. -- the
            # trailing backtick and sentence punctuation are the page's, not the
            # invocation's, and feeding them to argparse invents a failure.
            tail = tail.split("`")[0].rstrip(" .,;:)")
            try:
                arguments = shlex.split(tail)
            except ValueError:
                continue
            if any(token.startswith("$") or token.startswith("<") for token in arguments):
                continue  # a placeholder the reader substitutes
            parser = parsers[script]()
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    parser.parse_args(arguments)
                except SystemExit:
                    unparsed.append(f"{script} {tail.strip()}")

    assert not unparsed, f"{page.name} documents invocations the parser rejects: {'; '.join(sorted(set(unparsed)))}"


def test_the_readme_connect_call_matches_the_signature_it_documents() -> None:
    """The README shows `eyvan.init` in three forms and names what it returns.

    All three were true when written and none is checked by running the snippet,
    so the signature is checked instead: a positional address, a `fleet`
    keyword, and a `base_url` on the result.
    """

    import inspect

    from eyvan.sdk.execution import ExecutionContext, init

    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")
    parameters = inspect.signature(init).parameters

    assert 'eyvan.init("http://10.0.0.5:8080")' in readme
    positional = next(iter(parameters.values()))
    assert positional.kind is inspect.Parameter.POSITIONAL_ONLY, "the README passes the address positionally"

    assert "eyvan.init(fleet=" in readme
    assert parameters["fleet"].kind is inspect.Parameter.KEYWORD_ONLY

    assert ".base_url" in readme
    assert "base_url" in {field.name for field in dataclasses.fields(ExecutionContext)}


def test_the_readme_does_not_teach_setting_the_url_it_already_defaults_to() -> None:
    """`os.environ["EYVAN_URL"] = "http://127.0.0.1:8080"` is what the start
    console used to print: an assignment of the value that wins anyway."""

    readme = (DOCS.parent / "README.md").read_text(encoding="utf-8")

    assert 'os.environ["EYVAN_URL"]' not in readme
