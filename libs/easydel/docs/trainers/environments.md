# RL Environments (`easydel.environments`)

`easydel.environments` provides multi-turn RL environments, a registry to build them by name, reward adapters for
GRPO-style trainers, ordering-safe reward shaping and an asynchronous episode collector. Environments that execute
model output run inside a sandbox session; see [Sandbox runtime](sandbox_runtime.md).

## Interface and registry

Every environment is an `AgenticEnvironment` (`easydel.trainers._shared.environment`) with `reset(seed)` returning
a `ResetResult` and `step(action)` returning a `StepResult(observation, reward, terminated, truncated, info)`.
Sandbox-backed environments subclass `SandboxEnvironment`, which adds `reset_async` / `step_async`, threaded
`reset_batch` / `step_batch`, `build_workspace_tools()` and `as_tool_env()` (wraps the env in `ToolEnvWrapper` with
bash / python / file tools bound to the session).

| function                                       | purpose                                                                                                                   |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `register_environment(name_or_names)`          | decorator; registers the class under `Registry` kind `"environment"`                                                      |
| `list_environments()`                          | registered names                                                                                                          |
| `build_environment(spec, **overrides)`         | `spec` is a name, a mapping with `"name"` or `"type"` plus constructor kwargs, or an instance (overrides then raise)       |
| `make_environment_factory(spec, **defaults)`   | reusable `environment_factory` callable for trainers                                                                      |
| `AutoEnvironment.from_name / from_task / from_dataset / as_factory` | resolve an environment from a name, a dataset row, a dataset, or build a trainer factory             |

`AutoEnvironment.from_task(row)` copies the first non-empty prompt column (`prompt`, `problem_statement`,
`task_prompt`, `question`, `instruction`, `query`) into `prompt` / `problem_statement` / `task_prompt`, and drops row
keys the target constructor does not accept. When no environment is named, `AutoEnvironment.infer_name` picks one
from the row keys, in this order:

| row contains                                              | environment      |
|-----------------------------------------------------------|------------------|
| `environment` / `env_name` / `env_type`                   | that name        |
| `harness`                                                 | `harness`        |
| `test_patch`, `fail_to_pass`, `pass_to_pass`, `repo_dir`  | `swe_repo`       |
| `verify_script`, `setup_script`                           | `terminal`       |
| `stdin_tests`, `test_code`                                | `code_judge`     |
| `initial_state` and `tools`                               | `synthetic_tool` |
| `answer`, `verifier`                                      | `python_tir`     |

Any other row raises `ValueError`.

```python
from easydel.environments import AutoEnvironment, SandboxPool

pool = SandboxPool(backend="local")
env = AutoEnvironment.from_task({"prompt": "Print 2+2", "stdin_tests": [{"input": "", "output": "4"}]}, runtime=pool)
factory = AutoEnvironment.as_factory("auto", runtime=pool)
```

### Reward functions and unscorable samples

`make_environment_reward_fn(environment="auto", *, runtime=None, backend=None, container_config=None,
max_workers=16, **defaults)` returns a `reward_funcs` callable for GRPO-family trainers:

- if the trainer passes `environment_rewards`, they are returned as is (no re-execution);
- otherwise one environment is built per completion (per-row columns of length `n` are forwarded), reset, and graded
  terminally in `max_workers` threads. Environments with a `grade_single_turn` method (`python_tir`) grade the final
  answer without executing code.
- **`None` means unscorable**: the reset failed, or the result carries `info["discard"]` or `info["env_error"]`
  (`is_unscorable(result)`). It is never reported as `0.0`. If every reset fails, `SandboxInfraError` is raised.
- the callable carries the attribute `discard_unscorable = True`. Under the GRPO default
  `discard_unscorable_groups=None`, a `None` from a function with this marker drops the whole prompt group; see
  the "Unscorable rewards" section of [RL safety](rl_safety.md).

Grader failures inside the built-in sandbox environments return `unscorable_step(error)`: a terminal step with
`info["env_error"] = True`, `info["discard"] = True` and a placeholder reward of `0.0` that must not be trained on.

## Built-in environments

| name (aliases)                     | class                       | graded by                                                         |
|------------------------------------|-----------------------------|-------------------------------------------------------------------|
| `code_judge` (`code`)              | `CodeJudgeEnvironment`      | stdin/stdout cases or Python `test_code`                          |
| `swe_repo` (`swe`)                 | `SWERepoEnvironment`        | `test_patch` + fail-to-pass / pass-to-pass commands               |
| `terminal` (`bash_task`)           | `TerminalEnvironment`       | `verify_script` after the episode                                 |
| `python_tir` (`tir`)               | `StatefulPythonEnvironment` | `answer` or `verifier(action)`                                    |
| `synthetic_tool` (`stateful_tool`) | `SyntheticToolEnvironment`  | `verifier(state, final_response)`                                 |
| `harness` (`cli_harness`)          | `HarnessEnvironment`        | `verify_script`, or test patch + F2P/P2P                          |
| `llm_judge`                        | `LLMJudgeEnvironment`       | `RubricJudge` score                                               |

### `code_judge`

Languages: `python`, `bash`, `javascript`, `cpp`, `rust`, `go`. The solution is the last fenced or `<code>` block of
the action (`extract_code_block`). `reward_mode="binary"` (default) gives `1.0` only when all cases pass;
`"pass_rate"` gives `passed / total` once the episode is done. With `max_steps > 1`, compiler errors and the first
failing case are returned as observations.

`test_code` is Python only and runs through an isolated test runner:

- a trusted runner is fed to `python -I -` over stdin; it marks itself non-dumpable, points fd 0 at `/dev/null` and
  closes other inherited descriptors;
- the solution runs in a child process, and test code calls proxies of its top-level names over pipes;
- the env draws a fresh nonce per evaluation; the reward is decided only by the single output line prefixed with that
  nonce. Zero or several nonce lines score `0`.

**Limitation (plain data only):** values crossing the boundary use a closed codec (`None`, `bool`, `int`, `float`,
`str`, `bytes`, `list`, `tuple`, `set`, `frozenset`, `dict`). A solution function returning anything else (a class
instance, a generator, ...) makes the call fail, and in-place mutation of arguments is not visible to the tests.

### `swe_repo`

Seeds a git working tree from `files`, a host `repo_dir`, or the repository already in the image (optionally checking
out `base_commit`), records the base `HEAD`, then executes `bash` / `read_file` / `write_file` / `edit_file` tool
calls until the agent answers without tool calls or reaches `max_steps` (default `12`). Default
`ContainerConfig(use_jail="required")`.

Grading, in order:

1. **Kill agent processes** (`kill_agent_processes`): the session's `kill_sandbox_processes()` must succeed, else the
   episode is unscorable.
2. Capture `git diff` as the model patch.
3. **Restore test infrastructure** (`restore_grading_files`): every path touched by `test_patch` and every
   `conftest.py`, `pytest.ini`, `.pytest.ini`, `pytest.toml`, `.pytest.toml`, `tox.ini`, `setup.cfg`,
   `pyproject.toml`, `noxfile.py`, `.coveragerc`, `.gitattributes`, `sitecustomize.py`, `usercustomize.py` or `*.pth`
   is deleted and rewritten from the blob ids recorded at reset (`capture_grading_snapshot`), or stays deleted if it
   did not exist at the base revision.
4. Apply `test_patch`; a patch that fails to apply scores `0.0`.
5. Kill agent processes a second time, then run the `fail_to_pass` and `pass_to_pass` commands.

The diff, restore and apply steps run git through `run_clean_git`: a fresh git dir that borrows only the object
store and index, with system/global config, hooks, attributes filters and fsmonitor disabled, so nothing the agent
wrote under `.git/` runs during grading. The test commands themselves still run in the agent's repository; a test
command that calls git reads that repository's config.

**Resolved** (reward `1.0`) iff at least one test list is non-empty, every fail-to-pass command passed **and** every
pass-to-pass command passed (`verification_resolved`). Info reports `f2p_passed/f2p_total/p2p_passed/p2p_total`.

### `terminal`

Seeds `files`, runs `setup_script` (failure -> reset `env_error`), executes ```` ```bash ```` / `<bash>` blocks or
`bash` tool calls, and runs `verify_script` once the agent submits without commands or hits `max_steps`. Default
`ContainerConfig(use_jail="required")`.

Before verification all agent processes are killed (failure -> unscorable). A fresh nonce is then drawn and the
script receives `ED_REWARD_NONCE`, `ED_REWARD_FILE` (a nonce-named, freshly emptied file) and
`AGENT_FINAL_RESPONSE`. The reward is read, first match wins:

1. a float in `$ED_REWARD_FILE` (clamped to `[0, 1]`);
2. the last stdout line `REWARD[$ED_REWARD_NONCE]=<float>` (clamped);
3. the exit status (`0` -> `1.0`).

Plain `REWARD=<x>` lines are never trusted. If the script text uses the old plain `REWARD=` marker (and mentions
neither `ED_REWARD_NONCE` nor `ED_REWARD_FILE`) and stdout contains such a line, the episode is unscorable.

```sh
# verify_script
test -f out.txt && echo "REWARD[$ED_REWARD_NONCE]=1" || echo 0.25 > "$ED_REWARD_FILE"
```

### `python_tir`

Multi-turn Python REPL: variables persist across turns; code blocks are executed and their output returned until the
agent answers without code (or reaches `max_steps`, default `6`). Grading uses `verifier(action)` when given, else
numeric comparison against `answer`, else the last `\boxed{...}` (or the stripped action) against `answer`.
`grade_single_turn(action)` grades without executing code; a raising verifier gives an unscorable step.

### `synthetic_tool`

No sandbox. Each `reset()` deep-copies `initial_state`; `tools` (mapping or sequence of callables) read and mutate
`state`; `verifier(state, final_response)` computes the reward when the agent calls `submit_result` or answers without
tool calls. `tool_schemas` exposes OpenAI-style schemas. Options: `format_penalty`, `halt_on_invalid_tool`,
`max_tool_calls_per_step`.

### `harness`

Runs an unmodified CLI agent from a `HarnessPreset` (`list_harnesses()`, `register_harness()`) inside the sandbox
(default `ContainerConfig(network=True, timeout=harness_timeout, use_jail="required")`). Model calls go through a
`HarnessTITOGateway`, an HTTP server serving `/v1/chat/completions`, `/v1/responses` and `/v1/messages` that records
token-in-token-out trajectories (`HarnessEnvironment.last_trajectories`). Each episode gets its own API key from
`open_episode()`; unknown keys get `401`.

| `HarnessTITOGateway` arg | meaning                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `host`, `port`           | bind address (default `127.0.0.1`, `0` = free port)                                           |
| `advertise_host`         | host/IP written into harness URLs instead of the bind address (bound port kept)               |
| `public_base_url`        | full URL written into harness URLs; overrides `advertise_host`                                |

With a container session, `reset` raises `ValueError` if the advertised URL is loopback
(`advertised_is_loopback`): bind a reachable interface and set `advertise_host` or `public_base_url`. Grading uses
`verify_script` (exit status) when set, else the `test_patch` + fail-to-pass/pass-to-pass rule of `swe_repo`.

### `llm_judge`

Single-turn; `step` scores the action with a `RubricJudge` (pass `judge=`, or `base_url` + `model` and optional
`criteria` / `client_kwargs` for a `JudgeClient`). When the judge cannot score, the step carries
`info["env_error"] = True` and `info["unscorable"] = True`. `make_judge_reward_fn` builds a judge-backed
`reward_funcs` callable directly; `PairwiseJudge` scores a group by pairwise win rate. With `max_pairs` below
the full `n * (n - 1) / 2`, the schedule gives every member at least one comparison and per-member counts differ
by at most one; `max_pairs < ceil(n / 2)` raises `ValueError`.

## Reward helpers (`rewards.py`)

| symbol                                       | behaviour                                                                                                     |
|----------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `correct_answer_floor(ceiling, correct)`     | `ceiling + (correct - ceiling) / 2`; raises unless `ceiling < correct`                                         |
| `compose_outcome_reward(correct, terms, ...)` | outcome + sum of finite shaping terms, then correct samples floored and incorrect samples capped at `incorrect_ceiling`, so correct always outranks incorrect |
| `structural_validity`, `validate_structural_config`, `StructuralValidity` | token-level structural checks                                                   |
| `gate_invalid(reward, valid, *, cap=0.0)`    | caps the reward of invalid samples                                                                            |
| `lexicographic_blend`                        | ordered combination of reward terms                                                                           |
| `think_budget_penalty`, `count_think_tokens` | thinking-length penalty                                                                                       |
| `RewardShaper(base_reward, ...)`             | `RewardProtocol` wrapper applying the above; `None` / non-finite base scores stay `None`; copies the base's `discard_unscorable` marker |

## Async collector

`AsyncEnvironmentRolloutCollector(environment_factory, *, generate_turn_fn, encode_fn, max_workers=8,
max_outstanding=32, max_turns=8, max_staleness=2, drop_env_errors=True)` steps episodes in background threads and
returns them in completion order.

- `publish_policy(state, version=...)` publishes a weight snapshot; each turn of a running episode reads the latest
  one, and episodes track their oldest and newest policy version.
- Generated tokens carry `action_mask=1` and behaviour log-probs; observation tokens carry `action_mask=0`.
- Episodes whose oldest version lags by more than `max_staleness` are dropped as stale; `env_error` episodes are
  dropped unless `drop_env_errors=False` (then kept with the flag set and `trajectory=None`). Kept env-error
  records never count toward `collect_episodes(count)` and are never collated or included in the
  `collect_collated_batch` metrics; if no episode has a trajectory, it raises `RuntimeError`.
- `collect_groups(num_groups=, group_size=, current_policy_version=)` pads a group with at least
  `ceil(group_size / 2)` valid episodes by repeating valid members and drops sparser groups. If it raises
  (for example a `TimeoutError` from `timeout_per_episode`), its in-flight submissions are cancelled where
  possible and the rest are discarded when they complete, so no later call returns them.
- `collect_episodes`, `collect_collated_batch`, `close()`.
- The collector is single-consumer: the drain loops of `collect_episodes` and `collect_groups` share a lock,
  so concurrent calls run one after another instead of taking each other's completions.
