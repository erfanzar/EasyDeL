# Writing an Eyvan capacity provider

A provider is one driver, one capability entry, and a transport — with no edit
to the scheduler, the planner, or the durable model. If you find yourself
reaching into `eyvan.server` or copying a driver to make your provider work,
that is a bug in this contract: open an issue rather than forking.

`aws-ec2` is the worked example to read, because it was written after this
contract existed: one driver, one capability entry, one transport, and two
real ownership bugs in it were found by the conformance suite rather than its
own tests.

## 1. The shape

```text
eyvan_acme_provider/
├── __init__.py        # driver + capability entry + transport registration
└── pyproject.toml     # entry point registering the package with eyvan
```

The driver implements the provider interface the capacity layer calls
(request capacity, describe, delete, reconcile); the capability entry tells
the core what your backend can do; the transport is how your cloud is
reached. Nothing else. The core never learns your cloud's name.

## 2. Turning it on

Install the package into the environment that creates the capacity driver.
Publish a named entry point in the `eyvan.capacity_providers` group, then
explicitly allow its entry-point name in `EYVAN_CAPACITY_PROVIDER_PLUGINS`
(comma-separated). Discovery is **off by default**: installing a distribution
or selecting a backend alone does not authorize importing its plugin code.
`*` trusts every advertised plugin and should not be used where dependencies
are not equally trusted. Plugin code runs inside the host process, not a sandbox.

Select an advertised backend with `EYVAN_CAPACITY_BACKEND` or the fleet's
capacity-backend configuration. Pools appear in `eyvan capacity list` once
quota/fleet state says they exist. Import failures, missing allowed plugins,
invalid contracts and backend-name collisions are reported; plugins cannot
silently replace built-in backends.

## 3. The version contract

The extension ABI is **1.0**, declared by
`CAPACITY_PROVIDER_ABI_VERSION` in `eyvan/capacity/drivers/registry.py`.
The entry point resolves to an object satisfying `CapacityProviderPlugin`
(in `eyvan/capacity/drivers/base.py`), which may simply be a module with:

- `abi_version`: a `"MAJOR.MINOR"` string;
- `backend_names`: the backend names the plugin serves;
- `create(name, *, state, settings)`: a factory returning a `CapacityDriver`.

No base-class inheritance is required. Keep the ABI declaration importable
without importing Eyvan; import driver dependencies inside `create` where
necessary. A different major version or a newer minor version is refused,
not silently adapted. An older minor within the same major is accepted.
`validate_provider_plugin` in the registry exposes the same contract gate
used by discovery. Changes that withdraw guarantees or change signatures
require a major ABI change; additive guarantees require a minor change.

The ABI version is **not** the provider distribution's release version, and
this interface does not guarantee automatic persistence of a provider package
version on every cell or workload. Record package/build identity explicitly
in qualification receipts: a proof run against provider X vN says nothing
about vN+1. Discovery/version-policy regressions live in
`tests/test_provider_abi.py`; behavioral qualification lives in
`tests/test_provider_abi_conformance.py` and the per-provider ABI suites.
Keep these checks green per release; they do not replace real-provider
ownership, fencing and cleanup validation.

## 4. What Eyvan guarantees to you

- The scheduler, the capacity planner, and the durable model will not name
  your provider or special-case it — enforced by
  `tests/test_provider_boundary.py`.
- Your driver decides machine-level truth (exists? booting? gone?) and the
  core trusts your answers, including "unavailable/inconclusive" — ambiguity
  is surfaced, never converted into a prettier verdict.
- Planning uses the in-tree backend capability table. ABI 1.0 plugins cannot
  register capability declarations; an unknown backend is limited to
  single-machine cells.

## 5. What you guarantee back

- Every machine you create is trackable to a cell: if you created it, the
  plane can name and reap it. Ownership bugs are the worst kind — the
  conformance suite exists to catch them.
- Refusals carry their reason. The renderer prints your diagnostic verbatim
  under `blocked … the plane is retrying`; a bare failure code helps nobody.
- Deletion is best-effort-and-reported: an uncertain delete result is
  retryable state, never success (see [artifact-safety](artifact-safety.md)
  for the same rule on the object side).

## Capability entries

`BackendCapabilities` and `_BACKEND_CAPABILITIES` live in
`eyvan/capacity/drivers/base.py`. The in-tree table defines planning limits,
including whether one cell may span several machines and its worker ceiling.
ABI 1.0 exposes no plugin capability field. Multi-worker atomic planning needs
an in-tree capability entry and a pool factory that supports the shape;
a plugin declaration alone cannot enable it.

## Qualification

Host and disposable-PostgreSQL suites exercise the core with fake providers;
your provider qualifies through the conformance suite plus a recorded spend
on real infrastructure. A provider claim without an immutable
exact-candidate receipt is not a claim — see [proofs](proofs.md).
