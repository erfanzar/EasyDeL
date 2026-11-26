# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Remote training payload; host imports and checkpoint hooks do not initialize JAX."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SCHEMA = "eyvan.spot-training/v1"
DEVICE_COUNT = 4


def validate_checkpoint_uri(uri: str) -> str:
    if not isinstance(uri, str) or any(ord(char) < 33 for char in uri):
        raise ValueError("invalid checkpoint URI")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "gs"
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", parsed.netloc)
        or parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
        or any(part in {"", ".", ".."} for part in parsed.path.lstrip("/").split("/"))
    ):
        raise ValueError("expected an explicit gs://bucket/prefix/checkpoint.json URI")
    return uri


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_checkpoint(checkpoint, *, run_id):
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA or checkpoint.get("run_id") != run_id:
        raise ValueError("checkpoint identity mismatch")
    if checkpoint.get("step") != 20:
        raise ValueError("checkpoint must contain step 20")
    weights = checkpoint.get("weights")
    if not isinstance(weights, list) or len(weights) != 32 or not all(_finite(value) for value in weights):
        raise ValueError("checkpoint must contain 32 finite weights")
    initial, saved = checkpoint.get("initial_loss"), checkpoint.get("checkpoint_loss")
    if not _finite(initial) or not _finite(saved) or not 0 <= saved < initial:
        raise ValueError("checkpoint must record finite, improving loss")
    if not isinstance(checkpoint.get("instance"), str) or not checkpoint["instance"]:
        raise ValueError("checkpoint instance is missing")
    return checkpoint


def metadata_value(path: str) -> str:
    if path not in {"project/project-id", "instance/scheduling/preemptible", "instance/name"}:
        raise ValueError("unexpected metadata path")
    request = Request(
        "http://metadata.google.internal/computeMetadata/v1/" + path, headers={"Metadata-Flavor": "Google"}
    )
    with urlopen(request, timeout=2) as response:
        raw = response.read(4097)
    if not raw or len(raw) > 4096:
        raise ValueError("invalid metadata response size")
    return raw.decode("ascii").strip()


def validate_instance(*, project):
    if metadata_value("project/project-id") != project:
        raise ValueError("instance project mismatch")
    if metadata_value("instance/scheduling/preemptible").lower() != "true":
        raise ValueError("this example requires Spot capacity")
    return metadata_value("instance/name")


def validate_runtime(*, project):
    instance = validate_instance(project=project)
    import jax

    if jax.default_backend() != "tpu" or jax.local_device_count() != DEVICE_COUNT:
        raise ValueError("this example requires four local TPU devices")
    return instance


def _blob(uri):
    from google.cloud import storage

    parsed = urlsplit(validate_checkpoint_uri(uri))
    return storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/"))


def _read_checkpoint(uri):
    blob = _blob(uri)
    if not blob.exists(timeout=5, retry=None):
        return None
    # Bound the checkpoint payload rather than downloading arbitrary remote data.
    raw = blob.download_as_bytes(start=0, end=65536, timeout=5, retry=None)
    if len(raw) > 65536:
        raise ValueError("checkpoint exceeds example size limit")
    return json.loads(raw)


def _create_checkpoint(uri, checkpoint):
    _blob(uri).upload_from_string(
        json.dumps(checkpoint, allow_nan=False),
        content_type="application/json",
        if_generation_match=0,
        timeout=5,
        retry=None,
    )


def _emit(marker, **fields):
    print(json.dumps({"marker": marker, **fields}, allow_nan=False), flush=True)


def train(project, run_id, uri):
    instance = validate_runtime(project=project)
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(20260911)
    x = jnp.asarray(rng.normal(size=(DEVICE_COUNT, 256, 32)), dtype=jnp.float32)
    y = x @ jnp.ones((32,), dtype=jnp.float32)

    def update(weights, inputs, targets):
        gradient = jax.grad(lambda w: jnp.mean((inputs @ w - targets) ** 2))(weights)
        return weights - 0.1 * jax.lax.pmean(gradient, "replica")

    step = jax.pmap(update, axis_name="replica")

    def loss(weights):
        return float(jnp.mean((jnp.einsum("dij,dj->di", x, weights) - y) ** 2))

    checkpoint = _read_checkpoint(uri)
    if checkpoint is None:
        weights = jnp.zeros((DEVICE_COUNT, 32), dtype=jnp.float32)
        initial = loss(weights)
        for _ in range(20):
            weights = step(weights, x, y)
        weights.block_until_ready()
        checkpoint = {
            "schema": SCHEMA,
            "run_id": run_id,
            "step": 20,
            "weights": np.asarray(weights[0]).tolist(),
            "initial_loss": initial,
            "checkpoint_loss": loss(weights),
            "instance": instance,
        }
        validate_checkpoint(checkpoint, run_id=run_id)
        _create_checkpoint(uri, checkpoint)
        _emit("checkpoint_ready", run_id=run_id, step=20, device_count=DEVICE_COUNT, instance=instance)
        while True:
            time.sleep(1)  # The scheduler's checkpoint/stop protocol interrupts this trainer.
    checkpoint = validate_checkpoint(checkpoint, run_id=run_id)
    weights = jnp.broadcast_to(jnp.asarray(checkpoint["weights"], dtype=jnp.float32), (DEVICE_COUNT, 32))
    for _ in range(20):
        weights = step(weights, x, y)
    weights.block_until_ready()
    control = jnp.zeros((DEVICE_COUNT, 32), dtype=jnp.float32)
    for _ in range(40):
        control = step(control, x, y)
    control.block_until_ready()
    np.testing.assert_allclose(np.asarray(weights), np.asarray(control), rtol=1e-5, atol=1e-6)
    final = loss(weights)
    if not math.isfinite(final) or final >= checkpoint["checkpoint_loss"]:
        raise ValueError("restored training did not improve the checkpoint loss")
    _emit(
        "training_restored",
        run_id=run_id,
        restored_step=20,
        final_step=40,
        device_count=DEVICE_COUNT,
        checkpoint_instance=checkpoint["instance"],
        restored_instance=instance,
        matches_uninterrupted_control=True,
        initial_loss=checkpoint["initial_loss"],
        checkpoint_loss=checkpoint["checkpoint_loss"],
        final_loss=final,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "hook", "urgent"))
    parser.add_argument("--gcp-project", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--devices", type=int, default=DEVICE_COUNT)
    args = parser.parse_args(argv)
    validate_checkpoint_uri(args.checkpoint_uri)
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", args.gcp_project):
        raise ValueError("invalid GCP project ID")
    if not re.fullmatch(r"[a-f0-9]{32}", args.run_id) or args.devices != DEVICE_COUNT:
        raise ValueError("invalid run identity or device count")
    if args.mode == "hook":
        # Do not acquire the accelerator: the batch process still owns it during its hook.
        if os.environ.get("EYVAN_CHECKPOINT_URI") != args.checkpoint_uri:
            raise ValueError("checkpoint hook URI mismatch")
        instance = validate_instance(project=args.gcp_project)
        validate_checkpoint(_read_checkpoint(args.checkpoint_uri), run_id=args.run_id)
        _emit("checkpoint_verified", run_id=args.run_id, step=20, instance=instance)
    elif args.mode == "train":
        train(args.gcp_project, args.run_id, args.checkpoint_uri)
    else:
        instance = validate_runtime(project=args.gcp_project)
        import jax.numpy as jnp

        value = float(jnp.sum(jnp.ones((4096,))).block_until_ready())
        if value != 4096:
            raise ValueError("urgent TPU computation mismatch")
        _emit("urgent_complete", run_id=args.run_id, device_count=DEVICE_COUNT, instance=instance)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Spot training payload failed: " + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
