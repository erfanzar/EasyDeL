# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Hyperparameter sweep on one TPU host via eray.swarm.

Launches four independent training jobs on a single v5p-8 (4 chips), one chip
per job, sweeping learning rate x warmup steps. Each job is a fully isolated
JAX runtime: ``jax.devices()`` inside a run sees only that run's chip, so the
four trainings run truly in parallel on one host.

Run from a driver connected to the eray Ray cluster:

    python eray_swarm_lr_sweep.py

Notes:
    - The model must fit a single chip; for a 2-chip job use an explicit plan,
      e.g. ``SwarmConfig(tpu_version="v5p-8", chip_split=(2, 1, 1))``.
    - Each run gets ``ERAY_RUN_ID`` / ``ERAY_RUN_NAME`` in its environment.
"""

import eray

SWEEP = [
    {"learning_rate": 3e-4, "warmup_steps": 100},
    {"learning_rate": 1e-4, "warmup_steps": 100},
    {"learning_rate": 3e-4, "warmup_steps": 500},
    {"learning_rate": 1e-4, "warmup_steps": 500},
]

SWEEP_RUNS = [
    eray.SwarmRun(
        name=f"lr{hp['learning_rate']:g}-w{hp['warmup_steps']}",
        f_kwargs={**hp, "run_name": f"lr{hp['learning_rate']:g}-w{hp['warmup_steps']}"},
        # Each run is a single-process world; don't join a distributed runtime.
        env={"ENABLE_DISTRIBUTED_INIT": "0"},
    )
    for hp in SWEEP
]


@eray.swarmed(
    eray.SwarmConfig(tpu_version="v5p-8"),  # 4 chips / 4 runs = 1 chip each
    runs=SWEEP_RUNS,
)
def train(learning_rate: float, warmup_steps: int, run_name: str) -> dict:
    """One sweep job. Runs in its own process owning one TPU chip."""
    import jax

    import easydel as ed

    assert len(jax.devices()) == 1, "each sweep job should only see its own chip"

    args = ed.TrainingArguments(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_directory=f"gs://your-bucket/sweeps/{run_name}",
        # ... the rest of your usual arguments (steps, batch size, dtype, ...)
    )

    # Build the model + dataset and train exactly as you normally would:
    #   model = ed.AutoEasyDeLModelForCausalLM.from_pretrained("...")
    #   trainer = ed.SFTTrainer(arguments=args, model=model, train_dataset=ds, ...)
    #   output = trainer.train()
    #   final_loss = float(output.metrics["loss"])
    final_loss = 0.0  # placeholder — return your real final metric

    return {
        "run": run_name,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "loss": final_loss,
    }


def main():
    status = train()  # launches all four runs; blocks until the sweep finishes

    if not isinstance(status, eray.JobSucceeded):
        raise SystemExit(f"sweep failed: {status}")

    for result in sorted(status.result, key=lambda r: r["loss"]):
        print(
            f"{result['run']:>16}  lr={result['learning_rate']:g}  warmup={result['warmup_steps']}  loss={result['loss']:.4f}"
        )


if __name__ == "__main__":
    main()
