# Sharding strategies

Five runnable recipes showing how to express SPMD parallelism in
spectrax. Each file works on a single CPU (trivially replicated) and
on any multi-device backend.

| File                             | Topic                                                 |
| -------------------------------- | ----------------------------------------------------- |
| `01_fsdp.py`                     | 1-D `(fsdp,)` mesh, `sharding=("fsdp", None)` Linears |
| `02_tensor_parallel.py`          | 1-D `(tp,)` mesh, column + row parallel MLP           |
| `03_fsdp_tp_hybrid.py`           | 2-D `(fsdp, tp)` mesh, both axes per weight           |
| `04_logical_rules.py`            | Logical axis names mapped by `logical_axis_rules`     |
| `05_with_sharding_constraint.py` | `with_sharding_constraint_by_name` + HLO dump         |

Run any one of them from the repository root, e.g.::

    python -m examples.05_shardings.01_fsdp
    python -m examples.05_shardings.02_tensor_parallel
    python -m examples.05_shardings.03_fsdp_tp_hybrid
    python -m examples.05_shardings.04_logical_rules
    python -m examples.05_shardings.05_with_sharding_constraint

Key APIs exercised:

- `spx.create_mesh`, `spx.SpxMesh`
- `spx.nn.Linear(..., sharding=(...))`
- `spx.sharding.get_partition_spec`, `get_named_sharding`
- `spx.sharding.logical_axis_rules`, `current_axis_rules`
- `spx.sharding.with_sharding_constraint_by_name`
