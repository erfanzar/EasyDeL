# Dataset Profiling

The `DatasetProfiler` gives a quick, non-mutating health check of a sharded
data source before a training run. It samples rows, reports sequence-length
distributions, field presence, truncation/padding rates, source mix, and
estimates token-packing efficiency for a target window size.

## Basic usage

```python
from easydel.data import profile_dataset, JsonShardedSource

source = JsonShardedSource("data/*.jsonl")
profile = profile_dataset(source, seq_length=2048, max_rows=5000)
```

`profile_dataset` prints and returns a `DatasetProfile`:

```md
DatasetProfile(
  num_rows_sampled=5000
  field_counts={'input_ids': 5000, 'attention_mask': 5000, ...}
  length_histogram={'input_ids': {42: 120, 128: 900, ...}, ...}
  length_percentiles={'input_ids': {'p50': 128.0, 'p90': 512.0, 'p99': 1024.0}, ...}
  truncation_rate=0.0010
  padding_rate=0.1200
  chat_template_fallback_rate=0.0000
  source_distribution={'openhermes': 3000, 'codealpaca': 2000}
  source_token_counts={'openhermes': 384000, 'codealpaca': 256000}
  total_input_tokens=640000
  estimated_packing_efficiency={'greedy': 0.82, 'pool': 0.89, 'first_fit': 0.91}
  packing_efficiency_gap={'pool_vs_greedy': 0.07, 'first_fit_vs_greedy': 0.09}
  seq_length=2048
)
```

## Programmatic access

Use `DatasetProfiler` directly when you want to inspect the result without
printing:

```python
from easydel.data import DatasetProfiler
from easydel.data.core.protocols import ShardedDataSource

profiler = DatasetProfiler(max_rows=10_000, seq_length=2048)
profile = profiler.profile(source)
print(profile.to_dict())
```

## Pipeline integration

Call `.profile()` at any point before `.load()` to inspect the current stage:

```python
from easydel.data import Pipeline

pipeline = Pipeline.from_config(config)
profiles = pipeline.source().tokenize().pack().profile(max_rows=5000)
# profiles is a dict: {dataset_name: DatasetProfile}
```

## Fields and interpretation

- `num_rows_sampled`: rows consumed (capped by `max_rows`).
- `field_counts`: how many sampled rows contained each field.
- `length_histogram`: exact `length -> count` histogram for `input_ids`,
  `attention_mask`, and `labels` (configurable via `length_fields`).
- `length_percentiles`: `p50`, `p90`, `p99` length summaries per field.
- `truncation_rate`: fraction of rows with a truthy `"truncated"` field.
  `0.0` when the field is absent.
- `padding_rate`: fraction of rows where `sum(attention_mask) < len(input_ids)`.
- `chat_template_fallback_rate`: fraction of rows with
  `chat_template_applied=False` or `template_fallback=True`.
- `source_distribution`: counts per `"__source__"` value (set by mix stages).
- `source_token_counts`: sum of `len(input_ids)` per `"__source__"` value.
- `total_input_tokens`: total `len(input_ids)` across all sampled rows.
- `estimated_packing_efficiency`: greedy/pool/first-fit efficiency estimates
  for the requested `seq_length`, computed by the same packers used at
  training time. Empty when `seq_length` is not provided.
- `packing_efficiency_gap`: `pool_vs_greedy` and `first_fit_vs_greedy`
  efficiency differences. A large positive gap means smarter packing pays off.

## Notes

- Profiling is additive and read-only: it never mutates the source or pipeline.
- Packing efficiency is an estimate based on `input_ids` lengths; it includes
  the synthetic EOS separators used by the packers.
- For large datasets, keep `max_rows` modest (the default is 10,000); the
  profiler streams rows and uses only CPU memory.
