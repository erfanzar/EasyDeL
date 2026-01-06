# EasyDeL TPU Cluster Autoscaling with Ray

This directory contains automated TPU cluster configuration and management tools for Google Cloud Platform, designed to work seamlessly with EasyDeL and eFormer for distributed training and inference on TPUs.

## Overview

The autoscaling system automatically discovers available TPU resources across GCP zones and generates Ray cluster configurations for each zone. It supports multiple TPU generations (v4, v5e, v5p, v6e) and handles quota management, preemptible instances, and multi-slice TPU deployments.

## Quick Start

### 1. Launch a Cluster

To launch a Ray cluster with TPUs in a specific zone:

```bash
# Recommended: run the repo-pinned Ray to avoid local/cluster version mismatches.
uv run --python 3.13 ray up autoscale/easydel-europe-west4-a.yaml --no-config-cache

# Or, if you manage Ray globally on your machine:
# ray up autoscale/easydel-europe-west4-a.yaml --no-config-cache
```

To connect to the cluster:

```bash
uv run --python 3.13 ray attach autoscale/easydel-europe-west4-a.yaml
```

To shut down the cluster:

```bash
uv run --python 3.13 ray down autoscale/easydel-europe-west4-a.yaml
```

Notes:
- Ray >=2.53 supports `.rayignore` for controlling cluster uploads (this repo includes one).
- If you run via `uv run` and want the pre-2.47 behavior where only the driver uses the uv environment, set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`.

## TPU Generations and Configurations

### Supported TPU Types

| Generation | Chip Count | Slices Available | Runtime Version | Use Case |
|------------|------------|-----------------|-----------------|----------|
| **v4** | 4 chips/node | 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 | tpu-ubuntu2204-base | Training large models |
| **v5e** | 4 chips/node | 8, 16, 32, 64, 128, 256 | v2-alpha-tpuv5-lite | Cost-effective inference |
| **v5p** | 4 chips/node | 8, 16, 32, 64, 128, 256, 512, 1024, 2048 | v2-alpha-tpuv5 | High-performance training |
| **v6e** | 4 chips/node | 8, 16, 32, 64, 128, 256 | v2-alpha-tpuv6e | Latest generation inference |

### Understanding TPU Slices

- **Slice Size**: Number indicates TPU cores (e.g., v5e-256 = 256 TPU cores)
- **Multi-Slice**: Large models can span multiple TPU pods
- **Preemptible**: All configurations use preemptible (spot) instances for cost savings

## Ray Dashboard

### Setting Up the Dashboard

1. Forward the dashboard port when attaching to the cluster:

```bash
uv run --python 3.13 ray attach autoscale/easydel-us-central1-a.yaml --port-forward 8265
```

1. Access the dashboard at `http://localhost:8265`

The dashboard provides:

- Real-time cluster status
- TPU utilization metrics
- Job execution monitoring
- Worker node management
- Resource allocation overview

## Using with EasyDeL/eFormer

### Basic Health Check

```python
# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

import ray
from eformer.executor.ray import TpuAcceleratorConfig, autoscale_execute_resumable

BUCKET_PATH = "gs://your-bucket/easydel-checkpoints"

execution_env = {
    "working_dir": ".",
    "env_vars": {
        "EASYDEL_AUTO": "1",
        "HF_TOKEN": "YOUR_HF_TOKEN",
        "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",
        "HF_HOME": "/dev/shm/huggingface",
        "HF_DATASETS_OFFLINE": "0",
        "WANDB_API_KEY": "YOUR_WANDB_KEY",
        "ENABLE_DISTRIBUTED_INIT": "1",
    },
    "excludes": [".git/**", ".venv", "**/__pycache__", "*.pt", "*.safetensors", "data/", "checkpoints/", "wandb/"],
}

ray.init(runtime_env=execution_env)

@autoscale_execute_resumable(TpuAcceleratorConfig("v4-512", 4, execution_env=execution_env))
def health_check():
    import jax
    print(f"Available devices: {jax.local_devices()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Process index: {jax.process_index()}")
    return jax.device_count()

if __name__ == "__main__":
    result = health_check()
    print(f"Total TPU cores available: {result}")
```

## Cluster Management

### Updating Cluster Configurations

To regenerate cluster configurations based on current TPU availability:

```bash
python autoscale/update-cluster.py \
    --project-id YOUR_PROJECT_ID \
    --output-dir autoscale/ \
    --families v4 v5e v5p v6e \
    --print-summary
```

### Monitoring Cluster Status

```bash
# Check cluster status
uv run --python 3.13 ray status

# View cluster resources
uv run --python 3.13 ray cluster-resources

# Monitor running jobs
uv run --python 3.13 ray job list
```

### Cost Optimization Tips

1. **Use Preemptible TPUs**: All configurations use preemptible instances by default (up to 70% cost savings)
2. **Right-size Your Slices**: Start with smaller slices and scale up as needed
3. **Regional Distribution**: Use zones with better availability for preemptible instances
4. **Automatic Shutdown**: Configure idle timeout in cluster YAML

## Environment Variables

Key environment variables for EasyDeL/eFormer:

| Variable | Description | Example |
|----------|-------------|---------|
| `EASYDEL_AUTO` | Enable automatic configuration | `"1"` |
| `HF_TOKEN` | Hugging Face access token | `"hf_xxx"` |
| `WANDB_API_KEY` | Weights & Biases API key | `"xxx"` |
| `TPU_WORKER_ID` | TPU worker identification | Auto-set |
| `ENABLE_DISTRIBUTED_INIT` | Enable distributed workload init | `"1"` |

## Troubleshooting

### Common Issues and Solutions

1. **TPU Allocation Failed**
   - Check quota in the specific zone
   - Try a different zone with availability
   - Ensure TPU API is enabled in your project

2. **Connection Timeout**
   - Verify firewall rules allow Ray ports (6379, 8265, 10001)
   - Check VPC network configuration

3. **Ray Version / SSH Key Issues**
   - Run `ray up/attach/down` via `uv run --python 3.13 ray ...` to keep local CLI and cluster Ray versions in sync
   - If you set `auth.ssh_private_key`, note Ray 2.51+ changed the default GCP SSH key naming/path (private key no longer uses `.pem`)

4. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use model parallelism with EasyScaler

5. **Preemption Handling**
   - Use `autoscale_execute_resumable` decorator
   - Configure checkpoint saving frequency
   - Implement automatic restart logic

## Advanced Configuration

### Custom Node Types

You can modify the cluster YAML to add custom node configurations:

```yaml
available_node_types:
  tpu_custom_v5e_64:
    min_workers: 2  # Always keep 2 workers running
    max_workers: 10
    resources:
      CPU: 120
      TPU: 4
      custom_resource: 1
    node_config:
      acceleratorType: v5litepod-64
      runtimeVersion: v2-alpha-tpuv5-lite
      schedulingConfig:
        preemptible: false  # Use on-demand for critical workloads
```

### Multi-Region Deployment

For fault tolerance and better availability:

```bash
# Launch clusters in multiple regions
uv run --python 3.13 ray up autoscale/easydel-us-central1-a.yaml --no-config-cache &
uv run --python 3.13 ray up autoscale/easydel-europe-west4-a.yaml --no-config-cache &
uv run --python 3.13 ray up autoscale/easydel-asia-northeast1-a.yaml --no-config-cache &
```

## Resources

- [EasyDeL Documentation](https://github.com/erfanzar/EasyDeL)
- [eFormer Documentation](https://github.com/erfanzar/eFormer)
- [Ray Documentation](https://docs.ray.io/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)

## Support

For issues or questions:

- Open an issue on [EasyDeL GitHub](https://github.com/erfanzar/EasyDeL/issues)
- Check existing [discussions](https://github.com/erfanzar/EasyDeL/discussions)
- Consult the [Ray community](https://discuss.ray.io/)
