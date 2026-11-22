# EasyDeL TPU Autoscale (moved into eray)

This toolchain now lives in the eray CLI:

```bash
eray autoscale generate [--zones us-central1-a] [--families v5p] [--spot|--on-demand]
eray autoscale up   ~/.eray/autoscale/easydel-us-central1-a.yaml
eray autoscale down ~/.eray/autoscale/easydel-us-central1-a.yaml
eray autoscale status <config>
```

Improvements over the old generator here: node types carry the
eray-canonical resource labels (`TPU-{fam}-{size}-head` with the casing the
eray pool scheduler requires; `TPU` = physical chips per host), spot is a
flag instead of a hardcode, GCP is queried through gcloud (no
google-api-python-client), and the YAML is assembled structurally instead of
by string concatenation. The template ships inside the eray package
(`eray/provision/templates/cluster-template.yaml`).

For reliable large slices prefer the queued-resource fleet path
(`eray qr` / `eray fleet` / `eray fleet watch`) — Ray's GCP provider notes
that multi-host TPU autoscaling through the launcher is best-effort.

`generate-cluster-configs.sh` in this directory delegates to the eray CLI
for muscle-memory compatibility.
