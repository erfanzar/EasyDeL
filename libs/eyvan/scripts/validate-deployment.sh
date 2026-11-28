#!/usr/bin/env bash
# Copyright (C) 2026 Erfan Zare Chavoshi
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail

readonly HELM_VERSION="v3.18.6"
readonly KUBECONFORM_VERSION="v0.7.0"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly CHART="${ROOT}/deploy/helm/eyvan"

resolve_tool() {
  local explicit="$1"
  local name="$2"
  local bundled="/tmp/eyvan-tools/${name}"
  if [[ -n "${explicit}" ]]; then
    printf '%s\n' "${explicit}"
  elif command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
  elif [[ -x "${bundled}" ]]; then
    printf '%s\n' "${bundled}"
  else
    echo "missing required ${name}; install the pinned validation binary" >&2
    return 1
  fi
}

HELM_BIN="$(resolve_tool "${HELM_BIN:-}" helm)"
KUBECONFORM_BIN="$(resolve_tool "${KUBECONFORM_BIN:-}" kubeconform)"
[[ -x "${HELM_BIN}" ]] || { echo "HELM_BIN is not executable: ${HELM_BIN}" >&2; exit 1; }
[[ -x "${KUBECONFORM_BIN}" ]] || { echo "KUBECONFORM_BIN is not executable: ${KUBECONFORM_BIN}" >&2; exit 1; }

helm_actual="$(${HELM_BIN} version --short)"
[[ "${helm_actual}" == "${HELM_VERSION}"+g* ]] || {
  echo "Helm version mismatch: require ${HELM_VERSION}, got ${helm_actual}" >&2
  exit 1
}
kubeconform_actual="$(${KUBECONFORM_BIN} -v)"
[[ "${kubeconform_actual}" == "${KUBECONFORM_VERSION}" ]] || {
  echo "kubeconform version mismatch: require ${KUBECONFORM_VERSION}, got ${kubeconform_actual}" >&2
  exit 1
}

tmp="$(mktemp -d "${TMPDIR:-/tmp}/eyvan-deployment-validation.XXXXXX")"
trap 'rm -rf "${tmp}"' EXIT
readonly digest="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
readonly content_sha="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

"${HELM_BIN}" lint "${CHART}"
"${HELM_BIN}" template eyvan "${CHART}" \
  --namespace eyvan-system >"${tmp}/helm-default-disabled.yaml"
"${HELM_BIN}" template eyvan "${CHART}" \
  --namespace eyvan-gcp \
  --set-string image.repository=registry.invalid/eyvan-control-plane \
  --set-string image.digest="${digest}" \
  --set-string agentImage.repository=registry.invalid/eyvan-agent \
  --set-string agentImage.digest="${digest}" \
  --set cellAgent.enabled=true \
  --set-string cellAgent.fleetStateUri=gs://state-bucket.invalid/eyvan/gcp-state.json \
  --set-string cellAgent.capacityBackend=gcp \
  --set-string cellAgent.nativeWorkerPackage=gs://artifacts.invalid/eyvan.whl \
  --set-string cellAgent.nativeWorkerPackageSha256="${content_sha}" \
  --set-string cellAgent.nativeWorkerPackageGeneration=123 \
  --set-string cellAgent.nativeWorkerWheelhouse=gs://artifacts.invalid/wheelhouse.zip \
  --set-string cellAgent.nativeWorkerWheelhouseSha256="${content_sha}" \
  --set-string cellAgent.nativeWorkerWheelhouseGeneration=456 >"${tmp}/helm-gcp.yaml"
"${HELM_BIN}" template eyvan "${CHART}" \
  --namespace eyvan-control \
  --set-string image.repository=registry.invalid/eyvan-control-plane \
  --set-string image.digest="${digest}" \
  --set-string agentImage.repository=registry.invalid/eyvan-agent \
  --set-string agentImage.digest="${digest}" \
  --set cellAgent.enabled=true \
  --set-string cellAgent.fleetStateUri=gs://state-bucket.invalid/eyvan/kubernetes-state.json \
  --set-string cellAgent.capacityBackend=kubernetes \
  --set-string cellAgent.kubernetes.workerNamespace=eyvan-workers \
  --set-string workerImage.repository=registry.invalid/eyvan-native-worker \
  --set-string workerImage.digest="${digest}" >"${tmp}/helm-kubernetes.yaml"

for raw in prerequisites.yaml migration-job.yaml cell-agent.yaml control-plane.yaml; do
  cat "${ROOT}/deploy/${raw}" >>"${tmp}/raw.yaml"
  printf '\n---\n' >>"${tmp}/raw.yaml"
done

for manifest in \
  "${tmp}/helm-default-disabled.yaml" \
  "${tmp}/helm-gcp.yaml" \
  "${tmp}/helm-kubernetes.yaml" \
  "${tmp}/raw.yaml"; do
  "${KUBECONFORM_BIN}" -strict -summary "${manifest}"
done

echo "Deployment manifests validated with Helm ${HELM_VERSION} and kubeconform ${KUBECONFORM_VERSION}."
