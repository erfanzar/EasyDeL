{{- define "eyvan.name" -}}
{{- default "eyvan" .Values.nameOverride -}}
{{- end -}}

{{- define "eyvan.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{- define "eyvan.controlPlaneImage" -}}
{{- if .Values.image.digest -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" .Values.image.digest) -}}
{{- fail "image.digest must be a registry-reported sha256 digest" -}}
{{- end -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
{{- end -}}

{{- define "eyvan.cellAgentImage" -}}
{{- if .Values.agentImage.digest -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" .Values.agentImage.digest) -}}
{{- fail "agentImage.digest must be a registry-reported sha256 digest" -}}
{{- end -}}
{{- printf "%s@%s" .Values.agentImage.repository .Values.agentImage.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.agentImage.repository .Values.agentImage.tag -}}
{{- end -}}
{{- end -}}


{{- define "eyvan.capacityBackend" -}}
{{- $backend := lower (trim .Values.cellAgent.capacityBackend) -}}
{{- if eq $backend "k8s" -}}
kubernetes
{{- else if has $backend (list "gcp" "gcp-tpu" "gcp-compute" "kubernetes") -}}
{{- $backend -}}
{{- else -}}
{{- fail "cellAgent.capacityBackend must be one of gcp, gcp-tpu, gcp-compute, kubernetes, or k8s" -}}
{{- end -}}
{{- end -}}

{{- define "eyvan.validateCellAgent" -}}
{{- if ne (toString .Values.cellAgent.replicas) "1" -}}
{{- fail "cellAgent.replicas must be exactly 1; one fleet-state object has one owner" -}}
{{- end -}}
{{- $fleetStateUri := .Values.cellAgent.fleetStateUri -}}
{{- if not (regexMatch "^gs://[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]/[^/?#]+(/[^/?#]+)*$" $fleetStateUri) -}}
{{- fail "cellAgent.fleetStateUri must be a canonical gs://bucket/non-empty-object URI when cellAgent.enabled=true" -}}
{{- end -}}
{{- $bucket := first (splitList "/" (trimPrefix "gs://" $fleetStateUri)) -}}
{{- if contains ".." $bucket -}}
{{- fail "cellAgent.fleetStateUri bucket name must not contain adjacent periods" -}}
{{- end -}}
{{- end -}}

{{- define "eyvan.validateNativeArtifacts" -}}
{{- $package := .Values.cellAgent.nativeWorkerPackage -}}
{{- $packageSha := .Values.cellAgent.nativeWorkerPackageSha256 -}}
{{- $packageGeneration := toString .Values.cellAgent.nativeWorkerPackageGeneration -}}
{{- if $package -}}
{{- if not (hasPrefix "gs://" $package) -}}
{{- fail "cellAgent.nativeWorkerPackage must be a gs:// wheel for verified bootstrap" -}}
{{- end -}}
{{- if not (regexMatch "^[0-9a-f]{64}$" $packageSha) -}}
{{- fail "cellAgent.nativeWorkerPackageSha256 must be 64 lowercase hexadecimal characters" -}}
{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*$" $packageGeneration) -}}
{{- fail "cellAgent.nativeWorkerPackageGeneration must be a positive decimal GCS generation" -}}
{{- end -}}
{{- else if or $packageSha .Values.cellAgent.nativeWorkerPackageGeneration -}}
{{- fail "native worker package digest/generation requires cellAgent.nativeWorkerPackage" -}}
{{- end -}}
{{- $wheelhouse := .Values.cellAgent.nativeWorkerWheelhouse -}}
{{- $wheelhouseSha := .Values.cellAgent.nativeWorkerWheelhouseSha256 -}}
{{- $wheelhouseGeneration := toString .Values.cellAgent.nativeWorkerWheelhouseGeneration -}}
{{- if $wheelhouse -}}
{{- if not $package -}}
{{- fail "cellAgent.nativeWorkerWheelhouse requires cellAgent.nativeWorkerPackage" -}}
{{- end -}}
{{- if not (hasPrefix "gs://" $wheelhouse) -}}
{{- fail "cellAgent.nativeWorkerWheelhouse must be a gs:// zip for verified bootstrap" -}}
{{- end -}}
{{- if not (regexMatch "^[0-9a-f]{64}$" $wheelhouseSha) -}}
{{- fail "cellAgent.nativeWorkerWheelhouseSha256 must be 64 lowercase hexadecimal characters" -}}
{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*$" $wheelhouseGeneration) -}}
{{- fail "cellAgent.nativeWorkerWheelhouseGeneration must be a positive decimal GCS generation" -}}
{{- end -}}
{{- else if or $wheelhouseSha .Values.cellAgent.nativeWorkerWheelhouseGeneration -}}
{{- fail "native worker wheelhouse digest/generation requires cellAgent.nativeWorkerWheelhouse" -}}
{{- end -}}
{{- end -}}

{{- define "eyvan.workerNamespace" -}}
{{- $namespace := default .Release.Namespace .Values.cellAgent.kubernetes.workerNamespace -}}
{{- if or (gt (len $namespace) 63) (not (regexMatch "^[a-z0-9]([-a-z0-9]*[a-z0-9])?$" $namespace)) -}}
{{- fail "cellAgent.kubernetes.workerNamespace must be an exact Kubernetes DNS-label namespace name" -}}
{{- end -}}
{{- $namespace -}}
{{- end -}}

{{- define "eyvan.kubernetesWorkerImage" -}}
{{- $repository := required "workerImage.repository is required for Kubernetes capacity" .Values.workerImage.repository -}}
{{- $digest := required "workerImage.digest is required for Kubernetes capacity" .Values.workerImage.digest -}}
{{- if not (regexMatch "^sha256:[0-9a-f]{64}$" $digest) -}}
{{- fail "workerImage.digest must be a registry-reported sha256 digest" -}}
{{- end -}}
{{- printf "%s@%s" $repository $digest -}}
{{- end -}}

{{- define "eyvan.controlPlaneLabels" -}}
app: eyvan-control-plane
app.kubernetes.io/name: {{ include "eyvan.name" . }}
app.kubernetes.io/component: control-plane
{{- end -}}

{{- define "eyvan.cellAgentLabels" -}}
app: eyvan-cell-agent
app.kubernetes.io/name: {{ include "eyvan.name" . }}
app.kubernetes.io/component: cell-agent
{{- end -}}
