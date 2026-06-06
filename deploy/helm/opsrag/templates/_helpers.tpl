{{/*
Common template helpers for the opsrag chart.
*/}}

{{- define "opsrag.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "opsrag.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "opsrag.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "opsrag.labels" -}}
helm.sh/chart: {{ include "opsrag.chart" . }}
{{ include "opsrag.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: opsrag
{{- end -}}

{{- define "opsrag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "opsrag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "opsrag.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "opsrag.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
The api container image reference.
*/}}
{{- define "opsrag.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | toString) -}}
{{- end -}}

{{/*
MCP enable env vars: one OPSRAG_MCP_<NAME>_ENABLED per integration, uppercased.
Rendered onto the api container so the running pod sees exactly the operator's
enable flags (contract: helm-values-schema.md "Wiring from values to container").
*/}}
{{- define "opsrag.mcpEnv" -}}
{{- range $name, $cfg := .Values.mcp }}
- name: OPSRAG_MCP_{{ $name | upper }}_ENABLED
  value: {{ $cfg.enabled | quote }}
{{- end }}
{{- end -}}
