#!/usr/bin/env bash
# seed-sample-corpus.sh -- index the bundled synthetic samples into the
# configured vector store so the User Story 1 quickstart can demonstrate a
# cited answer.
#
# Indexing is an operator action that writes directly to the vector store
# via the indexer entrypoint (opsrag.ingestion.indexer) -- it does NOT go
# through the authenticated HTTP API, so no OIDC token is needed. Run it
# inside the API container (which has the providers + network wired):
#
#     docker compose -f deploy/compose/docker-compose.yaml \
#         exec opsrag-api scripts/seed-sample-corpus.sh
#
# or locally, with the project's virtualenv active and config.yaml present.
#
# Environment:
#   SAMPLES_DIR   directory to index (default: <repo>/samples or /app/samples)
#   OPSRAG_CONFIG path to config.yaml (honoured by the indexer; default config.yaml)
#   PYTHON        python interpreter to use (default: python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"

# Resolve the samples directory: explicit override, else repo-local, else the
# container's /app/samples.
if [[ -n "${SAMPLES_DIR:-}" ]]; then
    :
elif [[ -d "${REPO_ROOT}/samples" ]]; then
    SAMPLES_DIR="${REPO_ROOT}/samples"
elif [[ -d "/app/samples" ]]; then
    SAMPLES_DIR="/app/samples"
else
    echo "seed-sample-corpus: cannot find a samples/ directory." >&2
    echo "  Set SAMPLES_DIR=/path/to/samples and retry." >&2
    exit 2
fi

if [[ ! -d "${SAMPLES_DIR}" ]]; then
    echo "seed-sample-corpus: ${SAMPLES_DIR} does not exist." >&2
    exit 2
fi

echo "seed-sample-corpus: indexing ${SAMPLES_DIR}"
echo "  interpreter : ${PYTHON}"
echo "  config      : ${OPSRAG_CONFIG:-config.yaml}"

# The indexer prints a per-run summary and exits non-zero if nothing indexed.
exec "${PYTHON}" -m opsrag.ingestion.indexer "${SAMPLES_DIR}"
