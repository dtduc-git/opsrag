#!/bin/sh
# opsrag container entrypoint.
#
# Multiplexes on the OPSRAG_ROLE environment variable and exec's the matching
# process. All roles serve the same FastAPI application
# (opsrag.api.server:app); the application reads OPSRAG_ROLE at startup to
# decide whether to run the serving path, the background indexing loop, the
# daily scheduler, and/or the Slack bot. Keeping a single app object across
# roles means one image, one import path, and per-role behaviour driven by
# configuration and env -- never hardcoded deployment facts.
#
# Notes:
#   - This file must be executable. The Dockerfile COPYs it and runs
#     `chmod +x /usr/local/bin/docker-entrypoint.sh`.
#   - POSIX sh only; no bashisms.
#   - Any extra arguments passed to the container are appended to the
#     resolved command, so e.g. `docker run <image> --reload` forwards
#     `--reload` to uvicorn.
set -eu

# Default role is the API server.
ROLE="${OPSRAG_ROLE:-api}"

# Host/port for the served application. Overridable via env; the API default
# port is 8080 (matches the EXPOSE in the Dockerfile and the quickstart).
HOST="${OPSRAG_HOST:-0.0.0.0}"
PORT="${OPSRAG_PORT:-8080}"

# Application import path. Kept as a single source of truth here.
APP="opsrag.api.server:app"

case "$ROLE" in
  job-indexer)
    # Ephemeral run-to-completion indexing Job (NOT a server). Replaces the
    # always-on `indexer` deployment: a k8s Job (or `docker compose run`)
    # builds the pipeline, indexes the target(s), writes progress to the
    # durable Postgres job-state, and exits. Arguments after the role select
    # the target, e.g. `--repo devops/foo --branch master` or `--all`.
    export OPSRAG_ROLE="$ROLE"
    echo "opsrag: starting job-indexer (run-to-completion) args='$*'" >&2
    exec python -m opsrag.job.indexer "$@"
    ;;
  api | backend | indexer | scheduler | slackbot | slack-bot | worker)
    # All recognised roles serve the same ASGI app. The app inspects
    # OPSRAG_ROLE at startup to enable/disable the indexing loop, the
    # APScheduler-driven daily job, and the Slack bot. Export it so the
    # process sees the same value this script resolved.
    export OPSRAG_ROLE="$ROLE"
    echo "opsrag: starting role='$ROLE' on ${HOST}:${PORT}" >&2
    exec uvicorn "$APP" --host "$HOST" --port "$PORT" "$@"
    ;;
  *)
    echo "opsrag: unknown OPSRAG_ROLE='$ROLE'" >&2
    echo "opsrag: valid roles are: api, backend, indexer, scheduler, slackbot, worker, job-indexer" >&2
    exit 1
    ;;
esac
