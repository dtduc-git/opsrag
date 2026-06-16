# =============================================================================
# opsrag container image -- multi-stage build.
#
# Stage 1 (builder): installs the Python dependency tree into a self-contained
#   virtualenv using uv (resolved from pyproject.toml + uv.lock).
# Stage 2 (runtime): a minimal Debian "slim" base running as a non-root user.
#
# Why debian-slim and not distroless:
#   The container entrypoint (docker-entrypoint.sh) multiplexes on the
#   OPSRAG_ROLE env var and therefore needs a POSIX shell at runtime.
#   gcr.io/distroless/python3-debian12 ships no shell, so the shell-based
#   role dispatcher cannot run there. We use the matching slim base instead,
#   drop to a non-root UID, and keep the image lean by copying only the
#   prebuilt virtualenv and the application source from the builder stage.
#
# All base images are public, vendor-neutral references. No private registry,
# no organization-specific image names.
# =============================================================================

# ------------------------------- Stage 1: build ------------------------------
FROM python:3.14-slim-bookworm AS builder

# uv reads these to build a relocatable, self-contained virtualenv that we can
# copy wholesale into the runtime stage.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Build-time system deps:
#   build-essential -- native wheels (e.g. psycopg, tokenizers) may compile
#   git             -- some dependency resolvers / VCS-pinned deps need it
# These stay in the builder stage only; they are not copied to the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official standalone installer image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the dependency manifests first to maximize layer caching: the
# expensive dependency install layer is reused across application source edits.
COPY pyproject.toml uv.lock README.md ./

# Create the virtualenv and install the locked dependency tree (no project yet).
# Extras pin the production-leaning stack; adjust via build arg if desired.
# discord + teams ship in the image so the published image supports every
# channel; the SDKs are lazy-imported, so a disabled channel never loads them.
# Telegram needs no extra (httpx is a core dep).
ARG OPSRAG_EXTRAS="fastembed,pgvector,vertex,bedrock,ner,cohere,mem0,login,litellm,discord,teams"
# Build the `--extra X --extra Y ...` flags from the comma list with a POSIX
# loop (the RUN shell is /bin/sh = dash, which lacks bash's ${var//,/ } pattern
# substitution -- that produced a "Bad substitution" error).
RUN uv venv "$VIRTUAL_ENV" \
    && EXTRA_FLAGS="" \
    && for e in $(echo "$OPSRAG_EXTRAS" | tr ',' ' '); do EXTRA_FLAGS="$EXTRA_FLAGS --extra $e"; done \
    && uv pip install --python "$VIRTUAL_ENV/bin/python" -r pyproject.toml $EXTRA_FLAGS

# Security hardening: bump transitive build/utility deps that image scanners flag
# (wheel -> malicious-wheel-unpack escalation; jaraco.context -> tar path
# traversal) to their patched releases. Neither is runtime-exploitable by the
# service (the bot never installs untrusted wheels / extracts untrusted tars),
# but pinning the patched versions keeps the published image CVE-clean.
RUN uv pip install --python "$VIRTUAL_ENV/bin/python" --upgrade wheel "jaraco.context"

# Bundle the spaCy model into the venv at BUILD time so the slim runtime (which
# has no pip) never tries to download it on first use. Both mem0's BM25
# lemmatization and the Q&A-cache NER load `en_core_web_sm`; without this they
# crash in the runtime image ("spaCy requires either pip or uv to download
# models"). Pinned to match spaCy 3.8.x. Skipped when spaCy isn't in the
# selected extras (e.g. a minimal build without `ner`/`mem0`).
RUN if "$VIRTUAL_ENV/bin/python" -c "import spacy" 2>/dev/null; then \
      uv pip install --python "$VIRTUAL_ENV/bin/python" \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" ; \
    else echo "spaCy not installed; skipping en_core_web_sm" ; fi

# Bundle the fastembed BM25 sparse model (Qdrant/bm25, ~160KB) at BUILD time so
# mem0's hybrid-search works OFFLINE in the slim runtime. Without this, the
# first memory search downloads it from Hugging Face to an ephemeral /tmp dir --
# which re-downloads on every restart and FAILS in egress-restricted/air-gapped
# clusters. Baked into a fixed cache dir the runtime points FASTEMBED_CACHE_PATH
# at. mkdir keeps the COPY valid even for minimal builds without fastembed.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN mkdir -p /opt/fastembed_cache \
    && if "$VIRTUAL_ENV/bin/python" -c "import fastembed" 2>/dev/null; then \
         "$VIRTUAL_ENV/bin/python" -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')" ; \
       else echo "fastembed not installed; skipping Qdrant/bm25 prefetch" ; fi

# Now copy the application source and install the project itself into the venv.
COPY opsrag ./opsrag
RUN uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e .

# ------------------------------ Stage 2: runtime -----------------------------
FROM python:3.14-slim-bookworm AS runtime

# Runtime needs git: SCM indexing in clone_mode shells out to `git clone`
# (curl is used by the compose healthcheck). ca-certificates for HTTPS clones.
# openssh-client so `git clone` over SSH works (use_ssh=true, e.g. when the
# HTTPS path is behind an SSO/Pomerium proxy).
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends git ca-certificates curl openssh-client \
    && rm -rf /var/lib/apt/lists/*
# `apt-get upgrade` pulls the latest security patches for the base image (e.g.
# libgnutls30 deb12u6 -> deb12u7, which fixes CVE-2026-33845 + CVE-2026-42010 the
# Trivy CRITICAL gate flags). Keeps the image current without a base-image bump.

# Patch the base image's SYSTEM python build tools that image scanners flag:
# pip (CVE-2026-6357 / CVE-2026-3219 / CVE-2025-8869), wheel (CVE-2026-24049),
# and setuptools' vendored jaraco.context (CVE-2026-23949). The app runs from
# /opt/venv, so these are never used at runtime, but patching them keeps the
# published image CVE-clean. (The venv's own copies are patched in the builder.)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /root/.cache/pip
RUN groupadd --gid 1000 opsrag \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin opsrag

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    OPSRAG_ROLE=api \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

WORKDIR /app

# Copy the prebuilt virtualenv and application source from the builder stage.
COPY --from=builder --chown=1000:1000 /opt/venv /opt/venv
COPY --from=builder --chown=1000:1000 /app/opsrag /app/opsrag
# Baked fastembed BM25 cache (mem0 hybrid search runs offline; see builder).
COPY --from=builder --chown=1000:1000 /opt/fastembed_cache /opt/fastembed_cache

# Operator scripts (seed-sample-corpus.sh, audit) and the synthetic sample
# corpus. The quickstart runs `scripts/seed-sample-corpus.sh` inside this
# container to index samples/ into the vector store.
COPY --chown=1000:1000 scripts /app/scripts
COPY --chown=1000:1000 samples /app/samples

# Install the role-dispatching entrypoint and make it executable.
COPY --chown=1000:1000 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Drop privileges: run as the non-root UID 1000 from here on.
USER 1000

# API listens on 8080 (see quickstart). Other roles read OPSRAG_ROLE in-process.
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
