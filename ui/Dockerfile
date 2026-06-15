# =============================================================================
# OpsRAG frontend image — React 19 + Vite + nginx (LOCAL DEV)
#
# Owner: Duke Dinh (DevOps/SRE)
# Production image is built from `platform-docker-images/images/opsrag/frontend-v2/`
# (vendored from this tree); keep this Dockerfile in sync with that one.
# =============================================================================

# Stage 1 — build the React bundle.
FROM node:26-alpine AS build

WORKDIR /app

# Install deps first so npm-only changes don't bust the layer.
COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Copy source + build configs.
# `public/` ships the OpsRAG brand-mark PNG that the redesign references
# from `<img src="/opsrag-logo.png">` (sidebar, chat empty state, assistant
# avatar, favicon). Pre-redesign Dockerfile excluded `public/` because the
# old UI had no static assets and relied entirely on runtime /api/ui-config.
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
COPY public ./public

# Vite emits to /app/dist.
RUN npm run build

# Stage 2 — serve via nginx. SSE-compatible config (chunked off, proxy_buffering off
# inside nginx.conf). Non-root base image (runs as uid 101): nginx listens on
# 3000 (>1024) so it needs no root -- satisfies the "image user should not be
# root" hardening check (the default nginx:alpine runs the master as root).
FROM nginxinc/nginx-unprivileged:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
