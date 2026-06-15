# Feature Specification: Image (vision) understanding across web + channels

**Feature Branch**: `003-vision-multimodal`

**Created**: 2026-06-16

**Status**: Draft

**Input**: User description: "When user send image, our OpsRAG can understand?
=> Yes, design the vision feature properly like the Channels feature. This is a
new feature."

## Clarifications

### Session 2026-06-16

- Q: Which surfaces should accept image input? → A: **Web UI + all channels**
  (Telegram / Discord / Slack / Teams).
- Q: How should an attached image be used in the answer pipeline? → A: **Vision
  pass-through (ephemeral)** — the image is sent to a vision-capable LLM
  alongside the question for that turn; no description-into-retrieval, no
  indexing.
- Q: Should images be stored, or handled ephemerally? → A: **Ephemeral** — fetch
  per turn, send to the model, drop. No persisted bytes; images do not reappear
  in web session history or the Channels browse view.
- Q: What if the configured model isn't vision-capable when an image arrives? →
  A: **Auto-route to a vision model** — a configurable vision model is used for
  that turn; if none is configured, the image is dropped and the user is told.
- Q: When a user sends an image with no caption/text? → A: **Auto-analyze** —
  treat a bare image as "Please analyze this image" and answer.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — SRE pastes an error screenshot into the web chat (Priority: P1)

An on-call engineer hits an opaque error in a dashboard. Instead of
transcribing it, they paste the screenshot straight into the OpsRAG web chat
and type "what's causing this?". OpsRAG sends the image to a vision-capable
model, reads the error, and answers with cited runbook context.

**Why this priority**: This is the headline capability and the most common
real workflow — a screenshot is faster and richer than retyping. If the web
path doesn't work, the feature has no front door.

**Independent Test**: In the web UI, attach (or clipboard-paste) a PNG of a
log/error panel, send with a question, and receive an answer that references
content visible only in the image.

**Acceptance Scenarios**:

1. **Given** a vision-capable model is active (or a vision model is configured),
   **When** the user attaches one image and sends a question,
   **Then** the answer reflects content from the image and the turn streams
   normally over SSE.
2. **Given** an image is attached with no text,
   **When** the user sends it,
   **Then** OpsRAG analyzes the image as if asked "Please analyze this image."
3. **Given** the turn completes,
   **When** the user reloads the page,
   **Then** the prior answer is present but the image thumbnail is gone
   (ephemeral — no server-side persistence).

---

### User Story 2 — Telegram/Discord user sends a photo to the bot (Priority: P1)

A user in an allowed Telegram chat (or Discord channel) sends a photo of a
broken manifest / topology diagram, with or without a caption. The bot
understands the image and replies in the channel.

**Why this priority**: Channels are how this user base actually interacts with
OpsRAG day-to-day; images are silently dropped today, which is surprising and
broken-feeling.

**Independent Test**: Send a photo + caption to the Telegram bot from an
allowed user; receive an answer that reflects the image. Repeat on Discord.

**Acceptance Scenarios**:

1. **Given** an allowed user sends a photo with a caption,
   **When** the bot processes it,
   **Then** the image bytes are fetched (using the platform's file API) only
   after the permission check passes, and the reply reflects the image.
2. **Given** a photo with no caption,
   **When** the bot processes it,
   **Then** it auto-analyzes the image.
3. **Given** a user who is NOT permitted to DM the bot sends a photo,
   **When** the bot receives it,
   **Then** no image is fetched and the message is silently denied (existing
   DM-allowlist behavior is unchanged).

---

### User Story 3 — Operator runs a non-vision default model (Priority: P2)

An operator runs OpsRAG with a text-only default model and has not configured a
vision model. A user sends an image.

**Why this priority**: Graceful degradation. The feature must never hard-fail a
turn just because vision isn't available.

**Independent Test**: Configure a non-vision model, no vision model; send an
image; confirm the text question is still answered with a clear notice.

**Acceptance Scenarios**:

1. **Given** no vision-capable model is available,
   **When** an image arrives,
   **Then** the image is dropped, the text question (or the synthesized
   "analyze this image") is answered as best as possible, and the reply
   includes a short notice that images can't be read with the current model.

---

### Edge Cases

- Too many images (> configured max) or an oversized image → the whole turn is
  rejected with a clear message (web: 400; channels: a friendly reply), per
  FR-013 (no silent partial drop).
- Unsupported mime type (e.g. PDF, video, audio) → rejected with a clear note;
  not sent to the model.
- A platform file fetch fails (expired file_id, network) → the turn proceeds as
  text-only with a notice; never crashes the dispatcher.
- Animated GIF / multi-frame → treated as a single image (first frame is
  acceptable); no special handling required.
- An image arrives mid-conversation on an existing thread → only the current
  turn sees the image (ephemeral); it is not re-sent on later turns.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to attach one or more images to a chat turn
  from the web UI (file picker, clipboard paste, and drag-and-drop) and from
  all four channel adapters (Telegram, Discord, Slack, Teams).
- **FR-002**: An attached image MUST be sent to a vision-capable LLM together
  with the turn's question (pass-through). The model's reading of the image MUST
  influence the answer.
- **FR-003**: Image bytes MUST be handled ephemerally — they MUST NOT be written
  to the LangGraph checkpoint, the session store, or any durable store. Only a
  text marker (e.g. "[attached image: <name>]") MAY be persisted in
  conversation history.
- **FR-004**: When an image is present but the active generation model is not
  vision-capable, the system MUST route that turn to a configured vision model
  if one exists.
- **FR-005**: When an image is present and no vision-capable model is available,
  the system MUST drop the image, answer the text, and append a short notice
  that images cannot be read with the current configuration.
- **FR-006**: An image sent with no accompanying text MUST be auto-analyzed
  using a synthesized prompt ("Please analyze this image.").
- **FR-007**: Channel adapters MUST fetch image bytes only AFTER the existing
  permission check passes for that message (no fetch on denied messages).
- **FR-008**: The feature MUST NOT change existing authorization — DM allowlist,
  channel allowlist, and scopes apply unchanged.
- **FR-009**: The system MUST enforce a configurable maximum image count per
  turn and a configurable maximum byte size per image, on both web and channels.
- **FR-010**: The system MUST accept a configurable set of image mime types
  (default: png, jpeg, gif, webp) and reject others with a clear message.
- **FR-011**: Vision behavior MUST be configurable via environment and YAML
  (enable/disable, vision model id, vision provider, max count, max bytes,
  allowed mime types) with no rebuild required, consistent with the project's
  model-selection flexibility.
- **FR-012**: Vision token/cost usage MUST be captured by the existing per-user
  usage telemetry, the same as text turns.
- **FR-013**: When some attached images are invalid (oversized / wrong type),
  the system MUST reject the whole turn with a clear message rather than
  silently dropping a subset. *(Chosen for predictability; revisit if noisy.)*
- **FR-014**: A failed platform image fetch MUST degrade to a text-only answer
  with a notice; it MUST NOT raise an unhandled error in the dispatcher.

### Key Entities

- **ImagePart** — an in-memory, ephemeral image: raw bytes + mime type (+ an
  optional source name for the history marker). Never serialized to a durable
  store.
- **ImageRef** — a lightweight, pre-fetch reference produced by a channel
  adapter: platform, file id / url, mime hint, size hint. Resolved to an
  `ImagePart` by the dispatcher after the permission check.
- **VisionConfig** — configuration: `enabled`, `model`, `provider`,
  `max_images`, `max_bytes`, `allowed_mime`. Env/YAML overridable.
- **Vision capability map** — a function `is_vision_capable(provider, model)`
  over known model-id patterns, used for auto-routing.

## Architecture Notes *(informative)*

- **Ephemeral transport**: images travel in the runnable `config`
  (`config["configurable"]["turn_images"]`), which LangGraph passes to nodes but
  does not checkpoint. The generator node reads them at generation time. This is
  the single mechanism that satisfies FR-003 with a Postgres checkpointer.
- **Neutral content format**: message `content` may be a string (fast path,
  unchanged) or a provider-neutral parts list
  `[{"type":"text",...},{"type":"image","mime_type","data"}]`. Each provider
  translates image parts to its native shape (Anthropic/Vertex-Claude base64
  blocks, Bedrock Converse image blocks, OpenAI/LiteLLM `image_url` data-URLs,
  Vertex-Gemini `Part.from_data`). Centralized in `opsrag/llms/content.py`.
- **Auto-routing**: factory pre-builds a `vision_llm` at startup when configured,
  threaded into the agent entry points and used by the generator when a turn has
  images and the active model isn't vision-capable.
- **Per-platform fetch**: Telegram `getFile` + file download; Discord attachment
  CDN url; Slack `url_private` + bearer token; Teams `contentUrl` (+ bot token).

## Success Criteria *(mandatory)*

- **SC-001**: A user can attach an image in the web UI and in Telegram/Discord
  and receive an answer that demonstrably depends on the image's content.
- **SC-002**: After a turn that included an image, no image bytes exist in the
  Postgres checkpoint or the session store (only a text marker), verified by an
  automated test.
- **SC-003**: With a non-vision model and no vision model configured, an
  image-bearing turn still returns a text answer plus a notice, with no error.
- **SC-004**: Permission-denied channel messages never trigger an image fetch.
- **SC-005**: All existing tests pass; new per-provider block-conversion tests,
  per-adapter extraction tests, and the ephemerality test pass.

## Out of Scope

- Persisting / indexing images into the RAG corpus (rejected: ephemeral chosen).
- Generating images, OCR-as-a-separate-service, or document (PDF) understanding.
- Image description fed into `knowledge_search` retrieval (rejected this round).
- Re-sending an image across multiple turns / image "memory".
