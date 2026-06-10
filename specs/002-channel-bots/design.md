# Channel Bots — Multi-Channel Chat Adapters (Slack / Telegram / Discord / Teams)

**Status:** approved design, in implementation
**Date:** 2026-06-10
**Owner:** OpsRAG

## 1. Goal

Today OpsRAG has one chat surface besides the web UI: a Slack bot
(`opsrag/slack_bot/`) that runs as a Socket Mode worker and calls the agent
pipeline (`query_with_session_events`) **in-process**. We want the same
experience on **Telegram, Microsoft Teams, and Discord**, at **full Slack
parity** (streaming progress, thread context, cited answers, 👍/👎 feedback,
per-channel allowlist + per-user quota).

The agent core is already channel-agnostic. What is Slack-bound is a thin
**transport + render + identity** ring. This design extracts that ring behind a
**`ChannelAdapter` port** (Ports & Adapters / hexagonal) so each platform is a
small adapter over one shared flow. A fix to thread-context or feedback lands
once and every channel inherits it.

## 2. Decisions (locked)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Full Slack parity** on every channel | consistent UX |
| D2 | **Ports & Adapters**: shared `channels/core` + `ChannelAdapter` port; Slack refactored onto it; TG/Discord/Teams are thin adapters | one flow, parity for free, no 4× maintenance |
| D3 | **One role-gated worker per channel** (`OPSRAG_ROLE=slackbot/telegrambot/discordbot`), each its own Deployment with its own `agent_graph`; **Teams = webhook mounted on the API role** (Bot Framework pushes inbound, needs the public ingress the API already has) | isolation + independent scale; Teams transport is fundamentally inbound |
| D4 | **Synthetic anonymous per-channel identity** (matches Slack Phase 1): `oid = "<channel>-bot:<workspace>:<user>"`, `is_anonymous=True` (traceable, not authenticated). Teams real-AAD identity is a future enhancement. | keep scope tight; fail-closed on admin gates |
| D5 | **Deps:** Telegram → `httpx` (no new dep); Discord → `discord.py` extra; Teams → `botbuilder-core`/`botbuilder-schema` extra. Image installs discord+teams so the published image supports all channels; **lazy imports** so a disabled channel never imports its SDK. | lean, vendor-neutral |
| D6 | Fix the **role-gating bug**: channel workers boot only when `OPSRAG_ROLE` matches the channel (today the Slack bot boots on `enabled` alone → N replicas = N Socket Mode connections = duplicate answers). | correctness |

## 3. Architecture

### 3.1 Package layout

```
opsrag/channels/
  __init__.py
  types.py          # InboundMessage, AgentResult, FeedbackEvent, ThreadMessage, MessageHandle, ReactionKind
  base.py           # ChannelAdapter (Protocol) + CoreSink (Protocol)
  config.py         # ChannelsConfig + per-channel sub-configs (Slack/Telegram/Discord/Teams)
  registry.py       # name -> "module:Class" string map; lazy importlib factory (enabled channels only)
  dispatcher.py     # ChannelDispatcher — the shared flow (was slack_bot/handler.py::_dispatch)
  permission.py     # ChannelPermission — generic allowlist + rolling per-user quota
  streaming.py      # ProgressStreamer — generic heartbeat ladder + finalize over adapter.edit
  feedback.py       # record_feedback() — investigation_cache + feedback_store writes
  boot.py           # build_and_start(role, cfg, agent_graph, providers, caches) — generic per-role boot
  adapters/
    __init__.py
    slack/           # refactor of slack_bot/ onto the port (WRAPS existing leaf modules)
    telegram/
    discord/
    teams/           # webhook handler + a FastAPI router mounted on the API role
opsrag/slack_bot/    # becomes a thin back-compat shim (re-exports from channels.adapters.slack
                     # + channels.permission) for one release; keeps existing imports/tests working
```

### 3.2 Neutral types (`channels/types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum

class ReactionKind(str, Enum):
    ACK = "ack"      # "I picked this up" (Slack 👀)
    DONE = "done"    # success (Slack ✅)
    ERROR = "error"  # failure (Slack ❌)

@dataclass(frozen=True)
class InboundMessage:
    channel_id: str            # platform chat/channel id
    user_id: str               # platform user id
    text: str                  # message text, bot-mention already stripped by the adapter
    message_id: str            # id of the inbound message (reaction / reply anchor)
    thread_id: str | None      # Slack thread_ts / Discord+Teams reply id; None => flat (DM)
    is_dm: bool
    workspace: str | None      # team/guild/tenant id; namespaces the synthetic oid
    raw: dict = field(default_factory=dict)   # escape hatch for adapter-specific bits

@dataclass(frozen=True)
class AgentResult:
    answer: str
    sources: list[dict]
    diagram_present: bool
    session_id: str | None
    investigation_id: str | None

@dataclass(frozen=True)
class FeedbackEvent:
    thumbs: str                # "up" | "down"
    investigation_id: str
    user_id: str
    thread_id: str | None
    raw: dict = field(default_factory=dict)

@dataclass(frozen=True)
class ThreadMessage:
    author: str                # display name ("Rootly", "Alice", ...)
    text: str
    is_self: bool              # True => our own past reply (filtered by core)

# MessageHandle: opaque, per-adapter (Slack (channel,ts) tuple / Telegram message_id /
# Discord Message / Teams ConversationReference). The core treats it as a token it
# hands back to edit()/finalize().
MessageHandle = object
```

### 3.3 The port (`channels/base.py`)

```python
from typing import Protocol, runtime_checkable
from opsrag.auth.pomerium import CurrentUser
from opsrag.channels.types import (
    InboundMessage, AgentResult, FeedbackEvent, ThreadMessage,
    MessageHandle, ReactionKind,
)

@runtime_checkable
class CoreSink(Protocol):
    """What the dispatcher hands the adapter so inbound events drive the flow."""
    async def on_message(self, msg: InboundMessage) -> None: ...
    async def on_feedback(self, fb: FeedbackEvent) -> None: ...

@runtime_checkable
class ChannelAdapter(Protocol):
    name: str  # "slack" | "telegram" | "discord" | "teams"

    # --- lifecycle: the adapter owns its transport, normalizes inbound platform
    #     events to InboundMessage / FeedbackEvent, and pushes them into `sink`.
    #     It MUST drop bot-loop messages (its own + other bots per channel policy)
    #     and set is_dm before calling sink.on_message.
    async def connect(self, sink: CoreSink) -> None: ...
    async def close(self) -> None: ...

    # --- outbound primitives (driven by ProgressStreamer + dispatcher)
    async def post_placeholder(self, channel_id: str, thread_id: str | None, text: str) -> MessageHandle: ...
    async def edit(self, handle: MessageHandle, text: str) -> None: ...                 # heartbeat tick
    async def finalize(self, handle: MessageHandle, result: AgentResult) -> None: ...   # adapter RENDERS here
    async def react(self, channel_id: str, message_id: str, kind: ReactionKind) -> None: ...  # best-effort
    async def fetch_thread(self, channel_id: str, thread_id: str, *, cap: int) -> list[ThreadMessage]: ...
    async def resolve_identity(self, msg: InboundMessage) -> CurrentUser: ...
    async def send_denial(self, msg: InboundMessage, reason: str) -> None: ...          # private/DM
    async def confirm_feedback(self, fb: FeedbackEvent, *, accepted: bool) -> None: ...  # ephemeral ack
```

**Rendering never leaves the adapter.** The core hands `finalize` the neutral
`AgentResult`; the adapter turns it into Block Kit / Markdown / embed / Adaptive
Card. The core stays format-blind. `post_placeholder` + `edit` carry only plain
text (the heartbeat ladder).

Not every channel supports every primitive: `react` is best-effort and may be a
no-op (Telegram/Teams have no reaction-on-message). `fetch_thread` returns `[]`
where the platform has no thread model surfaced (Telegram DMs are flat). The
core already treats both as best-effort.

### 3.4 The shared flow (`channels/dispatcher.py`)

`ChannelDispatcher` is `slack_bot/handler.py::_dispatch` with every Slack call
replaced by a port call. It holds `adapter, agent_graph, providers, permission,
config, qa_cache, investigation_cache, semantic_router, feedback_store,
web_ui_base_url`. It IS the `CoreSink`.

`on_message(msg: InboundMessage)` — identical seven stages:
1. `permission.allow(msg)` → deny path calls `adapter.send_denial`.
2. text already stripped by adapter; empty → no-op.
3. `handle = adapter.post_placeholder(...)`; `adapter.react(ACK)`; build
   `ProgressStreamer(adapter, handle, ...)`; `start_heartbeat()`.
4. if `msg.thread_id and not is_dm`: `adapter.fetch_thread(...)` → serialize the
   non-self messages into the "PRIOR THREAD MESSAGES:" block (the serialization +
   greedy-newest truncation logic moves verbatim from `thread_context.py`; the
   *fetch* is the port call, the *assembly* is shared core).
5. `current_user = adapter.resolve_identity(msg)`.
6. `query_with_session_events(self.agent_graph, query=combined, user_id=oid,
   thread_id=_session_thread_id(msg), ...)` — **unchanged, in-process**. Collect
   `final` / `error` events.
7. on success → `AgentResult` → `streamer.finalize_result(result)` (which calls
   `adapter.finalize`); `adapter.react(DONE)`; `permission.record_usage(user)`.
   on error → `streamer.finalize_text(ERROR_TEXT + kind)`; `adapter.react(ERROR)`.

`on_feedback(fb)` — `slack_bot/handler.py::on_block_action` minus the Slack
parsing: write `investigation_cache.record_feedback` + `feedback_store.record`
(user_id namespaced `"<channel>:<user>"`), then `adapter.confirm_feedback`.

`_session_thread_id(msg)` generalizes the Slack rule: DM → `"<ch>-dm:<channel>"`;
threaded → `"<ch>-thread:<channel>:<thread_or_msg>"`. The `<ch>` prefix keeps
sessions disjoint across platforms.

### 3.5 Permission (`channels/permission.py`)

`ChannelPermission(allowed_channels, per_user_daily_quota)` — the rolling-window
quota + allowlist logic from `SlackBotPermission`, but typed over `InboundMessage`
(`msg.is_dm`, `msg.channel_id`, `msg.user_id`). Bot-loop filtering moves to the
adapter (it knows its own bot id), so the core only sees real user messages.

**Back-compat:** `opsrag/slack_bot/permission.py::SlackBotPermission` stays as a
thin subclass that accepts the Slack **event dict** shape (`{"channel","user",
"bot_id",...}`) used by `test_slack_bot_channel_resolution.py`, normalizes it to
an `InboundMessage`-like view, and delegates. That test must stay green unchanged.

### 3.6 Streaming (`channels/streaming.py`)

`ProgressStreamer(adapter, handle, *, heartbeat_interval_s=30)` — `SlackProgressStreamer`
with `client.update_message` → `adapter.edit`. Same ladder phrases, same
"stop after last rung", same idempotent `finalize`. Add `finalize_result(AgentResult)`
(calls `adapter.finalize`) and keep `finalize_text(str)` for the error path.

### 3.7 Registry + boot (`channels/registry.py`, `channels/boot.py`)

```python
# registry.py — string-based so disabled channels never import their SDK
ADAPTERS = {
    "slack":    "opsrag.channels.adapters.slack.adapter:SlackAdapter",
    "telegram": "opsrag.channels.adapters.telegram.adapter:TelegramAdapter",
    "discord":  "opsrag.channels.adapters.discord.adapter:DiscordAdapter",
    "teams":    "opsrag.channels.adapters.teams.adapter:TeamsAdapter",
}
ROLE_TO_CHANNEL = {"slackbot": "slack", "telegrambot": "telegram", "discordbot": "discord"}
# teams has no worker role — it's a webhook on the API role.
```

`boot.build_and_start(role, cfg, agent_graph, providers, caches)` (called from
the FastAPI lifespan): if `role` maps to a channel AND `cfg.channels.<name>.enabled`,
importlib-load the adapter, build `ChannelPermission` + `ChannelDispatcher`,
`await adapter.connect(dispatcher)`, return the adapter for lifespan shutdown.
**Role-gating fix lives here**: a channel worker starts iff `OPSRAG_ROLE`
matches; on the `api` role none of the outbound workers start. Teams' webhook
router is mounted on the API role separately (see §6).

## 4. Per-channel adapter specs

### 4.1 Slack (`adapters/slack/`) — refactor, WRAP don't rewrite

- **Transport:** keep `client.py` (`SlackBotClient`, Socket Mode) **byte-for-byte**.
  The adapter holds a `SlackBotClient` and implements the port by delegating:
  `post_placeholder`→`post_message`, `edit`→`update_message`, `react`→`add_reaction`
  (map ACK→`eyes`, DONE→`white_check_mark`, ERROR→`x`), `fetch_thread`→
  `fetch_thread_replies` then map raw msgs to `ThreadMessage` (reuse the
  self-filtering + display-name logic from `thread_context.py`).
- **Inbound normalization:** `SlackBotClient._on_request` already routes
  `app_mention`/`message.im`/`block_actions`. Rework `start(sink)` so the client
  builds `InboundMessage`/`FeedbackEvent` and calls `sink.on_message`/`on_feedback`
  (replacing today's `dispatcher.on_app_mention` etc.). Strip `<@MENTION>` here.
- **Render:** `finalize` calls `format_answer_as_slack_blocks(...)` (render.py
  **unchanged**) and `update_message(blocks=...)`.
- **Identity:** `slack_user_to_current_user` (identity.py unchanged).
- **Feedback:** parse `block_actions` value `up:<id>`/`down:<id>` → `FeedbackEvent`;
  `confirm_feedback` POSTs the ephemeral `response_url` note.
- **Deps:** `slack-sdk` (already present). **No behavior change** — the existing
  permission test + any manual Slack smoke must pass.

### 4.2 Telegram (`adapters/telegram/`) — httpx, no new dep

- **Transport:** Bot API over `httpx`. Long-poll `getUpdates?timeout=50&offset=N`
  in a loop task (no public ingress). `connect(sink)` starts the poll loop;
  `close()` cancels it. Persist `offset` in memory (next = last update_id + 1).
- **Inbound:** updates carry `message` (text, `chat.id`, `from.id`, `message_thread_id`
  for forum topics, `chat.type` in `private|group|supergroup`). DM ⇔ `chat.type==private`.
  Trigger on: DMs always; in groups, on `@botusername` mention or reply-to-bot.
  Strip the `@botusername` token. Bot-loop: ignore `from.is_bot`.
- **Outbound:** `sendMessage` (returns `message_id`) → handle; `editMessageText`
  for heartbeat + finalize. Parse mode **MarkdownV2** (escape reserved chars) or
  HTML — use **HTML** (simpler escaping); render answer+sources+footer as HTML,
  feedback as an **inline keyboard** (`reply_markup.inline_keyboard` with
  `callback_data="up:<id>"`/`"down:<id>"`). Telegram message cap 4096 chars →
  truncate with a "view in UI" link (reuse the truncation idea from render.py).
- **Feedback:** `callback_query` updates → `FeedbackEvent`; `confirm_feedback` →
  `answerCallbackQuery(text=...)` (the toast) + optionally `editMessageReplyMarkup`.
- **Identity:** `telegram-bot:<chat_id>:<user_id>`, anonymous.
- **react:** no-op (Telegram has no message reactions in Bot API v1 path) — return.
- **Config/secrets:** `bot_token_env` (default `OPSRAG_TELEGRAM_BOT_TOKEN`),
  `allowlist` (chat ids, may be negative for groups), `per_user_daily_quota`.

### 4.3 Discord (`adapters/discord/`) — discord.py gateway

- **Transport:** `discord.py` `commands.Bot`/`Client` over the gateway websocket
  (outbound, no ingress). `connect(sink)` → `await client.start(token)` in a task,
  with `on_message` / `on_interaction` handlers that build neutral types and call
  the sink. Intents: `message_content` (privileged — document enabling it).
- **Inbound:** trigger on DM (`isinstance(channel, DMChannel)`) or bot @mention in
  a guild text channel. `thread_id` = the Discord `Thread` id when the message is in
  a thread, else None. Strip the mention. Bot-loop: ignore `message.author.bot`.
- **Outbound:** `channel.send(text)` → `Message` (handle); `message.edit(content=)`
  for heartbeat + finalize. Render answer as Markdown inside an **Embed** (description
  ≤ 4096; total ≤ 6000; sources as fields), feedback as **Buttons** (discord.py
  `ui.View` with two `Button`s, `custom_id="up:<id>"/"down:<id>"`). 2000-char plain
  cap if not using embeds → use embed.
- **Feedback:** button interaction → `FeedbackEvent`; `confirm_feedback` →
  `interaction.response.send_message(ephemeral=True, ...)`.
- **Identity:** `discord-bot:<guild_id_or_dm>:<user_id>`, anonymous.
- **Config/secrets:** `bot_token_env` (default `OPSRAG_DISCORD_BOT_TOKEN`),
  `allowlist` (channel ids), `per_user_daily_quota`.
- **Deps:** `discord` extra `["discord.py>=2.4"]`; lazy import.

### 4.4 Teams (`adapters/teams/`) — Bot Framework webhook on the API

- **Transport:** Microsoft Bot Framework **pushes** activities to a public HTTPS
  endpoint. So Teams is NOT an outbound worker — it is a FastAPI router
  `POST /api/channels/teams/messages` **mounted on the `api` role**. The router
  validates the inbound JWT (Bot Connector auth via `botbuilder-core`
  `CloudAdapter`/`BotFrameworkAuthentication`) and converts the `Activity` to
  `InboundMessage`, then calls the shared dispatcher.
- **Streaming nuance:** Teams supports message **update** via the connector
  (`update_activity`), so the placeholder + heartbeat + finalize edit-in-place
  pattern works. `post_placeholder` sends an Activity and keeps its
  `ConversationReference`/activity id as the handle; `edit` = `update_activity`.
  (If update proves flaky in a tenant, fall back to "typing" indicator +
  single final message — keep behind a config flag, default edit.)
- **Inbound:** `activity.type=="message"`; `conversation.id` = channel_id;
  `from.id` = user_id; `conversation.isGroup`/conversationType → is_dm (`personal`).
  Strip `<at>bot</at>`. Bot-loop: ignore activities from the bot's own id.
- **Render:** **Adaptive Card** (answer TextBlock w/ markdown, sources as a
  FactSet/links, footer) + an `Action.Submit` pair for 👍/👎 carrying
  `{"feedback":"up","id":"<id>"}`. Plain-text fallback in `activity.text`.
- **Feedback:** `activity.type=="message"` with `value` from `Action.Submit`
  (Teams delivers card actions as a message activity with `.value`) → `FeedbackEvent`;
  `confirm_feedback` → send/replace a small confirm Activity.
- **Identity:** `teams-bot:<tenant_id>:<user_id>`, anonymous (real AAD oid = future).
- **Config/secrets:** `app_id_env` (`OPSRAG_TEAMS_APP_ID`), `app_password_env`
  (`OPSRAG_TEAMS_APP_PASSWORD`), `allowlist` (conversation ids), `per_user_daily_quota`.
- **Deps:** `teams` extra `["botbuilder-core>=4.15","botbuilder-schema>=4.15"]`; lazy import.
- **Operator setup (docs):** create an Azure Bot resource + app registration,
  set the messaging endpoint to `https://<opsrag-host>/api/channels/teams/messages`,
  upload a Teams app manifest. Documented in `docs/channels/teams.md`.

## 5. Config schema (`channels/config.py`)

Unified block on `OpsRAGConfig`:

```yaml
channels:
  slack:    { enabled: false, bot_token_env: OPSRAG_SLACK_BOT_TOKEN, app_token_env: OPSRAG_SLACK_APP_TOKEN,
              allowlist: [], per_user_daily_quota: 200, streaming_enabled: true, workspace_url: null, web_ui_base_url: "" }
  telegram: { enabled: false, bot_token_env: OPSRAG_TELEGRAM_BOT_TOKEN, allowlist: [], per_user_daily_quota: 200, web_ui_base_url: "" }
  discord:  { enabled: false, bot_token_env: OPSRAG_DISCORD_BOT_TOKEN, allowlist: [], per_user_daily_quota: 200, web_ui_base_url: "" }
  teams:    { enabled: false, app_id_env: OPSRAG_TEAMS_APP_ID, app_password_env: OPSRAG_TEAMS_APP_PASSWORD,
              allowlist: [], per_user_daily_quota: 200, web_ui_base_url: "" }
```

- Each sub-config validates its allowlist loosely (channel-id strings; Slack keeps
  the strict `C...` validator). Secrets only via `*_env` (Constitution Principle VI).
- **Back-compat:** keep the top-level `slack_bot: SlackBotConfig` field. At config
  load, if `slack_bot.enabled` and `channels.slack` is default, mirror it into
  `channels.slack` (deprecation log). Both paths boot the same Slack adapter.
  Existing `config.yaml`/`config-example.yaml` `slack_bot:` blocks keep working.

## 6. Deployment

- **Entrypoint** (`docker-entrypoint.sh`): add `telegrambot`, `discordbot` to the
  recognised-roles case (alongside `slackbot`). All still `exec uvicorn` the same
  app; the app reads `OPSRAG_ROLE` and boots the matching channel worker.
- **Lifespan** (`opsrag/api/server.py`): replace the inline Slack block (~1341)
  with `app.state.channel = await channels.boot.build_and_start(role, cfg, ...)`.
  On the **api** role, additionally `app.include_router(teams_router)` when
  `cfg.channels.teams.enabled`. Shutdown calls `adapter.close()`.
- **Helm:** generalize `slackbot-deployment.yaml` into per-channel Deployments
  (`slackbot`, `telegrambot`, `discordbot`) gated by `.Values.channels.<name>.enabled`,
  each with `OPSRAG_ROLE=<name>bot`. Teams needs **no** Deployment (served by the
  API) but its messaging endpoint needs the API Ingress reachable publicly — document.
  `values.yaml`: replace `slackBot:` with a `channels:` map (keep `slackBot` as a
  deprecated alias mapping to `channels.slack` for one release). `values.schema.json`
  updated. `configmap.yaml`: render the `channels:` block into the mounted
  `config.yaml` so enabling a channel in values flips it in-app (today's configmap
  doesn't render the bot config at all — fix).
- **Deps/image:** add `discord` + `teams` extras to `pyproject.toml`; `Dockerfile`
  installs them so the published image supports all channels. Telegram needs nothing
  new.

## 7. Identity & security

- All channels: **synthetic anonymous** `CurrentUser` (`is_anonymous=True`),
  `oid="<channel>-bot:<workspace>:<user>"`. Admin-gated actions stay fail-closed.
- **Allowlist + per-user quota** enforced in the shared `ChannelPermission` (cost
  choke point — an open bot can burn the LLM budget). DMs bypass the allowlist but
  not the quota, matching Slack.
- **Role-gating** (D6) ensures exactly one process per channel → no duplicate answers.
- **No secrets in code/docs** — tokens via `*_env`, allowlists are ids not names,
  no org-specific identifiers. Vendor-neutrality audit must stay green.
- Teams webhook **must** validate the Bot Connector JWT (reject unauthenticated POSTs).

## 8. Testing strategy (carefully — this is the real safety net)

Existing coverage is thin (one permission test), so the refactor's net is NEW:

1. **`FakeAdapter`** (`channels/adapters/fake.py`, test-only or under adapters):
   records `posted/edited/finalized/reactions/denials/confirms`, lets a test feed
   `InboundMessage`/`FeedbackEvent` into the dispatcher, returns scripted threads.
2. **Core dispatcher tests** (`tests/unit/test_channel_dispatcher.py`) over the
   FakeAdapter + a fake `query_with_session_events` (monkeypatch / stub graph):
   - permission deny → `send_denial`, agent NOT called.
   - empty text → no-op.
   - happy path → placeholder posted, heartbeat started, identity resolved, agent
     called with the right `thread_id`/`user_id`, `finalize_result` got the answer,
     DONE reaction, `record_usage` called once.
   - agent error / empty final → `finalize_text(ERROR)`, ERROR reaction, quota NOT burned.
   - thread context → `fetch_thread` result serialized + prepended; self-messages dropped.
   - feedback → `record_feedback` + `feedback_store.record` + `confirm_feedback`;
     malformed value ignored.
   - `_session_thread_id` matrix (dm / new-thread / existing-thread, per channel prefix).
3. **`ChannelPermission` tests** — port the 7 Slack cases to neutral `InboundMessage`;
   **keep** `test_slack_bot_channel_resolution.py` green via the back-compat shim.
4. **Per-adapter unit tests** (no network) — each:
   - inbound normalization: platform update/activity → correct `InboundMessage`
     (DM vs group, mention strip, bot-loop drop, thread id).
   - render: `AgentResult` → expected payload shape (Block Kit / HTML / Embed /
     Adaptive Card) incl. sources + feedback controls + truncation.
   - feedback parse: platform action → `FeedbackEvent`.
   - identity oid format.
   Mock the SDK/transport (httpx `MockTransport` for Telegram; stub `discord`/
   `botbuilder` objects; do not hit the network).
5. **Registry/boot test** — role → adapter mapping; disabled channel never imports
   its SDK (assert `importlib` not called); api role starts no worker.
6. **Full suite + ruff** must pass; vendor-neutrality audit must pass.

Every adapter ships with its tests in the same change. "Done" = green `pytest` +
green `ruff` + green audit, verified by the orchestrator (not just the sub-builder).

## 9. Phased rollout / work split

- **P0 Foundation** (sequential, critical path): `channels/{types,base,permission,
  streaming,feedback,dispatcher,registry,boot,config}.py` + `FakeAdapter` + core
  tests green.
- **P1 Slack refactor** (sequential, after P0): `adapters/slack/` wrapping existing
  leaf modules + `slack_bot/` back-compat shim + shared scaffolding (server.py boot,
  entrypoint roles, `channels` config wired into `OpsRAGConfig`, Helm per-channel
  Deployments, configmap/values/schema, pyproject extras, Dockerfile). Existing
  permission test green; no Slack behavior change.
- **P2 Adapters** (parallel, after P1 — each additive in its own dir + test file):
  Telegram, Discord, Teams.
- **P3 Verify** (sequential): full `pytest` + `ruff` + audit; adversarial review of
  the new tests (are they vacuous? do they assert behavior?); fix gaps.

Each phase is independently shippable; a channel can be merged on its own once P1
lands.
