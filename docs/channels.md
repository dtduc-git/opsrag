# Channel bots (Slack / Telegram / Discord / Teams)

OpsRAG can answer questions from chat platforms, not just the web UI. Each
platform ("channel") gets the **same** experience as the web `/query`: streaming
"thinking" progress, thread-aware context, cited answers, thumbs up/down
feedback, and a per-channel allowlist plus per-user daily quota.

All four channels share one transport-agnostic core (`opsrag/channels/`) and
call the agent pipeline **in-process** (`query_with_session_events`) — the same
retrieval, grounding, and citations as the web UI, with no extra hop. Each
platform is a thin **adapter** over a common `ChannelAdapter` port, so a fix to
thread-context, quota, or feedback lands once and every channel inherits it.

## Architecture at a glance

| Channel | Transport | Public ingress? | Runs as |
|---|---|---|---|
| Slack | Socket Mode (outbound websocket) | no | worker, `OPSRAG_ROLE=slackbot` |
| Telegram | Bot API long-poll over HTTPS (outbound) | no | worker, `OPSRAG_ROLE=telegrambot` |
| Discord | Gateway websocket (outbound) | no | worker, `OPSRAG_ROLE=discordbot` |
| Teams | Bot Framework webhook (inbound) | **yes** | route on the `api` role |

Slack, Telegram, and Discord connect **outbound**, so they need no public
endpoint and run as their own role-gated worker Deployment. **Teams is the
exception**: the Microsoft Bot Framework *pushes* activities to a public HTTPS
endpoint, so Teams is served as a route (`POST /api/channels/teams/messages`) on
the existing `api` role rather than as a separate worker.

A channel worker starts **only** when `OPSRAG_ROLE` matches the channel *and*
that channel is enabled in config. This guarantees exactly one process holds the
connection — running the bot on a multi-replica `api` Deployment would otherwise
open N connections and answer every message N times.

## Configuration

All channels are configured under the unified `channels:` block (see
`config-example.yaml`). Every channel shares `enabled`, `allowlist`,
`per_user_daily_quota`, and `web_ui_base_url`; secrets are referenced **only** by
the name of the env var that carries them (never inline a token).

```yaml
channels:
  slack:
    enabled: false
    bot_token_env: OPSRAG_SLACK_BOT_TOKEN
    app_token_env: OPSRAG_SLACK_APP_TOKEN
    allowlist: []                # Slack channel ids (C...); DMs bypass
    per_user_daily_quota: 200
    streaming_enabled: true
    web_ui_base_url: ""          # deep-link base for "View in OpsRAG UI"
  telegram:
    enabled: false
    bot_token_env: OPSRAG_TELEGRAM_BOT_TOKEN
    allowlist: []                # chat ids (may be negative for groups)
    per_user_daily_quota: 200
    web_ui_base_url: ""
  discord:
    enabled: false
    bot_token_env: OPSRAG_DISCORD_BOT_TOKEN
    allowlist: []                # channel ids
    per_user_daily_quota: 200
    web_ui_base_url: ""
  teams:
    enabled: false               # webhook on the `api` role (no worker)
    app_id_env: OPSRAG_TEAMS_APP_ID
    app_password_env: OPSRAG_TEAMS_APP_PASSWORD
    allowlist: []                # conversation ids
    per_user_daily_quota: 200
    web_ui_base_url: ""
```

`allowlist` is the cost-control choke point: an empty allowlist denies every
public channel/group (DMs are always allowed, since the sender is implicitly
identified, but they still count against the per-user quota). Populate it with
the channel/chat/conversation ids you explicitly want to answer in.

### Identity

Channel users map to a **synthetic, anonymous** identity:
`oid = "<channel>-bot:<workspace>:<user>"` (for example
`telegram-bot:-1001234:42`). It is *traceable* (you can group usage by platform
and user) but not *authenticated* — admin-gated actions stay fail-closed, exactly
as for an anonymous web visitor. Mapping a channel user to a real SSO identity is
a future enhancement.

## Per-channel setup

### Slack

1. Create a Slack app (from scratch) at <https://api.slack.com/apps>.
2. Enable **Socket Mode** and generate an app-level token (`xapp-...`) with the
   `connections:write` scope.
3. Add bot scopes: `app_mentions:read`, `chat:write`, `reactions:write`,
   `channels:history`, `groups:history`, `im:history`, `users:read`. Install the
   app and copy the bot token (`xoxb-...`).
4. Subscribe to bot events: `app_mention`, `message.im`.
5. Set `OPSRAG_SLACK_BOT_TOKEN` and `OPSRAG_SLACK_APP_TOKEN`, enable
   `channels.slack`, and list the channel ids (`C...`) in `allowlist`. Invite the
   bot to those channels.
6. Run a worker with `OPSRAG_ROLE=slackbot`.

> Migrating from the old `slack_bot:` block? It still works — when
> `slack_bot.enabled` is true and `channels.slack` is left at defaults, it is
> mirrored into `channels.slack` at load (with a deprecation warning). Prefer the
> `channels.slack` block for new deployments.

### Telegram

1. Talk to [@BotFather](https://t.me/BotFather), `/newbot`, and copy the token.
2. For the bot to see non-command group messages, either `/setprivacy` -> Disable
   in BotFather, or simply @-mention the bot (it triggers on a mention or a reply
   to its own message; DMs always trigger).
3. Set `OPSRAG_TELEGRAM_BOT_TOKEN`, enable `channels.telegram`, and add the chat
   ids to `allowlist` (group ids are negative; get an id by adding the bot and
   reading the logged `channel_id`). DMs bypass the allowlist.
4. Run a worker with `OPSRAG_ROLE=telegrambot`. It long-polls `getUpdates` — no
   public webhook or ingress required. (Telegram has no thread-replies API, so
   in-chat history is not pulled as context; the answer still uses the message
   plus full RAG retrieval.)

### Discord

1. Create an application at <https://discord.com/developers/applications>, add a
   **Bot**, and copy the bot token.
2. Under the bot settings, enable the **Message Content Intent** (a privileged
   intent required to read message text).
3. Invite the bot with the `bot` scope and the `Send Messages`,
   `Read Message History`, and `Add Reactions` permissions.
4. Set `OPSRAG_DISCORD_BOT_TOKEN`, enable `channels.discord`, and add the channel
   ids to `allowlist`. The bot answers DMs and @-mentions in allowlisted
   channels, rendering answers as an embed with thumbs up/down buttons.
5. Run a worker with `OPSRAG_ROLE=discordbot` (gateway websocket, no ingress).

### Microsoft Teams

Teams is inbound-via-webhook, so it is served on the `api` role and needs the
API reachable over public HTTPS.

1. Create an **Azure Bot** resource (Azure Portal -> Create -> Azure Bot). Use a
   multi-tenant Microsoft App ID; copy the **App ID** and create a client
   **secret** (App password).
2. Set the bot's **Messaging endpoint** to
   `https://<your-opsrag-host>/api/channels/teams/messages`.
3. Enable the **Microsoft Teams** channel on the bot resource.
4. Build a Teams app manifest referencing the bot's App ID and upload/sideload it
   to Teams (Developer Portal or an app package).
5. Set `OPSRAG_TEAMS_APP_ID` and `OPSRAG_TEAMS_APP_PASSWORD`, enable
   `channels.teams`, and add conversation ids to `allowlist`.
6. No extra worker: the `api` role mounts the webhook when `channels.teams` is
   enabled. Every inbound activity's Bot Connector JWT is validated (unverified
   requests are rejected with 401), and answers render as Adaptive Cards with
   thumbs up/down submit actions.

## Deployment (Helm)

The chart renders one Deployment per enabled outbound channel, each with the
matching `OPSRAG_ROLE`. Teams needs no Deployment — just the API Ingress reachable
publicly.

```yaml
channels:
  slack:    { enabled: true }
  telegram: { enabled: false }
  discord:  { enabled: false }
  teams:    { enabled: false }
```

Provide the token env vars via your secret (`api.envFromSecret`); the same image
serves every role. The deprecated `slackBot:` values alias still maps to
`channels.slack` for one release.

## Reading channel conversations in the web UI

Conversations that happen in **shared channels** (Slack/Discord/Teams channels,
Telegram groups) are browsable **read-only** in the web UI under the **Channels**
page by any authenticated user (scope `chat`; in `open` mode, anyone). The list
and each conversation's messages come from `GET /channels/conversations` and
`GET /channels/conversations/{thread_id}/messages`.

Only **shared-channel** conversations are exposed. A session's privacy is encoded
in its `thread_id` prefix: shared-channel sessions are `<platform>-thread:…`,
while private 1:1 DMs are `<platform>-dm:…`. The endpoints expose **only** the
`-thread:` sessions and validate the prefix server-side, so private DMs and
private web conversations physically cannot be reached through them (the messages
endpoint returns 404 for any non-`-thread:` id).

The surface is read-only — there is no reply, continue, or delete. Consistent
with the synthetic identity model above, the only identity shown is the
**platform** ("Slack channel", "Telegram group", …); the channel users
themselves remain anonymous.

## Security notes

- **Secrets** are referenced only by `*_env` var name — never written to config,
  the ConfigMap, or the repo.
- **Allowlist + quota** are the cost gate: an open bot can burn the LLM budget,
  so an empty allowlist denies all public channels and every user has a daily cap.
- **Role-gating** ensures exactly one process per channel — no duplicate answers.
- **Identity** is synthetic and anonymous; admin-gated actions remain fail-closed.
- **Teams** rejects any inbound activity without a valid Bot Connector JWT.
