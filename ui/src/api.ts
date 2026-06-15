const BASE = "/api";

// ── Auth scopes ─────────────────────────────────────────────────────────
// The backend's scope vocabulary (mirrors opsrag/auth/scopes.py). The UI
// uses these for nav gating only — server-side `require_scope` stays
// authoritative. In OPEN mode the backend grants every scope (or returns
// anonymous-with-all-access), so nav gating is transparent.
export type Scope = "chat" | "investigate" | "mcp" | "admin";

// ── Login-required signal ───────────────────────────────────────────────
// When `apiFetch` sees a 401 {error:"unauthenticated"} envelope it sets the
// login hash and broadcasts this event so the AuthGate can re-evaluate and
// drop the user onto the Login page without a full reload. We also clear any
// stale identity so the gate doesn't flash an authenticated shell.
export const LOGIN_HASH = "#/login";
export const AUTH_REQUIRED_EVENT = "opsrag:auth-required";

let authRequired = false;

/** True once a 401-unauthenticated has been observed this session. */
export function isAuthRequired(): boolean {
  return authRequired;
}

/** Reset the auth-required latch (e.g. after a successful (re)login). */
export function clearAuthRequired(): void {
  authRequired = false;
}

function triggerLogin(): void {
  authRequired = true;
  // Route to the login view via hash (the SPA has no router lib).
  if (window.location.hash !== LOGIN_HASH) {
    try { window.location.hash = LOGIN_HASH; } catch { /* non-browser env */ }
  }
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}

// Detect the backend's unauthenticated envelope. oidc_enforcement.py returns
// `401 {error:"unauthenticated", reason, request_id}`; we treat ANY 401 as a
// login signal but only consume the body when it's JSON so the SSE path
// (which must NOT call .json()) stays intact.
function is401(resp: Response): boolean {
  return resp.status === 401;
}

/**
 * Central fetch wrapper. On a 401 it flips the app into login mode (sets the
 * login hash + clears stale auth) and throws so callers stop. In OPEN mode
 * the backend never returns 401, so this is fully transparent — behaves like
 * a bare `fetch`.
 *
 * NOTE: this does NOT read the body on success, so streaming/SSE callers can
 * pass `{ stream: true }` to skip the 401 short-circuit's body handling and
 * read `resp.body` themselves. The 401 check only inspects `resp.status`.
 */
export async function apiFetch(
  input: string,
  init?: RequestInit,
): Promise<Response> {
  const resp = await fetch(input, init);
  if (is401(resp)) {
    triggerLogin();
    // Surface a typed error so SSE callers (which never reach `.json()`)
    // still abort cleanly before trying to read a body that won't come.
    throw new UnauthenticatedError();
  }
  return resp;
}

/** Thrown by `apiFetch` (and SSE generators) when the backend returns 401. */
export class UnauthenticatedError extends Error {
  constructor() {
    super("unauthenticated");
    this.name = "UnauthenticatedError";
  }
}

export interface UIConfig {
  brand_name: string;
  brand_subtitle: string;
  assistant_name: string;
  favicon_url: string;
  accent_color: string;
  confluence_base_url: string;
  slack_workspace_url: string;
  rootly_web_url: string;
  gitlab_base_url: string;
  model_name?: string | null;
  // Config-driven feature gate: true only when the deployment enabled a
  // live-telemetry MCP integration (datadog / prometheus / k8s / ...).
  // The Investigate tab is hidden when false. Optional for backward compat
  // with older backends that don't return it (treated as false).
  investigation_enabled?: boolean;
}

export async function fetchUIConfig(): Promise<UIConfig> {
  const r = await fetch(`${BASE}/ui-config`);
  if (!r.ok) throw new Error(`ui-config failed: ${r.status}`);
  return r.json();
}

export interface CacheSummary {
  qa: { name?: string; points_count?: number; threshold?: number; default_ttl_seconds?: number; available?: boolean; error?: string };
  investigation: { available: boolean; total: number };
  tool: { hits: number; misses: number; negative_hits: number; evictions: number; skipped_too_big: number; size: number; max_entries: number; by_tool: Record<string, { hits: number; misses: number; negative_hits: number }> };
}

export async function fetchCacheSummary(): Promise<CacheSummary> {
  const r = await fetch(`${BASE}/cache/summary`);
  if (!r.ok) throw new Error(`cache-summary failed: ${r.status}`);
  return r.json();
}

// --- Agent guidance (deployment-wide custom instructions; live-editable) ---
export interface AgentGuidance {
  custom_instructions: string;
  updated_at?: string | null;
  updated_by?: string | null;
  source: "db" | "config" | "none";
}

export async function fetchAgentGuidance(): Promise<AgentGuidance> {
  const r = await apiFetch(`${BASE}/admin/agent-guidance`);
  if (!r.ok) throw new Error(`agent-guidance failed: ${r.status}`);
  return r.json();
}

export async function saveAgentGuidance(text: string): Promise<AgentGuidance> {
  const r = await apiFetch(`${BASE}/admin/agent-guidance`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ custom_instructions: text }),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { const j = await r.json(); detail = j.detail?.reason || j.detail || detail; } catch {}
    throw new Error(`save failed: ${detail}`);
  }
  return r.json();
}

export interface CachePurgeRequest {
  target: "qa" | "investigation" | "tool" | "all";
  strategy: "all" | "older_than" | "repo" | "quality_low" | "thumbs_down" | "question_contains" | "tool_name";
  older_than_hours?: number;
  repo?: string;
  question_substring?: string;
  tool_name?: string;
}

export interface CachePurgeResponse {
  target: string;
  strategy: string;
  purged_qa: number;
  purged_investigation: number;
  purged_tool: number;
  detail?: string;
}

export async function purgeCache(req: CachePurgeRequest): Promise<CachePurgeResponse> {
  const r = await fetch(`${BASE}/cache/purge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { const j = await r.json(); detail = j.detail || detail; } catch {}
    throw new Error(`purge failed: ${detail}`);
  }
  return r.json();
}

export interface Session {
  thread_id: string;
  // Optional for the public Channels view: the channel-conversations endpoint
  // strips the synthetic bot `user_id` (channel users are anonymous), so it's
  // absent there but present for owned web sessions.
  user_id?: string;
  checkpoint_count: number;
  // Enriched by the session store so the conversation list can show a real
  // title/preview/time instead of the opaque thread id. Optional for safety.
  title?: string | null;
  preview?: string | null;
  updated_at?: string | null;
  created_at?: string | null;
  turn_count?: number;
  // Present only on public-channel conversations (slack|discord|telegram|teams).
  // Undefined for ordinary web sessions.
  platform?: ChannelPlatform | null;
}

// The chat-bot platforms that surface shared-channel conversations in the
// read-only Channels browser. Mirrors the backend's PUBLIC_CHANNEL_THREAD
// platforms (opsrag/channels/origin.py).
export type ChannelPlatform = "slack" | "discord" | "telegram" | "teams";

export interface PurposeUsage {
  call_count: number;
  input_tokens: number;
  output_tokens: number;
  avg_latency_ms: number;
}

export interface ModelUsage {
  input_tokens: number;
  output_tokens: number;
  call_count: number;
  avg_latency_ms: number;
  estimated_cost_usd: number;
  by_purpose?: Record<string, PurposeUsage>;
}

export interface PurposeBucket extends PurposeUsage {
  category: "indexing" | "query";
  estimated_cost_usd: number;
}

export interface UsageSummary {
  total_input_tokens: number;
  total_output_tokens: number;
  total_calls: number;
  total_estimated_cost_usd: number;
  indexing_cost_usd?: number;
  query_cost_usd?: number;
  uptime_seconds: number;
  active_sessions: number;
  models: Record<string, ModelUsage>;
  by_purpose?: Record<string, PurposeBucket>;
}

// One week bucket for the Home "Usage this month" mini bar chart.
// Oldest-first; the last entry is the current week. See GET /usage/weekly.
export interface UsageWeek {
  week_start: string;     // ISO date (Monday) of the week
  tokens: number;         // input + output
  input_tokens: number;
  output_tokens: number;
  call_count: number;
  cost_usd: number;
}

export interface IndexingRepo {
  repo: string;
  branch: string;
  status: string;
  source_type?: string;        // "git" | "confluence" | ... — defaults to "git"
  display_name?: string | null; // human-readable label, e.g. "slack:#devops"
  total_files: number;
  indexed_files: number;       // files that produced ≥1 chunk
  skipped_files: number;       // files no parser claimed / parse errors
  processed_files?: number;    // indexed + skipped — recently added to API
  total_chunks: number;
  entities_found: number;
  percent: number;
  elapsed_seconds: number;
  error: string | null;
}

export interface IndexingSummary {
  total_repos: number;
  total_files: number;
  total_indexed: number;
  total_chunks: number;
  repos: IndexingRepo[];
}

export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
}

export async function fetchSessions(userId: string): Promise<Session[]> {
  const resp = await fetch(`${BASE}/sessions/${encodeURIComponent(userId)}`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.sessions ?? [];
}

export interface ReplayedMessage {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  source_urls?: (string | null)[];
  grounded?: boolean;
  query_type?: string | null;
  investigation_id?: string | null;
  // Original Pomerium author of THIS turn. Null for legacy / pre-Pomerium
  // sessions. UI uses these to render "You" vs the teammate's name.
  author_email?: string | null;
  author_name?: string | null;
  // ISO 8601 UTC timestamp of the LangGraph checkpoint that produced
  // this turn. Same value on the user+assistant pair (they share one
  // checkpoint cycle). Null only on legacy rows without a `ts` field.
  ts?: string | null;
}


export interface InvestigationFeedbackResponse {
  investigation_id: string;
  recorded: boolean;
  detail?: string | null;
}

export async function postInvestigationFeedback(
  investigationId: string,
  thumbs: "up" | "down",
  correction?: string,
): Promise<InvestigationFeedbackResponse> {
  const resp = await fetch(`${BASE}/investigation/${encodeURIComponent(investigationId)}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thumbs, correction: correction || null }),
  });
  if (!resp.ok) {
    return { investigation_id: investigationId, recorded: false, detail: `HTTP ${resp.status}` };
  }
  return resp.json();
}

// T1.3 — activated thumbs-up/down with extra context written
// into Postgres `opsrag_feedback` for SRE triage. Same endpoint as
// `postInvestigationFeedback` above (kept for back-compat with code that
// only has an investigation_id); the new entry point lets callers pass
// thread_id / query / answer snippets so SREs can see context without
// re-running the investigation.
export interface FeedbackRequest {
  investigation_id: string;
  direction: 1 | -1;
  note?: string;
  query_snippet?: string;
  answer_snippet?: string;
  thread_id?: string | null;
  user_id?: string | null;
}

export interface FeedbackResponse {
  ok: boolean;
  feedback_id?: number | null;
  detail?: string | null;
}

export async function postFeedback(req: FeedbackRequest): Promise<FeedbackResponse> {
  // Map the new ergonomic shape onto the existing investigation feedback
  // endpoint (so backend stays single-handler).
  const body = {
    thumbs: req.direction === 1 ? "up" : "down",
    correction: req.note ?? null,
    thread_id: req.thread_id ?? null,
    user_id: req.user_id ?? null,
    query_snippet: req.query_snippet ?? null,
    answer_snippet: req.answer_snippet ?? null,
  };
  try {
    const resp = await fetch(
      `${BASE}/investigation/${encodeURIComponent(req.investigation_id)}/feedback`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      },
    );
    if (!resp.ok) {
      return { ok: false, detail: `HTTP ${resp.status}` };
    }
    const data = (await resp.json()) as { recorded?: boolean; feedback_id?: number | null; detail?: string | null };
    return { ok: Boolean(data.recorded), feedback_id: data.feedback_id ?? null, detail: data.detail ?? null };
  } catch (err) {
    // Fire-and-forget UX — never propagate exceptions to the caller.
    return { ok: false, detail: err instanceof Error ? err.message : "network error" };
  }
}

// T1.3 — SRE triage list endpoint.
export interface FeedbackListItem {
  id: number;
  investigation_id: string;
  thread_id: string | null;
  user_id: string | null;
  direction: number;
  note: string | null;
  created_at: string | null;
  query_snippet: string | null;
  answer_snippet: string | null;
}

export async function fetchRecentFeedback(direction?: 1 | -1, limit: number = 50): Promise<FeedbackListItem[]> {
  const params = new URLSearchParams();
  if (direction === 1 || direction === -1) params.set("direction", String(direction));
  params.set("limit", String(limit));
  const resp = await fetch(`${BASE}/feedback?${params.toString()}`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return (data.items ?? []) as FeedbackListItem[];
}

// ── T1.6 — Feedback-as-correction ─────────────────────────
// When a user clicks 👎 and types the correct answer in the chat UI, the
// correction is QUEUED for operator review (it is not live until approved).
// On approval an operator injects it as a modestly-boosted (1.8×) Qdrant chunk.
export interface CorrectionRequest {
  question: string;
  wrong_answer: string;
  correct_answer: string;
  evidence_url?: string | null;
  thread_id?: string | null;
  user_id?: string | null;
}

export interface CorrectionResponse {
  ok: boolean;
  pending_id: number;
  status: string;
  message: string;
  feedback_id?: number | null;
}

export async function postCorrection(req: CorrectionRequest): Promise<CorrectionResponse> {
  const body = {
    question: req.question,
    wrong_answer: req.wrong_answer ?? "",
    correct_answer: req.correct_answer,
    evidence_url: req.evidence_url ?? null,
    thread_id: req.thread_id ?? null,
    user_id: req.user_id ?? null,
  };
  const resp = await fetch(`${BASE}/correction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail: string;
    try { const j = await resp.json(); detail = j.detail || `HTTP ${resp.status}`; }
    catch { detail = `HTTP ${resp.status}`; }
    throw new Error(detail);
  }
  return resp.json();
}

export interface CorrectionListItem {
  chunk_id: string;
  original_question: string | null;
  wrong_answer: string | null;
  correct_answer: string | null;
  user_id: string | null;
  evidence_url: string | null;
  created_at: number | null;
}

export async function fetchCorrections(limit: number = 50): Promise<CorrectionListItem[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  const resp = await fetch(`${BASE}/corrections?${params.toString()}`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return (data.items ?? []) as CorrectionListItem[];
}

export async function deleteCorrection(chunkId: string): Promise<boolean> {
  const resp = await fetch(`${BASE}/corrections/${encodeURIComponent(chunkId)}`, {
    method: "DELETE",
  });
  if (!resp.ok) return false;
  const data = await resp.json();
  return Boolean(data.deleted);
}

export async function fetchSessionMessages(threadId: string): Promise<ReplayedMessage[]> {
  const resp = await fetch(`${BASE}/sessions/${encodeURIComponent(threadId)}/messages`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.messages ?? [];
}

export async function deleteSession(threadId: string): Promise<void> {
  await fetch(`${BASE}/sessions/${encodeURIComponent(threadId)}`, { method: "DELETE" });
}

// ── Public channel conversations (read-only) ────────────────────────────
// Shared chat-channel conversations (Slack/Discord/Telegram/Teams) anyone with
// the `chat` scope can browse. Auth is the same-origin session cookie. The
// backend exposes only `<platform>-thread:` conversations and validates the
// prefix server-side, so private 1:1 DMs and web threads can't leak here. The
// list response strips the synthetic bot user_id (identity is platform-only).

export async function fetchChannelConversations(): Promise<Session[]> {
  const resp = await apiFetch(`${BASE}/channels/conversations`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return (data.conversations ?? []) as Session[];
}

export async function fetchChannelMessages(threadId: string): Promise<ReplayedMessage[]> {
  const resp = await apiFetch(`${BASE}/channels/conversations/${encodeURIComponent(threadId)}/messages`);
  if (!resp.ok) return [];
  const data = await resp.json();
  return (data.messages ?? []) as ReplayedMessage[];
}

// Human-readable label for a channel platform badge.
export function channelPlatformLabel(p: ChannelPlatform | null | undefined): string {
  switch (p) {
    case "slack": return "Slack channel";
    case "discord": return "Discord channel";
    case "telegram": return "Telegram group";
    case "teams": return "Teams channel";
    default: return "Channel";
  }
}

export async function fetchUsage(): Promise<UsageSummary> {
  const resp = await fetch(`${BASE}/usage`);
  return resp.json();
}

export async function fetchUsageWeekly(): Promise<UsageWeek[]> {
  const resp = await fetch(`${BASE}/usage/weekly`);
  const data = await resp.json();
  return Array.isArray(data?.weeks) ? data.weeks : [];
}

export async function fetchIndexing(): Promise<IndexingSummary> {
  const resp = await fetch(`${BASE}/indexing/status`);
  return resp.json();
}

export async function indexRepo(
  repo: string,
  branch: string,
): Promise<{ repo: string; branch: string; chunks_indexed: number }> {
  const resp = await fetch(`${BASE}/index/repo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, branch }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Index failed (${resp.status}): ${detail || resp.statusText}`);
  }
  return resp.json();
}

export async function indexSource(
  sourceType: string,
  scope: string,
): Promise<{ source_type: string; scope: string; chunks_indexed: number }> {
  const resp = await fetch(`${BASE}/index/source`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_type: sourceType, scope }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Index source failed (${resp.status}): ${detail || resp.statusText}`);
  }
  return resp.json();
}

export async function* streamQuery(
  query: string,
  userId: string,
  threadId?: string,
): AsyncGenerator<SSEEvent> {
  // Use apiFetch so a 401 flips the app into login mode BEFORE we start
  // reading the stream. apiFetch only inspects resp.status (never .json()),
  // so the SSE body stays untouched for the happy path below.
  const resp = await apiFetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query, user_id: userId, thread_id: threadId ?? null, stream: true,
    }),
  });
  if (!resp.ok || !resp.body) throw new Error(`Query failed: ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      if (!part.trim()) continue;
      let event = "", data = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (event) {
        try { yield { event, data: JSON.parse(data || "{}") }; }
        catch { yield { event, data: {} }; }
      }
    }
  }
}


// ── Hypothesis-driven investigation (Datadog Bits AI SRE-style) ─────────

export interface InvestigateRequest {
  alert_text: string;
  service_hint?: string | null;
  namespace_hint?: string | null;
  env_hint?: string | null;
  runbook_urls?: string[];
}

export interface InvestigationEvidence {
  source_id: string;
  chunk_id: string;
  snippet: string;
  score: number;
  repo: string;
}

export interface InvestigationNode {
  id: string;
  statement: string;
  status: "pending" | "validated" | "invalidated" | "inconclusive";
  depth: number;
  parent_id: string | null;
  children: string[];
  confidence: number;
  judge_rationale: string;
  termination_reason: string | null;
  evidence_count: number;
  evidence: InvestigationEvidence[];
  hypothesis_source?: "llm" | "runbook" | "past_investigation";
}

export interface InvestigationSummary {
  investigation_id: string;
  duration_sec: number;
  tree_size: { total_nodes: number; validated: number; invalidated: number; inconclusive: number; pending: number };
  max_depth_reached: number;
  tool_calls: { total: number; retrieval: number; llm_query_gen: number; llm_judge: number; llm_synth: number };
  tokens: { input: number; output: number; total: number };
  circuit_breakers_hit: string[];
  outcome: string;
}

export interface InvestigationResponse {
  investigation_id: string;
  alert_text: string;
  service_hint: string | null;
  namespace_hint: string | null;
  env_hint: string | null;
  bootstrap_findings: string[];
  nodes: InvestigationNode[];
  root_ids: string[];
  final_chain_node_ids: string[];
  final_root_cause: string | null;
  outcome: string;
  summary: InvestigationSummary;
}

// ── Investigation history (Tier B replay) ────────────────────────

export interface InvestigationHistoryItem {
  investigation_id: string;
  alert_text: string;
  service_hint: string | null;
  namespace_hint: string | null;
  env_hint: string | null;
  outcome: string;
  final_root_cause: string;
  created_at: number;
  age_seconds: number;
}

export async function fetchInvestigationHistory(limit: number = 50): Promise<InvestigationHistoryItem[]> {
  const r = await fetch(`${BASE}/investigations?limit=${limit}`);
  if (!r.ok) throw new Error(`history failed: ${r.status}`);
  const d = await r.json();
  return d.investigations || [];
}

// ── M1: Identity (/api/me) ──────────────────────────────────────────────
// Backend contract: `/api/me` returns either
//   { anonymous: true }                                        — no identity
// OR a full MeResponse-shaped object when authenticated via Pomerium.
// On network errors we degrade to anonymous (never throw) so the UI keeps
// rendering even when the identity sidecar is down.

export interface MeResponse {
  oid: string | null;
  email: string | null;
  name: string | null;
  picture_url: string | null;
  groups: string[];
  is_anonymous: boolean;
  tracking_enabled: boolean;
  is_admin: boolean;
  // M-auth — roles/scopes resolved server-side from group→role mappings.
  // `scopes` drives nav gating; `roles` drives the Home scope-preview pill.
  // Default [] so legacy backends (pre-roles) degrade to "no scopes".
  roles: string[];
  scopes: Scope[];
}

// In OPEN mode the anonymous identity still has every scope so the demo
// runs unrestricted (nav gate is transparent). The backend is the source of
// truth; this is only the network-failure fallback.
const ALL_SCOPES: Scope[] = ["chat", "investigate", "mcp", "admin"];

const ANONYMOUS_ME: MeResponse = {
  oid: null,
  email: null,
  name: null,
  picture_url: null,
  groups: [],
  is_anonymous: true,
  tracking_enabled: false,
  is_admin: false,
  roles: [],
  scopes: ALL_SCOPES,
};

// Coerce an arbitrary list into the known Scope vocabulary.
function parseScopes(raw: unknown): Scope[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter((s): s is Scope =>
    s === "chat" || s === "investigate" || s === "mcp" || s === "admin",
  );
}

export async function fetchMe(): Promise<MeResponse> {
  try {
    const r = await fetch(`${BASE}/me`);
    if (!r.ok) {
      // 401 in login/oidc mode -> latch "auth required" so AuthGate shows the
      // login page on boot (not just after a later action 401s).
      if (r.status === 401) triggerLogin();
      return ANONYMOUS_ME;
    }
    const data = await r.json();
    // Backend may return a minimal `{anonymous: true}` shape for unauthenticated
    // callers; normalise to the full MeResponse so the UI doesn't need to branch.
    if (data && data.anonymous === true) return ANONYMOUS_ME;
    return {
      oid: data.oid ?? null,
      email: data.email ?? null,
      name: data.name ?? null,
      picture_url: data.picture_url ?? null,
      groups: Array.isArray(data.groups) ? data.groups : [],
      is_anonymous: Boolean(data.is_anonymous ?? false),
      tracking_enabled: Boolean(data.tracking_enabled ?? true),
      is_admin: Boolean(data.is_admin ?? false),
      roles: Array.isArray(data.roles) ? data.roles : [],
      scopes: parseScopes(data.scopes),
    };
  } catch {
    return ANONYMOUS_ME;
  }
}

// ── M-auth: login / logout / SSO providers ──────────────────────────────
// Backend contract (per design DESIGN 1 / FINDING explore:ui-auth):
//   POST /api/auth/login           — { username, password } → 200 on success,
//                                     401 on bad credentials.
//   POST /api/auth/logout          — clears the session cookie.
//   GET  /api/auth/sso/{provider}/login — top-level redirect into the IdP.
// Auth is cookie-based (HttpOnly session), so the browser carries it on
// every subsequent request automatically — no token handling in JS.

export type SSOProvider = "google" | "microsoft" | "github";

export interface LoginResult {
  ok: boolean;
  detail?: string;
}

export async function login(username: string, password: string): Promise<LoginResult> {
  try {
    // Backend /auth/login expects form-encoded `email` + `password`
    // (FastAPI Form fields) and sets the session cookie on the response.
    const form = new URLSearchParams();
    form.set("email", username.trim());
    form.set("password", password);
    const r = await fetch(`${BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      credentials: "same-origin",
      body: form.toString(),
    });
    if (r.ok) {
      clearAuthRequired();
      return { ok: true };
    }
    let detail = `HTTP ${r.status}`;
    try { const j = await r.json(); detail = j.detail || j.error || detail; } catch { /* non-JSON */ }
    return { ok: false, detail };
  } catch (err) {
    return { ok: false, detail: err instanceof Error ? err.message : "network error" };
  }
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${BASE}/auth/logout`, { method: "POST" });
  } catch {
    // Best-effort — the gate redirects to login regardless.
  }
}

// Build the top-level SSO login URL. Callers do `window.location.href = ...`
// so the IdP redirect happens as a top-level GET (cookie SameSite=Lax safe).
export function ssoLoginUrl(provider: SSOProvider): string {
  return `${BASE}/auth/sso/${provider}/login`;
}

// Which SSO providers the backend advertises. Best-effort: if the backend
// exposes a providers endpoint we use it; otherwise the LoginPage falls back
// to probing each provider URL and hiding the ones that fail.
export interface AuthMethods {
  passwordEnabled: boolean;
  providers: SSOProvider[];
}

export async function fetchAuthProviders(): Promise<AuthMethods> {
  try {
    const r = await fetch(`${BASE}/auth/providers`);
    if (!r.ok) return { passwordEnabled: true, providers: [] };
    const data = await r.json();
    const raw = Array.isArray(data) ? data : (data.providers ?? data.sso ?? []);
    const providers = (Array.isArray(raw) ? raw : []).filter(
      (p): p is SSOProvider => p === "google" || p === "microsoft" || p === "github",
    );
    // Default password on unless the backend explicitly disables it (SSO-only).
    const passwordEnabled = data && typeof data.password_enabled === "boolean"
      ? data.password_enabled
      : true;
    return { passwordEnabled, providers };
  } catch {
    return { passwordEnabled: true, providers: [] };
  }
}

// ── M2: Per-user usage ──────────────────────────────────────────────────

export interface UsageByUser {
  user_oid: string;
  email: string;
  display_name: string | null;
  query_count: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd_micros: number;
  last_active_at: string | null;
}

// `UsageMine` is the same shape as a single-row UsageByUser — kept as a
// distinct alias so call sites read clearly.
export type UsageMine = UsageByUser;

export async function fetchUsageByUser(): Promise<UsageByUser[]> {
  try {
    const r = await fetch(`${BASE}/admin/usage`);
    if (r.status === 403) return [];
    if (!r.ok) return [];
    const data = await r.json();
    // Accept either a bare array or {users: [...]} envelope.
    if (Array.isArray(data)) return data as UsageByUser[];
    return (data.users ?? []) as UsageByUser[];
  } catch {
    return [];
  }
}

export async function fetchUsageMine(): Promise<UsageMine | null> {
  try {
    const r = await fetch(`${BASE}/me/usage`);
    if (r.status === 401 || r.status === 403 || r.status === 404) return null;
    if (!r.ok) return null;
    const data = await r.json();
    if (!data || data.anonymous === true) return null;
    return data as UsageMine;
  } catch {
    return null;
  }
}

// Render cost stored as micro-CENTS (1e-8 USD) for display. Default 4 dp
// because most per-user costs sit below $1 and 2dp loses signal.
// IMPORTANT: divide by 100_000_000 — the backend's `cost_usd_micros`
// column is in micro-cents (per opsrag/llms/pricing.py docstring),
// NOT micro-dollars. Earlier this divided by 1_000_000 which inflated
// every cost by 100x.
export function microsToUsd(micros: number, dp: number = 4): string {
  if (!micros || micros === 0) return "$0.00";
  const usd = micros / 100_000_000;
  // Force trailing-zero padding so columns line up.
  return "$" + usd.toFixed(dp);
}

// ── M3: MCP tokens ──────────────────────────────────────────────────────

export interface MCPToken {
  id: string;
  name: string;
  created_at: string;
  expires_at: string | null;
  last_used_at: string | null;
}

export interface MCPTokenCreated extends MCPToken {
  // Only present in the response to POST /api/mcp/tokens — the raw secret.
  // Never re-fetched; UI must surface it once and never store it.
  token: string;
}

export async function listMcpTokens(): Promise<MCPToken[]> {
  const r = await fetch(`${BASE}/mcp/tokens`);
  if (!r.ok) {
    if (r.status === 401 || r.status === 403) return [];
    throw new Error(`list tokens failed: ${r.status}`);
  }
  const data = await r.json();
  if (Array.isArray(data)) return data as MCPToken[];
  return (data.tokens ?? []) as MCPToken[];
}

export async function createMcpToken(
  name: string,
  expires_in_days: number | null,
): Promise<MCPTokenCreated> {
  const r = await fetch(`${BASE}/mcp/tokens`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, expires_in_days }),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { const j = await r.json(); detail = j.detail || detail; } catch {}
    throw new Error(`create token failed: ${detail}`);
  }
  return r.json();
}

export async function revokeMcpToken(id: string): Promise<void> {
  const r = await fetch(`${BASE}/mcp/tokens/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!r.ok && r.status !== 404) {
    throw new Error(`revoke token failed: ${r.status}`);
  }
}

// ── Centralized-MCP audit log (admin-only) ──────────────────────────────
// Read-only view over `opsrag_mcp_audit`: who (user + token) called which
// read-only tool, when, and the result. Args are never stored literally —
// only `args_hash` (sha256). Both endpoints require the `admin` scope.

export interface McpAuditRow {
  occurred_at: string | null;
  user_oid: string | null;
  token_id: string | null;
  tool_name: string;
  args_hash: string | null;
  latency_ms: number | null;
  status: string;       // ok | denied | error
  error: string | null;
}

export interface McpAuditList {
  rows: McpAuditRow[];
  total: number;
  limit: number;
  offset: number;
}

export interface McpAuditSummary {
  total_calls: number;
  error_count: number;
  denied_count: number;
  distinct_users: number;
  distinct_tools: number;
  top_tools: { tool_name: string; calls: number }[];
}

export interface McpAuditFilters {
  tool?: string;
  user?: string;
  status?: string;
  sinceMinutes?: number;
  limit?: number;
  offset?: number;
}

export async function fetchMcpAudit(f: McpAuditFilters = {}): Promise<McpAuditList> {
  const q = new URLSearchParams();
  if (f.tool) q.set("tool", f.tool);
  if (f.user) q.set("user", f.user);
  if (f.status) q.set("status", f.status);
  if (f.sinceMinutes) q.set("since_minutes", String(f.sinceMinutes));
  q.set("limit", String(f.limit ?? 100));
  q.set("offset", String(f.offset ?? 0));
  const r = await fetch(`${BASE}/mcp/audit?${q.toString()}`);
  if (!r.ok) {
    if (r.status === 401 || r.status === 403) {
      return { rows: [], total: 0, limit: f.limit ?? 100, offset: f.offset ?? 0 };
    }
    throw new Error(`mcp audit failed: ${r.status}`);
  }
  return r.json();
}

export async function fetchMcpAuditSummary(sinceMinutes?: number): Promise<McpAuditSummary> {
  const q = sinceMinutes ? `?since_minutes=${sinceMinutes}` : "";
  const r = await fetch(`${BASE}/mcp/audit/summary${q}`);
  if (!r.ok) {
    if (r.status === 401 || r.status === 403) {
      return { total_calls: 0, error_count: 0, denied_count: 0, distinct_users: 0, distinct_tools: 0, top_tools: [] };
    }
    throw new Error(`mcp audit summary failed: ${r.status}`);
  }
  return r.json();
}


// --- Admin: users & roles (RBAC) --------------------------------------
// All admin-gated; apiFetch flips the app into login mode on a 401.
export interface AdminUser {
  id: string;
  email: string;
  name: string | null;
  roles: string[];
  scopes: string[];
  has_password: boolean;
  email_verified: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface RoleInfo {
  role: string;
  label: string;
  description: string;
  scopes: string[];
}

export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const r = await apiFetch(`${BASE}/admin/users`);
  if (!r.ok) throw new Error(`list users failed: ${r.status}`);
  const data = await r.json();
  return (data.users ?? []) as AdminUser[];
}

export async function fetchRoleCatalog(): Promise<RoleInfo[]> {
  const r = await apiFetch(`${BASE}/admin/roles`);
  if (!r.ok) throw new Error(`role catalog failed: ${r.status}`);
  const data = await r.json();
  return (data.roles ?? []) as RoleInfo[];
}

export async function updateUserRoles(
  userId: string,
  roles: string[],
): Promise<AdminUser> {
  const r = await apiFetch(`${BASE}/admin/users/${encodeURIComponent(userId)}/roles`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roles }),
  });
  if (!r.ok) {
    let detail: unknown = `${r.status}`;
    try { const j = await r.json(); detail = j.detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : `update roles failed: ${r.status}`);
  }
  return r.json();
}


// ── Runbooks (hand-authored + RAG via /api/runbooks) ────────────────────

export const RUNBOOK_FAILURE_CLASSES = [
  "deploy_regression",
  "dependency_outage",
  "infra_change",
  "resource_exhaustion",
  "config_change",
  "external_vendor",
  "data_quality",
  "unknown_recovered",
] as const;
export type RunbookFailureClass = typeof RUNBOOK_FAILURE_CLASSES[number];

// Human-readable labels for the snake_case failure-class enum. The raw value
// is the API/storage contract; this is purely for display in the UI.
export const RUNBOOK_FAILURE_CLASS_LABELS: Record<string, string> = {
  deploy_regression: "Deploy regression",
  dependency_outage: "Dependency outage",
  infra_change: "Infrastructure change",
  resource_exhaustion: "Resource exhaustion",
  config_change: "Config change",
  external_vendor: "External vendor",
  data_quality: "Data quality",
  unknown_recovered: "Unknown / recovered",
};

// Display label for a failure-class value; falls back to title-casing any
// value the map doesn't know (forward-compatible with new backend enums).
export function failureClassLabel(value: string | null | undefined): string {
  if (!value) return "";
  return (
    RUNBOOK_FAILURE_CLASS_LABELS[value] ||
    value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

export const RUNBOOK_SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"] as const;
export type RunbookSeverity = typeof RUNBOOK_SEVERITIES[number];

export interface Runbook {
  id: string;
  title: string;
  body_markdown: string;
  service: string | null;
  issue_kind: RunbookFailureClass | null;
  severity_min: RunbookSeverity | null;
  priority: number;
  tags: string[];
  source: "hand" | "imported" | "auto";
  author_email: string | null;
  source_investigation_id: string | null;
  enabled: boolean;
  created_at: string;
  updated_at: string;
  used_count: number;
  thumbs_up_count: number;
  thumbs_down_count: number;
  last_used_at: string | null;
}

export interface RunbookCreatePayload {
  title: string;
  body_markdown: string;
  service?: string | null;
  issue_kind?: RunbookFailureClass | null;
  severity_min?: RunbookSeverity | null;
  priority?: number;
  tags?: string[];
}

export interface RunbookUpdatePayload extends Partial<RunbookCreatePayload> {
  enabled?: boolean;
  change_note?: string;
}

export interface RunbookVersion {
  id: number;
  runbook_id: string;
  version_num: number;
  title: string;
  body_markdown: string;
  service: string | null;
  issue_kind: string | null;
  severity_min: string | null;
  priority: number | null;
  tags: string[];
  edited_by: string | null;
  edited_at: string;
  change_note: string | null;
}

export interface RunbookListResponse {
  count: number;
  runbooks: Runbook[];
}

export interface PromoteFromInvestigationResponse {
  draft_markdown: string;
  suggested_title: string | null;
  suggested_service: string | null;
  suggested_issue_kind: string | null;
  source_investigation_id: string;
}

export async function listRunbooks(opts?: {
  service?: string;
  issue_kind?: string;
  enabled_only?: boolean;
  limit?: number;
}): Promise<RunbookListResponse> {
  const qs = new URLSearchParams();
  if (opts?.service) qs.set("service", opts.service);
  if (opts?.issue_kind) qs.set("issue_kind", opts.issue_kind);
  if (opts?.enabled_only !== undefined) qs.set("enabled_only", String(opts.enabled_only));
  if (opts?.limit) qs.set("limit", String(opts.limit));
  const url = qs.toString() ? `${BASE}/runbooks?${qs}` : `${BASE}/runbooks`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`listRunbooks ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function getRunbook(id: string): Promise<Runbook> {
  const resp = await fetch(`${BASE}/runbooks/${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(`getRunbook ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function createRunbook(body: RunbookCreatePayload): Promise<Runbook> {
  const resp = await fetch(`${BASE}/runbooks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`createRunbook ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function updateRunbook(id: string, body: RunbookUpdatePayload): Promise<Runbook> {
  const resp = await fetch(`${BASE}/runbooks/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`updateRunbook ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function deleteRunbook(id: string, opts?: { hard?: boolean }): Promise<void> {
  const url = opts?.hard
    ? `${BASE}/runbooks/${encodeURIComponent(id)}?hard=true`
    : `${BASE}/runbooks/${encodeURIComponent(id)}`;
  const resp = await fetch(url, { method: "DELETE" });
  if (!resp.ok && resp.status !== 204)
    throw new Error(`deleteRunbook ${resp.status}: ${await resp.text()}`);
}

export async function fetchRunbookVersions(id: string, limit: number = 50): Promise<RunbookVersion[]> {
  const resp = await fetch(`${BASE}/runbooks/${encodeURIComponent(id)}/versions?limit=${limit}`);
  if (!resp.ok) throw new Error(`fetchRunbookVersions ${resp.status}: ${await resp.text()}`);
  const j = await resp.json();
  return j.versions || [];
}

export async function promoteInvestigationToRunbook(
  investigationId: string,
  overrides?: { title?: string; service?: string; issue_kind?: string; severity_min?: string }
): Promise<PromoteFromInvestigationResponse> {
  const resp = await fetch(`${BASE}/runbooks/from-investigation/${encodeURIComponent(investigationId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(overrides || {}),
  });
  if (!resp.ok) throw new Error(`promoteInvestigationToRunbook ${resp.status}: ${await resp.text()}`);
  return resp.json();
}


// ── Knowledge Graph status ──────────────────────────────────────────────
// Backed by GET /graph/stats. The graph store is provider-selected; the
// default `none` (NullGraphStore) reports enabled=false so the UI renders
// an honest "no graph backend configured" state. A real backend (neo4j)
// fills labels[] / relationship_types[].
export interface GraphStats {
  provider: string;
  enabled: boolean;
  label_count: number;
  relationship_type_count: number;
  labels: string[];
  relationship_types: string[];
}

export async function fetchGraphStats(): Promise<GraphStats> {
  // Use apiFetch (not bare fetch) so a 401 in login mode flips the app into
  // login mode instead of surfacing as a misleading "graph disabled" error.
  // The endpoint is authed by the session COOKIE via the OIDC/session
  // middleware (same as every other authenticated read); same-origin fetch
  // carries the HttpOnly session cookie automatically.
  const r = await apiFetch(`${BASE}/graph/stats`);
  if (!r.ok) throw new Error(`graph/stats failed: ${r.status}`);
  return r.json();
}

// ── Knowledge Graph view (filtered subgraph for rendering) ──────────────
// Backed by GET /graph/view?view=business|public|private. Returns a small,
// view-scoped subgraph (nodes + edges) the UI draws with ECharts. `provider`
// mirrors /graph/stats ("neo4j" when a real backend is wired, "disabled"
// when knowledge_graph.provider=none). `truncated` flags that the backend
// capped the result set.
export interface GraphNode { id: string; name: string; type: string }
export interface GraphEdge { source: string; target: string; type: string }
export interface GraphView {
  provider: string;
  view: string;
  truncated: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export async function fetchGraphView(
  view: "business" | "public" | "private",
): Promise<GraphView> {
  const r = await apiFetch(`${BASE}/graph/view?view=${view}`);
  if (!r.ok) throw new Error(`graph/view failed: ${r.status}`);
  return r.json();
}

// ── MCP Integrations catalog ────────────────────────────────────────────
// Backed by GET /integrations. Lists every integration in the registry
// with its enabled state (derived from active config, same source /readyz
// uses), tool count, and whether it exposes an upstream health probe.
export interface Integration {
  name: string;
  display_name: string;
  enabled: boolean;
  tool_count: number;
  tool_names: string[];
  has_health_probe: boolean;
  required_env: string[];
}

export interface IntegrationsSummary {
  integrations: Integration[];
  enabled_count: number;
  total: number;
}

export async function fetchIntegrations(): Promise<IntegrationsSummary> {
  const r = await fetch(`${BASE}/integrations`);
  if (!r.ok) throw new Error(`integrations failed: ${r.status}`);
  return r.json();
}

// ── Indexing job history ────────────────────────────────────────────────
// Backed by GET /indexing/jobs. Distinct from fetchIndexing()/IndexingSummary
// (the current per-source catalog behind the Sources page): this is the
// newest-first run history with status + timing + error, for the
// Operations -> Indexing Jobs page.
export interface IndexingJob {
  id: number;
  repo: string;
  branch: string;
  source_type: string;
  display_name: string | null;
  status: "running" | "success" | "failed";
  started_at: number;            // epoch seconds
  finished_at: number | null;    // epoch seconds, null while running
  duration_seconds: number;
  chunks_indexed: number;
  files_indexed: number;
  error: string | null;
  kind: "run" | "restored";
}

export interface IndexingJobsSummary {
  jobs: IndexingJob[];
  total: number;
  running: number;
  failed: number;
}

export async function fetchIndexingJobs(): Promise<IndexingJobsSummary> {
  const r = await fetch(`${BASE}/indexing/jobs`);
  if (!r.ok) throw new Error(`indexing/jobs failed: ${r.status}`);
  return r.json();
}
