import { useState, useEffect, useRef, useCallback } from "react";
import Sidebar, { type Page } from "./components/Sidebar";
import ChatMessage, { type Message, type RichComponent } from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import UsagePage from "./components/UsagePage";
import MCPAuditPage from "./components/MCPAuditPage";
import ErrorBoundary from "./components/ErrorBoundary";
import IndexingPage from "./components/IndexingPage";
import AgentGuidancePage from "./components/AgentGuidancePage";
import CachePage from "./components/CachePage";
import InvestigationPage from "./components/InvestigationPage";
import MCPTokensPage from "./components/MCPTokensPage";
import UsersRolesPage from "./components/UsersRolesPage";
import { RunbookListPage } from "./components/RunbookListPage";
import { RunbookEditorPage } from "./components/RunbookEditorPage";
import HomePage from "./components/HomePage";
import KnowledgeGraphPage from "./components/KnowledgeGraphPage";
import IntegrationsPage from "./components/IntegrationsPage";
import RetrievalQualityPage from "./components/RetrievalQualityPage";
import IndexingJobsPage from "./components/IndexingJobsPage";
import ConversationListPane from "./components/ConversationListPane";
import InvestigationListPane from "./components/InvestigationListPane";
import { fetchSessions, fetchSessionMessages, deleteSession, streamQuery, fetchUIConfig, fetchInvestigationHistory, logout, fetchChannelConversations, fetchChannelMessages, channelPlatformLabel, type Session, type UIConfig, type InvestigationHistoryItem, type MeResponse } from "./api";
import type { ProgressStep, CacheHitInfo } from "./components/ThinkingProgress";
// IconBolt previously powered the chat empty-state glyph; the premium
// redesign replaces it with the brand logo PNG, so the import is dead.

// `Page` is defined and exported by Sidebar.tsx (single source of truth for
// the navigation model) and imported above.
const USER_ID = "default";
const MODEL_NAME = "gemini-2.5-flash";

// Per-suggestion accent class — picks a side-rail color in the premium
// design (purple for Pipelines/Terraform, orange for Alerts, green for
// Helm). Each card gets a different class so the empty state is
// visually rhythmic instead of monotone.
const SUGGESTED_PROMPTS: Array<{ cat: string; text: string; tone?: "q" | "h" | "t" }> = [
  { cat: "Pipelines", text: "How do I enable a new GitLab CI pipeline using the templates?" },
  { cat: "Helm",      text: "What's the deployment process for a Helm chart in our gitops setup?", tone: "h" },
  { cat: "Alerts",    text: "When HighErrorRate fires on a service, what should I check first?", tone: "q" },
  { cat: "Terraform", text: "Show me the Terraform module for `my-service` and its variables.", tone: "t" },
];

// Hash routing:
//   ``  / `home`         → Home dashboard (default landing)
//   `chat`               → chat (no specific thread)
//   `chat/<thread_id>`   → chat, opening this session — shareable URL
//   `sources` / `graph` / `integrations` / `indexing` / `connections` /
//   `cache` / `usage` / `quality` / `investigate` / `docs` → that page
//   `mcp-tokens` (legacy) → connections
const SIMPLE_PAGES: Page[] = [
  "home", "usage", "indexing", "cache", "investigate", "docs", "channels",
  "connections", "sources", "graph", "integrations", "quality", "mcpaudit", "users", "guidance",
];

function parseHash(): { page: Page; threadId: string | null; runbookId: string | null; investigationId: string | null } {
  const h = window.location.hash.replace(/^#\/?/, "");
  if (h === "chat" || h.startsWith("chat/")) {
    const tid = h.startsWith("chat/") ? h.slice("chat/".length).trim() : "";
    return { page: "chat", threadId: tid || null, runbookId: null, investigationId: null };
  }
  // #/runbooks/new | #/runbooks/edit/<uuid> | #/runbooks
  if (h === "runbooks/new") {
    return { page: "runbook-edit", threadId: null, runbookId: null, investigationId: null };
  }
  if (h.startsWith("runbooks/edit/")) {
    const rid = h.slice("runbooks/edit/".length).trim();
    return { page: "runbook-edit", threadId: null, runbookId: rid || null, investigationId: null };
  }
  if (h === "runbooks") {
    return { page: "runbooks", threadId: null, runbookId: null, investigationId: null };
  }
  // #/investigate?id=<uuid> — resume a running investigation across refresh
  if (h.startsWith("investigate?")) {
    const qs = new URLSearchParams(h.slice("investigate".length).replace(/^\?/, ""));
    const inv = qs.get("id");
    return { page: "investigate", threadId: null, runbookId: null, investigationId: inv || null };
  }
  // Legacy alias: the old "MCP Tokens" page is now "Connections".
  if (h === "mcp-tokens") {
    return { page: "connections", threadId: null, runbookId: null, investigationId: null };
  }
  if ((SIMPLE_PAGES as string[]).includes(h)) {
    return { page: h as Page, threadId: null, runbookId: null, investigationId: null };
  }
  return { page: "home", threadId: null, runbookId: null, investigationId: null };
}

function pageFromHash(): Page {
  return parseHash().page;
}

interface AppProps {
  // Identity is owned by <AuthGate> and threaded in. `me` is always present
  // here (the gate only renders App once /me resolves). `reloadMe` lets the
  // shell re-fetch identity (used after sign-out).
  me: MeResponse;
  reloadMe: () => Promise<void>;
}

export default function App({ me, reloadMe }: AppProps) {
  const [page, setPageState] = useState<Page>(pageFromHash);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeThread, setActiveThread] = useState<string | null>(() => parseHash().threadId);
  // Track which thread we've already loaded from URL so a programmatic
  // setActiveThread (e.g. after `handleSend` creates a new thread) doesn't
  // trigger a re-load loop with the hash listener.
  const lastLoadedFromHashRef = useRef<string | null>(null);
  // Investigation history sidebar (Tier B replay).
  const [investigations, setInvestigations] = useState<InvestigationHistoryItem[]>([]);
  const [activeInvestigation, setActiveInvestigation] = useState<string | null>(
    () => parseHash().investigationId,
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  // Read-only public Channels browser (shared chat-channel conversations).
  // Separate state from the owned-web `sessions`/`messages` so the two panes
  // never cross-contaminate. No new/delete/reply here — strictly read-only.
  const [channelConversations, setChannelConversations] = useState<Session[]>([]);
  const [activeChannelThread, setActiveChannelThread] = useState<string | null>(null);
  const [channelMessages, setChannelMessages] = useState<Message[]>([]);
  const channelBodyRef = useRef<HTMLDivElement>(null);
  // Sidebar collapse — persisted across reloads. Premium design dropped
  // dark mode entirely so the localStorage key for theme is no longer
  // read; we keep the same `opsrag-` prefix for the new key.
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(
    () => localStorage.getItem("opsrag-sidebar-collapsed") === "1",
  );
  // Classic UI variant is hidden — kept on disk but unreachable. The
  // toggle in the sidebar is removed and main.tsx always mounts the
  // premium tree. (See AppClassic.tsx for the dormant classic build.)
  const [uiConfig, setUIConfig] = useState<UIConfig | null>(null);
  // Identity comes from <AuthGate> via props now (was a local fetchMe on
  // mount). The gate owns the lifecycle so the shell never flashes the wrong
  // chrome before /me resolves.
  const endRef = useRef<HTMLDivElement>(null);
  // The scrollable messages container (.chat-body). We scroll THIS element
  // (not the window) to the bottom on mount + whenever messages change, so
  // the latest turn is visible while the sticky header + composer stay put.
  const chatBodyRef = useRef<HTMLDivElement>(null);

  // Load messages for a given thread. Extracted so both Sidebar.onSelect
  // and the hash listener can share the same logic.
  const loadThreadMessages = useCallback(async (id: string) => {
    setMessages([]);
    try {
      const replayed = await fetchSessionMessages(id);
      setMessages(replayed.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
        sourceUrls: m.source_urls,
        grounded: m.grounded,
        queryType: m.query_type ?? null,
        investigationId: m.investigation_id ?? null,
        authorEmail: m.author_email ?? null,
        authorName: m.author_name ?? null,
        ts: m.ts ?? null,
      })));
    } catch {
      // Swallow — empty pane is fine fallback. New turns still work.
    }
  }, []);

  // Page + thread router. Pushes to window.location.hash so the URL is
  // shareable. Skips no-op writes to avoid extra hashchange events.
  const setPage = (p: Page) => {
    setPageState(p);
    const target = p === "chat"
      ? (activeThread ? `chat/${activeThread}` : "chat")
      : p;
    const current = window.location.hash.replace(/^#\/?/, "");
    if (current !== target) {
      window.location.hash = target;
    }
  };

  // Sync URL when activeThread changes (e.g. handleSend just created a
  // new thread, or a session was selected from the sidebar). Only fires
  // when we're on the chat page.
  useEffect(() => {
    if (page !== "chat") return;
    const target = activeThread ? `chat/${activeThread}` : "chat";
    const current = window.location.hash.replace(/^#\/?/, "");
    if (current !== target) {
      window.location.hash = target;
    }
  }, [activeThread, page]);

  // Listen for hashchange — handles browser back/forward AND copy-pasted
  // links. If the new hash names a different thread, load it.
  useEffect(() => {
    const onHash = () => {
      const { page: nextPage, threadId: nextTid } = parseHash();
      setPageState(nextPage);
      if (nextPage === "chat" && nextTid && nextTid !== activeThread) {
        setActiveThread(nextTid);
        if (lastLoadedFromHashRef.current !== nextTid) {
          lastLoadedFromHashRef.current = nextTid;
          loadThreadMessages(nextTid);
        }
      } else if (nextPage === "chat" && !nextTid && activeThread) {
        setActiveThread(null);
        setMessages([]);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [activeThread, loadThreadMessages]);

  // Mount-time: if URL had a thread on first paint, load it immediately.
  useEffect(() => {
    const { threadId } = parseHash();
    if (threadId && lastLoadedFromHashRef.current !== threadId) {
      lastLoadedFromHashRef.current = threadId;
      loadThreadMessages(threadId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Premium design is light-only. Strip any stale data-theme="dark"
    // that pre-redesign sessions persisted, so the chrome stays consistent.
    document.documentElement.removeAttribute("data-theme");
    localStorage.removeItem("opsrag-theme");
    localStorage.setItem("opsrag-sidebar-collapsed", sidebarCollapsed ? "1" : "0");
  }, [sidebarCollapsed]);


  // Sign-out: POST /api/auth/logout, then ask the gate to re-resolve /me.
  // After logout the backend (in login/oidc mode) returns 401, which
  // apiFetch turns into the login wall; in open mode it just re-renders the
  // anonymous shell. Defensive — never throws.
  const handleSignOut = useCallback(async () => {
    await logout();
    await reloadMe();
  }, [reloadMe]);

  useEffect(() => {
    fetchUIConfig().then((cfg) => {
      setUIConfig(cfg);
      document.title = `${cfg.brand_name} · ${cfg.brand_subtitle}`;
      if (cfg.favicon_url) {
        let link = document.querySelector("link[rel='icon']") as HTMLLinkElement | null;
        if (!link) {
          link = document.createElement("link");
          link.rel = "icon";
          document.head.appendChild(link);
        }
        link.href = cfg.favicon_url;
      }
      if (cfg.accent_color) {
        document.documentElement.style.setProperty("--primary", cfg.accent_color);
      }
    }).catch(() => { /* fall back to baked defaults */ });
  }, []);

  useEffect(() => {
    // Pin the messages list to the bottom. Scroll the container directly
    // (instant, never the window) so opening a long conversation lands on
    // the latest message rather than the title. New live turns then track
    // the bottom as content streams in.
    const el = chatBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
    else endRef.current?.scrollIntoView();
  }, [messages]);

  const handleNewConversation = useCallback(() => {
    setActiveThread(null);
    setMessages([]);
    setPageState("chat");
    if (window.location.hash.replace(/^#\/?/, "") !== "chat") {
      window.location.hash = "chat";
    }
  }, []);

  // Brand click → Home dashboard (the landing page). Distinct from
  // "New Chat", which lands on an empty chat composer.
  const handleGoHome = useCallback(() => {
    setPageState("home");
    if (window.location.hash.replace(/^#\/?/, "") !== "home") {
      window.location.hash = "home";
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    const s = await fetchSessions(USER_ID);
    setSessions(s);
  }, []);

  const refreshInvestigations = useCallback(async () => {
    try {
      const items = await fetchInvestigationHistory(50);
      setInvestigations(items);
    } catch { /* sidebar list is best-effort */ }
  }, []);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);
  // Pull investigation history the first time the user opens the page,
  // not on every mount — keeps the chat-only flow cheap.
  useEffect(() => {
    if (page === "investigate") refreshInvestigations();
  }, [page, refreshInvestigations]);

  // ── Public Channels browser (read-only) ──────────────────────────────
  const refreshChannelConversations = useCallback(async () => {
    try {
      const c = await fetchChannelConversations();
      setChannelConversations(c);
    } catch { /* best-effort — empty list is a fine fallback */ }
  }, []);

  const loadChannelMessages = useCallback(async (id: string) => {
    setChannelMessages([]);
    try {
      const replayed = await fetchChannelMessages(id);
      setChannelMessages(replayed.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
        sourceUrls: m.source_urls,
        grounded: m.grounded,
        queryType: m.query_type ?? null,
        investigationId: m.investigation_id ?? null,
        // Channel users are anonymous — never surface author identity here.
        authorEmail: null,
        authorName: null,
        ts: m.ts ?? null,
      })));
    } catch { /* empty pane is a fine fallback */ }
  }, []);

  // Lazy-load the channel list the first time the page is opened.
  useEffect(() => {
    if (page === "channels") refreshChannelConversations();
  }, [page, refreshChannelConversations]);

  // Keep the channel detail pinned to the bottom as messages load.
  useEffect(() => {
    const el = channelBodyRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [channelMessages]);

  const handleSend = async (query: string) => {
    // Stamp live turns with a timestamp so each message shows its time
    // (replayed/historical turns already carry `ts` from the backend).
    const nowIso = new Date().toISOString();
    const userMsg: Message = { role: "user", content: query, ts: nowIso };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true, progress: [], ts: nowIso };
    setMessages((p) => [...p, userMsg, assistantMsg]);
    setLoading(true);

    try {
      let answer = "", sources: string[] = [], sourceUrls: (string | null)[] = [], grounded = false, queryType: string | null = null, threadId = activeThread;
      let investigationId: string | null = null;
      let cacheIsStale = false;
      let queryCategory: string | null = null;
      const progress: ProgressStep[] = [];
      let cacheHit: CacheHitInfo | null = null;
      // — backend can emit `render_component` SSE events for
      // tool outputs that should be rendered inline (e.g. Prometheus
      // timeseries → chart). We accumulate them and attach to the
      // assistant message so ChatMessage's registry can dispatch.
      const richComponents: RichComponent[] = [];

      const updateAssistant = () => {
        setMessages((p) => {
          const u = [...p];
          u[u.length - 1] = {
            ...u[u.length - 1],
            content: answer,
            progress: [...progress],
            cacheHit,
            streaming: true,
            richComponents: richComponents.length ? [...richComponents] : undefined,
          };
          return u;
        });
      };

      for await (const evt of streamQuery(query, USER_ID, activeThread ?? undefined)) {
        if (evt.event === "node_start") {
          const node = evt.data.node as string;
          const label = (evt.data.label as string) ?? node;
          progress.push({ node, label, status: "active", startedAt: Date.now() });
          updateAssistant();
        } else if (evt.event === "node_end") {
          const node = evt.data.node as string;
          // Mark the most-recent active step for this node as done
          for (let i = progress.length - 1; i >= 0; i--) {
            if (progress[i].node === node && progress[i].status === "active") {
              progress[i] = { ...progress[i], status: "done", endedAt: Date.now() };
              break;
            }
          }
          updateAssistant();
        } else if (evt.event === "reasoner_token") {
          // Append the LLM's streaming reasoning to the most-recent
          // reasoner step. Match by node === "reasoner" OR label
          // containing "reason" — works whether or not the step is
          // still active by the time the token arrives (avoids races
          // with the node_end event that LangGraph sometimes emits
          // before the final custom-event flush).
          const delta = (evt.data.delta as string) ?? "";
          if (delta) {
            for (let i = progress.length - 1; i >= 0; i--) {
              const step = progress[i];
              if (step.node === "reasoner" || /reason/i.test(step.label)) {
                progress[i] = {
                  ...step,
                  streamingText: (step.streamingText ?? "") + delta,
                };
                break;
              }
            }
            updateAssistant();
          }
        } else if (evt.event === "cache_hit") {
          cacheHit = {
            similarity: (evt.data.similarity as number) ?? 0,
            ageSeconds: (evt.data.age_seconds as number) ?? 0,
          };
          if (evt.data.is_stale) cacheIsStale = true;
          updateAssistant();
        } else if (evt.event === "chunk") {
          answer += (evt.data.text as string) ?? "";
          updateAssistant();
        } else if (evt.event === "render_component") {
          // Backend says "render this UI component inline" — push onto
          // the message's richComponents and re-render. Component
          // type→implementation mapping lives in ChatMessage; App.tsx
          // is intentionally type-agnostic so new components don't
          // need plumbing here.
          const type = evt.data.component as string | undefined;
          const props = (evt.data.props as Record<string, unknown> | undefined) ?? {};
          if (type) {
            richComponents.push({ type, props });
            updateAssistant();
          }
        } else if (evt.event === "done") {
          answer = (evt.data.answer as string) ?? answer;
          sources = (evt.data.sources as string[]) ?? [];
          sourceUrls = (evt.data.source_urls as (string | null)[]) ?? [];
          grounded = (evt.data.grounded as boolean) ?? false;
          queryType = (evt.data.query_type as string) ?? null;
          threadId = (evt.data.thread_id as string) ?? threadId;
          investigationId = (evt.data.investigation_id as string) ?? null;
          if (evt.data.cache_is_stale) cacheIsStale = true;
          queryCategory = (evt.data.query_category as string) ?? queryCategory;
        } else if (evt.event === "error") {
          answer = `Error: ${evt.data.detail ?? "Unknown"}`;
        }
      }

      // Finalize: any still-active steps roll over to done (defensive
      // against missed node_end events).
      const finalizedProgress = progress.map((s) =>
        s.status === "active" ? { ...s, status: "done" as const, endedAt: Date.now() } : s
      );
      setMessages((p) => {
        const u = [...p];
        u[u.length - 1] = {
          role: "assistant", content: answer, sources, sourceUrls, grounded, queryType,
          streaming: false, progress: finalizedProgress, cacheHit,
          investigationId, cacheIsStale, queryCategory,
          richComponents: richComponents.length ? [...richComponents] : undefined,
          ts: new Date().toISOString(),
        };
        return u;
      });
      if (threadId && threadId !== activeThread) setActiveThread(threadId);
      refreshSessions();
    } catch (err) {
      setMessages((p) => { const u = [...p]; u[u.length - 1] = { role: "assistant", content: `Connection error: ${err instanceof Error ? err.message : "Unknown"}`, streaming: false, ts: new Date().toISOString() }; return u; });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        page={page}
        onPageChange={(p) => setPage(p)}
        onBrandClick={handleGoHome}
        sessions={sessions}
        activeThread={activeThread}
        onSelect={async (id) => {
          setActiveThread(id);
          setPageState("chat");
          window.location.hash = `chat/${id}`;
          lastLoadedFromHashRef.current = id;
          await loadThreadMessages(id);
        }}
        onNew={() => { setActiveThread(null); setMessages([]); setPage("chat"); }}
        onDelete={async (id) => { await deleteSession(id); if (activeThread === id) { setActiveThread(null); setMessages([]); } refreshSessions(); }}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((c) => !c)}
        modelName={uiConfig?.model_name ?? MODEL_NAME}
        brandName={uiConfig?.brand_name ?? "OpsRAG"}
        brandSubtitle={uiConfig?.brand_subtitle ?? "DevOps Intelligence"}
        investigations={investigations}
        activeInvestigation={activeInvestigation}
        onSelectInvestigation={(id) => { setActiveInvestigation(id); setPage("investigate"); }}
        onNewInvestigation={() => { setActiveInvestigation(null); setPage("investigate"); }}
        me={me}
        scopes={me.scopes}
        investigationEnabled={uiConfig?.investigation_enabled ?? false}
        onSignOut={handleSignOut}
      />

      <main className="main">
        {page === "home" && (
          <HomePage
            me={me}
            sessions={sessions}
            brandName={uiConfig?.brand_name ?? "OpsRAG"}
            investigationEnabled={uiConfig?.investigation_enabled ?? false}
            onNavigate={(p) => setPage(p as Page)}
            onNewChat={handleNewConversation}
            onOpenChat={async (id) => {
              setActiveThread(id);
              setPageState("chat");
              window.location.hash = `chat/${id}`;
              lastLoadedFromHashRef.current = id;
              await loadThreadMessages(id);
            }}
          />
        )}

        {page === "chat" && (
          <section className="page">
            <div className="topbar">
              <h1>
                Conversations <span className="dim">· {
                  activeThread
                    ? (sessions.find((s) => s.thread_id === activeThread)?.title
                        ?? `conversation ${activeThread.slice(0, 8)}`)
                    : "new conversation"
                }</span>
              </h1>
              <div className="actions">
                {loading && <span className="hdr-badge">streaming</span>}
                <button className="topbar-btn primary" onClick={handleNewConversation} title="Start a new conversation">
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 1v10M1 6h10"/></svg>
                  New Conversation
                </button>
              </div>
            </div>
            <div className="md-body">
              <aside className="md-list">
                <ConversationListPane
                  sessions={sessions}
                  activeThread={activeThread}
                  onSelect={async (id) => {
                    setActiveThread(id);
                    setPageState("chat");
                    window.location.hash = `chat/${id}`;
                    lastLoadedFromHashRef.current = id;
                    await loadThreadMessages(id);
                  }}
                  onNew={handleNewConversation}
                  onDelete={async (id) => {
                    await deleteSession(id);
                    if (activeThread === id) { setActiveThread(null); setMessages([]); }
                    refreshSessions();
                  }}
                />
              </aside>
              <div className="md-detail">
            <div className="chat-wrap">
              <div className="chat-body" ref={chatBodyRef}>
                {messages.length === 0 ? (
                  <div className="chat-empty">
                    <div className="chat-empty-inner">
                      <div className="glyph">
                        <img src="/opsrag-logo.png" alt="OpsRAG" />
                      </div>
                      <h2>Ask <em>{uiConfig?.brand_name ?? "OpsRAG"}</em> anything.</h2>
                      <p>
                        Query your DevOps knowledge — runbooks, Terraform modules,
                        Helm charts, K8s manifests, incident postmortems. Every
                        answer cited, every source linked.
                      </p>
                      <div className="suggestions">
                        {SUGGESTED_PROMPTS.map((p) => (
                          <button
                            key={p.text}
                            className={`suggestion ${p.tone ?? ""}`}
                            onClick={() => handleSend(p.text)}
                          >
                            <div className="tag">{p.cat}</div>
                            <div className="text">{p.text}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="chat-messages">
                    {messages.map((msg, i) => {
                      // For assistant messages, surface the preceding user
                      // turn so the feedback widget can persist a query
                      // snippet alongside the rating. (T1.3 /.)
                      const preceding = i > 0 && messages[i - 1].role === "user" ? messages[i - 1].content : null;
                      return (
                        <ChatMessage
                          key={i}
                          msg={msg}
                          ctx={{
                            threadId: activeThread,
                            precedingUserQuery: preceding,
                            viewerEmail: me?.email ?? null,
                            viewerName: me?.name ?? null,
                          }}
                        />
                      );
                    })}
                    <div ref={endRef} />
                  </div>
                )}
              </div>
              <ChatInput onSend={handleSend} disabled={loading} />
            </div>
              </div>
            </div>
          </section>
        )}

        {page === "channels" && (() => {
          const active = activeChannelThread
            ? channelConversations.find((c) => c.thread_id === activeChannelThread) ?? null
            : null;
          const activePlatformLabel = channelPlatformLabel(active?.platform ?? null);
          return (
          <section className="page">
            <div className="topbar">
              <h1>
                Channels <span className="dim">· {
                  active
                    ? (active.title ?? `conversation ${active.thread_id.slice(0, 8)}`)
                    : "shared chat-channel conversations"
                }</span>
              </h1>
              <div className="actions">
                <span className="hdr-badge">read-only</span>
              </div>
            </div>
            <div className="md-body">
              <aside className="md-list">
                <ConversationListPane
                  sessions={channelConversations}
                  activeThread={activeChannelThread}
                  onSelect={async (id) => {
                    setActiveChannelThread(id);
                    await loadChannelMessages(id);
                  }}
                  // Read-only browse: no new / delete affordances. These are
                  // required props on the shared pane, so they're inert no-ops.
                  onNew={() => { /* read-only — no new */ }}
                  onDelete={() => { /* read-only — no delete */ }}
                  readOnly
                />
              </aside>
              <div className="md-detail">
                <div className="chat-wrap">
                  <div className="chat-body" ref={channelBodyRef}>
                    {!activeChannelThread ? (
                      <div className="chat-empty">
                        <div className="chat-empty-inner">
                          <div className="glyph">
                            <img src="/opsrag-logo.png" alt="OpsRAG" />
                          </div>
                          <h2>Browse <em>channel</em> conversations.</h2>
                          <p>
                            Read conversations that happened in shared
                            Slack / Discord / Telegram / Teams channels via the
                            chat bots. Private DMs and web threads stay private.
                            Select a conversation to read it — this view is
                            read-only.
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="chat-messages">
                        <div className="msg-meta-row" style={{ marginBottom: 8 }}>
                          <span className="badge badge-type">{activePlatformLabel}</span>
                        </div>
                        {channelMessages.map((msg, i) => (
                          <ChatMessage
                            key={i}
                            msg={msg}
                            // Read-only ctx: omit threadId so the feedback path
                            // can't render, and set readOnly to also drop Copy.
                            // Identity is platform-only — no viewer email/name.
                            ctx={{ readOnly: true }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </section>
          );
        })()}

        {page === "usage" && (
          <section className="page">
            <div className="topbar">
              <h1>Usage &amp; Cost <span className="dim">· live telemetry</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Live · auto-refresh · per-purpose cost tracking</div>
              <h2>Spend split by <em>purpose</em>, attributed by <em>user</em>.</h2>
              <p>
                Every LLM call is recorded with its purpose (indexing,
                generation, reranking, …) and the user who triggered it.
                Same telemetry powers the monthly cost rollup and the
                live activity feed.
              </p>
            </div>
            <UsagePage me={me} />
          </section>
        )}

        {page === "mcpaudit" && (
          <section className="page">
            <div className="topbar">
              <h1>MCP Audit <span className="dim">· centralized MCP governance</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Read-only · token-scoped · fully audited</div>
              <h2>Who called <em>which tool</em>, and <em>when</em>.</h2>
              <p>
                Every call through the centralized MCP is recorded — the user and
                token behind it, the tool, the latency, and the result. Arguments
                are never stored, only a sha256 hash, so secrets never reach the log.
              </p>
            </div>
            <MCPAuditPage me={me} />
          </section>
        )}

        {page === "connections" && (
          <section className="page">
            <div className="topbar">
              <h1>MCP Tokens <span className="dim">· personal tokens for MCP clients</span></h1>
              <div className="actions">
                <button
                  className="topbar-btn primary"
                  onClick={() => window.dispatchEvent(new Event("opsrag:open-mcp-create"))}
                  title="Generate a new MCP token"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 1v10M1 6h10"/></svg>
                  Generate new token
                </button>
              </div>
            </div>
            <div className="hero">
              <div className="eyebrow"><span className="dot"></span> Personal access tokens</div>
              <h2>Use OpsRAG inside <em>Claude Code</em>, <em>Cursor</em>, or any MCP-compatible tool.</h2>
              <p>
                Generate a personal token, drop it into your MCP client config,
                and query your DevOps knowledge from inside your editor. Treat
                tokens like passwords.
              </p>
            </div>
            <MCPTokensPage me={me} />
          </section>
        )}

        {page === "users" && (
          <section className="page">
            <div className="topbar">
              <h1>Users &amp; Roles <span className="dim">· access control</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow"><span className="dot"></span> Role-based access control</div>
              <h2>Decide who can <em>chat</em>, <em>investigate</em>, and <em>connect</em>.</h2>
              <p>
                Assign roles to each user — roles bundle the scopes the server
                enforces on every request. Changes take effect when the user's
                session next refreshes (≤15 min) or they sign in again.
              </p>
            </div>
            <UsersRolesPage me={me} />
          </section>
        )}

        {page === "indexing" && (
          <section className="page">
            <div className="topbar">
              <h1>Indexing Jobs <span className="dim">· run history</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Live · run history</div>
              <h2>Every ingestion run, <em>accounted</em> for.</h2>
              <p>
                When each job ran, how long it took, and whether it succeeded or
                failed — with the full error on failure. For the catalog of indexed
                sources, see <em>Sources</em>.
              </p>
            </div>
            <IndexingJobsPage />
          </section>
        )}

        {page === "guidance" && (
          <section className="page">
            <div className="topbar">
              <h1>Agent Guidance <span className="dim">· deployment-wide custom instructions</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow"><span className="dot"></span> Always-applied · answers + chat · live-editable</div>
              <h2>Teach OpsRAG your <em>edge cases</em>.</h2>
              <p>
                Free-text instructions the agent always honors — like a
                <code> CLAUDE.md</code> for your deployment. Org conventions,
                escalation / on-call policy, tone, "always check X" rules. Edits
                apply on the next query; no redeploy.
              </p>
            </div>
            <AgentGuidancePage />
          </section>
        )}

        {page === "cache" && (
          <section className="page">
            <div className="topbar">
              <h1>Cache Control <span className="dim">· Q&amp;A · investigation · tool-output</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow"><span className="dot"></span> Three cache layers · multi-strategy purge</div>
              <h2>Save <em>$</em> per cache hit. Lose <em>0</em> faithfulness.</h2>
              <p>
                Semantic-cache Q&amp;A at 0.93+ similarity threshold. Investigation
                cache for repeat alerts. Tool-output micro-cache for sub-call
                deduplication. Cloudflare-style multi-strategy purge.
              </p>
            </div>
            <CachePage />
          </section>
        )}

        {page === "investigate" && (
          <section className="page">
            <div className="topbar">
              <h1>Investigations <span className="dim">· agentic root-cause analysis</span></h1>
              <div className="actions">
                <button
                  className="topbar-btn primary"
                  onClick={() => setActiveInvestigation(null)}
                  title="Start a new investigation"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 1v10M1 6h10"/></svg>
                  New investigation
                </button>
              </div>
            </div>
            <div className="md-body">
              <aside className="md-list">
                <InvestigationListPane
                  investigations={investigations}
                  activeInvestigation={activeInvestigation}
                  onSelect={(id) => setActiveInvestigation(id)}
                  onNew={() => setActiveInvestigation(null)}
                />
              </aside>
              <div className="md-detail">
                <ErrorBoundary
                  label="investigation"
                  resetKey={activeInvestigation ?? "new"}
                >
                  <InvestigationPage
                    loadInvestigationId={activeInvestigation}
                    onInvestigationCompleted={(id) => {
                      setActiveInvestigation(id);
                      refreshInvestigations();
                    }}
                    onNewInvestigation={() => setActiveInvestigation(null)}
                  />
                </ErrorBoundary>
              </div>
            </div>
          </section>
        )}

        {page === "runbooks" && (
          <section className="page">
            <div className="topbar">
              <h1>Runbooks <span className="dim">· hand-authored playbooks</span></h1>
              <div className="actions">
                <button
                  className="topbar-btn primary"
                  onClick={() => { window.location.hash = "runbooks/new"; }}
                  title="Author a new runbook"
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M6 1v10M1 6h10"/></svg>
                  New runbook
                </button>
              </div>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Authoritative playbooks</div>
              <h2>Runbooks the agent <em>trusts first</em>.</h2>
              <p>
                Curated SRE playbooks that take priority over RAG-indexed docs during
                Investigate-mode root-cause analysis. Scope each to a service, failure
                class, and severity floor so the right one fires for the right incident.
              </p>
            </div>
            <RunbookListPage
              onEdit={(id) => {
                window.location.hash = id ? `runbooks/edit/${id}` : "runbooks/new";
              }}
            />
          </section>
        )}

        {page === "runbook-edit" && (
          <RunbookEditorPage
            runbookId={parseHash().runbookId}
            onSaved={() => { window.location.hash = "runbooks"; }}
            onCancel={() => { window.location.hash = "runbooks"; }}
          />
        )}

        {page === "sources" && (
          <section className="page">
            <div className="topbar">
              <h1>Sources <span className="dim">· indexed knowledge</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Vector store grounded in your own knowledge</div>
              <h2>Every answer cited back to a <em>source</em>.</h2>
              <p>
                Repositories, wikis, and runbooks ingested into the vector
                store. Each row shows what's indexed and how many chunks it
                produced. Trigger or re-run ingestion from <em>Indexing Jobs</em>.
              </p>
            </div>
            <IndexingPage />
          </section>
        )}

        {page === "graph" && (
          <section className="page">
            <div className="topbar">
              <h1>Knowledge Graph <span className="dim">· entity &amp; relationship structure</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow"><span className="dot"></span> Graph-augmented retrieval</div>
              <h2>Retrieval that knows how things <em>connect</em>.</h2>
              <p>
                Beyond vector similarity, OpsRAG links entities and relationships so
                the agent can traverse dependencies, ownership, and blast radius. The
                always-on <em>entity-graph</em> (services, repositories, environments)
                augments retrieval out of the box; point <code>config.knowledge_graph.provider</code>
                at Neo4j for a richer, queryable graph.
              </p>
            </div>
            <KnowledgeGraphPage />
          </section>
        )}

        {page === "integrations" && (
          <section className="page">
            <div className="topbar">
              <h1>Integrations <span className="dim">· MCP connectors</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Model Context Protocol · 14 connectors</div>
              <h2>Plug OpsRAG into your <em>whole</em> stack.</h2>
              <p>
                MCP integrations connect the agent to GitLab, Datadog, Cloudflare,
                Kubernetes, Elasticsearch and more. Enable them in config; each
                enabled integration is health-probed by <span className="mono">/readyz</span>.
              </p>
            </div>
            <IntegrationsPage />
          </section>
        )}

        {page === "quality" && (
          <section className="page">
            <div className="topbar">
              <h1>Retrieval Quality <span className="dim">· feedback &amp; corrections</span></h1>
            </div>
            <div className="hero">
              <div className="eyebrow green"><span className="dot"></span> Feedback-driven grounding</div>
              <h2>Turn corrections into better <em>retrieval</em>.</h2>
              <p>
                Thumbs feedback and user-submitted corrections show how well grounded
                answers land. Each correction is stored as a high-weight chunk that
                steers future retrieval toward the right answer.
              </p>
            </div>
            <RetrievalQualityPage me={me} />
          </section>
        )}

        {page === "docs" && (
          <section className="page">
            <div className="topbar">
              <h1>API Documentation <span className="dim">· OpenAPI v3</span></h1>
              <div className="actions">
                <span className="sub">/api/openapi.json</span>
              </div>
            </div>
            <div className="docs-page">
              <iframe
                className="docs-iframe"
                title="OpsRAG API Documentation"
                srcDoc={`<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
<title>OpsRAG API</title>
<style>body{margin:0;}</style>
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  window.ui = SwaggerUIBundle({
    url: '/api/openapi.json',
    dom_id: '#swagger-ui',
    deepLinking: true,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    layout: 'BaseLayout'
  });
</script>
</body></html>`}
              />
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
