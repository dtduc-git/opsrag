import type { ReactNode } from "react";
import type { Session, InvestigationHistoryItem, MeResponse, Scope } from "../api";

// Command-center navigation model. The old single all-in-one "Workspace"
// group is replaced by a categorized sidebar: a standalone Home entry on
// top, then labeled sections (ASK / WORKFLOWS / KNOWLEDGE / OPERATIONS /
// INSIGHTS). Layout follows the approved design-5 mockup; the theme is
// unchanged.
export type Page =
  | "home"
  | "chat"
  | "investigate"
  | "runbooks"
  | "runbook-edit"
  | "sources"
  | "graph"
  | "integrations"
  | "indexing"
  | "connections"
  | "cache"
  | "usage"
  | "quality"
  | "mcpaudit"
  | "users"
  | "guidance"
  | "docs";

interface Props {
  page: Page;
  onPageChange: (p: Page) => void;
  sessions: Session[];
  activeThread: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  modelName: string;
  brandName?: string;
  brandSubtitle?: string;
  investigations?: InvestigationHistoryItem[];
  activeInvestigation?: string | null;
  onSelectInvestigation?: (id: string) => void;
  onNewInvestigation?: () => void;
  me?: MeResponse | null;
  // Effective scopes for the current viewer (from /me). Drives nav gating:
  // items list a `requires` scope and are hidden when it's absent. In OPEN
  // mode the backend grants every scope so nothing is hidden.
  scopes?: Scope[];
  // Config-driven feature gate from /ui-config: hides the Investigate tab
  // unless the deployment enabled a live-telemetry MCP integration. Separate
  // from the scope gate (authz) -- this is "is the feature wired at all".
  investigationEnabled?: boolean;
  // POSTs /api/auth/logout then re-gates. When provided (and authenticated),
  // an in-app Sign-out control is shown in the footer. Replaces the old
  // build-time VITE_SIGN_OUT_URL href flow.
  onSignOut?: () => void;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
  // Click the brand block (logo + name) to return to the Home dashboard.
  onBrandClick?: () => void;
}

// Sign-out URL for the deployment's SSO/identity proxy, supplied at build
// time via VITE_SIGN_OUT_URL. Unset by default -> no sign-out CTA (the
// local demo runs without SSO). No deployment host is baked in.
const SIGN_OUT_URL = import.meta.env.VITE_SIGN_OUT_URL ?? "";

function initialOf(s: string | null | undefined): string {
  const t = (s ?? "").trim();
  if (!t) return "?";
  return t[0].toUpperCase();
}

// Inline SVG icons (kept local so the sidebar doesn't depend on the legacy
// icons.tsx). 16x16, 1.5 stroke, currentColor — matches the existing set.
const I = {
  home: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"><path d="M2.5 7L8 2.5 13.5 7M4 6v7h8V6"/></svg>,
  newchat: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M14 8a5 5 0 0 1-5 5l-3 2v-2a5 5 0 1 1 8-5z"/></svg>,
  conversations: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M2 4h12M2 8h12M2 12h8"/></svg>,
  investigate: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 2.5a1.5 1.5 0 0 0-1.5 1.5c0 5 4 9 9 9A1.5 1.5 0 0 0 13 11.5l-2-1.3-1.5 1.3a8 8 0 0 1-3-3l1.3-1.5L6.5 4.5A1.5 1.5 0 0 0 5 3z"/></svg>,
  runbooks: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"><path d="M4 1h6l3 3v11H4z"/><path d="M10 1v3h3"/></svg>,
  sources: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><ellipse cx="8" cy="4" rx="6" ry="2"/><path d="M2 4v8c0 1.1 2.7 2 6 2s6-.9 6-2V4"/><path d="M2 8c0 1.1 2.7 2 6 2s6-.9 6-2"/></svg>,
  graph: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="3.5" cy="3.5" r="1.8"/><circle cx="12.5" cy="4" r="1.8"/><circle cx="8" cy="12.5" r="1.8"/><path d="M5 4.5l6 0M4.3 5l3 6M11.5 5.5l-3 5.5"/></svg>,
  integrations: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"><path d="M8 1.5l5.5 2.2v3.6c0 3.3-2.3 5.6-5.5 6.7-3.2-1.1-5.5-3.4-5.5-6.7V3.7z"/></svg>,
  indexing: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="3" width="12" height="2.5" rx="0.5"/><rect x="2" y="7" width="12" height="2.5" rx="0.5"/><rect x="2" y="11" width="12" height="2.5" rx="0.5"/></svg>,
  connections: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M6 10l4-4M5.5 8L4 9.5a2.1 2.1 0 1 0 3 3L8.5 11M10.5 8L12 6.5a2.1 2.1 0 1 0-3-3L7.5 5"/></svg>,
  mcptokens: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="5" cy="8" r="3"/><path d="M8 8h6M12 8v2.5M14 8v2"/></svg>,
  users: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="6" cy="5" r="2.3"/><path d="M2 13c0-2.2 1.8-3.6 4-3.6s4 1.4 4 3.6"/><path d="M11 3.2a2.3 2.3 0 0 1 0 4.4M14 13c0-1.9-1.2-3.1-2.8-3.5"/></svg>,
  cache: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 4l6-2.5L14 4l-6 2.5z"/><path d="M2 4v5l6 2.5L14 9V4"/><path d="M2 9v3l6 2.5L14 12V9"/></svg>,
  usage: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M2 14V2M2 14h12M5 11l3-4 2 2 3-5"/></svg>,
  quality: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M3 14V8M7 14V4M11 14v-4"/><path d="M2 14h12"/></svg>,
  audit: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round"><path d="M8 1.5l5.5 2.2v3.6c0 3.3-2.3 5.6-5.5 6.7-3.2-1.1-5.5-3.4-5.5-6.7V3.7z"/><path d="M5.8 8l1.6 1.6L10.5 6.3"/></svg>,
  docs: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 2h7l3 3v9H3z"/><path d="M6 7h5M6 9.5h5M6 12h3"/></svg>,
  guidance: <svg className="ic" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M3 2h7l3 3v9H3z"/><path d="M6 6.5h4M6 9h4"/><path d="M11.5 12.2l1 1 2-2.2"/></svg>,
  plus: <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"><path d="M6 1v10M1 6h10"/></svg>,
  collapse: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M10 4l-4 4 4 4"/></svg>,
  expand: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M6 4l4 4-4 4"/></svg>,
  signout: <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M10 4V2H2v12h8v-2 M6 8h9 M12 5l3 3-3 3"/></svg>,
  close: <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 3l6 6 M9 3l-6 6"/></svg>,
};

interface NavItem {
  page: Page;
  label: string;
  icon: ReactNode;
  title?: string;
  // Optional numeric badge (e.g. session/investigation counts).
  badge?: number;
  // When set, clicking fires this instead of plain navigation (New Chat).
  action?: () => void;
  // Extra pages that should also light this item as active.
  alsoActiveOn?: Page[];
  // Scope required to SEE this item. Omitted = always visible. Hiding is
  // UX-only; the backend's require_scope stays authoritative.
  requires?: Scope;
  // When explicitly false, the item is hidden because the deployment hasn't
  // enabled the backing integration (config-driven feature gate, distinct
  // from the scope/authz gate). Omitted/true = not feature-gated.
  featureEnabled?: boolean;
}

interface NavSection {
  label: string;
  items: NavItem[];
}

export default function Sidebar({
  page, onPageChange, sessions, modelName,
  brandName = "OpsRAG", brandSubtitle = "GraphRAG for SRE",
  investigations = [],
  me = null,
  scopes,
  investigationEnabled = false,
  onSignOut,
  collapsed = false,
  onToggleCollapsed,
  onBrandClick,
}: Props) {
  // In-app sign-out when the gate gave us a handler and the viewer is a real
  // (non-anonymous) identity. The legacy VITE_SIGN_OUT_URL <a href> is kept
  // as a fallback only when no onSignOut handler is wired.
  const isAuthed = !!me && !!me.email && !me.is_anonymous;
  const showSignoutButton = isAuthed && !!onSignOut;
  const showSignoutLink = isAuthed && !onSignOut && SIGN_OUT_URL !== "";

  // Scope gate. When `scopes` is undefined we fail OPEN (show everything) so
  // the gate degrades safely if identity didn't carry scopes. An item with
  // no `requires` is always visible.
  const hasScope = (s?: Scope): boolean => {
    if (!s) return true;
    if (!scopes) return true;
    return scopes.includes(s);
  };

  // Categorized navigation. Each item declares the scope required to SEE it:
  //   Conversations → chat · Investigations → investigate
  //   Operations (Integrations/Indexing/Connections/Cache) → admin
  //   Insights (Usage/Retrieval Quality) → admin
  // Runbooks/Sources/Graph stay ungated (read-only knowledge). Hiding is
  // UX-only; require_scope on the backend remains authoritative.
  const allSections: NavSection[] = [
    {
      label: "Workspace",
      items: [
        { page: "chat", label: "Conversations", icon: I.conversations, title: "Your conversations", badge: sessions.length || undefined, requires: "chat" },
        { page: "investigate", label: "Investigations", icon: I.investigate, title: "Agentic root-cause analysis", badge: investigations.length || undefined, requires: "investigate", featureEnabled: investigationEnabled },
        { page: "runbooks", label: "Runbooks", icon: I.runbooks, title: "Hand-authored SRE playbooks", alsoActiveOn: ["runbook-edit"] },
        { page: "connections", label: "MCP Tokens", icon: I.mcptokens, title: "Personal access tokens for MCP clients (Claude Code, Cursor)", requires: "mcp" },
      ],
    },
    {
      label: "Knowledge",
      items: [
        { page: "sources", label: "Sources", icon: I.sources, title: "Indexed knowledge sources" },
        // Knowledge Graph tab disabled per product decision (2026-06-01): the
        // shallow entity-graph wasn't earning its place in the sidebar. The
        // page + /graph API remain in the codebase, just not navigable.
      ],
    },
    {
      label: "Operations",
      items: [
        { page: "integrations", label: "Integrations", icon: I.integrations, title: "MCP integrations", requires: "admin" },
        { page: "indexing", label: "Indexing Jobs", icon: I.indexing, title: "Repository ingestion jobs", requires: "admin" },
        { page: "users", label: "Users & Roles", icon: I.users, title: "Manage users and their roles (RBAC)", requires: "admin" },
        { page: "guidance", label: "Agent Guidance", icon: I.guidance, title: "Deployment-wide custom instructions (always-applied to answers + chat)", requires: "admin" },
        { page: "cache", label: "Cache", icon: I.cache, title: "Q&A / investigation / tool-output caches", requires: "admin" },
      ],
    },
    {
      label: "Insights",
      items: [
        { page: "usage", label: "Usage & Cost", icon: I.usage, title: "Your LLM usage & cost (admins also see org-wide)", requires: "chat" },
        { page: "quality", label: "Retrieval Quality", icon: I.quality, title: "Feedback and corrections", requires: "admin" },
        { page: "mcpaudit", label: "MCP Audit", icon: I.audit, title: "Centralized MCP tool-call audit log (who called which read-only tool, when)", requires: "admin" },
      ],
    },
  ];

  // Filter items the viewer lacks scope for OR whose backing feature the
  // deployment hasn't enabled, then drop any section left empty.
  const sections: NavSection[] = allSections
    .map((sec) => ({
      ...sec,
      items: sec.items.filter((it) => hasScope(it.requires) && it.featureEnabled !== false),
    }))
    .filter((sec) => sec.items.length > 0);

  const isActive = (it: NavItem): boolean => {
    if (it.label === "New Chat") return false; // New Chat is an action, never the active row
    if (page === it.page) return true;
    return !!it.alsoActiveOn?.includes(page);
  };

  const handleClick = (it: NavItem) => {
    if (it.action) { it.action(); return; }
    onPageChange(it.page);
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <button
          type="button"
          className="brand-home-btn"
          onClick={onBrandClick}
          disabled={!onBrandClick}
          aria-label={`${brandName} — go to home`}
          title="Go to home"
        >
          <div className="brand-mark">
            <img src="/opsrag-logo.png" alt="OpsRAG logo" />
          </div>
          <div className="brand-text collapse-only-hide">
            <div className="brand-name">{brandName}</div>
            <div className="brand-sub">{brandSubtitle}</div>
          </div>
        </button>
        {onToggleCollapsed && (
          <button
            className="brand-collapse-btn"
            onClick={onToggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? I.expand : I.collapse}
          </button>
        )}
      </div>

      <nav className="sidebar-nav">
        {/* Standalone Home entry on top (matches the design-5 mockup). */}
        <button
          type="button"
          className={`nav-item nav-home ${page === "home" ? "active" : ""}`}
          onClick={() => onPageChange("home")}
          title="Home dashboard"
        >
          {I.home}<span className="collapse-only-hide">Home</span>
        </button>

        {sections.map((sec) => (
          <div className="nav-group" key={sec.label}>
            <div className="nav-label">{sec.label}</div>
            {sec.items.map((it) => (
              <button
                type="button"
                key={`${sec.label}:${it.label}`}
                className={`nav-item ${isActive(it) ? "active" : ""}`}
                onClick={() => handleClick(it)}
                title={it.title ?? it.label}
              >
                {it.icon}
                <span className="collapse-only-hide">{it.label}</span>
                {it.badge ? <span className="nav-badge collapse-only-hide">{it.badge}</span> : null}
              </button>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-spacer" />

      {/* Footer card — identity + live model + API docs + (in prod) sign-out. */}
      <div className="sidebar-foot">
        <div className="user">
          <div className="avatar">{initialOf(me?.name || me?.email || "?")}</div>
          <div className="uinfo collapse-only-hide">
            <div className="uname">{me?.name || (me?.email ? me.email.split("@")[0] : "Anonymous")}</div>
            <div className="umail">{me?.email || "anonymous mode"}</div>
          </div>
        </div>
        <div className="health collapse-only-hide">
          <span title="Active model">{modelName}</span>
          <span className="ok">Live</span>
        </div>
        <button
          type="button"
          className={`sidebar-docs-link collapse-only-hide ${page === "docs" ? "active" : ""}`}
          onClick={() => onPageChange("docs")}
          title="API documentation"
        >
          {I.docs}<span>API Docs</span>
        </button>
        {showSignoutButton && (
          <button
            type="button"
            className="signout collapse-only-hide"
            onClick={onSignOut}
            title="Sign out"
          >
            {I.signout}<span>Sign out</span>
          </button>
        )}
        {showSignoutLink && (
          <a className="signout collapse-only-hide" href={SIGN_OUT_URL} title="Sign out">
            {I.signout}<span>Sign out</span>
          </a>
        )}
      </div>
    </aside>
  );
}
