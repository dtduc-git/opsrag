import { useState, useEffect, useCallback } from "react";
import {
  listMcpTokens, createMcpToken, revokeMcpToken,
  type MCPToken, type MCPTokenCreated, type MeResponse,
} from "../api";
import { IconKey, IconCopy, IconCheck, IconClose, IconTrash } from "./icons";

// Public MCP endpoint. Surfaced as a `const` so a follow-up change can
// pull it from /api/ui-config without re-wiring every reference.
// MCP server URL — uses the Streamable HTTP transport (single endpoint
// at `/api/mcp/messages`). The older HTTP+SSE transport's `/api/mcp/sse`
// path also works for raw curl but Claude Code's `sse` transport mode
// expects responses on the SSE channel (not inline in POST), which our
// server doesn't currently implement. Stick with `type: "http"`.
// MCP server URL for this deployment. Configurable at build time via
// VITE_MCP_SERVER_URL; defaults to the current origin's /api/mcp/messages
// so it works out of the box without baking in any deployment host.
const MCP_SERVER_URL =
  import.meta.env.VITE_MCP_SERVER_URL ||
  (typeof window !== "undefined"
    ? `${window.location.origin}/api/mcp/messages`
    : "/api/mcp/messages");

interface Props {
  me?: MeResponse | null;
}

const EXPIRY_OPTIONS: { label: string; days: number | null }[] = [
  { label: "Never", days: null },
  { label: "30 days", days: 30 },
  { label: "90 days", days: 90 },
  { label: "1 year", days: 365 },
];

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

function fmtRelativeOrDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const delta = Date.now() - d.getTime();
    const mins = Math.round(delta / 60_000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.round(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.round(hrs / 24);
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString();
  } catch {
    return iso;
  }
}

export default function MCPTokensPage({ me = null }: Props) {
  const [tokens, setTokens] = useState<MCPToken[] | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [showExplainer, setShowExplainer] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmRevokeId, setConfirmRevokeId] = useState<string | null>(null);

  const isAnonymous = !me || me.is_anonymous;

  const refresh = useCallback(async () => {
    if (isAnonymous) {
      setTokens([]);
      return;
    }
    try {
      const t = await listMcpTokens();
      setTokens(t);
      setError(null);
      // First-time UX: auto-open the setup guide when the user has no
      // tokens yet (they're about to set one up). Once they have a
      // token the guide stays collapsed so the list is front-and-centre.
      if (t.length === 0) setShowSetup(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setTokens([]);
    }
  }, [isAnonymous]);

  useEffect(() => { refresh(); }, [refresh]);

  const onRevoke = async (id: string) => {
    try {
      await revokeMcpToken(id);
      setConfirmRevokeId(null);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  // Topbar (App.tsx) owns the "Generate new token" button per the
  // premium redesign — it fires `opsrag:open-mcp-create` to open
  // this page's create modal without lifting state up.
  useEffect(() => {
    const handler = () => { if (!isAnonymous) setShowCreate(true); };
    window.addEventListener("opsrag:open-mcp-create", handler);
    return () => window.removeEventListener("opsrag:open-mcp-create", handler);
  }, [isAnonymous]);

  return (
    <div className="page">
      {isAnonymous && (
        <div className="card-section" style={{ padding: 22, fontSize: 13, color: "var(--text-2)" }}>
          <strong>Sign in</strong> to manage MCP tokens.
        </div>
      )}

      {error && (
        <div className="card-section" style={{ padding: 14, fontSize: 13, color: "var(--danger)" }}>
          {error}
        </div>
      )}

      {!isAnonymous && (
        <div className="card-section" style={{ padding: 18 }}>
          <button
            className="explainer-toggle"
            onClick={() => setShowSetup((s) => !s)}
            style={{ width: "100%", textAlign: "left", fontWeight: 600 }}
          >
            {showSetup ? "▼" : "▸"} Setup guide — wire OpsRAG MCP into Claude Code / Cursor
          </button>
          {showSetup && (
            <div className="explainer-body" style={{ marginTop: 10 }}>
              <p style={{ marginBottom: 6 }}>
                <strong>Prerequisites</strong>
              </p>
              <ul>
                <li>The MCP server must be reachable from your network. Depending on the deployment it may sit behind a VPN or a zero-trust proxy.</li>
                <li>You need a Claude Code, Cursor, Continue, or other MCP-compatible client installed.</li>
              </ul>

              <p style={{ marginTop: 14, marginBottom: 6 }}>
                <strong>Step 1 — Generate your token</strong>
              </p>
              <p style={{ margin: 0 }}>
                Click <em>Generate new token</em> above. Give it a name that identifies the
                machine/tool (e.g. <code>laptop-claude-code</code>). <strong>The plaintext is
                shown once</strong> — copy it immediately.
              </p>

              <p style={{ marginTop: 14, marginBottom: 6 }}>
                <strong>Step 2 — Drop the config into your client</strong>
              </p>
              <p style={{ margin: "0 0 6px" }}>Config file paths by client:</p>
              <ul>
                <li><strong>Claude Code</strong>: <code>~/.config/claude-code/mcp.json</code> (Linux/macOS) or <code>%APPDATA%\claude-code\mcp.json</code> (Windows)</li>
                <li><strong>Cursor</strong>: Settings → MCP → Add server, or edit <code>~/.cursor/mcp.json</code></li>
                <li><strong>Continue</strong>: <code>~/.continue/config.json</code> under <code>experimental.modelContextProtocolServers</code></li>
              </ul>
              <p style={{ margin: "10px 0 6px" }}>Paste this snippet into <code>mcpServers</code> (replace <code>opsrag_xxx</code> with your token):</p>
              <div className="token-readonly-block">
                <pre style={{ margin: 0, fontSize: 12, whiteSpace: "pre-wrap" }}>{`{
  "mcpServers": {
    "opsrag": {
      "type": "http",
      "url": "${MCP_SERVER_URL}",
      "headers": { "Authorization": "Bearer opsrag_xxx" }
    }
  }
}`}</pre>
              </div>
              <p style={{ margin: "10px 0 6px" }}>Or run this single command (Claude Code):</p>
              <div className="token-readonly-block">
                <pre style={{ margin: 0, fontSize: 12, whiteSpace: "pre-wrap" }}>{`claude mcp add --transport http opsrag ${MCP_SERVER_URL} \\
  --header "Authorization: Bearer opsrag_xxx" -s user`}</pre>
              </div>

              <p style={{ marginTop: 14, marginBottom: 6 }}>
                <strong>Step 3 — Restart your client</strong>
              </p>
              <p style={{ margin: 0 }}>
                Most clients hot-reload <code>mcp.json</code>; if not, fully quit and relaunch.
              </p>

              <p style={{ marginTop: 14, marginBottom: 6 }}>
                <strong>Step 4 — Verify</strong>
              </p>
              <p style={{ margin: "0 0 6px" }}>From your client, ask:</p>
              <ul>
                <li><em>"List the OpsRAG MCP tools available to you."</em> — should report ~39 read-only tools (runbook_list, runbook_load, rootly_get_alert, k8s_list_pods, prometheus_query, etc.).</li>
                <li><em>"Use OpsRAG to find the runbook for CloudSQL Long Running Transaction"</em> — should invoke <code>runbook_list</code> then <code>runbook_load</code> and quote the runbook verbatim.</li>
              </ul>
              <p style={{ margin: "10px 0 0" }}>
                Or test the connection directly via curl:
              </p>
              <div className="token-readonly-block">
                <pre style={{ margin: 0, fontSize: 12, whiteSpace: "pre-wrap" }}>{`curl -s -H "Authorization: Bearer opsrag_xxx" \\
     -H "Content-Type: application/json" \\
     -X POST \\
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \\
     ${MCP_SERVER_URL} | jq '.result.tools | length'
# Expect: 39  (number of tools currently exposed)`}</pre>
              </div>

              <p style={{ marginTop: 14, marginBottom: 6 }}>
                <strong>Troubleshooting</strong>
              </p>
              <ul>
                <li><strong>403 "authorized network"</strong> — your network is not allowed by the deployment's access policy; connect via the required VPN / zero-trust path and retry.</li>
                <li><strong>401 unauthorized</strong> — token is wrong, expired, or revoked. Generate a new one.</li>
                <li><strong>Connection refused / DNS error</strong> — the MCP server URL isn't reachable from your network; confirm the URL and any required VPN/proxy.</li>
                <li><strong>Client reports 0 tools</strong> — restart the client; check the config file's JSON is valid.</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {!isAnonymous && tokens !== null && tokens.length === 0 && (
        <div className="card-section" style={{ padding: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>
            No tokens yet.
          </div>
          <div style={{ fontSize: 13, color: "var(--text-2)", marginBottom: 16 }}>
            Generate your first one to get started.
          </div>
          <button
            className="explainer-toggle"
            onClick={() => setShowExplainer((s) => !s)}
          >
            {showExplainer ? "▼" : "▸"} What is an MCP token?
          </button>
          {showExplainer && (
            <div className="explainer-body">
              <p>
                An MCP token lets an external tool (Claude Code, Cursor, Continue,
                or any Model Context Protocol client) call OpsRAG on your behalf.
              </p>
              <ul>
                <li><strong>Stored hashed</strong> — the server only keeps a SHA-256
                  hash of your token. If the database leaks, your token can't be reused.</li>
                <li><strong>Revocable</strong> — you can delete a token at any time and
                  it stops working immediately. Lost a laptop? Revoke and regenerate.</li>
                <li><strong>Scoped to your identity</strong> — every call made with the
                  token is attributed to you in usage tracking. No shared accounts.</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {!isAnonymous && tokens !== null && tokens.length > 0 && (
        <div className="card-section">
          <div className="card-section-title">
            <span>Your tokens</span>
            <span style={{ color: "var(--text-3)", fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
              {tokens.length} active
            </span>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>Name</th>
                <th>Created</th>
                <th>Expires</th>
                <th>Last used</th>
                <th style={{ textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {tokens.map((t) => (
                <tr key={t.id}>
                  <td>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                      <IconKey /> {t.name}
                    </span>
                  </td>
                  <td style={{ fontSize: 12, color: "var(--text-2)" }}>{fmtDate(t.created_at)}</td>
                  <td style={{ fontSize: 12, color: "var(--text-2)" }}>
                    {t.expires_at ? fmtDate(t.expires_at) : <span style={{ color: "var(--text-3)" }}>Never</span>}
                  </td>
                  <td style={{ fontSize: 12, color: "var(--text-2)" }}>{fmtRelativeOrDate(t.last_used_at)}</td>
                  <td style={{ textAlign: "right" }}>
                    {confirmRevokeId === t.id ? (
                      <span style={{ display: "inline-flex", gap: 6 }}>
                        <button className="btn btn-danger" onClick={() => onRevoke(t.id)}>Confirm</button>
                        <button className="btn" onClick={() => setConfirmRevokeId(null)}>Cancel</button>
                      </span>
                    ) : (
                      <button
                        className="btn"
                        onClick={() => setConfirmRevokeId(t.id)}
                        title="Revoke this token (immediate)"
                      >
                        <IconTrash /> Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showCreate && (
        <CreateTokenModal
          onClose={() => { setShowCreate(false); refresh(); }}
        />
      )}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Create-token modal: two phases — form, then "save this secret" reveal.
// ──────────────────────────────────────────────────────────────────────

function CreateTokenModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [expiryIdx, setExpiryIdx] = useState(0); // index into EXPIRY_OPTIONS
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [created, setCreated] = useState<MCPTokenCreated | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const exp = EXPIRY_OPTIONS[expiryIdx]?.days ?? null;
      const result = await createMcpToken(name.trim().slice(0, 60), exp);
      setCreated(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="details-overlay" onClick={onClose}>
      <div
        className="token-modal"
        role="dialog"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="token-modal-header">
          <div className="token-modal-title">
            {created ? "Token created" : "Generate MCP token"}
          </div>
          <button className="details-close" onClick={onClose} aria-label="Close">
            <IconClose />
          </button>
        </div>

        {!created && (
          <form onSubmit={onSubmit} className="token-modal-body">
            <label className="token-modal-label" htmlFor="token-name-input">Name</label>
            <input
              id="token-name-input"
              className="token-modal-input"
              type="text"
              maxLength={60}
              required
              placeholder="e.g. claude-code-laptop"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
            />
            <div className="token-modal-hint">A short label so you remember which tool uses this. Max 60 chars.</div>

            <label className="token-modal-label" htmlFor="token-expiry-select" style={{ marginTop: 14 }}>Expiry</label>
            <select
              id="token-expiry-select"
              className="token-modal-input"
              value={expiryIdx}
              onChange={(e) => setExpiryIdx(Number(e.target.value))}
            >
              {EXPIRY_OPTIONS.map((o, i) => (
                <option key={o.label} value={i}>{o.label}</option>
              ))}
            </select>

            {error && <div className="token-modal-error">{error}</div>}

            <div className="token-modal-actions">
              <button type="button" className="btn" onClick={onClose} disabled={submitting}>Cancel</button>
              <button type="submit" className="btn-primary" disabled={submitting || !name.trim()}>
                {submitting ? "Creating…" : "Generate token"}
              </button>
            </div>
          </form>
        )}

        {created && (
          <TokenRevealView token={created} onDone={onClose} />
        )}
      </div>
    </div>
  );
}

function TokenRevealView({ token, onDone }: { token: MCPTokenCreated; onDone: () => void }) {
  const [copiedToken, setCopiedToken] = useState(false);
  const [copiedConfig, setCopiedConfig] = useState(false);

  const configSnippet = JSON.stringify(
    {
      mcpServers: {
        opsrag: {
          type: "http",
          url: MCP_SERVER_URL,
          headers: { Authorization: `Bearer ${token.token}` },
        },
      },
    },
    null,
    2,
  );

  // Fastest path for Claude Code users: a single-command setup that
  // does what the JSON config block does, without the user having to
  // find their mcp.json or relaunch the app.
  const cliCommand = `claude mcp add --transport http opsrag ${MCP_SERVER_URL} --header "Authorization: Bearer ${token.token}" -s user`;

  const [copiedCli, setCopiedCli] = useState(false);

  const doCopy = async (text: string, which: "token" | "config" | "cli") => {
    try {
      await navigator.clipboard.writeText(text);
      if (which === "token") {
        setCopiedToken(true);
        setTimeout(() => setCopiedToken(false), 1500);
      } else if (which === "config") {
        setCopiedConfig(true);
        setTimeout(() => setCopiedConfig(false), 1500);
      } else {
        setCopiedCli(true);
        setTimeout(() => setCopiedCli(false), 1500);
      }
    } catch {
      // Clipboard API can fail in non-secure contexts; user can still
      // select & copy manually.
    }
  };

  return (
    <div className="token-modal-body">
      <div className="token-warning">
        <strong>This token will not be shown again.</strong> Save it now.
      </div>

      <div className="token-modal-label">Your token ({token.name})</div>
      <div className="token-readonly">
        <code>{token.token}</code>
        <button
          type="button"
          className={`copy-btn ${copiedToken ? "copied" : ""}`}
          onClick={() => doCopy(token.token, "token")}
        >
          {copiedToken ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
        </button>
      </div>

      <div className="token-modal-label" style={{ marginTop: 18 }}>One-line setup (Claude Code)</div>
      <div className="token-modal-hint">
        Fastest path — paste this into your terminal. Works for Claude Code on any OS.
      </div>
      <div className="token-readonly token-readonly-block">
        <pre><code>{cliCommand}</code></pre>
        <button
          type="button"
          className={`copy-btn ${copiedCli ? "copied" : ""}`}
          onClick={() => doCopy(cliCommand, "cli")}
        >
          {copiedCli ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
        </button>
      </div>

      <div className="token-modal-label" style={{ marginTop: 18 }}>Or — manual MCP client config</div>
      <div className="token-modal-hint">
        For Cursor / Continue / other MCP clients — drop into your client's config
        (e.g. <code>~/.cursor/mcp.json</code>, <code>~/.continue/config.json</code>).
      </div>
      <div className="token-readonly token-readonly-block">
        <pre><code>{configSnippet}</code></pre>
        <button
          type="button"
          className={`copy-btn ${copiedConfig ? "copied" : ""}`}
          onClick={() => doCopy(configSnippet, "config")}
        >
          {copiedConfig ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
        </button>
      </div>

      <div className="token-modal-actions">
        <button type="button" className="btn-primary" onClick={onDone}>Done</button>
      </div>
    </div>
  );
}
