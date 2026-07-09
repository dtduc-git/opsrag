import { Fragment, useEffect, useMemo, useState } from "react";
import {
  fetchAdminUsers,
  fetchRoleCatalog,
  fetchConnectorCatalog,
  updateUserRoles,
  updateUserConnectors,
  type AdminUser,
  type RoleInfo,
  type ConnectorInfo,
  type MeResponse,
} from "../api";

interface Props {
  me?: MeResponse | null;
}

// Order-insensitive role-set equality (draft vs saved).
function sameRoles(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const sb = new Set(b);
  return a.every((r) => sb.has(r));
}

// Tri-state a connector can be in for a given user.
type ConnState = "default" | "allow" | "deny";

interface ConnDraft {
  allow: string[];
  deny: string[];
}

function connStateOf(draft: ConnDraft, name: string): ConnState {
  if (draft.allow.includes(name)) return "allow";
  if (draft.deny.includes(name)) return "deny";
  return "default";
}

// A connector is never in both lists; selecting one clears the other.
function withConnState(draft: ConnDraft, name: string, next: ConnState): ConnDraft {
  const allow = draft.allow.filter((n) => n !== name);
  const deny = draft.deny.filter((n) => n !== name);
  if (next === "allow") allow.push(name);
  if (next === "deny") deny.push(name);
  return { allow, deny };
}

function sameConnectors(a: ConnDraft, b: ConnDraft): boolean {
  return sameRoles(a.allow, b.allow) && sameRoles(a.deny, b.deny);
}

export default function UsersRolesPage({ me }: Props) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [roles, setRoles] = useState<RoleInfo[]>([]);
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Per-user editing state, keyed by user id.
  const [draft, setDraft] = useState<Record<string, string[]>>({});
  const [savingId, setSavingId] = useState<string | null>(null);
  const [rowMsg, setRowMsg] = useState<Record<string, { ok: boolean; text: string }>>({});

  // Per-user connector overrides, keyed by user id.
  const [connDraft, setConnDraft] = useState<Record<string, ConnDraft>>({});
  const [connSavingId, setConnSavingId] = useState<string | null>(null);
  const [connRowMsg, setConnRowMsg] = useState<Record<string, { ok: boolean; text: string }>>({});
  // Connector editor is collapsed per user by default -- keeps the list scannable.
  const [connOpen, setConnOpen] = useState<Record<string, boolean>>({});
  const toggleConn = (id: string) =>
    setConnOpen((m) => ({ ...m, [id]: !m[id] }));

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [u, r, c] = await Promise.all([
        fetchAdminUsers(),
        fetchRoleCatalog(),
        fetchConnectorCatalog(),
      ]);
      setUsers(u);
      setRoles(r);
      setConnectors(c);
      setDraft(Object.fromEntries(u.map((x) => [x.id, [...x.roles]])));
      setConnDraft(
        Object.fromEntries(
          u.map((x) => [x.id, { allow: [...x.connectors_allow], deny: [...x.connectors_deny] }]),
        ),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load users.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // role -> scopes, from the catalog (for the live scope preview).
  const roleScopes = useMemo(() => {
    const m: Record<string, string[]> = {};
    roles.forEach((r) => { m[r.role] = r.scopes; });
    return m;
  }, [roles]);

  const previewScopes = (sel: string[]): string[] => {
    const s = new Set<string>();
    sel.forEach((r) => (roleScopes[r] || []).forEach((sc) => s.add(sc)));
    return Array.from(s).sort();
  };

  const toggle = (userId: string, role: string) => {
    setRowMsg((m) => ({ ...m, [userId]: undefined as never }));
    setDraft((d) => {
      const cur = d[userId] || [];
      const next = cur.includes(role) ? cur.filter((r) => r !== role) : [...cur, role];
      return { ...d, [userId]: next };
    });
  };

  const save = async (user: AdminUser) => {
    const sel = draft[user.id] || [];
    setSavingId(user.id);
    setRowMsg((m) => ({ ...m, [user.id]: undefined as never }));
    try {
      const updated = await updateUserRoles(user.id, sel);
      setUsers((us) => us.map((u) => (u.id === user.id ? updated : u)));
      setDraft((d) => ({ ...d, [user.id]: [...updated.roles] }));
      // Roles changes recompute effective connectors server-side; resync the draft.
      setConnDraft((d) => ({
        ...d,
        [user.id]: { allow: [...updated.connectors_allow], deny: [...updated.connectors_deny] },
      }));
      setRowMsg((m) => ({ ...m, [user.id]: { ok: true, text: "Saved · they re-auth to apply" } }));
    } catch (e) {
      setRowMsg((m) => ({
        ...m,
        [user.id]: { ok: false, text: e instanceof Error ? e.message : "Save failed" },
      }));
    } finally {
      setSavingId(null);
    }
  };

  const reset = (user: AdminUser) => {
    setDraft((d) => ({ ...d, [user.id]: [...user.roles] }));
    setRowMsg((m) => ({ ...m, [user.id]: undefined as never }));
  };

  const setConnector = (userId: string, name: string, next: ConnState) => {
    setConnRowMsg((m) => ({ ...m, [userId]: undefined as never }));
    setConnDraft((d) => {
      const cur = d[userId] || { allow: [], deny: [] };
      return { ...d, [userId]: withConnState(cur, name, next) };
    });
  };

  const saveConnectors = async (user: AdminUser) => {
    const cur = connDraft[user.id] || { allow: [], deny: [] };
    setConnSavingId(user.id);
    setConnRowMsg((m) => ({ ...m, [user.id]: undefined as never }));
    try {
      const updated = await updateUserConnectors(user.id, cur.allow, cur.deny);
      setUsers((us) => us.map((u) => (u.id === user.id ? updated : u)));
      setConnDraft((d) => ({
        ...d,
        [user.id]: { allow: [...updated.connectors_allow], deny: [...updated.connectors_deny] },
      }));
      setConnRowMsg((m) => ({ ...m, [user.id]: { ok: true, text: "Saved · they re-auth to apply" } }));
    } catch (e) {
      setConnRowMsg((m) => ({
        ...m,
        [user.id]: { ok: false, text: e instanceof Error ? e.message : "Save failed" },
      }));
    } finally {
      setConnSavingId(null);
    }
  };

  const resetConnectors = (user: AdminUser) => {
    setConnDraft((d) => ({
      ...d,
      [user.id]: { allow: [...user.connectors_allow], deny: [...user.connectors_deny] },
    }));
    setConnRowMsg((m) => ({ ...m, [user.id]: undefined as never }));
  };

  if (loading) {
    return <div className="card-section" style={{ padding: 22, color: "var(--text-2)" }}>Loading users…</div>;
  }
  if (error) {
    return <div className="card-section" style={{ padding: 16, color: "var(--danger)" }}>{error}</div>;
  }

  const TRI: { value: ConnState; label: string }[] = [
    { value: "default", label: "Default" },
    { value: "allow", label: "Allow" },
    { value: "deny", label: "Deny" },
  ];

  return (
    <div className="card-section ur-card">
      <div className="card-section-title ur-head">
        <span>Users</span>
        <span className="dim">{users.length} total</span>
      </div>

      <table className="ur-table">
        <thead>
          <tr>
            <th>User</th>
            <th>Auth</th>
            <th>Roles</th>
            <th>Effective scopes</th>
            <th style={{ textAlign: "right" }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {users.map((u) => {
            const sel = draft[u.id] || [];
            const dirty = !sameRoles(sel, u.roles);
            const isYou = !!me?.oid && me.oid === u.id;
            const msg = rowMsg[u.id];

            const cur = connDraft[u.id] || { allow: [], deny: [] };
            const connDirty = !sameConnectors(cur, {
              allow: u.connectors_allow,
              deny: u.connectors_deny,
            });
            const connMsg = connRowMsg[u.id];
            const isConnOpen = !!connOpen[u.id];

            return (
              <Fragment key={u.id}>
                <tr className="ur-main-row">
                  <td>
                    <div className="ur-user">
                      <span className="ur-email">{u.email}</span>
                      {isYou && <span className="ur-you">you</span>}
                    </div>
                    {u.name && u.name !== u.email && <div className="ur-name">{u.name}</div>}
                  </td>
                  <td>
                    <span className="ur-auth">{u.has_password ? "password" : "SSO"}</span>
                  </td>
                  <td>
                    <div className="ur-roles">
                      {roles.map((r) => {
                        const on = sel.includes(r.role);
                        // Lockout guard mirror: you can't drop your own admin.
                        const locked = isYou && r.role === "admin" && on;
                        return (
                          <button
                            key={r.role}
                            type="button"
                            className={`ur-chip${on ? " on" : ""}`}
                            disabled={locked || savingId === u.id}
                            title={locked ? "You can't remove your own admin role" : r.description}
                            onClick={() => toggle(u.id, r.role)}
                          >
                            {r.label}
                          </button>
                        );
                      })}
                    </div>
                  </td>
                  <td>
                    <div className="ur-scopes">
                      {previewScopes(sel).map((s) => (
                        <span key={s} className="ur-scope">{s}</span>
                      ))}
                      {sel.length === 0 && <span className="dim" style={{ fontSize: 12 }}>none</span>}
                    </div>
                  </td>
                  <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    {msg && (
                      <span className={`ur-msg${msg.ok ? " ok" : " err"}`}>{msg.text}</span>
                    )}
                    <button
                      className="btn secondary ur-btn"
                      disabled={!dirty || savingId === u.id}
                      onClick={() => reset(u)}
                    >
                      Reset
                    </button>
                    <button
                      className="btn btn-primary ur-btn"
                      disabled={!dirty || savingId === u.id}
                      onClick={() => save(u)}
                    >
                      {savingId === u.id ? "Saving…" : "Save"}
                    </button>
                  </td>
                </tr>

                <tr className="ur-conn-row">
                  <td colSpan={5}>
                    <div className="ur-conn">
                      {/* Collapsed summary: connector count + the effective set,
                          with a disclosure that reveals the tri-state editor. */}
                      <button
                        type="button"
                        className="ur-conn-summary"
                        aria-expanded={isConnOpen}
                        onClick={() => toggleConn(u.id)}
                      >
                        <span className={`ur-conn-caret${isConnOpen ? " open" : ""}`} aria-hidden>▸</span>
                        <span className="ur-conn-title">Connector access</span>
                        <span className="ur-conn-summary-chips">
                          {u.effective_connectors.length === 0 ? (
                            <span className="dim" style={{ fontSize: 12 }}>none allowed</span>
                          ) : (
                            u.effective_connectors.map((n) => (
                              <span key={n} className="ur-scope">{n}</span>
                            ))
                          )}
                        </span>
                        {(cur.allow.length > 0 || cur.deny.length > 0) && (
                          <span className="ur-conn-override" title="This user has per-user connector overrides">
                            {cur.allow.length + cur.deny.length} override{cur.allow.length + cur.deny.length === 1 ? "" : "s"}
                          </span>
                        )}
                        <span className="ur-conn-summary-cta">{isConnOpen ? "Close" : "Manage"}</span>
                      </button>

                      {isConnOpen && (
                        <div className="ur-conn-editor">
                          <div className="ur-conn-head">
                            <span className="dim" style={{ fontSize: 12 }}>
                              Allow grants a connector to this user; Deny blocks it (Deny wins over roles and admin). Default follows their roles.
                            </span>
                            {connMsg && (
                              <span className={`ur-msg${connMsg.ok ? " ok" : " err"}`}>{connMsg.text}</span>
                            )}
                            <span className="ur-conn-actions">
                              <button
                                className="btn secondary ur-btn"
                                disabled={!connDirty || connSavingId === u.id}
                                onClick={() => resetConnectors(u)}
                              >
                                Reset
                              </button>
                              <button
                                className="btn btn-primary ur-btn"
                                disabled={!connDirty || connSavingId === u.id}
                                onClick={() => saveConnectors(u)}
                              >
                                {connSavingId === u.id ? "Saving…" : "Save"}
                              </button>
                            </span>
                          </div>

                          {connectors.length === 0 ? (
                            <span className="dim" style={{ fontSize: 12 }}>No connectors enabled on this deployment.</span>
                          ) : (
                            <div className="ur-conn-grid">
                              {connectors.map((c) => {
                                const state = connStateOf(cur, c.name);
                                return (
                                  <div key={c.name} className="ur-conn-item">
                                    <div className="ur-conn-label">
                                      <span className="ur-conn-name">{c.label}</span>
                                      {c.restricted && (
                                        <span
                                          className="ur-conn-restricted"
                                          title="Restricted: off by default, needs an explicit grant"
                                        >
                                          restricted
                                        </span>
                                      )}
                                    </div>
                                    <div
                                      className="ur-tri"
                                      role="radiogroup"
                                      aria-label={`Access for ${c.label}`}
                                    >
                                      {TRI.map((t) => (
                                        <button
                                          key={t.value}
                                          type="button"
                                          role="radio"
                                          aria-checked={state === t.value}
                                          className={`ur-tri-btn${state === t.value ? ` on ${t.value}` : ""}`}
                                          disabled={connSavingId === u.id}
                                          onClick={() => setConnector(u.id, c.name, t.value)}
                                        >
                                          {t.label}
                                        </button>
                                      ))}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </td>
                </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
