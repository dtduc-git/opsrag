import { useEffect, useMemo, useState } from "react";
import {
  fetchAdminUsers,
  fetchRoleCatalog,
  updateUserRoles,
  type AdminUser,
  type RoleInfo,
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

export default function UsersRolesPage({ me }: Props) {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [roles, setRoles] = useState<RoleInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Per-user editing state, keyed by user id.
  const [draft, setDraft] = useState<Record<string, string[]>>({});
  const [savingId, setSavingId] = useState<string | null>(null);
  const [rowMsg, setRowMsg] = useState<Record<string, { ok: boolean; text: string }>>({});

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [u, r] = await Promise.all([fetchAdminUsers(), fetchRoleCatalog()]);
      setUsers(u);
      setRoles(r);
      setDraft(Object.fromEntries(u.map((x) => [x.id, [...x.roles]])));
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

  if (loading) {
    return <div className="card-section" style={{ padding: 22, color: "var(--text-2)" }}>Loading users…</div>;
  }
  if (error) {
    return <div className="card-section" style={{ padding: 16, color: "var(--danger)" }}>{error}</div>;
  }

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
            return (
              <tr key={u.id}>
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
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
