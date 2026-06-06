import { useEffect, useState } from "react";
import {
  login,
  ssoLoginUrl,
  fetchAuthProviders,
  type SSOProvider,
} from "../api";

interface Props {
  brandName?: string;
  brandSubtitle?: string;
  // Called after a successful password login so the shell can re-fetch /me
  // and drop the gate. SSO logins navigate away (top-level redirect), so they
  // come back through a fresh page load — no callback needed there.
  onLoggedIn?: () => void;
}

// Provider display metadata. We only show buttons for providers the backend
// advertises (best-effort) so the form never offers a dead SSO option.
const PROVIDER_META: Record<SSOProvider, { label: string; mark: React.ReactNode }> = {
  google: {
    label: "Continue with Google",
    mark: (
      <svg width="16" height="16" viewBox="0 0 18 18" aria-hidden>
        <path fill="#4285F4" d="M17.6 9.2c0-.6-.05-1.18-.16-1.74H9v3.3h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.9c1.7-1.57 2.66-3.88 2.66-6.54z" />
        <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.9-2.26c-.8.54-1.84.86-3.06.86-2.35 0-4.34-1.59-5.05-3.72H.94v2.33A9 9 0 0 0 9 18z" />
        <path fill="#FBBC05" d="M3.95 10.7a5.4 5.4 0 0 1 0-3.42V4.96H.94a9 9 0 0 0 0 8.08l3.01-2.34z" />
        <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58A9 9 0 0 0 .94 4.96l3.01 2.33C4.66 5.16 6.65 3.58 9 3.58z" />
      </svg>
    ),
  },
  microsoft: {
    label: "Continue with Microsoft",
    mark: (
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
        <path fill="#F25022" d="M1 1h6.5v6.5H1z" />
        <path fill="#7FBA00" d="M8.5 1H15v6.5H8.5z" />
        <path fill="#00A4EF" d="M1 8.5h6.5V15H1z" />
        <path fill="#FFB900" d="M8.5 8.5H15V15H8.5z" />
      </svg>
    ),
  },
  github: {
    label: "Continue with GitHub",
    mark: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.32c-2.23.48-2.7-1.07-2.7-1.07-.36-.93-.89-1.18-.89-1.18-.73-.5.05-.49.05-.49.8.06 1.23.83 1.23.83.72 1.23 1.88.87 2.34.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.83-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.22 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.52.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48v2.2c0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
      </svg>
    ),
  },
};

const PROVIDER_ORDER: SSOProvider[] = ["google", "microsoft", "github"];

export default function LoginPage({
  brandName = "OpsRAG",
  brandSubtitle = "DevOps Intelligence",
  onLoggedIn,
}: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Available login methods (password / SSO / both), from the backend so the
  // page shows exactly what's enabled -- supports easy mode switching.
  const [providers, setProviders] = useState<SSOProvider[]>([]);
  const [passwordEnabled, setPasswordEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;
    fetchAuthProviders()
      .then((m) => { if (!cancelled) { setProviders(m.providers); setPasswordEnabled(m.passwordEnabled); } })
      .catch(() => { if (!cancelled) { setProviders([]); setPasswordEnabled(true); } });
    return () => { cancelled = true; };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    const res = await login(username.trim(), password);
    setSubmitting(false);
    if (res.ok) {
      onLoggedIn?.();
    } else {
      setError(res.detail || "Sign-in failed. Check your credentials.");
    }
  };

  const orderedProviders = PROVIDER_ORDER.filter((p) => providers.includes(p));

  return (
    <div className="login-shell">
      <div className="login-card">
        <div className="login-brand">
          <div className="login-mark">
            <img src="/opsrag-logo.png" alt={`${brandName} logo`} />
          </div>
          <div className="login-brand-text">
            <div className="login-brand-name">{brandName}</div>
            <div className="login-brand-sub">{brandSubtitle}</div>
          </div>
        </div>

        <h1 className="login-headline">Sign in to <em>{brandName}</em>.</h1>
        <p className="login-lede">
          Query your DevOps knowledge — runbooks, Terraform, Helm, K8s, and
          incident postmortems. Every answer cited.
        </p>

        {passwordEnabled && (
          <form className="login-form" onSubmit={handleSubmit}>
            <label className="login-field">
              <span>Username or email</span>
              <input
                type="text"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="you@example.com"
                disabled={submitting}
                autoFocus
              />
            </label>
            <label className="login-field">
              <span>Password</span>
              <input
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                disabled={submitting}
              />
            </label>

            {error && <div className="login-error" role="alert">{error}</div>}

            <button
              type="submit"
              className="btn btn-primary login-submit"
              disabled={submitting || !username.trim() || !password}
            >
              {submitting ? "Signing in…" : "Sign in"}
            </button>
          </form>
        )}

        {orderedProviders.length > 0 && (
          <>
            {passwordEnabled && <div className="login-divider"><span>or</span></div>}
            <div className="login-sso">
              {orderedProviders.map((p) => (
                <a
                  key={p}
                  className="login-sso-btn"
                  href={ssoLoginUrl(p)}
                >
                  <span className="login-sso-mark">{PROVIDER_META[p].mark}</span>
                  {PROVIDER_META[p].label}
                </a>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
