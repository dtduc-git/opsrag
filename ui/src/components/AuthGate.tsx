import { useCallback, useEffect, useState } from "react";
import {
  fetchMe,
  fetchUIConfig,
  isAuthRequired,
  clearAuthRequired,
  AUTH_REQUIRED_EVENT,
  LOGIN_HASH,
  type MeResponse,
  type UIConfig,
} from "../api";
import LoginPage from "./LoginPage";

interface Props {
  // Render-prop: the gate owns identity and hands the resolved `me` plus a
  // re-fetch callback to the shell so App doesn't double-fetch /me.
  children: (me: MeResponse, reloadMe: () => Promise<void>) => React.ReactNode;
}

type Phase = "loading" | "login" | "ready";

// <AuthGate> wraps the app shell. On boot it fetches /me:
//   • OPEN mode  → /me returns a (possibly anonymous) identity WITH scopes,
//                  never 401s → phase "ready", gate is transparent.
//   • login/oidc → an unauthenticated caller trips a 401 envelope somewhere
//                  (or /me itself reports auth is required) → phase "login".
//
// We only show the Login page when the backend has actually demanded auth:
// either apiFetch latched a 401 (isAuthRequired) or the URL hash is the login
// route. An anonymous-but-allowed identity (open demo) renders the shell.
export default function AuthGate({ children }: Props) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [me, setMe] = useState<MeResponse | null>(null);
  const [uiConfig, setUIConfig] = useState<UIConfig | null>(null);

  // Pull brand for the login screen (best-effort; falls back to defaults).
  useEffect(() => {
    fetchUIConfig().then(setUIConfig).catch(() => { /* baked defaults */ });
  }, []);

  const resolve = useCallback(async () => {
    const identity = await fetchMe();
    setMe(identity);
    // Decide whether to wall off the shell. The login wall only appears when
    // the backend is enforcing auth: a 401 was latched, OR the hash explicitly
    // requests login. In OPEN mode neither is true, so we go straight to ready
    // even for an anonymous identity.
    const wantsLogin =
      isAuthRequired() || window.location.hash === LOGIN_HASH;
    setPhase(wantsLogin ? "login" : "ready");
  }, []);

  // Boot + re-evaluate whenever apiFetch signals a 401.
  useEffect(() => {
    resolve();
    const onAuthRequired = () => { setPhase("login"); };
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, [resolve]);

  // After a password login succeeds: clear the latch, drop the login hash,
  // and re-fetch identity to enter the shell.
  const handleLoggedIn = useCallback(async () => {
    clearAuthRequired();
    if (window.location.hash === LOGIN_HASH) {
      window.location.hash = "";
    }
    await resolve();
  }, [resolve]);

  // Allow the shell to refresh identity (e.g. after sign-out triggers a
  // re-gate, or scopes change). Sign-out itself sets phase via the event.
  const reloadMe = useCallback(async () => {
    await resolve();
  }, [resolve]);

  if (phase === "loading") {
    return (
      <div className="login-shell">
        <div className="login-loading">
          <div className="login-mark login-mark-lg">
            <img src="/opsrag-logo.png" alt="loading" />
          </div>
          <h1 className="login-loading-title">OpsRAG</h1>
          <p className="login-loading-sub">Warming up your DevOps intelligence…</p>
          <div className="login-loading-bar"><span /></div>
        </div>
      </div>
    );
  }

  if (phase === "login") {
    return (
      <LoginPage
        brandName={uiConfig?.brand_name ?? "OpsRAG"}
        brandSubtitle={uiConfig?.brand_subtitle ?? "DevOps Intelligence"}
        onLoggedIn={handleLoggedIn}
      />
    );
  }

  // phase === "ready" — `me` is always set here (resolve sets it before phase).
  return <>{children(me as MeResponse, reloadMe)}</>;
}
