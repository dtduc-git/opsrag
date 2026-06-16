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

// <AuthGate> wraps the app shell. On boot it fetches /me. Auth is ALWAYS
// required (login or oidc) -- there is no open/anonymous mode -- so:
//   • signed in      → /me returns a real identity (oid set) → phase "ready".
//   • not signed in  → /me is anonymous / 401 → phase "login" (LoginPage).
//
// We show the Login page whenever the identity is anonymous / has no oid, or a
// 401 was latched by apiFetch, or the URL hash is the login route.
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
    // Auth is ALWAYS required (login or oidc) -- there is no open/anonymous
    // mode. Any anonymous / no-identity result means "not signed in", so wall
    // off the shell and show the login page. (Also honors a latched 401 or the
    // explicit login hash.)
    const wantsLogin =
      identity.is_anonymous || !identity.oid ||
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
