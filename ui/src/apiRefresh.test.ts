// Silent-refresh + single-flight behaviour of apiFetch / refreshSession.
//
// Node test env: `window` is undefined at import time, so api.ts's import-time
// fetch wrapper is skipped and apiFetch/refreshSession call the global `fetch`,
// which we stub per-test. `vi.resetModules()` gives each test a fresh
// `_refreshInFlight` module-scoped latch (kept in its OWN file so it never
// resets the modules the pure-function specs in api.test.ts statically import).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => {
  vi.resetModules();
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("refreshSession (single-flight)", () => {
  it("collapses concurrent refreshes into ONE /auth/refresh POST", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { refreshSession } = await import("./api");

    // 5 requests 401 at once (idle lapse) -> they must share one refresh, or
    // token rotation would invalidate all but the first and bounce to login.
    const results = await Promise.all(
      Array.from({ length: 5 }, () => refreshSession()),
    );

    expect(results).toEqual([true, true, true, true, true]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/auth/refresh",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("returns false on a 401 refresh and re-arms for the next lapse", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    const { refreshSession } = await import("./api");

    expect(await refreshSession()).toBe(false);
    // The single-flight slot is released once settled, so a later lapse
    // refreshes again rather than reusing the failed result forever.
    expect(await refreshSession()).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns false when the refresh POST throws", async () => {
    const fetchMock = vi.fn(async () => { throw new Error("network down"); });
    vi.stubGlobal("fetch", fetchMock);
    const { refreshSession } = await import("./api");
    expect(await refreshSession()).toBe(false);
  });
});

describe("apiFetch (401 -> silent refresh -> retry)", () => {
  it("refreshes once and retries the original request, then returns it", async () => {
    const seq: string[] = [];
    let thingCalls = 0;
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      seq.push(`${init?.method ?? "GET"} ${url}`);
      if (url === "/api/auth/refresh") return new Response(null, { status: 200 });
      thingCalls += 1;
      return new Response("ok", { status: thingCalls === 1 ? 401 : 200 });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await import("./api");

    const resp = await apiFetch("/api/thing");
    expect(resp.status).toBe(200);
    expect(await resp.text()).toBe("ok");
    expect(seq).toEqual([
      "GET /api/thing",
      "POST /api/auth/refresh",
      "GET /api/thing",
    ]);
  });

  it("does NOT refresh when the first response is not a 401", async () => {
    const fetchMock = vi.fn(async () => new Response("ok", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch } = await import("./api");
    await apiFetch("/api/thing");
    expect(fetchMock).toHaveBeenCalledTimes(1); // no refresh, no retry
  });

  it("triggers login and throws when the refresh also fails", async () => {
    const dispatched: string[] = [];
    vi.stubGlobal("Event", class { type: string; constructor(t: string) { this.type = t; } });
    vi.stubGlobal("window", {
      __opsragCsrf: true, // skip api.ts's import-time fetch wrapper
      location: { hash: "" },
      dispatchEvent: (e: { type: string }) => { dispatched.push(e.type); return true; },
    });
    const fetchMock = vi.fn(async () => new Response(null, { status: 401 }));
    vi.stubGlobal("fetch", fetchMock);
    const { apiFetch, UnauthenticatedError } = await import("./api");

    await expect(apiFetch("/api/thing")).rejects.toBeInstanceOf(UnauthenticatedError);
    const w = (globalThis as unknown as { window: { location: { hash: string } } }).window;
    expect(w.location.hash).toBe("#/login");
    expect(dispatched).toContain("opsrag:auth-required");
  });
});
