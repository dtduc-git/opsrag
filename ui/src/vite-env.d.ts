/// <reference types="vite/client" />

// Build-time configuration injected via Vite env vars. All optional; the app
// has sensible defaults so it runs without any of them (local demo).
interface ImportMetaEnv {
  readonly VITE_API_URL?: string;        // dev-server proxy target for /api
  readonly VITE_SIGN_OUT_URL?: string;   // SSO/proxy sign-out URL (empty => no sign-out CTA)
  readonly VITE_MCP_SERVER_URL?: string; // MCP server URL shown in the tokens page
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
