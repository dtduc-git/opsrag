import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Minimal harness for the pure-function specs (slackPermalink, normalizeTitle,
// prettifySlackTokens, displayAuthorLabel). `node` env is enough — none of
// them touch the DOM; ChatMessage.tsx's other imports are guarded against a
// missing `window` (see api.ts's `typeof window !== "undefined"` check).
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
