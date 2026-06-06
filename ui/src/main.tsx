import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import AuthGate from "./components/AuthGate";
import "./index.css";

// Premium-only entry. The classic UI (AppClassic.tsx +
// SidebarClassic.tsx + index-classic.css) is kept on disk for future
// revival but is no longer reachable: no toggle in the sidebar, no
// dynamic import here. To bring it back, restore the variant-aware
// router from git history and re-add the toggle in Sidebar.tsx.
//
// <AuthGate> wraps the shell: it resolves /me on boot and renders the Login
// page only when the backend demands auth. In OPEN mode it is transparent
// and renders <App> immediately. It owns identity and threads `me` down.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AuthGate>
      {(me, reloadMe) => <App me={me} reloadMe={reloadMe} />}
    </AuthGate>
  </React.StrictMode>
);
