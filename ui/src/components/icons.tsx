/* Inline SVG icons — no external dependency. Stroke-based, currentColor. */

const stroke = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export const IconBolt = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z" /></svg>
);

export const IconChat = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
);

export const IconChart = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M3 3v18h18" /><path d="M7 14l3-3 4 4 5-7" /></svg>
);

export const IconStack = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" /></svg>
);

export const IconPlus = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M12 5v14M5 12h14" /></svg>
);

export const IconSend = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M22 2L11 13" /><path d="M22 2l-7 20-4-9-9-4z" /></svg>
);

export const IconSun = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
  </svg>
);

export const IconMoon = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
);

export const IconCopy = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

export const IconThumbUp = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M7 10v12" />
    <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
  </svg>
);

export const IconThumbDown = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M17 14V2" />
    <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
  </svg>
);

export const IconCheck = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M20 6L9 17l-5-5" /></svg>
);

export const IconTrash = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
  </svg>
);

export const IconFile = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <path d="M14 2v6h6" />
  </svg>
);

export const IconClose = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M18 6L6 18M6 6l12 12" /></svg>
);

export const IconHash = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M4 9h16M4 15h16M10 3L8 21M16 3l-2 18" /></svg>
);

export const IconMessage = () => (
  <svg viewBox="0 0 24 24" {...stroke}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
);

export const IconCoin = () => (
  <svg viewBox="0 0 24 24" {...stroke}><circle cx="12" cy="12" r="10"/><path d="M12 6v12M9 9h4.5a2.5 2.5 0 0 1 0 5H9M9 14h5.5a2.5 2.5 0 0 1 0 5H9"/></svg>
);

export const IconClock = () => (
  <svg viewBox="0 0 24 24" {...stroke}><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
);

export const IconDatabase = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <ellipse cx="12" cy="5" rx="9" ry="3"/>
    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>
  </svg>
);

export const IconSync = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M3 12a9 9 0 0 1 15-6.7L21 8"/>
    <path d="M21 3v5h-5"/>
    <path d="M21 12a9 9 0 0 1-15 6.7L3 16"/>
    <path d="M3 21v-5h5"/>
  </svg>
);

// Lucide-style key icon — used for the MCP Tokens nav entry.
export const IconKey = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <circle cx="7.5" cy="15.5" r="4.5" />
    <path d="M21 2l-9.6 9.6" />
    <path d="M15.5 7.5l3 3" />
  </svg>
);

// Sign-out / log-out icon.
export const IconLogout = () => (
  <svg viewBox="0 0 24 24" {...stroke}>
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <path d="M16 17l5-5-5-5" />
    <path d="M21 12H9" />
  </svg>
);
