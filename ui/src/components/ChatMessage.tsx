import { useState, useRef, useEffect, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import { IconCopy, IconCheck, IconFile, IconThumbUp, IconThumbDown } from "./icons";
import ThinkingProgress, { type ProgressStep, type CacheHitInfo } from "./ThinkingProgress";
import TimeseriesChart, { type TimeseriesChartProps } from "./TimeseriesChart";
import InvestigationPlan, { type InvestigationPlanProps } from "./InvestigationPlan";
import DiagramComponent, { type DiagramData } from "./DiagramComponent";
import { postFeedback, postCorrection } from "../api";

// — generic envelope for backend-emitted UI components. Today
// the only `type` is `TimeseriesChart`; future tool integrations
// (k8s topology, alert timelines, etc.) can plug into the same field
// without touching streaming code by extending the registry below.
export interface RichComponent {
  type: string;
  props: Record<string, unknown>;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  sourceUrls?: (string | null)[];
  grounded?: boolean;
  queryType?: string | null;
  streaming?: boolean;
  progress?: ProgressStep[];
  cacheHit?: CacheHitInfo | null;
  investigationId?: string | null;
  cacheIsStale?: boolean;
  queryCategory?: string | null;
  richComponents?: RichComponent[];
  // Pomerium author of this user-role turn (replayed sessions). Null
  // for legacy / pre-Pomerium history. Compared with the viewer's
  // `me.email` to decide "You" vs the teammate's name.
  authorEmail?: string | null;
  authorName?: string | null;
  // ISO 8601 UTC timestamp of this turn's LangGraph checkpoint. Surfaced
  // by the UI as a subtle inline date next to the author label, both
  // for diagnostic value (when was this asked?) and as a hint that
  // older replayed answers may reference state that's since changed.
  ts?: string | null;
  // User-attached image thumbnails (vision). Client-only: the server is
  // ephemeral and never persists image bytes, so these survive only for the
  // current session (they won't reappear on reload / replay). `dataUrl` is a
  // base64 data: URL captured at send time in ChatInput.
  images?: { mime: string; dataUrl: string }[];
}

// Registry of types the assistant message can render inline.
// Defensive: an unknown `type` falls through to a small warning so a
// new component shipped on the backend doesn't crash older UI builds.
const RICH_COMPONENT_REGISTRY: Record<string, (props: Record<string, unknown>) => ReactNode> = {
  TimeseriesChart: (props) => <TimeseriesChart {...(props as unknown as TimeseriesChartProps)} />,
  InvestigationPlan: (props) => <InvestigationPlan {...(props as unknown as InvestigationPlanProps)} />,
};

function renderRichComponent(comp: RichComponent, key: string | number): ReactNode {
  const renderer = RICH_COMPONENT_REGISTRY[comp.type];
  if (!renderer) {
    // Stay quiet in production — surface only in dev to help find
    // mismatched versions.
    return (
      <div key={key} className="rich-component-unknown">
        Unknown component type: <code>{comp.type}</code>
      </div>
    );
  }
  return <div key={key} className="rich-component-wrap">{renderer(comp.props)}</div>;
}

export interface ChatMessageContext {
  // Optional context that lets feedback include the user's question
  // + thread id without forcing the message itself to carry duplicate
  // state. Passed by the parent (App.tsx) which knows the surrounding
  // turn structure.
  threadId?: string | null;
  precedingUserQuery?: string | null;
  // Email + name of the viewer (from Pomerium / /api/me). When a user
  // message's authorEmail matches viewerEmail we render "You";
  // otherwise we render the original author's display name.
  viewerEmail?: string | null;
  viewerName?: string | null;
  // Read-only render (e.g. the public Channels browser): suppress the
  // interactive feedback (👍/👎 + correction) AND copy affordances. The
  // feedback path is also gated on `threadId` being present, but `readOnly`
  // makes the intent explicit and also hides the Copy button.
  readOnly?: boolean;
}

function displayAuthorLabel(msg: Message, ctx?: ChatMessageContext): string {
  if (msg.role !== "user") return "OpsRAG";
  const author = msg.authorEmail ?? null;
  const viewer = ctx?.viewerEmail ?? null;
  // Both null → legacy / anonymous mode → render "You" as before.
  if (!author && !viewer) return "You";
  // Either matches → it's the current viewer's turn.
  if (author && viewer && author.toLowerCase() === viewer.toLowerCase()) return "You";
  // No author info on the message but a viewer is logged in → can't
  // tell whose message this is. Fall back to "You" to avoid showing
  // misleading information.
  if (!author) return "You";
  // Different author: prefer human-readable name, fall back to email's
  // local-part, fall back to email itself.
  if (msg.authorName) return msg.authorName;
  const at = author.indexOf("@");
  return at > 0 ? author.slice(0, at) : author;
}

function displayAuthorInitial(msg: Message, ctx?: ChatMessageContext): string {
  if (msg.role !== "user") return "";
  const label = displayAuthorLabel(msg, ctx);
  if (label === "You") return "U";
  return label.charAt(0).toUpperCase();
}

// Render the message ts inline next to the author label as an ABSOLUTE
// local clock time so turns can be correlated with incident/log timelines.
//   - today:             "2:18 PM"
//   - earlier this year: "May 21, 2:18 PM"
//   - older:             "May 21 2024, 2:18 PM"
// The full ISO string is the element's `title` for precise hover detail.
function formatMessageTs(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (d.toDateString() === now.toDateString()) return time;
  const sameYear = d.getFullYear() === now.getFullYear();
  const date = d.toLocaleDateString(
    undefined,
    sameYear ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric" },
  );
  return `${date}, ${time}`;
}

function CodeBlock({ language, children }: { language: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = String(children).replace(/\n$/, "");

  const onCopy = async () => {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1400); }
    catch { /* ignore */ }
  };

  // — structured-JSON diagrams (preferred path). LLM emits
  // ```diagram-json fenced block whose body parses to
  // `{nodes: [...], edges: [...]}` per DiagramData. Auto-layout via
  // dagre, render via React Flow. If parse fails we fall through to
  // the raw-code branch — never a "syntax error" placeholder.
  //
  // Accept BOTH `diagram-json` (the prompt-specified label) AND `diagram`
  // (a common LLM auto-completion — Gemini drops the `-json` suffix on
  // ~30% of generations despite the system prompt). The schema check
  // below (nodes[] + edges[]) is what actually distinguishes a diagram
  // payload from a generic code block.
  if (language === "diagram-json" || language === "diagram") {
    let data: DiagramData | null = null;
    let parseError = "";
    try {
      const parsed = JSON.parse(text);
      if (parsed && Array.isArray(parsed.nodes) && Array.isArray(parsed.edges)) {
        data = parsed as DiagramData;
      } else {
        parseError = "diagram-json missing nodes[] or edges[]";
      }
    } catch (exc) {
      parseError = String((exc as Error).message || exc);
    }
    if (data) {
      return (
        <div className="code-block diagram-block">
          <div className="code-head">
            <span>diagram</span>
            <button className={`code-copy ${copied ? "copied" : ""}`} onClick={onCopy}>
              {copied ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
            </button>
          </div>
          <DiagramComponent data={data} />
        </div>
      );
    }
    // Fall through with a small hint above the raw source.
    return (
      <div className="code-block">
        <div className="code-head">
          <span>diagram-json (incomplete)</span>
          <button className={`code-copy ${copied ? "copied" : ""}`} onClick={onCopy}>
            {copied ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
          </button>
        </div>
        {parseError && <div className="diagram-parse-error" title={parseError}>diagram still streaming…</div>}
        <pre><code>{children}</code></pre>
      </div>
    );
  }

  // Legacy mermaid blocks kept for compatibility with older cached
  // answers; new content should use diagram-json. The mermaid path
  // also stays useful for sequence diagrams / Gantt / state machines
  // where structured-JSON would be over-engineered.
  if (language === "mermaid") {
    return <MermaidBlock source={text} onCopy={onCopy} copied={copied} />;
  }

  return (
    <div className="code-block">
      <div className="code-head">
        <span>{language || "code"}</span>
        <button className={`code-copy ${copied ? "copied" : ""}`} onClick={onCopy}>
          {copied ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
        </button>
      </div>
      <pre><code>{children}</code></pre>
    </div>
  );
}

// Normalize common LLM mistakes in mermaid source so partial-correct
// diagrams still render. Cheap regex pass — does not handle every quirk
// but covers the failures we've observed in production:
//   - unicode arrows  →  -->
//   - chained "A→ label → B" → "A -- label --> B"
// Keeps a copy of the original around in `lastRaw` for the Copy button.
function normalizeMermaid(src: string): string {
  let s = src;
  // Replace unicode arrows with mermaid's ASCII -->
  s = s.replace(/[→⟶⇒]/g, "-->");
  // Already-arrowed chains like "A--> label --> B" → "A -- label --> B"
  // (the LLM emits the label between two arrows; mermaid wants it before the arrow head)
  s = s.replace(/(\w)\s*-->\s*([^\n>]+?)\s*-->\s*(\w)/g,
    (_m, a, label, b) => `${a} -- "${label.trim()}" --> ${b}`);
  return s;
}

function MermaidBlock({ source, onCopy, copied }: { source: string; onCopy: () => void; copied: boolean }) {
  const [svg, setSvg] = useState<string>("");
  const [error, setError] = useState<string>("");
  const idRef = useRef(`mermaid-${Math.random().toString(36).slice(2)}`);
  const normalized = normalizeMermaid(source);

  useEffect(() => {
    let cancelled = false;
    // Debounce — during streaming the source grows char-by-char. Rendering
    // every keystroke is wasteful and (worse) mermaid's `render()` leaks
    // error DOM nodes into the document body when the source is partial /
    // invalid. Wait 250ms after the last change before attempting render.
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const m = (await import("mermaid")).default;
          m.initialize({
            startOnLoad: false,
            theme: "default",
            // strict — mermaid sanitizes the source before rendering;
            // SVG output cannot include arbitrary HTML from user input.
            securityLevel: "strict",
            flowchart: { useMaxWidth: true },
            // suppress mermaid's auto-injected error placeholders.
            suppressErrorRendering: true,
          });
          // Validate first via parse — pure check, NO DOM side effects.
          // This prevents the "Syntax error in text" elements mermaid
          // otherwise appends to the body when render() chokes on
          // partial / streaming-incomplete source.
          try {
            const parseResult = await m.parse(normalized, { suppressErrors: true });
            if (parseResult === false) {
              if (!cancelled) {
                setSvg("");
                setError("");  // not a real error — just incomplete source
              }
              return;
            }
          } catch {
            // parse threw — invalid syntax. Show raw source, no error chip.
            if (!cancelled) {
              setSvg("");
              setError("");
            }
            return;
          }
          const { svg: rendered } = await m.render(idRef.current, normalized);
          if (!cancelled) {
            setSvg(rendered);
            setError("");
          }
        } catch (exc) {
          if (!cancelled) setError(String((exc as Error).message || exc));
        }
      })();
    }, 250);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [normalized]);

  // Best-effort cleanup of any orphan error placeholders mermaid may have
  // injected before suppressErrorRendering was wired (or in older versions).
  // Runs once per mount; cheap, idempotent.
  useEffect(() => {
    const sweep = () => {
      document.querySelectorAll('div[id^="dmermaid-"]').forEach((el) => el.remove());
    };
    sweep();
    const t = setInterval(sweep, 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="code-block mermaid-block">
      <div className="code-head">
        <span>mermaid</span>
        <button className={`code-copy ${copied ? "copied" : ""}`} onClick={onCopy}>
          {copied ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
        </button>
      </div>
      {svg && !error ? (
        // SVG is sanitized by mermaid itself (strict mode above).
        <div className="mermaid-svg" dangerouslySetInnerHTML={{ __html: svg }} />
      ) : (
        <pre><code>{source}</code></pre>
      )}
      {error && (
        <div className="mermaid-error" title={error}>
          (mermaid render failed — showing source)
        </div>
      )}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1400); }
    catch { /* ignore */ }
  };
  return (
    <button className={`msg-action ${copied ? "copied" : ""}`} onClick={onCopy} title="Copy answer">
      {copied ? <><IconCheck /> Copied</> : <><IconCopy /> Copy</>}
    </button>
  );
}

function FeedbackButtons({
  investigationId,
  threadId,
  userQuery,
  assistantAnswer,
}: {
  investigationId: string;
  threadId?: string | null;
  userQuery?: string | null;
  assistantAnswer?: string | null;
}) {
  // T1.3 — thumbs feedback UX. Click optimistically shows the
  // selected state; the POST is dispatched without awaiting so the user
  // gets instant visual confirmation.
  //
  // T1.6 — when the user clicks 👎, instead of just an
  // optional one-line note, we open a richer CORRECTION form. The user
  // can type the correct answer + optional evidence URL; on submit we
  // POST to /correction which stores the corrected fact as a 2.5×-boost
  // Qdrant chunk that dominates retrieval for future similar questions.
  // The 👎 audit-log POST still happens in parallel — we want the
  // Postgres feedback record AND the Qdrant correction.
  const [picked, setPicked] = useState<"up" | "down" | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [correctAnswer, setCorrectAnswer] = useState("");
  const [evidenceUrl, setEvidenceUrl] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const [savedPendingId, setSavedPendingId] = useState<number | null>(null);

  const trimSnippet = (s: string | null | undefined) => (s ?? "").slice(0, 400);

  const submitThumbs = (direction: 1 | -1, note?: string) => {
    // Fire-and-forget — instant UX. Errors are swallowed by `postFeedback`.
    void postFeedback({
      investigation_id: investigationId,
      direction,
      note,
      thread_id: threadId ?? null,
      query_snippet: trimSnippet(userQuery),
      answer_snippet: trimSnippet(assistantAnswer),
    });
  };

  const click = (thumbs: "up" | "down") => {
    if (picked) return;
    setPicked(thumbs);
    const direction: 1 | -1 = thumbs === "up" ? 1 : -1;
    submitThumbs(direction);
    if (thumbs === "down") setFormOpen(true);
  };

  const submitCorrection = async () => {
    const trimmed = correctAnswer.trim();
    if (!trimmed) {
      setSaveErr("Correct answer is required.");
      return;
    }
    setSaving(true);
    setSaveErr(null);
    try {
      // Use the preceding user query as the question anchor so the
      // synthetic chunk's vector is near the user's intent. Fall back to
      // the investigation id if we somehow lost the original query
      // (shouldn't happen in normal flow — App.tsx threads it through).
      const question = (userQuery ?? "").trim() || investigationId;
      const resp = await postCorrection({
        question,
        wrong_answer: assistantAnswer ?? "",
        correct_answer: trimmed,
        evidence_url: evidenceUrl.trim() || null,
        thread_id: threadId ?? null,
      });
      setSavedPendingId(resp.pending_id);
      setFormOpen(false);
      // ALSO log the note into the existing feedback audit — that way
      // the SRE list view sees "user said the answer was X" in plain text.
      submitThumbs(-1, trimmed);
    } catch (err) {
      setSaveErr(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  // Rendered as a Fragment so the two pills sit on the SAME flex row as the
  // Copy button (matching chrome). The correction form / saved note wrap to a
  // full-width line below (msg-actions is flex-wrap; .correction-form is 100%).
  return (
    <>
      <button
        className={`msg-action fb-pill fb-up ${picked === "up" ? "selected" : ""}`}
        disabled={picked !== null}
        onClick={() => click("up")}
        title="Good answer"
      ><IconThumbUp /> Helpful</button>
      <button
        className={`msg-action fb-pill fb-down ${picked === "down" ? "selected" : ""}`}
        disabled={picked !== null}
        onClick={() => click("down")}
        title="Bad answer — leave the correct one to teach OpsRAG"
      ><IconThumbDown /> Not helpful</button>
      {formOpen && (
        <div className="correction-form">
          <div className="correction-form-label">
            What is the correct answer?
            <span className="correction-form-hint">
              {" "}Submitted for operator review — approved corrections weigh into future answers.
            </span>
          </div>
          <textarea
            className="correction-form-textarea"
            placeholder="e.g. The consumer of topic my-service.box-event is my-service-event-listener (saas/my-service/src/listeners/box.ts:42)…"
            value={correctAnswer}
            onChange={(e) => setCorrectAnswer(e.target.value)}
            autoFocus
            maxLength={8000}
            rows={3}
          />
          <input
            type="text"
            className="correction-form-input"
            placeholder="Evidence URL (Confluence, GitLab file, optional)"
            value={evidenceUrl}
            onChange={(e) => setEvidenceUrl(e.target.value)}
            maxLength={2000}
          />
          <div className="correction-form-actions">
            <button
              className="correction-form-submit"
              disabled={saving || !correctAnswer.trim()}
              onClick={() => { void submitCorrection(); }}
            >{saving ? "Saving…" : "Save correction"}</button>
            <button
              className="correction-form-cancel"
              disabled={saving}
              onClick={() => setFormOpen(false)}
            >Cancel</button>
          </div>
          {saveErr && <div className="correction-form-error">{saveErr}</div>}
        </div>
      )}
      {savedPendingId !== null && (
        <span className="correction-saved" title={`pending_id=${savedPendingId}`}>
          ✓ Correction submitted for review. It goes live once an operator approves it.
        </span>
      )}
    </>
  );
}

export default function ChatMessage({ msg, ctx }: { msg: Message; ctx?: ChatMessageContext }) {
  const isUser = msg.role === "user";

  const authorLabel = displayAuthorLabel(msg, ctx);
  const authorInitial = displayAuthorInitial(msg, ctx);
  const isTeammate = isUser && authorLabel !== "You";

  return (
    <div className={`message ${msg.role}`}>
      <div
        className={`avatar ${isUser ? "user" : "bot"}${isTeammate ? " teammate" : ""}`}
        title={isUser && msg.authorEmail ? msg.authorEmail : undefined}
      >
        {isUser ? authorInitial : <img src="/opsrag-logo.png" alt="OpsRAG" />}
      </div>

      <div className="msg-body">
        <div className="msg-meta">
          <span className="name" title={isUser && msg.authorEmail ? msg.authorEmail : undefined}>{authorLabel}</span>
          {msg.ts && <span className="msg-ts" title={msg.ts}>{formatMessageTs(msg.ts)}</span>}
        </div>

        {!isUser && (msg.progress?.length || msg.cacheHit) && (
          <ThinkingProgress
            steps={msg.progress ?? []}
            cacheHit={msg.cacheHit ?? null}
            finished={!msg.streaming}
          />
        )}

        <div className="msg-bubble">
          {msg.streaming && !msg.content ? (
            <div className="thinking">
              <div className="dot" /><div className="dot" /><div className="dot" />
            </div>
          ) : isUser ? (
            <p>{msg.content}</p>
          ) : (
            <ReactMarkdown
              components={{
                code({ className, children, ...props }: { className?: string; children?: ReactNode; node?: unknown; inline?: boolean }) {
                  // Allow hyphens in language names (e.g. `diagram-json`).
                  // `\w` excludes `-`; without `[\w-]+` the dispatcher
                  // would only see `diagram` and miss the JSON renderer.
                  const langMatch = (className || "").match(/language-([\w-]+)/);
                  const isBlock = langMatch || (typeof children === "string" && children.includes("\n"));
                  if (isBlock) return <CodeBlock language={langMatch ? langMatch[1] : ""}>{children}</CodeBlock>;
                  return <code className={className} {...props}>{children}</code>;
                },
              }}
            >
              {msg.content + (msg.streaming ? "▍" : "")}
            </ReactMarkdown>
          )}
        </div>

        {isUser && msg.images && msg.images.length > 0 && (
          <div className="msg-images">
            {msg.images.map((img, i) => (
              <img
                key={i}
                className="msg-image"
                src={img.dataUrl}
                alt={`attached image ${i + 1}`}
              />
            ))}
          </div>
        )}

        {!isUser && msg.richComponents && msg.richComponents.length > 0 && (
          <div className="rich-components">
            {msg.richComponents.map((comp, i) => renderRichComponent(comp, i))}
          </div>
        )}

        {!isUser && !msg.streaming && (
          <>
            {(msg.sources?.length || msg.grounded || msg.queryType) ? (
              <div className="msg-meta-row">
                {msg.sources?.map((src, i) => {
                  // Format: "<owner>/<repo>/path/to/file.ext"
                  const parts = src.split("/");
                  const file = parts[parts.length - 1];
                  // Show "repo · file" if we can identify a repo prefix.
                  const repoSlug = parts.length >= 3 ? parts.slice(0, 2).join("/") : null;
                  const url = msg.sourceUrls?.[i] ?? null;
                  const inner = (
                    <>
                      <IconFile />
                      {repoSlug && <span style={{ color: "var(--text-3)" }}>{repoSlug.split("/").pop()} · </span>}
                      {file}
                    </>
                  );
                  if (url) {
                    return (
                      <a key={src} className="src-tag src-tag-link" href={url} target="_blank" rel="noopener noreferrer" title={`${src}\n${url}`}>
                        {inner}
                      </a>
                    );
                  }
                  return (
                    <span key={src} className="src-tag" title={src}>
                      {inner}
                    </span>
                  );
                })}
                {msg.grounded && <span className="badge badge-grounded">Grounded</span>}
                {msg.cacheIsStale && (
                  <span className="badge badge-stale" title="Served from cache past TTL — refreshing in background; ask again in a moment for the latest answer.">
                    Updating…
                  </span>
                )}
                {msg.queryCategory && (
                  <span className={`badge badge-cat-${msg.queryCategory}`} title={`Query classified as ${msg.queryCategory}`}>
                    {msg.queryCategory}
                  </span>
                )}
                {msg.queryType && <span className="badge badge-type">{msg.queryType}</span>}
              </div>
            ) : null}
            <div className="msg-actions">
              {!ctx?.readOnly && <CopyButton text={msg.content} />}
              {/* Feedback (👍/👎 + correction) for any COMPLETED answer, not
                  just investigations. Chat answers have no investigation id,
                  so the thread id is the audit anchor; a 👎 still opens the
                  correction form that writes a high-weight Qdrant chunk.
                  Suppressed entirely in read-only views (public Channels). */}
              {!ctx?.readOnly && !msg.streaming && (msg.investigationId || ctx?.threadId) && (
                <FeedbackButtons
                  investigationId={msg.investigationId ?? ctx?.threadId ?? ""}
                  threadId={ctx?.threadId ?? null}
                  userQuery={ctx?.precedingUserQuery ?? null}
                  assistantAnswer={msg.content}
                />
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
