// prettifySlackTokens — degrade raw Slack mrkdwn tokens to readable
// Markdown/plain text CLIENT-SIDE, for LEGACY stored transcripts. New convos
// are cleaned server-side (normalize_slack_text); old rows still hold raw
// <@U…>, <!subteam^S…>, <!here>, <#C…>, <url|text>, *bold*. react-markdown +
// remark-gfm don't parse Slack's <…>, so those leak as literal angle-bracket
// noise. Belt-and-braces: CANNOT resolve <@U…> to a real name (no browser
// directory) — it only makes tokens non-ugly.
//
// SAFETY: aggressive rewrites (esp. *bold* → **bold**, which would corrupt
// intentional *italic*) run ONLY when a genuine Slack angle-bracket token is
// present. Clean Markdown → returned untouched (safe to call on every finalized
// assistant message). Code is never rewritten (``` fenced + `inline` skipped).
// Pure: no I/O, no throw. :emoji: is intentionally LEFT AS-IS.

const HAS_SLACK_TOKEN = /<[@#!]|<https?:\/\/[^>\s]+\|/;         // is this legacy Slack?
const CODE_SPAN = /(```[\s\S]*?```|`[^`\n]*`)/g;                // fenced + inline code (kept)

const RE_LINK    = /<(https?:\/\/[^|>\s]+)(?:\|([^>]*))?>/g;    // <url> | <url|text>
const RE_SUBTEAM = /<!subteam\^[A-Z0-9]+(?:\|([^>]+))?>/g;      // <!subteam^S1|oncall>
const RE_SPECIAL = /<!(here|channel|everyone)\b[^>]*>/g;        // <!here> <!channel>
const RE_CHANNEL = /<#[CG][A-Z0-9]+(?:\|([^>]+))?>/g;          // <#C1|general>
const RE_USER    = /<@([UW][A-Z0-9]+)>/g;                       // <@U1> / <@W1>
// *bold* → **bold** (Slack single-* is bold). Guards ALIGNED to the Python
// normalizer (slack_text.py:36): zero-width lookbehind + `\*(?![*\w])` closing,
// so `*x*y` is left alone and adjacent spans aren't missed.
const RE_BOLD    = /(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])/g;

function degrade(seg: string): string {
  return seg
    .replace(RE_LINK,    (_m, url, text) => (text && text.trim() ? `[${text}](${url})` : url))
    .replace(RE_SUBTEAM, (_m, label) => `@${label || "oncall"}`)
    .replace(RE_SPECIAL, (_m, kw) => `@${kw}`)
    .replace(RE_CHANNEL, (_m, name) => (name ? `#${name}` : "#channel"))
    .replace(RE_USER,    (_m, id) => `@${id}`)
    .replace(RE_BOLD,    (_m, inner) => `**${inner}**`);
}

export function prettifySlackTokens(text: string): string {
  if (!text || !HAS_SLACK_TOKEN.test(text)) return text;  // clean markdown → untouched
  const parts = text.split(CODE_SPAN);   // [prose, code, prose, code, …]
  for (let i = 0; i < parts.length; i += 2) parts[i] = degrade(parts[i]); // even = prose
  return parts.join("");
}
