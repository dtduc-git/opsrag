"""Slack mrkdwn -> clean-markdown normalizer for ingested query text.

Pure (no I/O, no async). Mirrors ``render.py:_to_mrkdwn``'s discipline:
protect fenced+inline code FIRST, run ordered rewrites, restore code LAST.
Slack inserts its own tokens (``<@U..>``, ``<url|text>``) with LITERAL angle
brackets, while user-typed ``<``/``>`` arrive HTML-escaped (``&lt;``/``&gt;``),
so literal-angle-bracket regexes only ever match real Slack tokens.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

# --- code protection (identical carve-out to render.py, applied first/last) ---
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
# D2: a streamed/truncated body can carry an UNTERMINATED fence -- protect it
# from the opening ``` to end-of-string so its tokens are never rewritten.
_UNCLOSED_FENCE_RE = re.compile(r"```.*\Z", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# --- Slack angle-bracket tokens (literal < >). Order below is load-bearing. ---
_USER_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|([^>]+))?>")
_SUBTEAM_RE = re.compile(r"<!subteam\^(S[A-Z0-9]+)(?:\|([^>]+))?>")
_ATCMD_RE = re.compile(r"<!(here|channel|everyone)(?:\|[^>]+)?>")
_BANG_LABEL_RE = re.compile(r"<!(?:[^>|]+)\|([^>]+)>")   # <!date^..|Feb 18> -> Feb 18
_BANG_BARE_RE = re.compile(r"<!(?:[^>|]+)>")              # <!date^..> -> drop
_CHANNEL_RE = re.compile(r"<#(C[A-Z0-9]+)(?:\|([^>]+))?>")
_MAILTO_RE = re.compile(r"<mailto:([^|>]+)(?:\|([^>]+))?>")
_LINK_RE = re.compile(r"<([^|>]+)\|([^>]+)>")            # generic <url|text>
# D1: bare autolinks -- http(s) kept verbatim; tel/sip/skype scheme stripped
# so a bare phone/uri reads naturally in the ingested query.
_BARE_URL_RE = re.compile(r"<((?:https?://|tel:|sip:|skype:)[^|>]+)>")

# --- block/inline markup ---
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*(?:&gt;|>){1,3}[ \t]?", re.MULTILINE)
_BOLD_RE = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_STRIKE_RE = re.compile(r"(?<![~\w])~(?!\s)([^~\n]+?)(?<!\s)~(?![~\w])")
_EMOJI_RE = re.compile(r"(?<![A-Za-z0-9]):([a-z_][a-z0-9_+\-]*):(?![A-Za-z0-9])")
_BULLET_RE = re.compile(r"^([ \t]*)[•·‣◦]\s+", re.MULTILINE)

# --- tidy ---
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_INTERNAL_SPACES_RE = re.compile(r"(?<=\S) {2,}(?=\S)")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_CODE_TOKEN_RE = re.compile(r"\x00CODE(\d+)\x00")

# Deliberately SMALL curated map -- everything else is stripped. Extend as
# real shortcodes surface in ingested Slack text.
_EMOJI_MAP: dict[str, str] = {
    "fire": "\U0001f525", "firef": "\U0001f525",
    "white_check_mark": "✅", "heavy_check_mark": "✅", "check": "✅",
    "x": "❌", "no_entry_sign": "\U0001f6ab",
    "warning": "⚠️", "rotating_light": "\U0001f6a8",
    "rocket": "\U0001f680", "tada": "\U0001f389", "eyes": "\U0001f440",
    "bulb": "\U0001f4a1", "memo": "\U0001f4dd", "wrench": "\U0001f527",
    "gear": "⚙️", "bell": "\U0001f514",
    "chart_with_upwards_trend": "\U0001f4c8",
    "green_circle": "\U0001f7e2", "large_green_circle": "\U0001f7e2",
    "yellow_circle": "\U0001f7e1", "red_circle": "\U0001f534",
}

_ID_SCAN_USER = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]+)?>")
_ID_SCAN_SUBTEAM = re.compile(r"<!subteam\^(S[A-Z0-9]+)(?:\|[^>]+)?>")
_ID_SCAN_CHANNEL = re.compile(r"<#(C[A-Z0-9]+)(?:\|[^>]+)?>")

_DEFAULT_SUBTEAM = "@oncall"
_TEL_SCHEMES = ("tel:", "sip:", "skype:")


def slack_ids(text: str) -> tuple[set[str], set[str], set[str]]:
    """Pure scan -> (user_ids, subteam_ids, channel_ids) present in ``text``.

    The async FirstResponder uses this to know which ids to resolve to names
    (via client.get_user_info / config) BEFORE calling ``normalize_slack_text``.
    """
    if not text:
        return set(), set(), set()
    return (
        {m.group(1) for m in _ID_SCAN_USER.finditer(text)},
        {m.group(1) for m in _ID_SCAN_SUBTEAM.finditer(text)},
        {m.group(1) for m in _ID_SCAN_CHANNEL.finditer(text)},
    )


def normalize_slack_text(
    text: str,
    user_names: Mapping[str, str] | None = None,
    subteam_names: Mapping[str, str] | None = None,
    channel_names: Mapping[str, str] | None = None,
) -> str:
    """Convert Slack mrkdwn + Slack tokens to clean markdown for the Web UI.

    ``*_names`` map Slack ids -> display names (resolved async by the caller).
    Missing names degrade gracefully: ``<@U..>`` -> ``@<id>``,
    ``<!subteam^S..>`` -> ``@oncall``, ``<#C..>`` -> ``#<id>``. A ``|label``
    inside a token always wins over the map (so an UNRESOLVED user with a
    pipe label renders the label, not a bare id).
    """
    if not text:
        return ""
    user_names = user_names or {}
    subteam_names = subteam_names or {}
    channel_names = channel_names or {}

    stash: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        stash.append(m.group(0))
        return f"\x00CODE{len(stash) - 1}\x00"

    # 1. Protect code: balanced fences, then any trailing UNCLOSED fence, then
    #    inline code (order so inline never sees fence bodies).
    work = _FENCED_CODE_RE.sub(_protect, text)
    work = _UNCLOSED_FENCE_RE.sub(_protect, work)
    work = _INLINE_CODE_RE.sub(_protect, work)

    # 2. Slack tokens (literal < >). Specific-before-generic so <#C..|x> /
    #    <!..|x> aren't grabbed by the generic <url|text> rule.
    def _user(m: re.Match[str]) -> str:
        uid, label = m.group(1), m.group(2)
        name = label or user_names.get(uid)
        return f"@{name}" if name else f"@{uid}"

    def _subteam(m: re.Match[str]) -> str:
        sid, label = m.group(1), m.group(2)
        name = label or subteam_names.get(sid)
        return f"@{name}" if name else _DEFAULT_SUBTEAM

    def _channel(m: re.Match[str]) -> str:
        cid, label = m.group(1), m.group(2)
        name = label or channel_names.get(cid)
        return f"#{name}" if name else f"#{cid}"

    def _mailto(m: re.Match[str]) -> str:
        email, label = m.group(1), m.group(2)
        if label and label.strip() and label != email:
            return f"[{label}](mailto:{email})"
        return email

    def _bare_url(m: re.Match[str]) -> str:
        uri = m.group(1)
        for pfx in _TEL_SCHEMES:
            if uri.startswith(pfx):
                return uri[len(pfx):]
        return uri  # http(s): keep the full clickable URL

    work = _USER_RE.sub(_user, work)
    work = _SUBTEAM_RE.sub(_subteam, work)
    work = _ATCMD_RE.sub(lambda m: f"@{m.group(1)}", work)
    work = _BANG_LABEL_RE.sub(lambda m: m.group(1), work)
    work = _BANG_BARE_RE.sub("", work)
    work = _CHANNEL_RE.sub(_channel, work)
    work = _MAILTO_RE.sub(_mailto, work)
    work = _LINK_RE.sub(lambda m: f"[{m.group(2)}]({m.group(1)})", work)
    work = _BARE_URL_RE.sub(_bare_url, work)

    # 3. Blockquotes BEFORE entity-unescape (Slack sends '>' as '&gt;').
    work = _BLOCKQUOTE_RE.sub("> ", work)

    # 4. Inline markup. *bold* -> **bold**, ~strike~ -> ~~strike~~.
    #    _italic_ left as-is (valid markdown; avoids snake_case corruption).
    work = _BOLD_RE.sub(lambda m: f"**{m.group(1)}**", work)
    work = _STRIKE_RE.sub(lambda m: f"~~{m.group(1)}~~", work)

    # 5. Emoji shortcodes -> unicode (curated) or strip.
    work = _EMOJI_RE.sub(lambda m: _EMOJI_MAP.get(m.group(1), ""), work)

    # 6. Unicode bullets -> markdown '- '.
    work = _BULLET_RE.sub(lambda m: f"{m.group(1)}- ", work)

    # 7. HTML-entity unescape (lt/gt before amp so '&amp;lt;' -> '&lt;').
    work = work.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # 8. Tidy: drop trailing ws, collapse emoji-strip gaps, cap blank runs.
    work = _TRAILING_WS_RE.sub("", work)
    work = _INTERNAL_SPACES_RE.sub(" ", work)
    work = _MULTI_BLANK_RE.sub("\n\n", work)

    # 9. Restore code LAST. Loop handles a placeholder nested inside an
    #    unclosed-fence stash (terminates: nesting indices strictly decrease).
    while _CODE_TOKEN_RE.search(work):
        work = _CODE_TOKEN_RE.sub(lambda m: stash[int(m.group(1))], work)
    return work
