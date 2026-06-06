"""Runbook-grounded hypothesis injection.

When the alert (resolved from Rootly) carries an `AlertRule_Runbook_URL`,
we treat the linked Confluence page as authoritative encoded knowledge
and convert its content into root-level hypothesis nodes that compete
with the LLM's hypothetical causes on equal footing.

Flow:
  1. Resolve runbook URL -> Confluence page (via ConfluenceClient).
  2. Render ADF -> readable text.
  3. Flash relevance gate: is this runbook about the same alert?
  4. Flash extraction: list up to 3 candidate root causes, each <= 200ch.
  5. Each cause becomes a HypothesisNode (depth=0, source="runbook")
     with one Citation pointing at the Confluence page.

Failure modes (all -> return [] silently, no error event):
  - Confluence creds missing (no CONFLUENCE_EMAIL / _API_TOKEN env)
  - URL doesn't resolve to a page (short URL gone, 404, etc.)
  - Page body too short ("TBD" / "WIP")
  - Relevance gate says no
  - Extraction returns 0 causes

The investigation graph proceeds normally without runbook hypotheses
in these cases -- they're an *augmentation*, not a dependency.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from opsrag.agents.investigation.state import Citation, HypothesisNode

_log = logging.getLogger("opsrag.agents.investigation.runbook_grounded")

_MIN_RUNBOOK_BODY_CHARS = 200
_MAX_CAUSES = 3
_MAX_RUNBOOK_TEXT_FOR_LLM = 8000  # truncate to keep Flash prompt tight


async def _fetch_runbook_text(url: str) -> tuple[str, str] | None:
    """Resolve URL -> Confluence page -> (title, plaintext). Returns None
    on any failure. Uses env-based auth (CONFLUENCE_EMAIL / _API_TOKEN)
    to avoid plumbing the client through every caller.
    """
    if not url:
        return None
    if "atlassian.net" not in url:
        # Non-Confluence runbook (e.g. GitHub README, Notion). We don't
        # have a generic fetcher; skip rather than guess.
        _log.info("runbook URL not on atlassian.net -- skipping: %s", url[:120])
        return None

    email = os.environ.get("CONFLUENCE_EMAIL", "").strip()
    token = os.environ.get("CONFLUENCE_API_TOKEN", "").strip()
    if not email or not token:
        _log.warning("Confluence creds missing -- runbook fetch disabled")
        return None

    # Derive base from the URL itself so we honour the user's tenant
    # without depending on config-local.yaml at agent-time.
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    try:
        from opsrag.sources.confluence.adf import render_adf
        from opsrag.sources.confluence.client import ConfluenceClient
    except Exception as exc:
        _log.warning("confluence helpers unavailable: %s", exc)
        return None

    client = ConfluenceClient(base_url=base, email=email, api_token=token)
    try:
        page = await client.get_page_by_url(url)
        if page is None:
            return None
        text = render_adf(page.body_adf).strip()
        if len(text) < _MIN_RUNBOOK_BODY_CHARS:
            _log.info(
                "runbook body too short (%d chars) -- skipping: %s",
                len(text), (page.title or url)[:80],
            )
            return None
        return (page.title or url, text)
    finally:
        await client.close()


_RELEVANCE_PROMPT = """\
You verify whether a runbook page is about the same alert the user is
investigating right now.

Alert: {alert_text!r}

Runbook title: {runbook_title!r}
Runbook excerpt (first 1.5k chars):
{runbook_excerpt}

Output exactly one JSON object:
  {{"relevant": true|false, "reason": "<one short sentence>"}}

Default to `false` when in doubt. The runbook is relevant only when its
description, alert signature, or symptoms clearly match the alert text.
"""

_EXTRACTION_PROMPT = """\
You read a runbook for an SRE alert and extract the candidate root
causes it enumerates. The investigation agent will turn each into a
hypothesis to test.

Alert: {alert_text!r}

Runbook title: {runbook_title!r}
Runbook (full text):
{runbook_text}

Rules:
- Pick concrete CAUSE statements (e.g. "Upstream service is rate-limiting"),
  not remediation steps (e.g. "Restart the pod").
- Output up to {max_causes} causes, one short sentence each (<= 180 chars).
- If the runbook is unstructured / vague / a stub ("contact #on-call"),
  return [].
- Output ONLY a JSON array of strings:
  ["cause 1", "cause 2", ...]
- No prose, no fences.
"""


async def _llm_json(llm, prompt: str, *, purpose: str, max_tokens: int = 800) -> Any:
    """Call Flash, strip ``` fences, parse JSON. Returns None on parse fail.
    Salvages the largest balanced `[...]` or `{...}` substring so
    Flash's occasional thinking-token preamble doesn't kill us.
    """
    try:
        resp = await llm.generate(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You answer in strict JSON only -- no prose, no fences.",
            temperature=0.0,
            max_tokens=max_tokens,
            purpose=purpose,
        )
    except Exception as exc:
        _log.warning("runbook %s LLM error: %s", purpose, exc)
        return None
    text = (getattr(resp, "content", "") or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Direct parse first (the common case).
    try:
        return json.loads(text)
    except Exception:
        pass
    # Salvage path: find the first `[` or `{` and the matching close.
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == open_ch:
                depth += 1
            elif text[i] == close_ch:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    _log.warning("runbook %s JSON parse failed; len=%d preview=%r", purpose, len(text), text[:200])
    return None


async def generate_runbook_hypotheses(
    alert_text: str,
    runbook_url: str,
    llm: Any,
) -> list[HypothesisNode]:
    """Produce 0-3 root hypotheses derived from the linked runbook.
    Returns [] on any failure so the investigation falls through to
    LLM-only hypothesis generation.

    We skip the relevance gate because the runbook URL was explicitly
    attached to the alert by the runbook author (it's in the alert's
    own `AlertRule_Runbook_URL` field). The downstream judge
    invalidates wrong hypotheses anyway, so erring toward "always try
    to extract" maximises bootstrap signal.
    """
    fetched = await _fetch_runbook_text(runbook_url)
    if not fetched:
        return []
    title, text = fetched

    excerpt = text[:1500]
    truncated_text = text[:_MAX_RUNBOOK_TEXT_FOR_LLM]
    if len(text) > _MAX_RUNBOOK_TEXT_FOR_LLM:
        truncated_text += f"\n\n[...truncated, {len(text)} total chars]"
    causes = await _llm_json(
        llm,
        _EXTRACTION_PROMPT.format(
            alert_text=alert_text,
            runbook_title=title,
            runbook_text=truncated_text,
            max_causes=_MAX_CAUSES,
        ),
        purpose="runbook_extract",
        max_tokens=2048,  # Flash thinking-tokens eat budget; give headroom
    )
    if not isinstance(causes, list):
        return []
    causes = [c.strip() for c in causes if isinstance(c, str) and c.strip()][:_MAX_CAUSES]
    if not causes:
        _log.info("runbook %r yielded 0 causes -- skipping injection", title[:60])
        return []

    # Construct the shared evidence Citation that every runbook
    # hypothesis points to. The chunk_id is synthetic (confluence pages
    # aren't chunked at investigation-time); we tag it so the source
    # display still resolves to the runbook URL.
    cite = Citation(
        source_id=f"confluence:runbook:{title[:60]}",
        chunk_id=f"runbook:{re.sub(r'[^A-Za-z0-9]', '_', runbook_url)[-32:]}",
        snippet=excerpt[:280],
        score=1.0,
        repo="confluence",
    )

    nodes: list[HypothesisNode] = []
    for cause in causes:
        nodes.append(HypothesisNode(
            statement=cause[:280],
            depth=0,
            parent_id=None,
            evidence=[cite],
            hypothesis_source="runbook",
        ))
    _log.info(
        "runbook hypothesis injection: title=%r n_causes=%d",
        title[:60], len(nodes),
    )
    return nodes
