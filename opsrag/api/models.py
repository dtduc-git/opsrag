"""API request / response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ImageInput(BaseModel):
    """A base64-encoded image attached to a chat turn (ephemeral)."""
    mime_type: str = Field(..., description="image/png | image/jpeg | image/gif | image/webp")
    data: str = Field(..., description="base64-encoded image bytes (no data: prefix)")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    user_id: str = Field("anonymous", description="Stable user id for session routing")
    thread_id: str | None = Field(None, description="Optional existing thread to resume")
    stream: bool = Field(False, description="If true, return SSE stream instead of JSON")
    images: list[ImageInput] | None = Field(
        None, description="Optional images for a vision-capable model (ephemeral)"
    )


class SourceChunk(BaseModel):
    source: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    source_urls: list[str | None] = Field(default_factory=list)
    sources_content: list[SourceChunk] = Field(default_factory=list)
    grounded: bool
    thread_id: str
    session_resumable: bool
    query_type: str | None = None
    # Cache observability
    cache_hit: bool = False
    cache_similarity: float | None = None
    cache_age_seconds: float | None = None
    # Investigation cache id for feedback wiring
    investigation_id: str | None = None
    # Forensic / live / procedural / mixed / unknown
    query_category: str | None = None
    # Stale-while-revalidate. True when the cache
    # hit was past TTL but served anyway; UI shows an "updating..." badge
    # and a background revalidation refreshes for the next user.
    cache_is_stale: bool = False
    # Externalized reasoner plan (populated when the reasoner
    # called `update_plan` during this turn). Forwarded to the UI for
    # the InvestigationPlan card.
    plan: list[dict] = Field(default_factory=list)


class SessionSummary(BaseModel):
    thread_id: str
    user_id: str
    checkpoint_count: int
    # Enriched list fields (derived in one checkpoint walk by the session
    # store) so the UI can render a real conversation list -- a human title
    # (first question), a preview (most recent answer), last-activity time,
    # and turn count -- instead of an opaque thread id. All optional so older
    # stores / empty threads still validate.
    title: str | None = None
    preview: str | None = None
    updated_at: str | None = None
    created_at: str | None = None
    turn_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class IndexRepoRequest(BaseModel):
    repo: str
    # None -> the handler falls back to the configured scm.default_branch
    # (so a repo on `master` isn't hard-coded to a non-existent `main`).
    branch: str | None = None
    patterns: list[str] | None = None


class AgentGuidanceRequest(BaseModel):
    """Admin "Agent Guidance" save -- deployment-wide custom instructions."""
    custom_instructions: str = Field(default="", max_length=20000)


class AgentGuidanceResponse(BaseModel):
    custom_instructions: str
    updated_at: str | None = None
    updated_by: str | None = None
    # "db" when an operator has saved a value, "config" when falling back to
    # the deployment.custom_instructions seed, "none" when neither is set.
    source: str = "none"


class IndexRepoResponse(BaseModel):
    repo: str
    branch: str
    chunks_indexed: int


class IndexSourceRequest(BaseModel):
    """Trigger a non-git source ingestion (Confluence, etc).

    `source_type` must match a provider registered in the
    `IngestionPipeline.sources` dict. `scope` is source-specific --
    Confluence space key, Rootly project id, etc.
    """

    source_type: str
    scope: str


class IndexSourceResponse(BaseModel):
    source_type: str
    scope: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    version: str


class UIConfigResponse(BaseModel):
    """Runtime config the React UI fetches at boot. All values come
    from `OpsRAGConfig.brand` + the source-link bases. Lets the same UI
    image deploy for any tenant -- no rebuild required."""

    brand_name: str
    brand_subtitle: str
    assistant_name: str
    # The active answer/reasoning model id (e.g. "us.anthropic.claude-opus-4-8"),
    # surfaced read-only so the UI footer shows the real model instead of a
    # hard-coded guess.
    model_name: str | None = None
    # Optional white-label / source-link fields. These default to None/""
    # in config (a vendor-neutral deployment sets none of them), so they
    # MUST be Optional -- typing them as required `str` made /ui-config raise
    # a 500 on boot whenever any was unset (which is the default).
    favicon_url: str | None = None
    accent_color: str | None = None
    # Source-link bases -- exposed read-only so the UI could in theory
    # render previews or open-in-source-system buttons. Not required
    # for the core chat experience but cheap to include.
    confluence_base_url: str | None = None
    slack_workspace_url: str | None = None
    rootly_web_url: str | None = None
    gitlab_base_url: str | None = None
    # Feature gate (config-driven, NOT hardcoded): the Investigate tab is
    # only meaningful when the operator enabled a live-telemetry MCP
    # integration (datadog / prometheus / kubernetes / ...). The UI hides
    # the tab when this is False. Defaults False so a vendor-neutral
    # deployment with no telemetry enabled never shows it.
    investigation_enabled: bool = False


# Investigation feedback API
class InvestigationFeedbackRequest(BaseModel):
    thumbs: str = Field(..., description="'up' or 'down'")
    correction: str | None = Field(None, max_length=2000, description="Optional free-text correction")
    # Extra context persisted into Postgres opsrag_feedback
    # so SREs can triage low-scored answers without re-fetching the original
    # investigation. All optional -- the route handler tolerates absence.
    thread_id: str | None = Field(None, description="Optional thread id for session correlation")
    user_id: str | None = Field(None, description="Optional user id of the rater")
    query_snippet: str | None = Field(None, max_length=400, description="First 400 chars of the user's question")
    answer_snippet: str | None = Field(None, max_length=400, description="First 400 chars of the assistant's answer")


class InvestigationFeedbackResponse(BaseModel):
    investigation_id: str
    recorded: bool
    detail: str | None = None
    # Postgres row id (if the row was persisted). Lets the UI / SRE
    # tooling correlate a UI click with the durable feedback record.
    feedback_id: int | None = None


# SRE review list endpoint
class FeedbackListItem(BaseModel):
    id: int
    investigation_id: str
    thread_id: str | None = None
    user_id: str | None = None
    direction: int
    note: str | None = None
    created_at: str | None = None
    query_snippet: str | None = None
    answer_snippet: str | None = None


class FeedbackListResponse(BaseModel):
    items: list[FeedbackListItem]
    direction_filter: int | None = None
    limit: int


class InvestigationCacheSummary(BaseModel):
    total: int
    available: bool
    stale: int = 0
    low_quality: int = 0


# Unified cache control panel
class CachePurgeRequest(BaseModel):
    """Multi-strategy cache purge. The `target` field selects which
    cache(s) to operate on; `strategy` + its arg field selects the
    filter. Combine in the body, e.g.:

      {"target": "qa", "strategy": "older_than", "older_than_hours": 168}
      {"target": "qa", "strategy": "repo", "repo": "confluence:SRE"}
      {"target": "qa", "strategy": "quality_low"}
      {"target": "qa", "strategy": "question_contains", "question_substring": "kafka"}
      {"target": "investigation", "strategy": "thumbs_down"}
      {"target": "tool", "strategy": "tool_name", "tool_name": "prometheus_query"}
      {"target": "all", "strategy": "all"}
    """
    target: str = Field(..., description="qa | investigation | tool | all")
    strategy: str = Field(..., description="all | older_than | repo | quality_low | thumbs_down | question_contains | tool_name")
    older_than_hours: int | None = Field(None, ge=1)
    repo: str | None = None
    question_substring: str | None = Field(None, min_length=2, max_length=200)
    tool_name: str | None = None


class CachePurgeResponse(BaseModel):
    target: str
    strategy: str
    purged_qa: int = 0
    purged_investigation: int = 0
    purged_tool: int = 0
    detail: str | None = None


class CacheSummaryResponse(BaseModel):
    qa: dict
    investigation: dict
    tool: dict
    stale_older_than_days: int = 180


# Feedback-as-correction: user replies "no, it's actually X"
# and the corrected answer is stored as a high-weight (2.5x) Qdrant chunk
# that dominates future retrieval for the same question.
class CorrectionRequest(BaseModel):
    """Submitted by the chat UI when the user clicks thumbs-down + types the
    correct answer. ``question`` is the user's original query (so the
    embedded chunk has question-anchor signal). ``wrong_answer`` is kept
    for audit context -- it isn't embedded."""

    question: str = Field(..., min_length=1, max_length=4000)
    wrong_answer: str = Field("", max_length=8000)
    correct_answer: str = Field(..., min_length=1, max_length=8000)
    evidence_url: str | None = Field(None, max_length=2000)
    thread_id: str | None = Field(None, max_length=200)
    user_id: str | None = Field(None, max_length=200)


class CorrectionResponse(BaseModel):
    ok: bool
    # Pending-queue row id. The correction is NOT live yet -- it awaits
    # operator approval before it is injected into retrieval.
    pending_id: int
    status: str = "pending"
    message: str
    # Audit-log row id from `opsrag_feedback`. Lets the UI correlate the
    # submission with the durable Postgres replay row.
    feedback_id: int | None = None


class PendingCorrectionItem(BaseModel):
    id: int
    question: str
    wrong_answer: str | None = None
    correct_answer: str
    evidence_url: str | None = None
    user_id: str | None = None
    created_at: str | None = None


class PendingCorrectionListResponse(BaseModel):
    items: list[PendingCorrectionItem]
    limit: int


class CorrectionReviewResponse(BaseModel):
    ok: bool
    id: int
    status: str
    # Set when an approval injected the boosted chunk into retrieval.
    chunk_id: str | None = None
    message: str


class CorrectionListItem(BaseModel):
    chunk_id: str
    original_question: str | None = None
    wrong_answer: str | None = None
    correct_answer: str | None = None
    user_id: str | None = None
    evidence_url: str | None = None
    created_at: float | None = None


class CorrectionListResponse(BaseModel):
    items: list[CorrectionListItem]
    limit: int
