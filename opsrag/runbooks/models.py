"""Pydantic models for runbook API + store.

The DB row layout is in `opsrag/db/migrations/0004_runbooks.sql`. These
models mirror that layout for serialization across the FastAPI boundary.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from opsrag.runbooks.taxonomy import (
    FAILURE_CLASSES,
    SEVERITIES,
)


class RunbookCreate(BaseModel):
    """Payload for POST /api/runbooks. All optional fields can be set
    later via PUT."""

    title: str = Field(..., min_length=1, max_length=200)
    body_markdown: str = Field(..., min_length=1, max_length=200_000)
    service: str | None = Field(default=None, max_length=80)
    issue_kind: str | None = Field(default=None)
    severity_min: str | None = Field(default=None)
    priority: int = Field(default=100, ge=0, le=10000)
    tags: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("issue_kind")
    @classmethod
    def _validate_issue_kind(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v not in FAILURE_CLASSES:
            raise ValueError(
                f"issue_kind must be one of {FAILURE_CLASSES} (got {v!r})"
            )
        return v

    @field_validator("severity_min")
    @classmethod
    def _validate_severity(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v not in SEVERITIES:
            raise ValueError(
                f"severity_min must be one of {SEVERITIES} (got {v!r})"
            )
        return v


class RunbookUpdate(BaseModel):
    """Payload for PUT /api/runbooks/<id>. Every field optional;
    only present fields are applied. `change_note` is appended to
    the version row but NOT stored on the main runbook."""

    title: str | None = Field(default=None, min_length=1, max_length=200)
    body_markdown: str | None = Field(default=None, min_length=1, max_length=200_000)
    service: str | None = Field(default=None, max_length=80)
    issue_kind: str | None = Field(default=None)
    severity_min: str | None = Field(default=None)
    priority: int | None = Field(default=None, ge=0, le=10000)
    tags: list[str] | None = Field(default=None, max_length=32)
    enabled: bool | None = Field(default=None)
    change_note: str | None = Field(default=None, max_length=500)

    @field_validator("issue_kind")
    @classmethod
    def _validate_issue_kind(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v not in FAILURE_CLASSES:
            raise ValueError(
                f"issue_kind must be one of {FAILURE_CLASSES} (got {v!r})"
            )
        return v

    @field_validator("severity_min")
    @classmethod
    def _validate_severity(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        if v not in SEVERITIES:
            raise ValueError(
                f"severity_min must be one of {SEVERITIES} (got {v!r})"
            )
        return v


class Runbook(BaseModel):
    """Full runbook row -- what GET /api/runbooks/<id> returns."""

    id: str
    title: str
    body_markdown: str
    service: str | None = None
    issue_kind: str | None = None
    severity_min: str | None = None
    priority: int = 100
    tags: list[str] = Field(default_factory=list)
    source: str = "hand"                       # 'hand' | 'imported' | 'auto'
    author_email: str | None = None
    source_investigation_id: str | None = None
    enabled: bool = True
    created_at: datetime
    updated_at: datetime
    used_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    last_used_at: datetime | None = None


class RunbookVersion(BaseModel):
    """One historical version of a runbook (append-only audit log)."""

    id: int                                    # bigserial
    runbook_id: str
    version_num: int
    title: str
    body_markdown: str
    service: str | None = None
    issue_kind: str | None = None
    severity_min: str | None = None
    priority: int | None = None
    tags: list[str] = Field(default_factory=list)
    edited_by: str | None = None
    edited_at: datetime
    change_note: str | None = None


class RunbookHit(BaseModel):
    """Retrieval result. `score` combines embedding similarity (when
    available) and tsv rank; hand-authored ALWAYS sort above RAG hits
    regardless of score (see lane.py)."""

    runbook: Runbook
    score: float                               # 0..1, higher = more relevant
    score_breakdown: dict = Field(default_factory=dict)  # for debugging
    # Origin lets the UI render different badges.
    origin: str = "hand"                       # 'hand' | 'rag'
