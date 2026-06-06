"""Prompt-template rendering against ``DeploymentContext``.

Per Constitution Principle VI, system prompts are org-agnostic
reasoning patterns parameterised via context. This module provides the
helper that engine code uses to render prompt templates with
deployment-specific values supplied at runtime.

Usage::

    from opsrag.agent.prompt_render import render

    PROMPT_TEMPLATE = '''
    You are an SRE assistant for {organization_label}. The services you
    can reason about are: {services_csv}. Recognise ticket references
    matching {ticket_prefix}-N.
    '''

    prompt = render(PROMPT_TEMPLATE, ctx=settings.deployment)

The helper:

1. Uses ``str.format_map(...)`` semantics so curly braces inside the
   template that aren't intended as placeholders should be escaped as
   ``{{`` / ``}}`` (standard Python format-string convention).
2. Exposes a flat mapping of context values keyed by friendly names
   (``services_csv``, ``environments_csv``, ``ticket_prefix``,
   ``tracker_prefix``, ``organization_label``, plus the source URL
   bases). Unknown placeholders raise ``KeyError`` at render time so
   missing-context bugs surface during startup, not at the first user
   query.
3. Has no default opinions about what a deployment looks like: when a
   field is empty, the rendered substitution is the empty string -- and
   prompt authors are expected to write templates that read naturally
   in that case (e.g. ``"services you can reason about (if listed):
   {services_csv}"``).

The helper does NOT use Jinja2 or any heavier template engine. Plain
``str.format_map`` keeps prompts readable in source and avoids a
runtime dependency. If a prompt needs more complex logic (loops,
conditionals), that's a signal it's mixing reasoning with deployment
knowledge -- restructure rather than add template power.
"""
from __future__ import annotations

from collections.abc import Mapping

from opsrag.context import DeploymentContext


class _ContextView(Mapping[str, str]):
    """Mapping facade that materialises template values on demand.

    Built as a Mapping (not a dict) so ``str.format_map`` can iterate
    keys, but missing keys still raise ``KeyError`` immediately rather
    than the default-empty behaviour of ``defaultdict(str)`` -- which
    would let stale ``{old_field}`` references silently render as
    empty strings.
    """

    def __init__(self, ctx: DeploymentContext, *, extra: dict[str, str] | None = None) -> None:
        self._ctx = ctx
        self._extra: dict[str, str] = dict(extra or {})
        self._cache: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        if key in self._cache:
            return self._cache[key]
        if key in self._extra:
            return self._extra[key]
        try:
            value = _materialise(self._ctx, key)
        except KeyError:
            raise KeyError(key)
        self._cache[key] = value
        return value

    def __iter__(self):
        return iter(_KNOWN_KEYS)

    def __len__(self) -> int:
        return len(_KNOWN_KEYS)


def _materialise(ctx: DeploymentContext, key: str) -> str:
    """Project a ``DeploymentContext`` field onto a friendly template
    key. The mapping is intentionally narrow: prompts only see the
    deployment values the principle says they may use."""
    if key == "organization_label":
        return ctx.organization_label or ""

    if key == "services_csv":
        return ", ".join(ctx.services)
    if key == "services_bullets":
        return "\n".join(f"- `{s}`" for s in ctx.services)

    if key == "environments_csv":
        return ", ".join(ctx.environments)
    if key == "environments_bullets":
        return "\n".join(f"- `{e}`" for e in ctx.environments)

    if key == "key_repos_csv":
        return ", ".join(ctx.key_repos)
    if key == "key_repos_bullets":
        return "\n".join(f"- `{r}`" for r in ctx.key_repos)

    if key == "ticket_prefix" or key == "tracker_prefix":
        return ctx.tracker.prefix
    if key == "ticket_web_base":
        return ctx.tracker.web_base_url or ""

    if key == "k8s_namespaces_csv":
        return ", ".join(ctx.kubernetes.namespaces)
    if key == "k8s_clusters_csv":
        return ", ".join(f"{env}={cluster}" for env, cluster in ctx.kubernetes.clusters.items())
    if key == "k8s_pod_label_selector":
        return ctx.kubernetes.pod_label_selector or ""

    if key == "gcp_projects_csv":
        return ", ".join(f"{env}={proj}" for env, proj in ctx.cloud.gcp_projects.items())

    if key == "source_url_confluence":
        return ctx.source_urls.confluence or ""
    if key == "source_url_slack":
        return ctx.source_urls.slack or ""
    if key == "source_url_gitlab":
        return ctx.source_urls.gitlab or ""
    if key == "source_url_rootly":
        return ctx.source_urls.rootly or ""
    if key == "source_url_github_org":
        return ctx.source_urls.github_org or ""

    raise KeyError(key)


# Stable set of template keys the helper exposes. Prompts may
# reference any of these; unknown placeholders raise ``KeyError`` at
# render time.
_KNOWN_KEYS: tuple[str, ...] = (
    "organization_label",
    "services_csv",
    "services_bullets",
    "environments_csv",
    "environments_bullets",
    "key_repos_csv",
    "key_repos_bullets",
    "ticket_prefix",
    "tracker_prefix",
    "ticket_web_base",
    "k8s_namespaces_csv",
    "k8s_clusters_csv",
    "k8s_pod_label_selector",
    "gcp_projects_csv",
    "source_url_confluence",
    "source_url_slack",
    "source_url_gitlab",
    "source_url_rootly",
    "source_url_github_org",
)


# ---------------------------------------------------------------------------
# Process-level active deployment context.
#
# Prompt modules are imported once and their templates are module-level
# constants, so threading a DeploymentContext through every node signature
# would touch the whole graph. Instead the app factory sets the active
# context ONCE at startup (set_active_deployment(settings.deployment)) and
# prompt modules call ``render(TEMPLATE)`` with no explicit ctx. This mirrors
# the existing process-global pattern used by ``opsrag.usage.tracker``.
#
# Default is an empty DeploymentContext: unset deployments render every
# placeholder as the empty string, which is exactly the behaviour prompt
# authors are told to write for (Constitution Principle VI). Tests and tools
# that never call the setter therefore get a safe, org-free render.
# ---------------------------------------------------------------------------
_active_deployment: DeploymentContext | None = None


def set_active_deployment(ctx: DeploymentContext | None) -> None:
    """Install the process-wide deployment context used by ``render`` when
    no explicit ``ctx`` is passed. Call once at app startup."""
    global _active_deployment
    _active_deployment = ctx


def active_deployment() -> DeploymentContext:
    """Return the active deployment context, or an empty one if unset."""
    return _active_deployment if _active_deployment is not None else DeploymentContext()


# Live, UI-editable override of the operator custom-instructions. ``None`` =
# no DB override -> fall back to the config seed (deployment.custom_instructions).
# A string (even "") = the operator set it via the admin "Agent Guidance" page;
# it wins over the config seed. Refreshed from Postgres on a short interval +
# updated immediately on PUT, so edits take effect on the next query, no restart.
_custom_instructions_live: str | None = None


def set_custom_instructions_live(value: str | None) -> None:
    """Install (or clear, with None) the live custom-instructions override."""
    global _custom_instructions_live
    _custom_instructions_live = value


def current_custom_instructions() -> str:
    """The effective custom-instructions text (live override if set, else the
    config seed). Empty string when neither is set."""
    if _custom_instructions_live is not None:
        return _custom_instructions_live
    try:
        return active_deployment().custom_instructions or ""
    except Exception:
        return ""


def custom_instructions_block() -> str:
    """Operator custom-instructions as a system-prompt addendum, or "" when
    unset. Appended to the answer + chat system prompts so deployment-wide
    guidance / edge-case rules ALWAYS apply, on top of retrieval. Reads the
    live (UI-editable) value, falling back to the config seed. Never raises."""
    ci = current_custom_instructions().strip()
    if not ci:
        return ""
    return (
        "\n\n## Operator guidance (deployment-specific -- always honor this)\n"
        f"{ci}"
    )


def render(
    template: str,
    *,
    ctx: DeploymentContext | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    """Render ``template`` with values projected from ``ctx``.

    Standard ``str.format`` placeholder syntax: ``{key}`` interpolates
    a value; ``{{`` and ``}}`` escape literal braces. Unknown
    placeholders raise ``KeyError`` -- prompts MUST only reference keys
    in ``_KNOWN_KEYS`` or supply them via ``extra``.

    ``ctx`` defaults to the process-wide active deployment context (see
    :func:`set_active_deployment`); pass it explicitly in tests or when
    rendering for a specific deployment.

    ``extra`` is a small escape hatch for call-site values that aren't
    deployment facts (e.g. the current date string, a retry counter).
    Use sparingly; deployment knowledge belongs on ``DeploymentContext``.
    """
    if ctx is None:
        ctx = active_deployment()
    view = _ContextView(ctx, extra=extra)
    return template.format_map(view)


def known_keys() -> tuple[str, ...]:
    """Return the stable set of template keys ``render`` exposes."""
    return _KNOWN_KEYS
