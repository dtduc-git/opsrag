"""Launch indexing as an ephemeral k8s Job (with an in-process dev fallback).

In production the API no longer indexes in-process (which forced the single
always-on ``indexer`` deployment). Instead ``POST /index/repo`` asks this
launcher to create a run-to-completion k8s Job. Backend pods stay pure-serving;
the durable Postgres job-state (migration 0009) is how the UI tracks progress.

**Source of truth = a CronJob.** The Helm chart ships one CronJob whose
``jobTemplate`` carries the full indexing pod spec (image, config volume,
secret/env, service account, resources, ttl). The launcher creates an ad-hoc
Job by CLONING that ``jobTemplate`` and overriding only the container args --
exactly what ``kubectl create job --from=cronjob/<name>`` does. This keeps the
ad-hoc Job byte-for-byte consistent with the scheduled reindex and means the
API never hand-builds (and drifts from) the pod spec.

Mode selection (``OPSRAG_INDEX_JOB_MODE``):
  - ``k8s``        -> always create a Job (errors if client/RBAC/cronjob missing)
  - ``inprocess``  -> always run in-process (legacy/dev behaviour)
  - ``auto`` (default) -> Job when in-cluster (KUBERNETES_SERVICE_HOST) AND the
    kubernetes_asyncio client imports AND a template CronJob is configured;
    otherwise in-process.

Job-create requires a narrow Role: get the template CronJob + create/list Jobs
in the namespace, bound to the API ServiceAccount (see the Helm rbac template).
"""
from __future__ import annotations

import logging
import os
import re

_log = logging.getLogger("opsrag.job.launcher")


def _slug(s: str) -> str:
    """k8s-safe name fragment (lowercase alnum + '-', <=40 chars)."""
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return (s[:40] or "x").strip("-")


class JobLauncher:
    """Creates indexing Jobs by cloning a template CronJob. Construct via
    :meth:`from_env`; ``None`` means 'no launcher -> caller runs in-process'."""

    def __init__(self, *, namespace: str, cronjob_name: str,
                 container_name: str = "indexer") -> None:
        self._namespace = namespace
        self._cronjob = cronjob_name
        self._container = container_name

    # -- construction --------------------------------------------------------
    @classmethod
    def from_env(cls) -> JobLauncher | None:
        mode = (os.environ.get("OPSRAG_INDEX_JOB_MODE") or "auto").strip().lower()
        if mode == "inprocess":
            return None
        in_cluster = bool(os.environ.get("KUBERNETES_SERVICE_HOST"))
        if mode == "auto" and not in_cluster:
            return None
        try:
            import kubernetes_asyncio  # noqa: F401
        except Exception:
            if mode == "k8s":
                _log.error("OPSRAG_INDEX_JOB_MODE=k8s but kubernetes_asyncio is not "
                           "installed; indexing cannot launch Jobs")
            return None
        cronjob = os.environ.get("OPSRAG_JOB_CRONJOB_NAME")
        if not cronjob:
            if mode == "k8s":
                _log.error("OPSRAG_INDEX_JOB_MODE=k8s but OPSRAG_JOB_CRONJOB_NAME is "
                           "unset; cannot clone a Job template")
            else:
                _log.warning("no OPSRAG_JOB_CRONJOB_NAME; indexing runs in-process")
            return None
        namespace = (os.environ.get("OPSRAG_JOB_NAMESPACE")
                     or _read_namespace() or "default")
        return cls(
            namespace=namespace,
            cronjob_name=cronjob,
            container_name=os.environ.get("OPSRAG_JOB_CONTAINER", "indexer"),
        )

    # -- public API ----------------------------------------------------------
    async def launch_repo(self, repo: str, branch: str) -> str:
        return await self._launch(
            name_hint=f"idx-{_slug(repo)}-{_slug(branch)}",
            args=["--repo", repo, "--branch", branch],
            labels={"opsrag.io/index-repo": _slug(repo), "opsrag.io/index-branch": _slug(branch)},
        )

    async def launch_source(self, source_type: str, scope: str) -> str:
        return await self._launch(
            name_hint=f"idx-{_slug(source_type)}-{_slug(scope)}",
            args=["--source", source_type, "--scope", scope],
            labels={"opsrag.io/index-source": _slug(source_type), "opsrag.io/index-scope": _slug(scope)},
        )

    async def launch_all(self) -> str:
        return await self._launch(name_hint="idx-all", args=["--all"],
                                  labels={"opsrag.io/index-all": "true"})

    # -- internals -----------------------------------------------------------
    async def _launch(self, *, name_hint: str, args: list[str], labels: dict) -> str:
        from kubernetes_asyncio import client, config

        try:
            config.load_incluster_config()
        except Exception:
            await config.load_kube_config()  # dev-with-kubeconfig

        all_labels = {"app.kubernetes.io/name": "opsrag",
                      "opsrag.io/component": "index-job", **labels}
        async with client.ApiClient() as api:
            batch = client.BatchV1Api(api)
            # Clone the template CronJob's jobTemplate -> a fresh Job.
            cj = await batch.read_namespaced_cron_job(self._cronjob, self._namespace)
            job_spec = cj.spec.job_template.spec
            # Override the indexer container's args (job-indexer + target).
            containers = job_spec.template.spec.containers or []
            target = next((c for c in containers if c.name == self._container),
                          containers[0] if containers else None)
            if target is None:
                raise RuntimeError(
                    f"template CronJob {self._cronjob} has no containers to launch")
            target.args = ["job-indexer", *args]
            # generateName -> unique suffix so re-indexing the same repo never
            # collides with an in-flight Job.
            job = client.V1Job(
                api_version="batch/v1", kind="Job",
                metadata=client.V1ObjectMeta(generate_name=f"{name_hint}-"[:50],
                                             labels=all_labels),
                spec=job_spec,
            )
            created = await batch.create_namespaced_job(self._namespace, job)
        name = created.metadata.name
        _log.info("launched indexing Job %s/%s (from cronjob %s) args=%s",
                  self._namespace, name, self._cronjob, args)
        return name


def _read_namespace() -> str | None:
    try:
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
            return f.read().strip() or None
    except Exception:
        return None
