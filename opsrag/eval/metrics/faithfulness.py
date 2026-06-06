"""FaithfulnessJudgeMetric -- LLM-as-judge via VertexGeminiJudge (Path B).

DeepEval ships a `FaithfulnessMetric` that defaults to OpenAI for the
judge. We can't use that (no OpenAI billing path on this project, and
the user's constraint is to stay on Vertex). Instead we wrap the same
underlying logic with our `VertexGeminiJudge` -- sends a structured-output
prompt to Gemini Pro asking for a 0-1 grounded score plus reasoning.

Faithfulness here = "every factual claim in the answer is supported by
the retrieval_context". 1.0 = fully grounded. 0.5 = partially grounded.
0.0 = hallucinated. Vacuous answers ("I don't know") score 1.0 (nothing
to fabricate).

Multi-shot averaging (Sprint 1 P0):
  Set OPSRAG_EVAL_FAITH_SHOTS=N (default 1) to run the judge N times
  concurrently and average the scores. N=3 typically halves sigma on
  borderline goldens. Cost is Nx judge calls; latency stays the same
  via asyncio.gather.
"""
from __future__ import annotations

import asyncio
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, Field

from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge

_PROMPT_TEMPLATE = """You are an evaluator scoring whether an AI answer is
factually grounded in the provided context. Return JSON only.

# Question
{question}

# Provided context (retrieved by RAG)
{context}

# AI answer
{answer}

# Task
Score on a 0.0-1.0 scale:
- 1.0 -- every factual claim in the answer is directly supported by the context.
- 0.7 -- most claims supported, minor extrapolations.
- 0.5 -- partial support; some claims unsupported but not contradictory.
- 0.3 -- major unsupported claims.
- 0.0 -- answer fabricates facts not in context, or contradicts it.

Hedging answers like "I don't know" or "the context does not contain..."
score 1.0 (refusing to fabricate is grounded behavior).

Return: {{"score": float, "reasoning": "1-2 sentence justification"}}"""


class _FaithfulnessResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class FaithfulnessJudgeMetric(BaseMetric):
    def __init__(
        self,
        judge: VertexGeminiJudge | None = None,
        threshold: float = 0.7,
    ):
        self.judge = judge or VertexGeminiJudge()
        self.threshold = threshold
        self.score: float = 0.0
        self.success: bool = False
        self.reason: str = ""
        self.error: str | None = None

    def _build_prompt(self, test_case: LLMTestCase) -> str:
        ctx = "\n---\n".join(test_case.retrieval_context or []) or "(empty)"
        return _PROMPT_TEMPLATE.format(
            question=test_case.input,
            context=ctx[:8000],  # cap to keep prompt under judge's context budget
            answer=test_case.actual_output or "(empty)",
        )

    def _shots(self) -> int:
        try:
            n = int(os.getenv("OPSRAG_EVAL_FAITH_SHOTS", "1"))
        except ValueError:
            n = 1
        return max(1, n)

    def measure(self, test_case: LLMTestCase) -> float:
        prompt = self._build_prompt(test_case)
        n = self._shots()
        try:
            if n == 1:
                result = self.judge.generate_schema(prompt, _FaithfulnessResponse)
                self.score = float(result.score)
                self.reason = result.reasoning
            else:
                # Multi-shot: N parallel sync calls via ThreadPoolExecutor.
                # ThreadPool keeps each call sync (no asyncio.run() event-loop
                # teardown noise) but fires concurrently -> wall time stays
                # ~1x single-shot judge latency, not Nx. sigma reduces ~sqrt(N).
                # Resilience: per-shot exceptions (Vertex 503, parse errors)
                # are excluded from the mean rather than counted as 0.0.
                # Only if ALL shots fail do we surface an error.
                def _one_shot(_):
                    try:
                        return self.judge.generate_schema(prompt, _FaithfulnessResponse)
                    except Exception as e:
                        return e
                with ThreadPoolExecutor(max_workers=n) as ex:
                    raw = list(ex.map(_one_shot, range(n)))
                results = [r for r in raw if not isinstance(r, Exception)]
                errors = [r for r in raw if isinstance(r, Exception)]
                if not results:
                    # All N shots failed -- surface the error.
                    raise errors[0]
                scores = [float(r.score) for r in results]
                self.score = statistics.mean(scores)
                ordered = sorted(zip(scores, results), key=lambda x: x[0])
                median_reason = ordered[len(ordered) // 2][1].reasoning
                err_note = f" errors={len(errors)}/{n}" if errors else ""
                self.reason = (
                    f"[shots={len(results)}/{n}{err_note} mean={self.score:.2f} "
                    f"scores={scores}] {median_reason}"
                )
        except Exception as exc:
            self.score = 0.0
            self.reason = f"judge error: {exc}"
            self.error = str(exc)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        prompt = self._build_prompt(test_case)
        n = self._shots()
        try:
            if n == 1:
                result = await self.judge.a_generate_schema(prompt, _FaithfulnessResponse)
                self.score = float(result.score)
                self.reason = result.reasoning
            else:
                results = await asyncio.gather(*[
                    self.judge.a_generate_schema(prompt, _FaithfulnessResponse)
                    for _ in range(n)
                ])
                scores = [float(r.score) for r in results]
                self.score = statistics.mean(scores)
                ordered = sorted(zip(scores, results), key=lambda x: x[0])
                median_reason = ordered[len(ordered) // 2][1].reasoning
                self.reason = (
                    f"[shots={n} mean={self.score:.2f} scores={scores}] "
                    f"{median_reason}"
                )
        except Exception as exc:
            self.score = 0.0
            self.reason = f"judge error: {exc}"
            self.error = str(exc)
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Faithfulness"
