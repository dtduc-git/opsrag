"""DeepEval LLM adapters -- bridge to OpsRAG's existing LLM stack.

Custom adapters used so eval calls hit the same Vertex SDK + ADC + quota
project as production. No LangChain in any path.
"""
from opsrag.eval.adapters.vertex_judge import VertexGeminiJudge

__all__ = ["VertexGeminiJudge"]
