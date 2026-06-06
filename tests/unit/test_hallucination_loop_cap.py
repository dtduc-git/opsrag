"""Regression: the hallucination regenerate loop must be bounded by
max_retries. Previously check_hallucination never incremented retry_count, so
a persistently-not-grounded verdict looped generate->verify->check forever."""
from opsrag.agent.nodes.hallucination import check_hallucination_node, hallucination_decision


class _LLM:
    """Always says 'not grounded' (the pathological case that used to loop)."""
    async def generate_structured(self, **kw):
        class R:  # noqa
            grounded = False
        return R()


class _Obs:
    async def log(self, *a, **k): ...


def test_decision_caps_at_max_retries():
    assert hallucination_decision({"generation_grounded": True}) == "grounded"
    assert hallucination_decision({"generation_grounded": False, "retry_count": 0, "max_retries": 2}) == "not_grounded"
    assert hallucination_decision({"generation_grounded": False, "retry_count": 2, "max_retries": 2}) == "max_retries_hit"


async def test_not_grounded_increments_retry_count():
    node = check_hallucination_node(_LLM(), _Obs())
    # simulate the loop: each not-grounded pass must bump retry_count
    state = {"generation": "an answer", "graded_chunks": [], "retry_count": 0, "max_retries": 2}
    for expected in (1, 2):
        out = await node(state)
        assert out["generation_grounded"] is False
        assert out["retry_count"] == expected
        state = {**state, **out}
    # now the decision must stop the loop
    assert hallucination_decision(state) == "max_retries_hit"


async def test_empty_answer_also_counts():
    node = check_hallucination_node(_LLM(), _Obs())
    out = await node({"generation": "", "retry_count": 1, "max_retries": 2})
    assert out["retry_count"] == 2
