"""Regression: the hallucination regenerate loop must be bounded -- by its OWN
counter (regen_count / max_regens), separate from the CRAG rewrite's retry_count
so the two loops don't cannibalize one shared budget. Previously it never
incremented anything, so a persistently-not-grounded verdict looped forever."""
from opsrag.agent.nodes.hallucination import check_hallucination_node, hallucination_decision


class _LLM:
    """Always says 'not grounded' (the pathological case that used to loop)."""
    async def generate_structured(self, **kw):
        class R:  # noqa
            grounded = False
        return R()


class _Obs:
    async def log(self, *a, **k): ...


def test_decision_caps_at_max_regens():
    assert hallucination_decision({"generation_grounded": True}) == "grounded"
    assert hallucination_decision({"generation_grounded": False, "regen_count": 0, "max_regens": 2}) == "not_grounded"
    assert hallucination_decision({"generation_grounded": False, "regen_count": 2, "max_regens": 2}) == "max_retries_hit"


async def test_not_grounded_increments_regen_count():
    node = check_hallucination_node(_LLM(), _Obs())
    # simulate the loop: each not-grounded pass must bump regen_count (NOT the
    # CRAG rewrite's retry_count -- the budgets are independent now)
    state = {"generation": "an answer", "graded_chunks": [], "regen_count": 0, "max_regens": 2}
    for expected in (1, 2):
        out = await node(state)
        assert out["generation_grounded"] is False
        assert out["regen_count"] == expected
        assert "retry_count" not in out  # rewrite budget untouched
        state = {**state, **out}
    # now the decision must stop the loop
    assert hallucination_decision(state) == "max_retries_hit"


async def test_empty_answer_also_counts():
    node = check_hallucination_node(_LLM(), _Obs())
    out = await node({"generation": "", "regen_count": 1, "max_regens": 2})
    assert out["regen_count"] == 2
