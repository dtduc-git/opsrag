import asyncio

from opsrag.llms.anthropic import AnthropicLLM
from opsrag.llms.content import ImagePart, build_user_content

PNG = b"\x89PNG\r\nfake"


def _msgs():
    return [{"role": "user", "content": build_user_content("q", [ImagePart(PNG, "image/png")])}]


def test_anthropic_builds_image_block(monkeypatch):
    captured = {}

    class _Resp:
        class _Usage:
            input_tokens = 1
            output_tokens = 1
        content = []
        model = "claude-sonnet-4-20250514"
        usage = _Usage()

    class _Messages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Client:
        messages = _Messages()

    llm = AnthropicLLM(api_key="x", model="claude-sonnet-4-20250514")
    monkeypatch.setattr(llm, "_get_client", lambda: _Client())

    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][0]["content"]
    assert isinstance(sent, list)
    assert sent[1]["type"] == "image"
    assert sent[1]["source"]["media_type"] == "image/png"


def test_bedrock_builds_image_block(monkeypatch):
    from opsrag.llms.bedrock import BedrockLLM
    captured = {}

    def _converse(**kwargs):
        captured.update(kwargs)
        return {"output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1}}

    llm = BedrockLLM.__new__(BedrockLLM)            # skip boto3 __init__
    llm._client = type("C", (), {"converse": staticmethod(_converse)})()
    llm._model = "anthropic.claude-sonnet-4-20250514-v1:0"
    llm._default_max_tokens = 4096

    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][0]["content"]
    assert sent[1]["image"]["format"] == "png"
    assert sent[1]["image"]["source"]["bytes"] == PNG


def test_openai_builds_image_url(monkeypatch):
    from opsrag.llms.openai import OpenAILLM
    captured = {}

    class _Msg:
        content = "ok"
    class _Choice:
        message = _Msg()
    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        model = "gpt-4o"
    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()
    class _Chat:
        completions = _Completions()
    class _Client:
        chat = _Chat()

    llm = OpenAILLM(api_key="x", model="gpt-4o")
    monkeypatch.setattr(llm, "_get_client", lambda: _Client())

    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][-1]["content"]      # last = the user message
    assert sent[1]["type"] == "image_url"
    assert sent[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_litellm_builds_image_url(monkeypatch):
    import sys
    import types
    captured = {}

    async def _acompletion(**kwargs):
        captured.update(kwargs)
        class _Msg:
            content = "ok"
        class _Choice:
            message = _Msg()
        class _Usage:
            prompt_tokens = 1
            completion_tokens = 1
        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            model = "gemini/gemini-3-flash-preview"
        return _Resp()

    fake = types.ModuleType("litellm")
    fake.acompletion = _acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)

    from opsrag.llms.litellm_provider import LiteLLMLLM
    llm = LiteLLMLLM(model="gemini/gemini-3-flash-preview")

    asyncio.run(llm.generate(messages=_msgs()))

    sent = captured["messages"][-1]["content"]
    assert sent[1]["type"] == "image_url"


def test_vertex_claude_builds_image_block(monkeypatch):
    from opsrag.llms.vertex import VertexAILLM

    captured = {}

    class _Resp:
        class _Usage:
            input_tokens = 1
            output_tokens = 1
        content = []
        model = "claude-sonnet-4@20250514"
        usage = _Usage()

    def _create(**kwargs):
        captured.update(kwargs)
        return _Resp()

    async def _noop():
        return None

    llm = VertexAILLM.__new__(VertexAILLM)
    llm._model = "claude-sonnet-4@20250514"
    llm._is_claude = True
    llm._project = "p"
    llm._location = "us-east5"
    llm._client = type("C", (), {"messages": type("M", (), {"create": staticmethod(_create)})()})()
    monkeypatch.setattr(llm, "_get_client", lambda: llm._client)
    monkeypatch.setattr(llm, "_fire_on_usage", lambda **k: _noop())

    asyncio.run(llm.generate(messages=_msgs()))
    sent = captured["messages"][0]["content"]
    assert sent[1]["type"] == "image" and sent[1]["source"]["type"] == "base64"
