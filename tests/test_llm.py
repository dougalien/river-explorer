import json

import pytest
import requests

import llm


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(content="hi", tool_calls=None, tokens=30):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return FakeResponse(200, {"choices": [{"message": msg}], "usage": {"total_tokens": tokens}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def client(session, model="gpt-oss:120b", budgets=(), fallback_models=(), base_url="https://ollama.com/v1"):
    cfg = llm.LLMConfig(api_key="sk-secret-123", model=model, fallback_models=fallback_models, base_url=base_url)
    return llm.ChatClient(cfg, budgets=budgets, session=session)


def test_request_shape_and_reply_parsing():
    s = FakeSession(ok("hello"))
    reply = client(s).complete([{"role": "user", "content": "yo"}], max_tokens=77, temperature=0.2)
    sent = s.requests[0]
    assert sent["url"] == "https://ollama.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-secret-123"
    assert sent["json"]["max_tokens"] == 77 and sent["json"]["temperature"] == 0.2
    assert sent["json"]["model"] == "gpt-oss:120b"
    assert reply.content == "hello" and reply.tokens == 30 and not reply.tool_calls


def test_openai_base_url_uses_its_token_field():
    s = FakeSession(ok())
    client(s, model="gpt-4.1-mini", base_url="https://api.openai.com/v1").complete([], max_tokens=9)
    assert s.requests[0]["json"]["max_completion_tokens"] == 9 and "max_tokens" not in s.requests[0]["json"]


def test_next_model_answers_when_the_first_is_over_its_limit():
    s = FakeSession(FakeResponse(429), FakeResponse(429), ok("from the backup"))
    reply = client(s, fallback_models=("gpt-oss:20b",)).complete([])
    assert reply.content == "from the backup"
    assert [r["json"]["model"] for r in s.requests] == ["gpt-oss:120b", "gpt-oss:120b", "gpt-oss:20b"]


def test_paid_only_model_is_skipped_without_retry():
    s = FakeSession(FakeResponse(403), ok("small model"))
    assert client(s, fallback_models=("gemma3:4b",)).complete([]).content == "small model"
    assert len(s.requests) == 2


def test_every_model_failing_raises_so_the_app_uses_its_built_in_guide():
    s = FakeSession(FakeResponse(429), FakeResponse(429), FakeResponse(403))
    with pytest.raises(llm.LLMError) as e:
        client(s, fallback_models=("gemma3:4b",)).complete([])
    assert "unavailable" in str(e.value)


def test_reasoning_models_get_no_temperature():
    s = FakeSession(ok())
    client(s, model="gpt-5.4-mini").complete([{"role": "user", "content": "x"}], temperature=0.2)
    assert "temperature" not in s.requests[0]["json"]


def test_tool_calls_are_parsed_and_bad_arguments_become_empty():
    calls = [{"id": "c1", "type": "function", "function": {"name": "search_gages", "arguments": '{"state": "MA"}'}},
             {"id": "c2", "type": "function", "function": {"name": "search_gages", "arguments": "not json"}}]
    reply = client(FakeSession(ok(content=None, tool_calls=calls))).complete([{"role": "user", "content": "x"}])
    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [
        ("c1", "search_gages", {"state": "MA"}), ("c2", "search_gages", {})]
    assert reply.message["tool_calls"] == calls


def test_budget_stops_calls_and_counts_tokens():
    b = llm.Budget(max_calls=2, max_tokens=None)
    c = client(FakeSession(ok(), ok(), ok()), budgets=[b])
    c.complete([{"role": "user", "content": "1"}])
    c.complete([{"role": "user", "content": "2"}])
    assert b.calls == 2 and b.tokens == 60 and b.left == 0
    with pytest.raises(llm.BudgetExceeded):
        c.complete([{"role": "user", "content": "3"}])


def test_token_budget_and_daily_reset():
    b = llm.Budget(max_calls=100, max_tokens=50, daily=True)
    b.record(60)
    with pytest.raises(llm.BudgetExceeded):
        b.check()
    b._day = b._day.replace(year=b._day.year - 1)  # pretend a new day has started
    b.check()
    assert b.calls == 0


def test_errors_are_friendly_and_never_leak_the_key():
    with pytest.raises(llm.LLMError) as e:
        client(FakeSession(FakeResponse(401, {"error": {"message": "bad key sk-secret-123"}}))).complete([])
    assert "sk-secret-123" not in str(e.value) and "key" in str(e.value).lower()
    with pytest.raises(llm.LLMError) as e:
        client(FakeSession(FakeResponse(400, {"error": {"message": "model gpt-9 does not exist"}}))).complete([])
    assert "gpt-9" in str(e.value)


def test_one_retry_on_busy_then_success_or_clear_failure():
    assert client(FakeSession(FakeResponse(429), ok("fine"))).complete([]).content == "fine"
    with pytest.raises(llm.LLMError) as e:
        client(FakeSession(FakeResponse(429), FakeResponse(429))).complete([])
    assert "busy" in str(e.value) or "quota" in str(e.value)
    with pytest.raises(llm.LLMError):
        client(FakeSession(requests.ConnectionError("down"), requests.ConnectionError("down"))).complete([])


def test_malformed_success_body():
    with pytest.raises(llm.LLMError):
        client(FakeSession(FakeResponse(200, {"nope": 1}))).complete([])


def test_model_without_tool_support_hands_over_to_the_next():
    s = FakeSession(FakeResponse(400, {"error": {"message": "gemma3:4b does not support tools"}}), ok("tools ok"))
    assert client(s, model="gemma3:4b", fallback_models=("gpt-oss:20b",)).complete([], tools=[{"type": "function"}]).content == "tools ok"
