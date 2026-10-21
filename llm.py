"""A small chat client for any OpenAI-compatible API, plus spending limits.

Plain HTTP on purpose: no vendor SDK to break when its API changes. The default is Ollama Cloud's free models
(https://ollama.com/v1). When a model is busy, over its free limit, or needs a paid plan, the next model in
`fallback_models` is tried; when all of them fail, LLMError tells the app to use its built-in guide.
Nothing here imports Streamlit.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

import requests

try:  # trust the operating system's certificate store; Python's bundled CA list can be stale
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


class LLMError(RuntimeError):
    """The AI helper could not answer. The message is safe to show to a student (it never contains the key)."""


class BudgetExceeded(LLMError):
    """A call or token limit has been reached."""


class ModelUnavailable(LLMError):
    """This model could not answer (busy, over its free limit, or not offered); another model may still work."""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = "gpt-oss:120b"
    base_url: str = "https://ollama.com/v1"
    timeout: float = 45.0
    fallback_models: tuple[str, ...] = ()  # tried in order when `model` is busy, over its limit, or unavailable


class Budget:
    """Counts AI calls and tokens. One per student session and one for the whole server (daily=True resets at midnight)."""

    def __init__(self, max_calls: int, max_tokens: int | None = None, daily: bool = False, what: str = "session"):
        self.max_calls, self.max_tokens, self.daily, self.what = max_calls, max_tokens, daily, what
        self.calls = 0
        self.tokens = 0
        self._day = date.today()
        self._lock = threading.Lock()

    def _roll(self) -> None:
        if self.daily and date.today() != self._day:
            self._day, self.calls, self.tokens = date.today(), 0, 0

    def check(self) -> None:
        with self._lock:
            self._roll()
            if self.calls >= self.max_calls or (self.max_tokens and self.tokens >= self.max_tokens):
                raise BudgetExceeded(f"The AI helper has reached its limit for this {self.what}.")

    def record(self, tokens: int) -> None:
        with self._lock:
            self._roll()
            self.calls += 1
            self.tokens += tokens

    @property
    def left(self) -> int:
        with self._lock:
            self._roll()
            return max(self.max_calls - self.calls, 0)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall]
    message: dict  # the assistant message, ready to append to the conversation
    tokens: int = 0


@dataclass
class ChatClient:
    config: LLMConfig
    budgets: Sequence[Budget] = field(default_factory=tuple)
    session: Any = None  # a requests-like object; tests pass a fake

    def complete(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str | None = None,
                 max_tokens: int = 500, json_mode: bool = False, temperature: float | None = None) -> Reply:
        for b in self.budgets:
            b.check()
        cfg = self.config
        body: dict[str, Any] = {"model": cfg.model, "messages": messages}
        if "api.openai.com" in cfg.base_url:
            body["max_completion_tokens"] = max_tokens
        else:  # Ollama and most other OpenAI-compatible services take the older name
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            if tool_choice:
                body["tool_choice"] = tool_choice
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if temperature is not None and not cfg.model.startswith(("gpt-5", "o1", "o3", "o4")):
            body["temperature"] = temperature
        data = self._post_with_fallback(body)
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("The AI helper sent back something unexpected.") from exc
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except (ValueError, KeyError):
                args = {}
            calls.append(ToolCall(id=tc.get("id", ""), name=tc["function"].get("name", ""),
                                  arguments=args if isinstance(args, dict) else {}))
        tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
        for b in self.budgets:
            b.record(tokens)
        return Reply(content=(msg.get("content") or "").strip(), tool_calls=calls, message=msg, tokens=tokens)

    def _post_with_fallback(self, body: dict) -> dict:
        """Try the main model, then each fallback model, until one answers."""
        models = [self.config.model] + [m for m in self.config.fallback_models if m != self.config.model]
        error: LLMError | None = None
        for model in models:
            try:
                return self._post({**body, "model": model})
            except ModelUnavailable as exc:
                error = exc
        raise LLMError(str(error) if error else "The AI helper is unavailable.")

    def _post(self, body: dict) -> dict:
        cfg = self.config
        http = self.session or requests
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        last = "unknown problem"
        for attempt in range(2):
            try:
                r = http.post(url, headers=headers, json=body, timeout=cfg.timeout)
            except requests.RequestException as exc:
                last = f"could not reach the AI service ({type(exc).__name__})"
                time.sleep(1.5)
                continue
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    last = "the AI service sent back something that was not JSON"
                    continue
            if r.status_code == 401:
                raise LLMError("The AI helper's key was rejected. Ask your instructor to check it.")
            if r.status_code in (402, 403, 404):  # this model needs a paid plan or no longer exists
                raise ModelUnavailable(f"The AI helper is unavailable: model {body.get('model')} cannot be used (HTTP {r.status_code}).")
            if r.status_code == 429 or r.status_code >= 500:
                last = "the AI service is busy or out of quota" if r.status_code == 429 else "the AI service had a hiccup"
                time.sleep(1.5)
                continue
            detail = ""
            try:
                detail = str(r.json().get("error", {}).get("message", ""))[:160]
            except (ValueError, AttributeError):
                pass
            if "does not support" in detail:  # e.g. a small model without tool calling: let the next model try
                raise ModelUnavailable(f"The AI helper is unavailable: {detail}")
            raise LLMError(f"The AI helper rejected the request (HTTP {r.status_code}). {detail}".strip())
        raise ModelUnavailable(f"The AI helper is unavailable: {last}.")


def check_model(client: ChatClient) -> str:
    """One tiny call that proves the key and model name work. Returns the model's reply."""
    return client.complete([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=16).content
