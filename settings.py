"""Where the app finds its settings: environment variables, then a local .env file, then Streamlit secrets.

Secrets (the AI key, the class passcode, an optional USGS key) never go in the code or in the browser.
Locally, put them in .env next to app.py. On Streamlit Community Cloud, put them in the app's Secrets box.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from llm import Budget, ChatClient, LLMConfig

load_dotenv(Path(__file__).with_name(".env"))


def get(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value and str(value).strip():
            return str(value).strip()
    except Exception:  # no secrets file, or not running under Streamlit
        pass
    return default


def get_int(name: str, default: int) -> int:
    try:
        return int(get(name, str(default)))
    except ValueError:
        return default


DEFAULT_MODELS = "gpt-oss:120b, gpt-oss:20b, gemma3:4b"   # free Ollama Cloud models, tried in this order


def llm_config() -> LLMConfig | None:
    """None when no AI key is set; the app then runs on plain keyword search and the built-in guide."""
    key = get("OLLAMA_API_KEY") or get("AI_API_KEY")
    if not key:
        return None
    models = [m.strip() for m in get("AI_MODELS", get("OLLAMA_MODELS", DEFAULT_MODELS)).split(",") if m.strip()]
    return LLMConfig(api_key=key, model=models[0], fallback_models=tuple(models[1:]),
                     base_url=get("AI_BASE_URL", "https://ollama.com/v1"))


def make_client(config: LLMConfig, session_budget: Budget, server_budget: Budget) -> ChatClient:
    return ChatClient(config=config, budgets=(session_budget, server_budget))


def passcode() -> str:
    """The class passcode that unlocks the AI helper; empty means anyone may use it."""
    return get("CLASS_PASSCODE")


def export_usgs_key() -> None:
    key = get("USGS_API_KEY")
    if key:
        os.environ["USGS_API_KEY"] = key
