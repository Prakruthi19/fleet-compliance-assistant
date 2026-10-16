"""
Thin, provider-agnostic LLM client: chat() for structured calls (router,
eval question generation) and stream() for answers.

Ollama is called through its native API rather than the OpenAI-compatible
one because only the native API exposes `think: false` (qwen3.x otherwise
spends hundreds of tokens reasoning — at ~7 tok/s on CPU that is the single
biggest latency cost), `keep_alive`, and JSON-schema constrained output.
"""
import json
import time
from functools import lru_cache
from typing import Iterator

import httpx

from regrag import config


def _ollama_body(messages: list[dict], schema: dict | None, stream: bool) -> dict:
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "think": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    if schema:
        body["format"] = schema
    return body


@lru_cache
def _azure():
    import os
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


def _azure_model() -> str:
    import os
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


@lru_cache
def _openai():
    """Any OpenAI-compatible host (Groq, Together, OpenRouter, ...)."""
    from openai import OpenAI
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, max_retries=8)  # free-tier 429s


def _openai_kwargs() -> dict:
    kwargs = {"max_tokens": config.LLM_MAX_TOKENS}
    if config.LLM_REASONING_EFFORT:
        kwargs["reasoning_effort"] = config.LLM_REASONING_EFFORT
    return kwargs


def chat(messages: list[dict], schema: dict | None = None) -> str:
    """One complete response. With `schema`, output is constrained to that JSON schema."""
    if config.LLM_PROVIDER == "azure":
        kwargs = {}
        if schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "out", "schema": schema}}
        resp = _azure().chat.completions.create(model=_azure_model(), messages=messages,
                                                temperature=0, **kwargs)
        return resp.choices[0].message.content
    if config.LLM_PROVIDER == "openai":
        kwargs = _openai_kwargs()
        if schema:
            # Strict json_schema support varies by host/model; JSON mode plus
            # the schema in the prompt works everywhere.
            messages = [{"role": "system", "content": "Reply with only a JSON object matching this schema:\n"
                         + json.dumps(schema)}] + messages
            kwargs["response_format"] = {"type": "json_object"}
        resp = _openai().chat.completions.create(model=config.LLM_MODEL, messages=messages,
                                                 temperature=0, **kwargs)
        return resp.choices[0].message.content
    for attempt in range(3):
        resp = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=_ollama_body(messages, schema, False),
                          timeout=600)
        # 500s here are almost always a failed model load under memory
        # pressure; a retry once memory frees up usually succeeds.
        if resp.status_code != 500 or attempt == 2:
            break
        time.sleep(10 * (attempt + 1))
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def chat_json(messages: list[dict], schema: dict) -> dict:
    return json.loads(chat(messages, schema))


def stream(messages: list[dict]) -> Iterator[str]:
    """Yield answer tokens as they are generated."""
    if config.LLM_PROVIDER == "azure":
        for ev in _azure().chat.completions.create(model=_azure_model(), messages=messages,
                                                   temperature=0, stream=True):
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content
        return
    if config.LLM_PROVIDER == "openai":
        for ev in _openai().chat.completions.create(model=config.LLM_MODEL, messages=messages,
                                                    temperature=0, stream=True, **_openai_kwargs()):
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content
        return
    with httpx.stream("POST", f"{config.OLLAMA_URL}/api/chat",
                      json=_ollama_body(messages, None, True), timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                msg = json.loads(line)
                if piece := msg.get("message", {}).get("content"):
                    yield piece


def warm() -> None:
    """Load the model into memory ahead of the first user request."""
    if config.LLM_PROVIDER == "ollama":
        httpx.post(f"{config.OLLAMA_URL}/api/chat",
                   json={"model": config.LLM_MODEL, "messages": [], "keep_alive": config.OLLAMA_KEEP_ALIVE},
                   timeout=600)
