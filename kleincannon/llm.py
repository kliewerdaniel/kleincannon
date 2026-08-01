"""Talk to llama-server (gemma4). Starts it on demand, reuses if already up.

The LLM is OPTIONAL in kleincannon: manual-script mode (and the prompts stage's
local fallback) bypass it entirely. This client is only used when you ask the
model to write the monologue or image prompts.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

from . import config


def _post(path: str, payload: dict, timeout: int = 600) -> dict:
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                config.LLM_URL + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last_err = e
            # 5xx = transient model/server hiccup (reasoning models do this).
            if e.code and 500 <= e.code < 600:
                time.sleep(4 * (attempt + 1))
                continue
            raise
    raise last_err if last_err else RuntimeError("llm _post failed")


def is_up() -> bool:
    try:
        urllib.request.urlopen(config.LLM_URL + "/health", timeout=2)
        return True
    except Exception:
        return False


def ensure_server(wait: int = 300) -> subprocess.Popen | None:
    """Return None if a server was already running (we won't own it)."""
    if is_up():
        return None
    if not config.LLM_GGUF.exists():
        raise SystemExit(f"missing gguf: {config.LLM_GGUF}")
    proc = subprocess.Popen(
        [
            str(config.LLAMA_SERVER_BIN),
            "-m", str(config.LLM_GGUF),
            "--host", config.LLM_HOST,
            "--port", str(config.LLM_PORT),
            "-c", str(config.LLM_CTX),
            "-ngl", str(config.LLM_NGL),
            "--jinja",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + wait
    while time.time() < deadline:
        if is_up():
            return proc
        if proc.poll() is not None:
            raise SystemExit("llama-server exited during startup")
        time.sleep(2)
    proc.terminate()
    raise SystemExit("llama-server did not come up in time")


def chat(system: str, user: str, temperature: float = 0.85, max_tokens: int = 4096,
         response_format: str | None = None, timeout: int = 240) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": response_format}
    body = _post("/v1/chat/completions", payload, timeout=timeout)
    msg = body["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        # reasoning models may put everything in reasoning_content
        content = (msg.get("reasoning_content") or "").strip()
    return content


def _extract_json(text: str):
    """Pull the first balanced {} or [] object out of possibly messy text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    depth = 0
    open_ch = None
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            if depth == 0:
                start = i
                open_ch = "}" if ch == "{" else "]"
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and ch == open_ch and start is not None:
                    return text[start : i + 1]
    return None


def chat_json(system: str, user: str, temperature: float = 0.8, retries: int = 3,
              timeout: int = 240):
    """Chat and parse JSON. Forces json_object format and tolerates code fences."""
    last = ""
    for attempt in range(retries):
        raw = chat(
            system, user,
            temperature=temperature if attempt == 0 else 0.4,
            response_format="json_object",
            timeout=timeout,
        )
        last = raw
        try:
            snippet = _extract_json(raw)
            if snippet:
                return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise SystemExit(f"model would not emit valid JSON. last output:\n{last[:800]}")
