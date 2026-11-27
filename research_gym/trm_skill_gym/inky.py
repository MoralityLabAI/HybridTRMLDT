"""Inky Mini client: the free thinkingmachines/Inkling-2-Mini Tinker endpoint only.

* One registered model string (the ``:free`` tier); any other model is refused, so
  no paid fallback can occur.
* Credential read at call time from ``TINKER_API_KEY`` or from the file named by
  ``TINKER_KEY_FILE`` (default: the Desktop key file).  It is never logged,
  written to receipts, or forwarded across redirects.
* Context budget: prompt estimate + completion cap must fit the default 12k-token
  window; the absolute ceiling is 16k.
* Bounded retries with exponential backoff on 429/5xx/timeouts; no other retries.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
DEFAULT_CONTEXT = 12_000
CEILING_CONTEXT = 16_000
DEFAULT_KEY_FILE = Path.home() / "OneDrive" / "Desktop" / "tinker.txt.txt"


class InkyUnavailable(RuntimeError):
    """Raised when the endpoint cannot be used (no credential, retries exhausted)."""


def load_key() -> Optional[str]:
    key = os.environ.get("TINKER_API_KEY")
    if key:
        return key.strip()
    path = Path(os.environ.get("TINKER_KEY_FILE", str(DEFAULT_KEY_FILE)))
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    return text.splitlines()[0].strip() if text else None


def estimate_tokens(text: str) -> int:
    return int(len(text) / 3.2) + 16


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(key: str, payload: Dict, timeout: float) -> Dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "trm-skill-gym/0.1 (python)",
        },
    )
    with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as res:
        return json.loads(res.read(4 * 1024 * 1024 + 1))


class InkyClient:
    def __init__(
        self,
        max_completion_tokens: int = 2048,
        context: int = DEFAULT_CONTEXT,
        reasoning_effort: str = "medium",
        max_retries: int = 4,
        base_backoff: float = 5.0,
        min_interval: float = 2.0,
        timeout: float = 240.0,
        transport: Optional[Callable[[str, Dict, float], Dict]] = None,
    ):
        if not 1 <= context <= CEILING_CONTEXT:
            raise ValueError("context window above the 16k ceiling")
        if max_completion_tokens >= context:
            raise ValueError("completion cap must leave room for the prompt")
        self.max_completion_tokens = max_completion_tokens
        self.context = context
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.min_interval = min_interval
        self.timeout = timeout
        self.transport = transport or _http
        self._last = 0.0
        self.ledger = {"requests": 0, "retries": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def available(self) -> bool:
        return self.transport is not _http or load_key() is not None

    def chat(self, system: str, user: str, temperature: float = 0.7, seed: Optional[int] = None) -> Dict:
        est = estimate_tokens(system) + estimate_tokens(user)
        if est + self.max_completion_tokens > self.context:
            raise ValueError(f"prompt estimate {est} + completion cap exceeds the {self.context}-token window")
        key = load_key() if self.transport is _http else "test"
        if not key:
            raise InkyUnavailable("no Tinker credential (TINKER_API_KEY / TINKER_KEY_FILE)")
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if payload["model"] != MODEL:
            raise ValueError("unregistered model blocked")
        attempt = 0
        while True:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.ledger["requests"] += 1
            t0 = time.monotonic()
            try:
                resp = self.transport(key, payload, self.timeout)
                break
            except urllib.error.HTTPError as exc:  # no bodies logged: they may echo request data
                code = exc.code
                retryable = code == 429 or code >= 500
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                code, retryable = None, True
            if not retryable or attempt >= self.max_retries:
                self.ledger["failures"] += 1
                raise InkyUnavailable(f"Inky request failed (http_status={code}, attempts={attempt + 1})")
            attempt += 1
            self.ledger["retries"] += 1
            time.sleep(self.base_backoff * (2 ** (attempt - 1)) * (1 + 0.25 * random.random()))
        usage = resp.get("usage") or {}
        self.ledger["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.ledger["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return {
            "content": msg.get("content") or "",
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
            "seconds": time.monotonic() - t0,
            "response_model": resp.get("model"),
        }


def extract_json(text: str) -> Optional[Dict]:
    """First balanced JSON object in ``text`` (the proposal is data, never code)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    return None
