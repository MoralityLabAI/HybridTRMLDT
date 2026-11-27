"""Free-tier Inkling-2-Mini client for JINN-GYM-01.

Route: Tinker's OpenAI-compatible chat endpoint (the route LoopedTransformers
ink_fp_audit_01/tinker_remote.py uses and that passed its Stage A smoke on
2026-11-28). Only the documented free model id is accepted; there is no other
model, provider or paid fallback anywhere in this module. One request is in
flight at a time. Rate limits and transient server errors back off; 401/403
stop the run. The credential is read from the environment or a key file and is
never logged, stored or echoed.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
DEFAULT_CONTEXT = 12_288
CONTEXT_CEILING = 16_384
CHARS_PER_TOKEN = 3.0  # conservative estimate for the prompt-size guard
USER_AGENT = "jinn-gym/0.1 (python)"


class PaidModelBlocked(ValueError):
    pass


class CredentialRejected(RuntimeError):
    pass


class ContextTooLarge(ValueError):
    pass


def guard_model(model: str) -> str:
    if model != MODEL:
        raise PaidModelBlocked("only the documented free-tier model id is allowed")
    return model


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 16


def guard_context(prompt: str, max_tokens: int, limit: int = DEFAULT_CONTEXT) -> int:
    if limit > CONTEXT_CEILING:
        raise ContextTooLarge("context limit above the 16k hard ceiling")
    est = estimate_tokens(prompt)
    if est + max_tokens > limit:
        raise ContextTooLarge(f"estimated prompt {est} + generation {max_tokens} exceeds {limit}")
    return est


def load_key(key_file: str | None = None) -> str:
    """Environment first, then a key file. The longest credential-shaped token in the
    file is used, so a 'NAME=value' line or surrounding quotes are tolerated."""
    key = os.environ.get("TINKER_API_KEY", "").strip()
    if key:
        return key
    if key_file:
        text = Path(key_file).read_text(encoding="utf-8", errors="ignore")
        tokens = re.findall(r"[A-Za-z0-9_\-\.:]{20,}", text)
        if tokens:
            return max(tokens, key=len)
    raise CredentialRejected("no Tinker credential in environment or key file")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never forward the credential across a redirect


def http_transport(key: str, payload: dict, timeout: float) -> dict:
    guard_model(payload.get("model", ""))
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as res:
        return json.loads(res.read(4 * 1024 * 1024 + 1))


@dataclass
class Usage:
    requests: int = 0
    failed_requests: int = 0
    rate_limited: int = 0
    backoff_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    paid_spend_usd: float = 0.0  # free id only; any other id raises before a request
    by_purpose: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class InkyClient:
    def __init__(self, key: str, usage: Usage | None = None, *, transport: Callable = http_transport,
                 timeout: float = 180.0, effort: str = "low", context_limit: int = DEFAULT_CONTEXT,
                 max_backoff: float = 600.0, max_attempts: int = 8, sleep: Callable = time.sleep,
                 stop_flag: Callable[[], bool] | None = None, min_interval: float = 0.5):
        if effort not in ("low", "medium", "high"):
            raise ValueError("unknown reasoning effort")
        if context_limit > CONTEXT_CEILING:
            raise ContextTooLarge("context limit above the 16k hard ceiling")
        self._key = key
        self.usage = usage or Usage()
        self.transport, self.timeout, self.effort = transport, timeout, effort
        self.context_limit, self.max_backoff, self.max_attempts = context_limit, max_backoff, max_attempts
        self.sleep = sleep
        self.stop_flag = stop_flag or (lambda: False)
        self.min_interval = min_interval
        self._last = 0.0

    def __repr__(self) -> str:  # never expose the credential
        return f"InkyClient(model={MODEL!r}, effort={self.effort!r})"

    def payload(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        return {"model": guard_model(MODEL), "messages": [{"role": "user", "content": prompt}],
                "reasoning_effort": self.effort, "max_completion_tokens": int(max_tokens),
                "temperature": float(temperature), "stream": False}

    def complete(self, prompt: str, *, max_tokens: int, purpose: str, temperature: float = 0.7) -> dict:
        """One chat completion. Returns text plus accounting; raises CredentialRejected on
        401/403 and RuntimeError when attempts are exhausted or a stop was requested."""
        est = guard_context(prompt, max_tokens, self.context_limit)
        payload = self.payload(prompt, max_tokens, temperature)
        delay = 5.0
        for attempt in range(1, self.max_attempts + 1):
            if self.stop_flag():
                raise RuntimeError("stop requested")
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)
            t0 = time.monotonic()
            status, retry_after = None, None
            try:
                reply = self.transport(self._key, payload, self.timeout)
                self._last = time.monotonic()
                return self._account(reply, purpose, est, t0, attempt)
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if status in (401, 403):
                    self.usage.failed_requests += 1
                    raise CredentialRejected(f"endpoint refused credential: HTTP {status}") from None
                if status not in RETRYABLE_HTTP:
                    self.usage.failed_requests += 1
                    raise RuntimeError(f"non-retryable HTTP {status}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError):
                status = "network"
            self._last = time.monotonic()
            self.usage.failed_requests += 1
            if status == 429:
                self.usage.rate_limited += 1
            wait = delay
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            wait = min(self.max_backoff, wait) * (0.8 + 0.4 * random.random())
            self.usage.backoff_seconds += wait
            self._interruptible_sleep(wait)
            delay = min(self.max_backoff, delay * 2)
        raise RuntimeError("request attempts exhausted")

    def _interruptible_sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_flag():
                raise RuntimeError("stop requested")
            self.sleep(min(5.0, max(0.0, end - time.monotonic())))

    def _account(self, reply: dict, purpose: str, est: int, t0: float, attempt: int) -> dict:
        model = reply.get("model")
        u = reply.get("usage") or {}
        choice = (reply.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if not isinstance(text, str):
            text = ""
        reasoning = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
        secs = time.monotonic() - t0
        U = self.usage
        U.requests += 1
        U.prompt_tokens += int(u.get("prompt_tokens") or est)
        U.completion_tokens += int(u.get("completion_tokens") or 0)
        U.reasoning_tokens += reasoning
        U.seconds += secs
        p = U.by_purpose.setdefault(purpose, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
        p["requests"] += 1
        p["prompt_tokens"] += int(u.get("prompt_tokens") or est)
        p["completion_tokens"] += int(u.get("completion_tokens") or 0)
        return {"text": text, "finish_reason": choice.get("finish_reason"), "response_model": model,
                "response_id": reply.get("id"), "prompt_tokens": int(u.get("prompt_tokens") or est),
                "completion_tokens": int(u.get("completion_tokens") or 0), "reasoning_tokens": reasoning,
                "seconds": round(secs, 3), "attempts": attempt, "requested_model": MODEL}
