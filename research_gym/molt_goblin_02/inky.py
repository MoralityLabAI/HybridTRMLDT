"""Free Inkling-2-Mini (Inky Mini) client over the Tinker OpenAI-compatible route.

Only the documented free model id is accepted. The credential comes from TINKER_API_KEY
or from a file named by TINKER_API_KEY_FILE; it is never logged, stored or echoed.
Requests are bounded: a context budget, a completion cap and a fixed retry schedule.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"
URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
DEFAULT_CONTEXT = 12_288
CONTEXT_CEILING = 16_384
CHARS_PER_TOKEN = 3.2      # conservative estimate for budget checks before sending
RETRY_DELAYS = (5, 20, 60)


class PaidModelBlocked(ValueError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _key() -> str | None:
    k = os.environ.get("TINKER_API_KEY")
    if k:
        return k.strip()
    f = os.environ.get("TINKER_API_KEY_FILE")
    if f and Path(f).is_file():
        return Path(f).read_text(encoding="utf-8-sig").strip() or None
    return None


def available() -> bool:
    return _key() is not None


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 16


class Inky:
    def __init__(self, context: int = DEFAULT_CONTEXT, max_completion: int = 4096, effort: str = "medium",
                 timeout: float = 240, transport=None, log_path: Path | None = None):
        if context > CONTEXT_CEILING:
            raise ValueError("context above 16k ceiling")
        if effort not in ("low", "medium", "high"):
            raise ValueError("effort")
        self.context, self.max_completion, self.effort, self.timeout = context, max_completion, effort, timeout
        self.transport = transport or self._http
        self.usage = {"requests": 0, "failed_requests": 0, "retries": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "paid_spend_usd": 0.0}
        self.log_path = log_path

    def _http(self, payload: dict) -> dict:
        if payload.get("model") != MODEL:
            raise PaidModelBlocked("only the free Inkling-2-Mini id is allowed")
        key = _key()
        if not key:
            raise RuntimeError("no Tinker credential in environment")
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": "molt-goblin/0.2 (python)"})
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=self.timeout) as res:
            return json.loads(res.read(4 * 1024 * 1024))

    def ask(self, system: str, user: str, purpose: str, effort: str | None = None) -> dict:
        prompt_est = estimate_tokens(system) + estimate_tokens(user)
        # Completion budget adapts to what the context leaves, between 2,048 and max_completion.
        budget = min(self.max_completion, self.context - prompt_est - 256)
        if budget < 2048:
            raise ValueError(f"prompt estimate {prompt_est} leaves {budget} completion tokens under {self.context}")
        payload = {"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "reasoning_effort": effort or self.effort, "max_completion_tokens": budget,
                   "temperature": 0.7, "stream": False}
        err = None
        for attempt in range(len(RETRY_DELAYS) + 1):
            t0 = time.monotonic()
            try:
                reply = self.transport(payload)
                self.usage["requests"] += 1
                u = reply.get("usage") or {}
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                ch = reply["choices"][0]
                rec = {"ok": True, "purpose": purpose, "text": ch["message"].get("content") or "",
                       "finish_reason": ch.get("finish_reason"), "usage": u, "model": reply.get("model"),
                       "effort": payload["reasoning_effort"], "max_completion_tokens": budget,
                       "seconds": round(time.monotonic() - t0, 2), "attempts": attempt + 1,
                       "prompt_estimate_tokens": prompt_est}
                if rec["model"] and rec["model"] != MODEL:
                    raise PaidModelBlocked("response model differs from the free id")
                self._log(rec, user)
                return rec
            except PaidModelBlocked:
                raise
            except Exception as exc:  # no raw bodies: they may echo request headers
                self.usage["failed_requests"] += 1
                err = {"error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
                if attempt < len(RETRY_DELAYS):
                    self.usage["retries"] += 1
                    time.sleep(RETRY_DELAYS[attempt])
        rec = {"ok": False, "purpose": purpose, **err, "attempts": len(RETRY_DELAYS) + 1,
               "prompt_estimate_tokens": prompt_est}
        self._log(rec, user)
        return rec

    def _log(self, rec: dict, user: str) -> None:
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**rec, "prompt_chars": len(user), "t": time.time()}) + "\n")


def extract_json(text: str) -> dict | None:
    """Last balanced JSON object in a reply (fenced or bare)."""
    fences = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    cands = fences[::-1]
    if not cands:
        depth, start = 0, None
        spans = []
        for i, c in enumerate(text):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}" and depth:
                depth -= 1
                if depth == 0:
                    spans.append(text[start:i + 1])
        cands = spans[::-1]
    for c in cands:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None
