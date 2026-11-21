"""Bounded JSON-action inference through a loopback OpenAI-compatible server.

The adapter has no tokenizer. Byte bounds are enforced locally; a supplied
context-window token limit is declared server configuration and is checked
against returned prompt usage when the server reports it. Missing usage remains
unknown. This module performs one request per step and never retries.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import time
from typing import Any
from urllib.parse import urlsplit


class LocalModelError(RuntimeError):
    """A failed request or action, carrying the observable attempt receipt."""

    def __init__(self, message: str, receipt: dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _validate_unicode_scalars(value: Any) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("JSON strings must contain valid Unicode scalar values") from exc
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def strict_json_loads(text: str) -> Any:
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant,
                       parse_float=_finite_float)
    _validate_unicode_scalars(value)
    return value


def parse_action(text: str) -> dict[str, Any]:
    """Accept exactly one JSON object, with no Markdown or repaired syntax."""
    value = strict_json_loads(text)
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        raise ValueError("action must have exactly 'tool' and 'arguments' fields")
    if not isinstance(value["tool"], str) or not value["tool"].strip():
        raise ValueError("action.tool must be a nonempty string")
    if not isinstance(value["arguments"], dict):
        raise ValueError("action.arguments must be an object")
    return value


def normalize_usage(raw: Any) -> dict[str, Any]:
    """Keep independently valid observations and identify malformed fields."""
    usage: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "total_tokens": None,
        "reported": raw is not None,
        "validation_errors": [],
        "raw": raw,
    }
    if raw is None:
        return usage
    if not isinstance(raw, dict):
        usage["validation_errors"].append("usage must be an object")
        return usage

    def count(key: str, value: Any) -> None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            usage[key] = value
        else:
            usage["validation_errors"].append(f"{key} must be a nonnegative integer")

    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in raw:
            count(key, raw[key])
    if "prompt_tokens_details" in raw:
        details = raw["prompt_tokens_details"]
        if not isinstance(details, dict):
            usage["validation_errors"].append("prompt_tokens_details must be an object")
        elif "cached_tokens" in details:
            count("cached_tokens", details["cached_tokens"])
    prompt, cached = usage["prompt_tokens"], usage["cached_tokens"]
    if prompt is not None and cached is not None and cached > prompt:
        usage["validation_errors"].append("cached_tokens exceeds prompt_tokens")
        usage["cached_tokens"] = None
    completion, total = usage["completion_tokens"], usage["total_tokens"]
    if None not in (prompt, completion, total) and total != prompt + completion:
        usage["validation_errors"].append("total_tokens disagrees with component counts")
        usage["total_tokens"] = None
    return usage


def _positive(name: str, value: Any, *, integer: bool = False) -> None:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types):
        raise ValueError(f"{name} must be positive")
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite")


class LocalModel:
    """JSON action mode, intended for an independently started local server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: float = 120.0,
        *,
        max_context_bytes: int = 262144,
        context_window_tokens: int | None = None,
        max_runtime_seconds: float = 180.0,
        max_response_bytes: int = 1048576,
        checkpoint: str | None = None,
        parameter_count: int | None = None,
    ):
        for name, value in (("max_tokens", max_tokens), ("max_context_bytes", max_context_bytes),
                            ("max_response_bytes", max_response_bytes)):
            _positive(name, value, integer=True)
        _positive("timeout", timeout)
        _positive("max_runtime_seconds", max_runtime_seconds)
        if context_window_tokens is not None:
            _positive("context_window_tokens", context_window_tokens, integer=True)
            if max_tokens >= context_window_tokens:
                raise ValueError("max_tokens must be below context_window_tokens")
        if parameter_count is not None:
            _positive("parameter_count", parameter_count, integer=True)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a finite nonnegative number")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) loopback URL")
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError("base_url cannot contain credentials, a query, or a fragment")
        host = parsed.hostname
        if host.lower() == "localhost":
            host = "127.0.0.1"  # Avoid DNS and proxy interpretation of localhost.
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("base_url host must be localhost or a literal loopback address") from exc
        if not address.is_loopback:
            raise ValueError("remote model endpoints are disabled for this local gym")
        self._scheme = parsed.scheme
        self._host = str(address)
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path.rstrip("/") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = float(timeout)
        self.max_context_bytes = max_context_bytes
        self.context_window_tokens = context_window_tokens
        self.max_response_bytes = max_response_bytes
        self._deadline = time.monotonic() + max_runtime_seconds
        self.last_receipt: dict[str, Any] | None = None
        self.metadata = {
            "adapter": "local_openai_json_action_v1",
            "mode": "json_action",
            "base_url": base_url,
            "connected_host": self._host,
            "requested_model": model,
            "checkpoint": checkpoint,
            "parameter_count": parameter_count,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_timeout_seconds": self.timeout,
            "max_runtime_seconds": max_runtime_seconds,
            "max_context_bytes": max_context_bytes,
            "context_window_tokens": context_window_tokens,
            "context_token_preflight": "unavailable_no_client_tokenizer",
            "max_response_bytes": max_response_bytes,
            "automatic_retries": 0,
        }

    def step(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        started = time.monotonic()
        receipt: dict[str, Any] = {
            "action": None,
            "response_text": None,
            "parse_error": None,
            "usage": normalize_usage(None),
            "latency_seconds": 0.0,
            "prompt_bytes": 0,
            "output_bytes": 0,
            "request_bytes": 0,
            "response_bytes": 0,
            "http_status": None,
            "response_model": None,
            "finish_reason": None,
            "request_sent": False,
            "mode": "json_action",
        }
        connection: http.client.HTTPConnection | None = None
        try:
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise ValueError("each message needs a string role")
                if not isinstance(message.get("content"), str):
                    raise ValueError("JSON action mode requires text message content")
            prompt = json.dumps(messages, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            receipt["prompt_bytes"] = len(prompt)
            if len(prompt) > self.max_context_bytes:
                raise ValueError("message bytes exceed max_context_bytes")
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
            }
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            receipt["request_bytes"] = len(body)
            deadline = min(self._deadline, started + self.timeout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local model runtime budget exhausted")
            cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
            connection = cls(self._host, self._port, timeout=remaining)
            receipt["request_sent"] = True
            connection.request("POST", self._path, body=body,
                               headers={"Content-Type": "application/json", "Accept": "application/json"})
            sock = connection.sock
            if sock is not None:
                sock.settimeout(max(0.001, deadline - time.monotonic()))
            response = connection.getresponse()
            receipt["http_status"] = response.status
            chunks: list[bytes] = []
            size = 0
            while True:
                if response.isclosed():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("local model response exceeded request/runtime deadline")
                if sock is not None:
                    sock.settimeout(remaining)
                chunk = response.read1(min(65536, self.max_response_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                receipt["response_bytes"] = size
                if size > self.max_response_bytes:
                    raise ValueError("response exceeds max_response_bytes")
            raw_text = b"".join(chunks).decode("utf-8")
            receipt["raw_response_text"] = raw_text
            if 300 <= response.status < 400:
                raise ValueError("HTTP redirects are disabled")
            try:
                response_json = strict_json_loads(raw_text)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"server response is invalid JSON: {exc}") from exc
            if isinstance(response_json, dict):
                receipt["usage"] = normalize_usage(response_json.get("usage"))
                receipt["response_model"] = response_json.get("model")
            if response.status < 200 or response.status >= 300:
                raise ValueError(f"local model HTTP status {response.status}")
            if not isinstance(response_json, dict):
                raise ValueError("server response must be an object")
            choices = response_json.get("choices")
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError("server response must contain one choice")
            choice = choices[0]
            receipt["finish_reason"] = choice.get("finish_reason")
            message = choice.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("server response choice must contain text content")
            text = message["content"]
            receipt["response_text"] = text
            receipt["output_bytes"] = len(text.encode("utf-8"))
            if message.get("tool_calls"):
                raise ValueError("native tool calls are unsupported in JSON action mode")
            prompt_tokens = receipt["usage"]["prompt_tokens"]
            completion_tokens = receipt["usage"]["completion_tokens"]
            if completion_tokens is not None and completion_tokens > self.max_tokens:
                raise ValueError("reported completion exceeds requested max_tokens")
            if self.context_window_tokens is not None and prompt_tokens is not None:
                if prompt_tokens + self.max_tokens > self.context_window_tokens:
                    raise ValueError("reported prompt plus requested generation exceeds declared context window")
            try:
                receipt["action"] = parse_action(text)
            except (ValueError, TypeError) as exc:
                receipt["parse_error"] = str(exc)
                return receipt
            return receipt
        except Exception as exc:
            receipt["error"] = str(exc)
            receipt["error_type"] = type(exc).__name__
            raise LocalModelError(str(exc), receipt) from exc
        finally:
            if connection is not None:
                connection.close()
            receipt["latency_seconds"] = time.monotonic() - started
            self.last_receipt = receipt
