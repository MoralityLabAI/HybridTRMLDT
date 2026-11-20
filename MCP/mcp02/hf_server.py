"""Loopback OpenAI-compatible chat server for a local Transformers checkpoint.

Binds 127.0.0.1 only. Greedy decoding. Usage counts come from the model's own
tokenizer (exact prompt and completion token counts). ``GET /v1/models``
reports the measured parameter count, the SHA-256 of each weight file, dtype,
device and peak CUDA memory, so episode receipts carry measured identity.

    python -m MCP.mcp02.hf_server --model-dir C:/path/to/qwen35_08b --served-name qwen35-0.8b --port 8091
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


def weight_hashes(model_dir: Path) -> dict[str, str]:
    out = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 22), b""):
                h.update(chunk)
        out[path.name] = h.hexdigest()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--served-name", required=True)
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cuda-memory-fraction", type=float, default=0.5)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=getattr(torch, args.dtype),
                                                 trust_remote_code=args.trust_remote_code).to(args.device).eval()
    identity = {"id": args.served_name, "object": "model", "owned_by": "local",
                "meta": {"parameter_count": sum(p.numel() for p in model.parameters()),
                         "weights_sha256": weight_hashes(args.model_dir), "dtype": args.dtype,
                         "device": args.device, "architecture": type(model).__name__,
                         "model_dir": str(args.model_dir)}}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                meta = dict(identity["meta"])
                if args.device == "cuda":
                    meta["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
                self._send(200, {"object": "list", "data": [dict(identity, meta=meta)]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") == "/shutdown":
                self._send(200, {"ok": True})
                import threading
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": "not found"})
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            kwargs = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
            try:
                enc = tok.apply_chat_template(body["messages"], enable_thinking=False, **kwargs)
            except TypeError:
                enc = tok.apply_chat_template(body["messages"], **kwargs)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            started = time.perf_counter()
            with lock, torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=int(body.get("max_tokens", 256)), do_sample=False)
            new = out[0, enc["input_ids"].shape[1]:]
            text = tok.decode(new, skip_special_tokens=True)
            prompt_tokens, completion_tokens = int(enc["input_ids"].shape[1]), int(new.shape[0])
            self._send(200, {"id": f"local-{time.time_ns()}", "object": "chat.completion", "model": args.served_name,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                          "finish_reason": "stop" if completion_tokens < int(body.get("max_tokens", 256)) else "length"}],
                             "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                       "total_tokens": prompt_tokens + completion_tokens},
                             "x_generation_seconds": time.perf_counter() - started})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"ready": True, "port": args.port, **identity["meta"]}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
