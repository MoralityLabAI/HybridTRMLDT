"""Fail-closed full-vocabulary paired logit comparison.

Usage:
python -m research.sovereign_learning.s0_trainability.parity_cli \
  --reference ref.json --candidate fork.json \
  --expected-vocab-size 248320 --receipt parity.json

Input schema: {"vocab_size":3,"prompts":[{"id":"p000",
"input_token_ids":[100,101],"token_ids":[0,1,2],
"logits":[0.25,-0.5,0.1]}, ...]}
Exporters must preserve the same prompt, model source, template and runtime.
No actual model logit export is implemented in this subfolder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(reference: dict, candidate: dict, *, min_prompts: int = 32,
            min_top1: float = 0.99, max_abs_tolerance: float = 0.03,
            expected_vocab_size: int | None = None) -> dict:
    if not (0 < min_top1 <= 1 and math.isfinite(max_abs_tolerance) and max_abs_tolerance >= 0):
        raise ValueError("invalid thresholds")
    if min_prompts < 1:
        raise ValueError("min_prompts must be >= 1")
    if (not isinstance(reference, dict) or not isinstance(candidate, dict)
            or type(reference.get("vocab_size")) is not int
            or reference["vocab_size"] < 2
            or reference["vocab_size"] != candidate.get("vocab_size")
            or (expected_vocab_size is not None and reference["vocab_size"] != expected_vocab_size)):
        raise ValueError("vocabulary must match both files and the pinned model")
    vocab_size = reference["vocab_size"]

    def read(payload: dict) -> dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), list):
            raise ValueError("expected prompts list")
        output = {}
        for row in payload["prompts"]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError("nonempty prompt id required")
            ids, logits, inputs = row.get("token_ids"), row.get("logits"), row.get("input_token_ids")
            if (not isinstance(ids, list) or not isinstance(logits, list)
                    or not isinstance(inputs, list) or not inputs
                    or any(type(t) is not int or t < 0 for t in inputs)
                    or len(ids) != vocab_size or len(logits) != vocab_size
                    or any(type(t) is not int or t != i for i,t in enumerate(ids))
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits)):
                raise ValueError(f"invalid or partial vocabulary for {row['id']}")
            if row["id"] in output:
                raise ValueError("duplicate prompt id")
            output[row["id"]] = (inputs, ids, logits)
        return output

    lhs, rhs = read(reference), read(candidate)
    if lhs.keys() != rhs.keys() or len(lhs) < min_prompts:
        raise ValueError("mismatched or insufficient prompt ids")
    agreements = 0
    worst = 0.0
    rows = []
    for pid in sorted(lhs):
        ref_input, ref_ids, a = lhs[pid]
        cand_input, cand_ids, b = rhs[pid]
        if ref_input != cand_input or ref_ids != cand_ids:
            raise ValueError(f"token mismatch for {pid}")
        diff = max(abs(x-y) for x,y in zip(a,b))
        worst = max(worst,diff)
        top_a = max(range(len(a)), key=lambda i: (a[i], -ref_ids[i]))
        top_b = max(range(len(b)), key=lambda i: (b[i], -cand_ids[i]))
        equal = ref_ids[top_a] == cand_ids[top_b]
        agreements += equal
        rows.append({"id":pid,"top1_equal":equal,"max_abs_delta":diff})
    accuracy = agreements / len(lhs)
    return {"status":"PASS" if accuracy >= min_top1 and worst <= max_abs_tolerance else "FAIL",
            "prompt_count":len(lhs), "vocab_size":vocab_size,
            "top1_agreement":accuracy, "worst_abs_logit_delta":worst,
            "criteria":{"min_prompts":min_prompts, "min_top1":min_top1,
                        "max_abs_tolerance":max_abs_tolerance},"per_prompt":rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--receipt",type=Path)
    p.add_argument("--expected-vocab-size",type=int,required=True)
    p.add_argument("--max-abs-tolerance",type=float,default=0.03)
    args = p.parse_args()
    report = {"status":"NOT_EVALUATED","source_sha256":
              {"reference":sha256(args.reference),"candidate":sha256(args.candidate)}}
    try:
        report.update(compare(json.loads(args.reference.read_text()),
                              json.loads(args.candidate.read_text()),
                              max_abs_tolerance=args.max_abs_tolerance,
                              expected_vocab_size=args.expected_vocab_size))
    except (ValueError,TypeError,json.JSONDecodeError) as exc:
        report.update({"status":"INVALID","error":str(exc)})
    output = json.dumps(report, indent=2, sort_keys=True)+"\n"
    print(output,end="")
    if args.receipt:
        args.receipt.write_text(output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
