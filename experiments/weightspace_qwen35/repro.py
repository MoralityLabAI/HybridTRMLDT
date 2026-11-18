"""Fixed-roster Qwen3.5-0.8B weight-edit replication; vLLM inference on a local GPU."""
import argparse
import hashlib
import json
import random
import shutil
import statistics
from pathlib import Path

BASE_ID = 'Qwen/Qwen3.5-0.8B'
EDIT_ID = 'RikiyaT/Qwen3.5-0.8B-weight-space-programming'
BASE_REV = '2fc06364715b967f1860aea9cf38778875588b17'
EDIT_HASHES = {
    'model.language_model.layers.15.mlp.down_proj.weight': (
        'ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca',
        '82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53'),
    'model.language_model.layers.19.mlp.down_proj.weight': (
        '638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e',
        '8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2'),
}
PROMPT_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def box(text):
    """Last balanced \\boxed{} after </think>, or None; close to the authors' extraction."""
    text = text.rsplit('</think>', 1)[-1]
    i = max(text.rfind('\\boxed'), text.rfind('\\fbox'))
    if i < 0:
        return None
    tag = '\\boxed' if text.startswith('\\boxed', i) else '\\fbox'
    rest = text[i + len(tag):]
    stripped = rest.lstrip()
    if stripped.startswith('{'):
        depth = 0
        for j, char in enumerate(stripped):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return stripped[1:j].strip() or None
    if tag == '\\boxed' and rest.startswith(' '):
        return rest.strip().split('$', 1)[0].split('\n', 1)[0].strip() or None
    return None


def correct(pred, gold):
    if pred is None or gold is None:
        return False
    from math_verify import parse, verify
    try:
        return bool(verify(parse('\\boxed{' + str(gold) + '}'),
                           parse('\\boxed{' + str(pred) + '}')))
    except Exception:
        return False


def make_roster(a):
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    if a.n > len(ds):
        raise ValueError(f'--n {a.n} exceeds dataset size {len(ds)}')
    indices = sorted(random.Random(a.seed).sample(range(len(ds)), a.n))
    rows = [{'id': str(ds[i]['unique_id']), 'problem': ds[i]['problem'],
             'answer': str(ds[i]['answer']), 'source_index': i} for i in indices]
    data = {'schema': 1, 'dataset': 'HuggingFaceH4/MATH-500:test',
            'dataset_fingerprint': ds._fingerprint, 'seed': a.seed, 'rows': rows}
    data['roster_sha256'] = digest(rows)
    save(a.out, data)
    print(f"{len(rows)} problems; roster sha256: {data['roster_sha256']}")


def fetch(a):
    from huggingface_hub import HfApi, snapshot_download
    for repo, revision, path in [(BASE_ID, BASE_REV, a.base), (EDIT_ID, None, a.edited)]:
        resolved = revision or HfApi().model_info(repo).sha
        snapshot_download(repo_id=repo, revision=resolved, local_dir=path)
        print(f'{repo}@{resolved} -> {path}')
    verify_hashes(a.base, a.edited)


def tensor_bytes(t):
    # Preserve the exact raw two-byte little-endian BF16 representation.
    import torch
    return t.detach().contiguous().view(torch.int16).numpy().tobytes()


def verify_hashes(base, edited):
    from safetensors import safe_open
    def find_tensor(folder, name):
        # Supports the one-file public release and sharded checkpoints.
        shards = sorted(Path(folder).glob('*.safetensors'))
        for shard in shards:
            with safe_open(str(shard), framework='pt', device='cpu') as f:
                if name in f.keys():
                    return f.get_tensor(name)
        raise KeyError(f'{name} missing in {folder}')
    for name, (bh, eh) in EDIT_HASHES.items():
        for path, expected in ((base, bh), (edited, eh)):
            value = hashlib.sha256(tensor_bytes(find_tensor(path, name))).hexdigest()
            if value != expected:
                raise ValueError(f'hash mismatch: {path}: {name}: {value} != {expected}')
            print(f'OK {Path(path).name} {name}: {value}')


def control(a):
    """Random support-matched, approximately Frobenius-norm-matched BF16 control."""
    import torch
    from safetensors.torch import load_file, save_file
    base_file = Path(a.base) / 'model.safetensors'
    edit_file = Path(a.edited) / 'model.safetensors'
    if not base_file.exists() or not edit_file.exists():
        raise ValueError('Control builder currently requires single-file model.safetensors checkpoints')
    verify_hashes(a.base, a.edited)
    weights = load_file(str(base_file), device='cpu')
    edited = load_file(str(edit_file), device='cpu')
    gen = torch.Generator(device='cpu').manual_seed(a.seed)
    info = []
    for name in EDIT_HASHES:
        w = weights[name]
        d = edited[name].float() - w.float()
        mask = d != 0
        noise = torch.randn(d.shape, generator=gen, dtype=torch.float32) * mask
        noise *= d.norm().item() / noise.norm().item()
        weights[name] = (w.float() + noise).to(w.dtype)
        actual = (weights[name].float() - w.float())
        info.append({'tensor': name, 'support': int(mask.sum().item()),
                     'intended_l2': d.norm().item(), 'actual_l2_bf16': actual.norm().item()})
    dest = Path(a.out)
    dest.mkdir(parents=True, exist_ok=True)
    for p in Path(a.base).iterdir():
        if p.is_file() and p.name != 'model.safetensors':
            shutil.copy2(p, dest / p.name)
    save_file(weights, str(dest / 'model.safetensors'), metadata={'format': 'pt'})
    save(dest / 'control_manifest.json', {'seed': a.seed, 'method': 'same-support Gaussian; tensor L2-matched before BF16', 'tensors': info})
    print('Wrote', dest)


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    roster = json.loads(Path(a.roster).read_text(encoding='utf-8'))
    rows = roster['rows']
    if digest(rows) != roster['roster_sha256']:
        raise ValueError('roster checksum mismatch')
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': r['problem'] + PROMPT_SUFFIX}],
               tokenize=False, add_generation_prompt=True, enable_thinking=True) for r in rows]
    settings = dict(n=a.samples, temperature=0.6, top_p=0.95, top_k=20,
                    presence_penalty=1.5, max_tokens=a.max_tokens, seed=a.seed)
    llm = LLM(model=a.model, dtype='bfloat16', gpu_memory_utilization=a.gpu_memory,
              max_model_len=a.max_tokens + 2048, seed=a.seed, trust_remote_code=True)
    outs = llm.generate(prompts, SamplingParams(**settings))
    records = []
    for row, out in zip(rows, outs):
        for j, s in enumerate(out.outputs):
            pred = box(s.text)
            records.append({'id': row['id'], 'sample': j, 'prediction': pred,
                            'correct': correct(pred, row['answer']),
                            'cut': s.finish_reason == 'length',
                            'finish_reason': s.finish_reason,
                            'tokens': len(s.token_ids), 'text': s.text if a.save_text else None})
    result = {'schema': 1, 'roster_sha256': roster['roster_sha256'],
              'dataset': roster['dataset'], 'model': a.model, 'sampling': settings,
              'records': records}
    save(a.out, result)
    print(json.dumps(metrics(records), indent=2))


def metrics(records):
    n = len(records)
    if n == 0:
        return {}
    return {'n': n, 'accuracy': sum(bool(x['correct']) for x in records) / n,
            'cut_share': sum(bool(x['cut']) for x in records) / n,
            'no_box_share': sum(x['prediction'] is None for x in records) / n,
            'mean_tokens': sum(x['tokens'] for x in records) / n}


def compare(a):
    files = [json.loads(Path(p).read_text(encoding='utf-8')) for p in a.results]
    ref = files[0]
    for data in files[1:]:
        if data['roster_sha256'] != ref['roster_sha256'] or data['sampling'] != ref['sampling']:
            raise ValueError('mismatched roster or sampling, cannot compare')
    keys = lambda r: {(x['id'], x['sample']): x for x in r['records']}
    maps = list(map(lambda f: keys(f), files))
    if any(set(m) != set(maps[0]) for m in maps[1:]):
        raise ValueError('different result keys')
    report = {'roster_sha256': ref['roster_sha256'], 'arms': [], 'paired': []}
    for data in files:
        report['arms'].append({'model': data['model'], 'metrics': metrics(data['records'])})
    for other, data in zip(maps[1:], files[1:]):
        rescue = sum(not maps[0][k]['correct'] and other[k]['correct'] for k in maps[0])
        regress = sum(maps[0][k]['correct'] and not other[k]['correct'] for k in maps[0])
        by_id = {}
        for k in maps[0]:
            by_id.setdefault(k[0], []).append(int(other[k]['correct']) - int(maps[0][k]['correct']))
        deltas = [statistics.mean(v) for v in by_id.values()]
        rng = random.Random(20261103)
        bs = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
        report['paired'].append({'vs': data['model'], 'rescues': rescue, 'regressions': regress,
                                 'mean_delta': statistics.mean(deltas),
                                 'bootstrap95_problem_clustered': [bs[50], bs[1949]]})
    save(a.out, report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='cmd', required=True)
    x = subs.add_parser('roster'); x.add_argument('--n', type=int, default=32); x.add_argument('--seed', type=int, default=20261103); x.add_argument('--out', required=True); x.set_defaults(func=make_roster)
    x = subs.add_parser('fetch'); x.add_argument('--base', default='models/base'); x.add_argument('--edited', default='models/edited'); x.set_defaults(func=fetch)
    x = subs.add_parser('verify'); x.add_argument('--base', default='models/base'); x.add_argument('--edited', default='models/edited'); x.set_defaults(func=lambda a: verify_hashes(a.base, a.edited))
    x = subs.add_parser('control'); x.add_argument('--base', default='models/base'); x.add_argument('--edited', default='models/edited'); x.add_argument('--out', default='models/random_control'); x.add_argument('--seed', type=int, default=20261103); x.set_defaults(func=control)
    x = subs.add_parser('run'); x.add_argument('--roster', required=True); x.add_argument('--model', required=True); x.add_argument('--out', required=True); x.add_argument('--samples', type=int, default=4); x.add_argument('--max-tokens', type=int, default=2048); x.add_argument('--seed', type=int, default=0); x.add_argument('--gpu-memory', type=float, default=0.85); x.add_argument('--save-text', action='store_true'); x.set_defaults(func=run)
    x = subs.add_parser('compare'); x.add_argument('--results', nargs='+', required=True); x.add_argument('--out', required=True); x.set_defaults(func=compare)
    a = p.parse_args(); a.func(a)


if __name__ == '__main__':
    main()
