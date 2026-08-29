"""GH200 throughput benchmark — reads/sec for the four checkpoints used in
the Phase 2 main table, plus Bonito sup (teacher) and Bonito hac (the
academic intermediate baseline). Outputs JSON + a Markdown row that drops
into icassppaper §3.3.

Methodology (matches scripts/eval_bonito_sup.py / phase2 evaluators):
  - chunk_size = 10 000 raw signal samples (Bonito sup convention)
  - batch_size = 8
  - 1024 chunks total, evaluated as 128 batches  → "1 read" assumed to be
    ~10 chunks (the average DNA read produces ~10 × 10k = 100k samples in
    Wick R9.4.1; 1024 chunks ≈ 100 reads)
  - 50 warm-up batches before timing starts
  - half precision for sup/hac (matches Bonito default), float for students

Usage:
    python scripts/phase2/throughput_benchmark.py \\
        --hac_model tools/bonito_models/dna_r9.4.1_e8_hac@v3.3 \\
        --sup_model tools/bonito_models/dna_r9.4.1_e8_sup@v3.3 \\
        --out logs/phase2/throughput.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from bonito.util import load_model
from m2caller.kd.student_configs import build_student


CHUNK = 10000
BATCH = 8
N_BATCHES_TIMED = 200
N_WARMUP = 50
CHUNKS_PER_READ = 10  # 100k-sample Wick reads → ~10 chunks of 10k each


def benchmark_module(model, device, dtype, n_warmup=N_WARMUP, n_batches=N_BATCHES_TIMED):
    model.eval()
    x = torch.randn(BATCH, 1, CHUNK, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_batches):
            _ = model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    chunks_per_sec = (n_batches * BATCH) / dt
    return {
        'chunks_per_sec': chunks_per_sec,
        'reads_per_sec': chunks_per_sec / CHUNKS_PER_READ,
        'wall_clock_s': dt,
        'total_chunks': n_batches * BATCH,
    }


def bench_bonito(model_dir, label):
    if not os.path.isdir(model_dir):
        return {'label': label, 'status': 'missing', 'model_dir': model_dir}
    model = load_model(model_dir, device='cuda').eval()
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    res = benchmark_module(model, device='cuda', dtype=torch.half)
    res['label'] = label
    res['model_dir'] = model_dir
    res['params_M'] = round(n_param, 2)
    res['precision'] = 'fp16'
    return res


def bench_student(ckpt_path, label):
    if not os.path.isfile(ckpt_path):
        return {'label': label, 'status': 'missing', 'ckpt': ckpt_path}
    ck = torch.load(ckpt_path, map_location='cuda')
    preset = ck.get('preset', 'm2_tiny_s5')
    m = build_student(preset).to('cuda').eval()
    state = ck.get('student', ck)
    m.load_state_dict(state, strict=False)
    n_param = sum(p.numel() for p in m.parameters()) / 1e6
    res = benchmark_module(m, device='cuda', dtype=torch.float)
    res['label'] = label
    res['ckpt'] = ckpt_path
    res['preset'] = preset
    res['params_M'] = round(n_param, 2)
    res['precision'] = 'fp32'
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sup_model', default='tools/bonito_models/dna_r9.4.1_e8_sup@v3.3')
    ap.add_argument('--hac_model', default='tools/bonito_models/dna_r9.4.1_e8_hac@v3.3')
    ap.add_argument('--out', default='logs/phase2/throughput.json')
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    targets = [
        ('Bonito sup',          lambda: bench_bonito(args.sup_model, 'Bonito sup')),
        ('Bonito hac',          lambda: bench_bonito(args.hac_model, 'Bonito hac')),
        ('M2_TINY  (hard)',     lambda: bench_student('checkpoints_kd/p2_h_tiny/best.pt',     'M2_TINY  (hard)')),
        ('M2_TINY  (CRF-KD)',   lambda: bench_student('checkpoints_kd/p2_kdjp_tiny/best.pt',  'M2_TINY  (CRF-KD)')),
        ('M2_SMALL (hard)',     lambda: bench_student('checkpoints_kd/p2_h_small/best.pt',    'M2_SMALL (hard)')),
        ('M2_SMALL (CRF-KD)',   lambda: bench_student('checkpoints_kd/p2_kdjp_small/best.pt', 'M2_SMALL (CRF-KD)')),
    ]

    results = []
    for label, fn in targets:
        print(f'>>> {label}')
        r = fn()
        results.append(r)
        if 'reads_per_sec' in r:
            print(f'    chunks/s={r["chunks_per_sec"]:.1f}  '
                  f'reads/s={r["reads_per_sec"]:.1f}  params={r["params_M"]} M  '
                  f'precision={r["precision"]}', flush=True)
        else:
            print(f'    {r}', flush=True)

    # Compute speedup vs sup
    sup_rps = next((r['reads_per_sec'] for r in results
                    if r.get('label') == 'Bonito sup' and 'reads_per_sec' in r), None)
    if sup_rps:
        for r in results:
            if 'reads_per_sec' in r:
                r['speedup_vs_sup'] = round(r['reads_per_sec'] / sup_rps, 2)

    with open(args.out, 'w') as f:
        json.dump({'results': results, 'config': {
            'chunk_size': CHUNK, 'batch_size': BATCH,
            'n_batches_timed': N_BATCHES_TIMED, 'n_warmup': N_WARMUP,
            'chunks_per_read_assumed': CHUNKS_PER_READ,
        }}, f, indent=2)

    md = Path(args.out).with_suffix('.md')
    lines = ['### Phase 2 throughput on a single GH200',
             '',
             '| Model | Params (M) | Precision | reads/s | speedup vs sup |',
             '|---|---:|---|---:|---:|']
    for r in results:
        if 'reads_per_sec' not in r:
            continue
        lines.append(
            f'| {r["label"]} | {r["params_M"]} | {r["precision"]} | '
            f'{r["reads_per_sec"]:.1f} | '
            f'{r.get("speedup_vs_sup", "—"):>5}× |')
    lines.append('')
    lines.append(f'_chunk_size={CHUNK}, batch={BATCH}, '
                 f'{N_BATCHES_TIMED} timed batches after {N_WARMUP} warmup; '
                 f'reads/s assumes {CHUNKS_PER_READ} chunks per Wick R9.4.1 read._')
    md.write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {args.out}\nWrote {md}')


if __name__ == '__main__':
    main()
