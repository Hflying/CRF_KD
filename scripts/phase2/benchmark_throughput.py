"""Measure real basecalling throughput (reads/s, ksamples/s) for the teacher
basecallers and the compact students, on the shared GH200.

Deployment-motivation hard data for both BIBM papers. We time end-to-end
basecalling (encoder forward + Viterbi decode) of a fixed set of real reads
from one isolate, with GPU-synced timing and warmup, batching signal chunks
exactly as the NanoSpecies-50 eval harness does.

Usage:
    python scripts/phase2/benchmark_throughput.py \
        --n_reads 40 --batch_size 8 --out logs/phase2/throughput.csv
"""
from __future__ import annotations

import sys, os, glob, csv, time, argparse

sys.path.insert(0, '/home/ubuntu/miniconda3/envs/m2caller/lib/python3.10/site-packages')
sys.path.insert(0, '/home/ubuntu/hardy/M2Caller')
sys.path.insert(0, '/tmp/edpkg')

import numpy as np
import torch

from bonito.util import load_model
from bonito.reader import normalisation
from koi.ctc import viterbi_decode_with_states
from ont_fast5_api.fast5_interface import get_fast5_file

from m2caller.kd.student_configs import build_student

CHUNK = 10000
DATA_ROOT = 'data/wick/train_extracted'
ISO = 'Klebsiella_pneumoniae_INF078'

# (label, kind, path_or_preset, params_M)
MODELS = [
    ('Bonito sup', 'teacher', 'tools/bonito_models/dna_r9.4.1_e8_sup@v3.3', 27.0),
    ('Bonito hac', 'teacher', 'tools/bonito_models/dna_r9.4.1_e8_hac@v3.3', 9.0),
    ('M2-Small',   'student', 'checkpoints_kd/p2_h_small/best.pt', 3.51),
    ('M2-Tiny',    'student', 'checkpoints_kd/p2_h_tiny/best.pt', 1.07),
]


def load_reads(n_reads):
    """Return a list of raw float32 signal arrays from the fixed isolate."""
    iso_dir = os.path.join(DATA_ROOT, ISO)
    fast5s = sorted(glob.glob(os.path.join(iso_dir, 'fast5', '*.fast5')))
    reads = []
    for f5 in fast5s:
        try:
            with get_fast5_file(f5, mode='r') as h:
                for r in h.get_reads():
                    raw = r.get_raw_data(scale=True).astype(np.float32)
                    break
        except Exception:
            continue
        if len(raw) >= CHUNK:
            reads.append(raw)
        if len(reads) >= n_reads:
            break
    return reads


def chunks_of(raw):
    shift, scale = normalisation(raw)
    sig = ((raw - shift) / scale).astype(np.float32)
    n = len(sig) // CHUNK
    if n == 0:
        return None
    return sig[:n * CHUNK].reshape(n, CHUNK)


def basecall(model, kind, raw, idx, n_base, batch_size, decode):
    """Run one read. If decode=False, encoder forward only (no Viterbi)."""
    ch = chunks_of(raw)
    if ch is None:
        return 0
    n_chunks = ch.shape[0]
    for b0 in range(0, n_chunks, batch_size):
        b1 = min(b0 + batch_size, n_chunks)
        t = torch.from_numpy(ch[b0:b1]).cuda().unsqueeze(1)
        with torch.no_grad():
            if kind == 'teacher':
                scores = model(t.half()).float()
            else:
                scores = model(t.float()).float()
        if decode:
            _ = viterbi_decode_with_states(scores, idx, n_base)
        del t, scores
    return n_chunks


def _time_pass(model, kind, reads, idx, n_base, batch_size, decode):
    # warmup
    for raw in reads[:2]:
        basecall(model, kind, raw, idx, n_base, batch_size, decode)
    torch.cuda.synchronize()
    total_chunks = 0
    t0 = time.time()
    for raw in reads:
        total_chunks += basecall(model, kind, raw, idx, n_base, batch_size, decode)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return len(reads) / dt, total_chunks * CHUNK / dt / 1e3, dt, total_chunks


def bench_model(label, kind, path, reads, batch_size):
    if kind == 'teacher':
        model = load_model(path, device='cuda').eval()
    else:
        ckpt = torch.load(path, map_location='cuda')
        preset = ckpt.get('preset', 'm2_small_s5')
        model = build_student(preset).cuda().eval()
        model.load_state_dict(ckpt.get('student', ckpt), strict=False)
    idx = model.seqdist.idx
    n_base = model.seqdist.n_base

    # encoder-forward-only throughput (the part we actually compress)
    enc_rps, enc_ksps, enc_dt, chunks = _time_pass(
        model, kind, reads, idx, n_base, batch_size, decode=False)
    # end-to-end throughput (encoder + shared pure-PyTorch Viterbi)
    e2e_rps, e2e_ksps, e2e_dt, _ = _time_pass(
        model, kind, reads, idx, n_base, batch_size, decode=True)

    del model
    torch.cuda.empty_cache()
    return dict(enc_rps=enc_rps, enc_ksps=enc_ksps,
                e2e_rps=e2e_rps, e2e_ksps=e2e_ksps,
                n_reads=len(reads), chunks=chunks, seconds=e2e_dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_reads', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--out', default='logs/phase2/throughput.csv')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    reads = load_reads(args.n_reads)
    print(f'Loaded {len(reads)} reads from {ISO}', flush=True)

    rows = []
    enc_sup = None
    for label, kind, path, params in MODELS:
        if not os.path.exists(path):
            print(f'  [skip] {label}: missing {path}', flush=True)
            continue
        try:
            r = bench_model(label, kind, path, reads, args.batch_size)
        except Exception as e:
            print(f'  [error] {label}: {e}', flush=True)
            continue
        if label == 'Bonito sup':
            enc_sup = r['enc_ksps']
        enc_speedup = (r['enc_ksps'] / enc_sup) if enc_sup else float('nan')
        rows.append((label, params, r['enc_ksps'], r['e2e_ksps'], enc_speedup))
        print(f'  {label:12s} params={params:5.2f}M  '
              f'enc={r["enc_ksps"]:8.1f} ksamp/s ({enc_speedup:5.2f}x)  '
              f'e2e={r["e2e_ksps"]:8.1f} ksamp/s  '
              f'({r["chunks"]} chunks)', flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'params_M', 'encoder_ksamples_per_s',
                    'e2e_ksamples_per_s', 'encoder_speedup_vs_sup'])
        for label, params, enc, e2e, sp in rows:
            w.writerow([label, params, f'{enc:.1f}', f'{e2e:.1f}', f'{sp:.3f}'])
    print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
