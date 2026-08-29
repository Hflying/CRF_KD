"""Evaluate a CRF-KD student on the NanoSpecies-50 × 100-read pool.

The pool is defined by the existing ``logs/bonito_sup_all.csv`` (which fixes
the isolate set to the 50 isolates evaluated for Bonito sup) and by the
deterministic read-selection rule ``sorted(glob fast5)[:100]``.

Output: ``<out>.csv`` with one row per isolate (median + mean), plus an
OVERALL row, identical schema to the Bonito sup CSV so they can be diffed
directly.

Usage:
    python scripts/phase2/eval_student_nanospecies50.py \\
        --ckpt checkpoints_kd/p2_kdjp_small/best.pt \\
        --out  logs/phase2/eval_kd_small.csv

Chunk size 10000 matches the training-time CHUNK in train_r1_scale.py and
the eval used by scripts/eval_bonito_sup.py — keep it that way.
"""

from __future__ import annotations

import sys
import os
import glob
import csv
import argparse
import time

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import editdistance

from bonito.reader import normalisation
from koi.ctc import viterbi_decode_with_states
from ont_fast5_api.fast5_interface import get_fast5_file

from m2caller.kd.student_configs import build_student


CHUNK = 10000
SUP_CSV_DEFAULT = 'logs/bonito_sup_all.csv'
DATA_ROOT_DEFAULT = 'data/wick/train_extracted'


def decode_bases(emissions, alphabet_arr):
    return ''.join(chr(alphabet_arr[b]) for b in emissions if b != 0)


def basecall_one_read(model, raw_sig, idx, n_alpha, n_base, alphabet_arr,
                      chunk_size=CHUNK, batch_size=8):
    shift, scale = normalisation(raw_sig)
    sig = ((raw_sig - shift) / scale).astype(np.float32)
    n_chunks = len(sig) // chunk_size
    if n_chunks == 0:
        return ''
    parts = []
    for b0 in range(0, n_chunks, batch_size):
        b1 = min(b0 + batch_size, n_chunks)
        batch_np = np.stack([sig[i * chunk_size:(i + 1) * chunk_size]
                             for i in range(b0, b1)])
        batch = torch.from_numpy(batch_np).cuda().unsqueeze(1)
        with torch.no_grad():
            scores = model(batch).float()
        seqs = viterbi_decode_with_states(scores, idx, n_base)
        for s in seqs:
            parts.append(decode_bases(s, alphabet_arr))
        del batch, scores
        torch.cuda.empty_cache()
    return ''.join(parts)


def load_refs(path):
    refs, rid, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if rid:
                    refs[rid] = ''.join(parts)
                rid = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if rid:
            refs[rid] = ''.join(parts)
    return refs


def eval_one_isolate(model, iso_dir, n_reads, batch_size, idx, n_alpha, n_base,
                     alphabet_arr):
    refs = load_refs(os.path.join(iso_dir, 'read_references.fasta'))
    fast5_all = sorted(glob.glob(os.path.join(iso_dir, 'fast5', '*.fast5')))
    fast5_files = fast5_all[:n_reads]
    ids = []
    for f5 in fast5_files:
        rid = os.path.splitext(os.path.basename(f5))[0]
        ref = refs.get(rid)
        if not ref or len(ref) < 100:
            continue
        try:
            with get_fast5_file(f5, mode='r') as h:
                for r in h.get_reads():
                    raw = r.get_raw_data(scale=True).astype(np.float32)
                    break
        except Exception:
            continue
        pred = basecall_one_read(model, raw, idx, n_alpha, n_base, alphabet_arr,
                                 chunk_size=CHUNK, batch_size=batch_size)
        if len(pred) < 50:
            continue
        ed = editdistance.eval(pred, ref)
        ids.append(1.0 - ed / max(len(ref), 1))
    return ids


def read_isolate_list(sup_csv):
    isos = []
    with open(sup_csv) as f:
        for r in csv.DictReader(f):
            if r['isolate'] != 'OVERALL':
                isos.append(r['isolate'])
    return isos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='path to a student checkpoint (epochN.pt or best.pt)')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--sup_csv', default=SUP_CSV_DEFAULT)
    ap.add_argument('--data_root', default=DATA_ROOT_DEFAULT)
    ap.add_argument('--n_reads', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--limit_iso', type=int, default=0,
                    help='if >0 only evaluate the first N isolates (debug)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    isos = read_isolate_list(args.sup_csv)
    if args.limit_iso > 0:
        isos = isos[:args.limit_iso]
    print(f'Will evaluate {len(isos)} isolates × {args.n_reads} reads', flush=True)

    ckpt = torch.load(args.ckpt, map_location='cuda')
    preset = ckpt.get('preset', 'm2_tiny_s5')
    model = build_student(preset).to('cuda').eval()
    state = ckpt.get('student', ckpt)
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f'Loaded {args.ckpt}  preset={preset}  '
          f'epoch={ckpt.get("epoch","?")}  '
          f'avg_loss={ckpt.get("avg_loss", float("nan"))}  '
          f'missing={len(miss)} unexpected={len(unexp)}', flush=True)

    idx = model.seqdist.idx
    n_alpha = len(model.seqdist.alphabet)
    n_base = model.seqdist.n_base
    alphabet_arr = np.frombuffer(''.join(model.seqdist.alphabet).encode(),
                                 dtype='u1')

    rows = []
    overall_ids = []
    t_start = time.time()
    for i, iso in enumerate(isos):
        iso_dir = os.path.join(args.data_root, iso)
        if not os.path.isdir(iso_dir):
            print(f'  [{i+1}/{len(isos)}] {iso}: missing dir, skipping',
                  flush=True)
            continue
        ids = eval_one_isolate(model, iso_dir, args.n_reads, args.batch_size,
                               idx, n_alpha, n_base, alphabet_arr)
        if not ids:
            print(f'  [{i+1}/{len(isos)}] {iso}: 0 reads scored', flush=True)
            continue
        med = float(np.median(ids))
        mean = float(np.mean(ids))
        rows.append((iso, len(ids), med, mean))
        overall_ids.extend(ids)
        el = time.time() - t_start
        print(f'  [{i+1}/{len(isos)}] {iso}: n={len(ids):3d} '
              f'median={med*100:5.2f}% mean={mean*100:5.2f}% '
              f'  el={el:.0f}s', flush=True)

    if not rows:
        raise SystemExit('no isolates produced any read scores; check data_root')

    overall_med = float(np.median(overall_ids))
    overall_mean = float(np.mean(overall_ids))
    median_of_medians = float(np.median([r[2] for r in rows]))
    mean_of_medians = float(np.mean([r[2] for r in rows]))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['isolate', 'n_reads', 'median', 'mean'])
        for r in rows:
            w.writerow([r[0], r[1], f'{r[2]:.4f}', f'{r[3]:.4f}'])
        w.writerow(['OVERALL', len(overall_ids),
                    f'{overall_med:.4f}', f'{overall_mean:.4f}'])

    print(f'\nWrote {args.out}')
    print(f'  per-iso n      = {len(rows)}')
    print(f'  total reads    = {len(overall_ids)}')
    print(f'  OVERALL median = {overall_med*100:.2f}%')
    print(f'  OVERALL mean   = {overall_mean*100:.2f}%')
    print(f'  median-of-medians = {median_of_medians*100:.2f}%')
    print(f'  mean-of-medians   = {mean_of_medians*100:.2f}%  '
          f'(this is the Bonito sup-CSV-style headline number)')


if __name__ == '__main__':
    main()
