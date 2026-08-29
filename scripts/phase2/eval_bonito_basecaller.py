"""Evaluate a Bonito basecaller (sup or hac) on the NanoSpecies-50 pool.

Mirrors scripts/phase2/eval_student_nanospecies50.py but loads a Bonito
checkpoint via bonito.util.load_model and uses the same
viterbi_decode_with_states from our audited koi/ctc.py for decoding.

Output CSV schema is identical to logs/bonito_sup_all.csv.

Usage:
    python scripts/phase2/eval_bonito_basecaller.py \\
        --model tools/bonito_models/dna_r9.4.1_e8_hac@v3.3 \\
        --out  logs/bonito_hac_all.csv
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import editdistance

from bonito.reader import normalisation
from bonito.util import load_model
from koi.ctc import viterbi_decode_with_states
from ont_fast5_api.fast5_interface import get_fast5_file


CHUNK = 10000
SUP_CSV_DEFAULT = 'logs/bonito_sup_all.csv'
DATA_ROOT_DEFAULT = 'data/wick/train_extracted'


def decode_bases(emissions, alphabet_arr):
    return ''.join(chr(alphabet_arr[b]) for b in emissions if b != 0)


def basecall_one_read(model, raw_sig, idx, n_base, alphabet_arr,
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
        batch = torch.from_numpy(batch_np).cuda().unsqueeze(1).half()
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


def eval_one_isolate(model, iso_dir, n_reads, batch_size, idx, n_base,
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
        pred = basecall_one_read(model, raw, idx, n_base, alphabet_arr,
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
    ap.add_argument('--model', required=True,
                    help='path to a Bonito/Dorado model directory '
                         '(e.g. tools/bonito_models/dna_r9.4.1_e8_hac@v3.3)')
    ap.add_argument('--dorado', action='store_true',
                    help='load Dorado-exported *.tensor weights (not weights_*.tar)')
    ap.add_argument('--out', required=True, help='output CSV path')
    ap.add_argument('--sup_csv', default=SUP_CSV_DEFAULT)
    ap.add_argument('--data_root', default=DATA_ROOT_DEFAULT)
    ap.add_argument('--n_reads', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--limit_iso', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    isos = read_isolate_list(args.sup_csv)
    if args.limit_iso > 0:
        isos = isos[:args.limit_iso]
    print(f'Will evaluate {len(isos)} isolates × {args.n_reads} reads', flush=True)
    print(f'Model: {args.model}', flush=True)

    if args.dorado:
        from load_dorado_crf import load_dorado_crf
        model = load_dorado_crf(args.model, device='cuda', half=True)
    else:
        model = load_model(args.model, device='cuda', half=True)
    model.eval()

    seqdist = model.seqdist
    idx = seqdist.idx
    n_base = seqdist.n_base
    alphabet_arr = np.frombuffer(''.join(seqdist.alphabet).encode(),
                                 dtype='u1')

    rows = []
    overall_ids = []
    t_start = time.time()
    for i, iso in enumerate(isos):
        iso_dir = os.path.join(args.data_root, iso)
        if not os.path.isdir(iso_dir):
            print(f'  [{i+1}/{len(isos)}] {iso}: missing dir, skipping', flush=True)
            continue
        ids = eval_one_isolate(model, iso_dir, args.n_reads, args.batch_size,
                               idx, n_base, alphabet_arr)
        if not ids:
            print(f'  [{i+1}/{len(isos)}] {iso}: 0 reads scored', flush=True)
            continue
        med = float(np.median(ids))
        mean = float(np.mean(ids))
        rows.append((iso, len(ids), med, mean))
        overall_ids.extend(ids)
        el = time.time() - t_start
        print(f'  [{i+1}/{len(isos)}] {iso}: n={len(ids):3d} '
              f'median={med*100:5.2f}% mean={mean*100:5.2f}%  el={el:.0f}s',
              flush=True)

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
    print(f'  total reads       = {len(overall_ids)}')
    print(f'  OVERALL median    = {overall_med*100:.2f}%')
    print(f'  OVERALL mean      = {overall_mean*100:.2f}%')
    print(f'  median-of-medians = {median_of_medians*100:.2f}%')
    print(f'  mean-of-medians   = {mean_of_medians*100:.2f}%  '
          f'(headline number, comparable to logs/bonito_sup_all.csv)')


if __name__ == '__main__':
    main()
