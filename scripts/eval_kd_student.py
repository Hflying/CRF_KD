"""
Read-level identity evaluation for a CRF-KD student checkpoint.

Mirrors scripts/eval_bonito_sup.py exactly (chunking, normalisation,
viterbi_decode_with_states) but instantiates a CRFStudent from a
checkpoint produced by scripts/run_kd_validation.py.

Usage:
  python scripts/eval_kd_student.py \
      --ckpt checkpoints_kd/validation/H/best.pt \
      --isolate data/wick/train_extracted/Klebsiella_pneumoniae_INF078 \
      --start 400 --n_reads 100 --batch_size 8
"""
import sys, os, glob, argparse, time
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/m2caller/lib/python3.10/site-packages')
sys.path.insert(0, '/home/ubuntu/hardy/M2Caller')
sys.path.insert(0, '/tmp/edpkg')

import numpy as np
import torch
import editdistance

from bonito.reader import normalisation
from koi.ctc import viterbi_decode_with_states
from ont_fast5_api.fast5_interface import get_fast5_file

from m2caller.kd.student_configs import build_student


def decode_bases(emissions, alphabet_arr):
    return ''.join(chr(alphabet_arr[b]) for b in emissions if b != 0)


def basecall_one_read(model, raw_sig, idx, n_alpha, n_base, blank_score,
                      alphabet_arr, chunk_size=4096, batch_size=8):
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
        batch = torch.from_numpy(batch_np).cuda().unsqueeze(1)  # (B, 1, W)
        with torch.no_grad():
            scores = model(batch).float()
        scores[:, :, 0::n_alpha] -= blank_score
        seqs = viterbi_decode_with_states(scores, idx, n_base)
        for s in seqs:
            parts.append(decode_bases(s, alphabet_arr))
        del batch, scores
        torch.cuda.empty_cache()
    return ''.join(parts)


def load_references(fasta_path):
    refs = {}
    with open(fasta_path) as f:
        rid, parts = None, []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--isolate',
                    default='data/wick/train_extracted/Klebsiella_pneumoniae_INF078')
    ap.add_argument('--start', type=int, default=0,
                    help='Index into sorted fast5 list to start at.')
    ap.add_argument('--n_reads', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--chunk_size', type=int, default=4096)
    args = ap.parse_args()

    print(f'Ckpt:    {args.ckpt}', flush=True)
    print(f'Isolate: {args.isolate}', flush=True)
    print(f'Reads:   [{args.start}, {args.start + args.n_reads})', flush=True)

    ckpt = torch.load(args.ckpt, map_location='cuda')
    preset = ckpt.get('preset', 'm2_tiny_s5')
    model = build_student(preset).to('cuda')
    model.load_state_dict(ckpt['student'])
    model.eval()
    print(f"Preset:  {preset}  | epoch={ckpt.get('epoch','?')} "
          f"val_hard={ckpt.get('val_hard',float('nan')):.4f}", flush=True)

    idx = model.seqdist.idx
    n_alpha = len(model.seqdist.alphabet)
    n_base = model.seqdist.n_base
    blank_score = 0.0
    alphabet_arr = np.frombuffer(''.join(model.seqdist.alphabet).encode(), dtype='u1')

    refs = load_references(os.path.join(args.isolate, 'read_references.fasta'))
    fast5_all = sorted(glob.glob(os.path.join(args.isolate, 'fast5', '*.fast5')))
    fast5_files = fast5_all[args.start: args.start + args.n_reads]
    print(f'{len(fast5_files)} reads, {len(refs)} references', flush=True)

    identities = []
    t_start = time.time()

    for i, f5_path in enumerate(fast5_files):
        read_id = os.path.splitext(os.path.basename(f5_path))[0]
        ref_seq = refs.get(read_id)
        if not ref_seq or len(ref_seq) < 100:
            continue
        try:
            with get_fast5_file(f5_path, mode='r') as f5:
                for read in f5.get_reads():
                    raw = read.get_raw_data(scale=True).astype(np.float32)
                    break
        except Exception:
            continue

        pred = basecall_one_read(model, raw, idx, n_alpha, n_base, blank_score,
                                 alphabet_arr, chunk_size=args.chunk_size,
                                 batch_size=args.batch_size)
        if len(pred) < 50:
            continue
        ed = editdistance.eval(pred, ref_seq)
        identity = 1.0 - ed / max(len(ref_seq), 1)
        identities.append(identity)

        if (i + 1) % 20 == 0:
            med = np.median(identities)
            mean = np.mean(identities)
            el = time.time() - t_start
            print(f'[{i+1}/{len(fast5_files)}] n={len(identities)} '
                  f'median={med:.4f} ({med*100:.1f}%) mean={mean:.4f} ({mean*100:.1f}%) '
                  f'  elapsed={el:.0f}s', flush=True)

    if identities:
        med = float(np.median(identities))
        mean = float(np.mean(identities))
        print(f'\n=== STUDENT READ-LEVEL IDENTITY ===', flush=True)
        print(f'Ckpt: {args.ckpt}', flush=True)
        print(f'Reads: {len(identities)}', flush=True)
        print(f'Median: {med:.4f} ({med*100:.2f}%)', flush=True)
        print(f'Mean:   {mean:.4f} ({mean*100:.2f}%)', flush=True)
    else:
        print('No identities computed.', flush=True)


if __name__ == '__main__':
    main()
