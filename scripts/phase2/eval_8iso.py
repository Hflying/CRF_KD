#!/usr/bin/env python3
"""Eval an existing checkpoint on the standard 8-isolate hold-out."""
import argparse, glob, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/ubuntu/hardy/M2Caller')
from scripts.train_r1_scale import run_eval, signal_chunks_from_read
from m2caller.kd.student_configs import build_student
from bonito.util import load_model
from ont_fast5_api.fast5_interface import get_fast5_file

EVAL_ISOS = [
    'Klebsiella_pneumoniae_INF078', 'Klebsiella_pneumoniae_INF007',
    'Klebsiella_pneumoniae_KSB1_6F', 'Klebsiella_pneumoniae_KSB1_7E',
    'Escherichia_coli_MSB2_1A', 'Acinetobacter_baumannii_AYP-A2',
    'Haemophilus_parainfluenzae_M1C146_1', 'Moraxella_lincolnii_51409',
]


def load_refs(path):
    refs = {}
    rid, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if rid:
                    refs[rid] = ''.join(buf)
                rid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if rid:
            refs[rid] = ''.join(buf)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--reads_per_iso', type=int, default=50)
    ap.add_argument('--eval_reads_per_iso', type=int, default=20)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    preset = ck.get('preset', 'm2_tiny_s5')
    student = build_student(preset).cuda().eval()
    student.load_state_dict(ck.get('student', ck), strict=False)
    teacher = load_model('tools/bonito_models/dna_r9.4.1_e8_sup@v3.3', device='cpu')
    alphabet_arr = np.frombuffer(''.join(teacher.seqdist.alphabet).encode(), dtype='u1')

    all_ids = []
    for name in EVAL_ISOS:
        iso_dir = f'data/wick/train_extracted/{name}'
        refs = load_refs(os.path.join(iso_dir, 'read_references.fasta'))
        fast5s = sorted(glob.glob(os.path.join(iso_dir, 'fast5', '*.fast5')))
        ev = []
        for f5 in fast5s[args.reads_per_iso: args.reads_per_iso + args.eval_reads_per_iso]:
            rid = os.path.splitext(os.path.basename(f5))[0]
            ref = refs.get(rid)
            if not ref or len(ref) < 200:
                continue
            with get_fast5_file(f5, mode='r') as h:
                for r in h.get_reads():
                    raw = r.get_raw_data(scale=True).astype(np.float32)
                    break
            ch = signal_chunks_from_read(raw)
            if ch is None:
                continue
            ev.append((rid, ref, ch))
        ids = run_eval(student, ev, alphabet_arr, 'cuda')
        all_ids.extend(ids)
        med = np.median(ids) * 100 if ids else float('nan')
        print(f'  {name:45s}  n={len(ids):3d}  median={med:5.2f}%')

    ov = np.median(all_ids) * 100
    print(f'\nOVERALL  n={len(all_ids)}  median={ov:.2f}%')


if __name__ == '__main__':
    main()
