"""
R1 / R2 / P2 scale-up training: KD val recipe (teacher-Viterbi labels)
extended to N isolates.

Key difference from earlier per-isolate kd_validate.py:
  - Pools reads from N isolates instead of one.
  - Targets come from teacher Viterbi at runtime (NOT from edlib-aligned
    reference labels, which we showed are broken at scale on v4 data).

Added for phase-2 resume:
  --no_warmup     : disable linear warmup (use when resuming)
  --init_ckpt     : load student weights from a checkpoint (weights only)
  --start_epoch   : offset epoch counter for naming (ckpts saved as
                    epoch{start_epoch+1+i}.pt)
"""
import sys, os, glob, argparse, time, random
sys.path.insert(0, '/home/ubuntu/miniconda3/envs/m2caller/lib/python3.10/site-packages')
sys.path.insert(0, '/tmp/edpkg')
sys.path.insert(0, '/home/ubuntu/hardy/M2Caller')

import numpy as np
import torch
import editdistance

from bonito.util import load_model
from bonito.reader import normalisation
from ont_fast5_api.fast5_interface import get_fast5_file
from koi.ctc import viterbi_decode_with_states

from m2caller.kd.student_configs import build_student
from m2caller.kd.crf_kd import CRFKDLoss
from m2caller.kd.distill_train import teacher_viterbi_targets

CHUNK = 10000
SEED = 42


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_refs(path):
    refs, rid, parts = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if rid: refs[rid] = ''.join(parts)
                rid = line[1:].split()[0]; parts = []
            else: parts.append(line)
        if rid: refs[rid] = ''.join(parts)
    return refs


def signal_chunks_from_read(raw):
    shift, scale = normalisation(raw)
    sig = ((raw - shift) / scale).astype(np.float32)
    n = len(sig) // CHUNK
    if n == 0: return None
    return sig[:n * CHUNK].reshape(n, CHUNK)


def load_isolate_chunks(iso_dir, max_reads, want_ref):
    refs = load_refs(os.path.join(iso_dir, 'read_references.fasta'))
    fast5s = sorted(glob.glob(os.path.join(iso_dir, 'fast5', '*.fast5')))
    out = []
    for f5 in fast5s:
        rid = os.path.splitext(os.path.basename(f5))[0]
        ref = refs.get(rid)
        if want_ref and (not ref or len(ref) < 200): continue
        try:
            with get_fast5_file(f5, mode='r') as h:
                for r in h.get_reads():
                    raw = r.get_raw_data(scale=True).astype(np.float32); break
        except Exception:
            continue
        chunks = signal_chunks_from_read(raw)
        if chunks is None: continue
        out.append((rid, ref, chunks))
        if len(out) >= max_reads: break
    return out


def viterbi_decode_to_string(scores, idx, n_base, alphabet_arr):
    seqs = viterbi_decode_with_states(scores, idx, n_base)
    return [''.join(chr(alphabet_arr[b]) for b in em if b != 0) for em in seqs]


def run_eval(student, eval_reads, alphabet_arr, device):
    student.eval()
    idx = student.seqdist.idx
    n_base = student.seqdist.n_base
    ids = []
    with torch.no_grad():
        for rid, ref, chunks in eval_reads:
            t = torch.from_numpy(chunks).float().to(device).unsqueeze(1)
            scores = student(t).float()
            preds = viterbi_decode_to_string(scores, idx, n_base, alphabet_arr)
            full = ''.join(preds)
            if len(full) < 50 or len(ref) < 100: continue
            ed = editdistance.eval(full, ref)
            ids.append(1.0 - ed / max(len(ref), 1))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', choices=['H', 'KD'], required=True)
    ap.add_argument('--data_root', default='data/wick/train_extracted')
    ap.add_argument('--reads_per_iso', type=int, default=50)
    ap.add_argument('--eval_reads_per_iso', type=int, default=20)
    ap.add_argument('--n_eval_iso', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup_steps', type=int, default=200)
    ap.add_argument('--no_warmup', action='store_true')
    ap.add_argument('--lambda_post', type=float, default=1.0)
    ap.add_argument('--lambda_hard', type=float, default=1.0)
    ap.add_argument('--joint_post', action='store_true')
    ap.add_argument('--preset', default='m2_tiny_s5')
    ap.add_argument('--backbone', default='auto', choices=['auto', 'mamba2', 'simple'],
                    help='student encoder: mamba2 (default) or simple CNN ablation')
    ap.add_argument('--out_dir', default='checkpoints_kd/r1_scale')
    ap.add_argument('--teacher_model', default='tools/bonito_models/dna_r9.4.1_e8_sup@v3.3')
    ap.add_argument('--init_ckpt', default=None,
                    help='load student weights from this .pt file (weights only, no optimizer)')
    ap.add_argument('--start_epoch', type=int, default=0,
                    help='name new ckpts epoch{start_epoch+1+i}.pt')
    ap.add_argument('--seed', type=int, default=SEED,
                    help='RNG seed (controls init + data shuffle); for run-to-run variance studies')
    ap.add_argument('--train_iso_names', default=None,
                    help='comma-separated isolate names to restrict TRAINING to '
                         '(default: all isolates under data_root)')
    ap.add_argument('--n_train_iso', type=int, default=0,
                    help='if >0, restrict training to first N isolates (after sorting)')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[{args.branch}] device={device}  preset={args.preset}', flush=True)

    isos = sorted(glob.glob(os.path.join(args.data_root, '*')))
    isos = [d for d in isos if os.path.isdir(os.path.join(d, 'fast5'))]
    if args.train_iso_names:
        want = set(args.train_iso_names.split(','))
        isos = [d for d in isos if os.path.basename(d) in want]
        print(f'[{args.branch}] restricted training to {len(isos)} named isolate(s): '
              f'{[os.path.basename(d) for d in isos]}', flush=True)
    elif args.n_train_iso > 0:
        isos = isos[:args.n_train_iso]
        print(f'[{args.branch}] restricted training to first {len(isos)} isolate(s)', flush=True)
    print(f'[{args.branch}] {len(isos)} isolates available (seed={args.seed})', flush=True)

    train_chunks = []
    for i, iso in enumerate(isos):
        reads = load_isolate_chunks(iso, args.reads_per_iso, want_ref=False)
        for rid, ref, chunks in reads:
            train_chunks.append(chunks)
        if (i + 1) % 10 == 0:
            n_total = sum(c.shape[0] for c in train_chunks)
            print(f'[{args.branch}]   loaded {i+1}/{len(isos)} isolates, {n_total} chunks so far', flush=True)
    all_chunks = np.concatenate(train_chunks, axis=0) if train_chunks else None
    n_chunks = all_chunks.shape[0] if all_chunks is not None else 0
    print(f'[{args.branch}] TRAIN: {len(isos)} iso × {args.reads_per_iso} reads → {n_chunks} chunks', flush=True)

    eval_iso_names = [
        'Klebsiella_pneumoniae_INF078', 'Klebsiella_pneumoniae_INF007',
        'Klebsiella_pneumoniae_KSB1_6F', 'Klebsiella_pneumoniae_KSB1_7E',
        'Escherichia_coli_MSB2_1A', 'Acinetobacter_baumannii_AYP-A2',
        'Haemophilus_parainfluenzae_M1C146_1', 'Moraxella_lincolnii_51409',
    ][:args.n_eval_iso]
    eval_per_iso = {}
    for name in eval_iso_names:
        iso_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(iso_dir): continue
        refs = load_refs(os.path.join(iso_dir, 'read_references.fasta'))
        fast5s = sorted(glob.glob(os.path.join(iso_dir, 'fast5', '*.fast5')))
        fast5s_eval = fast5s[args.reads_per_iso : args.reads_per_iso + args.eval_reads_per_iso]
        ev = []
        for f5 in fast5s_eval:
            rid = os.path.splitext(os.path.basename(f5))[0]
            ref = refs.get(rid)
            if not ref or len(ref) < 200: continue
            try:
                with get_fast5_file(f5, mode='r') as h:
                    for r in h.get_reads():
                        raw = r.get_raw_data(scale=True).astype(np.float32); break
            except Exception: continue
            ch = signal_chunks_from_read(raw)
            if ch is None: continue
            ev.append((rid, ref, ch))
        eval_per_iso[name] = ev
        print(f'[{args.branch}]   eval {name}: {len(ev)} reads loaded', flush=True)

    print(f'[{args.branch}] loading teacher: {args.teacher_model}', flush=True)
    teacher = load_model(args.teacher_model, device='cuda').eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    alphabet_arr = np.frombuffer(''.join(teacher.seqdist.alphabet).encode(), dtype='u1')

    student = build_student(args.preset, backbone=args.backbone).to(device).train()
    if args.init_ckpt:
        print(f'[{args.branch}] loading init ckpt: {args.init_ckpt}', flush=True)
        ck = torch.load(args.init_ckpt, map_location=device)
        sd = ck.get('student', ck)
        missing, unexpected = student.load_state_dict(sd, strict=False)
        if missing: print(f'  missing keys: {len(missing)}', flush=True)
        if unexpected: print(f'  unexpected keys: {len(unexpected)}', flush=True)
    print(f'[{args.branch}] student: state_len={student.seqdist.state_len} '
          f'backbone={args.backbone} '
          f'params={sum(p.numel() for p in student.parameters())/1e6:.2f}M', flush=True)

    if args.branch == 'H':
        defaults = dict(lambda_post=0.0, lambda_fwd=0.0, lambda_hard=1.0)
    else:
        defaults = dict(lambda_post=args.lambda_post, lambda_fwd=0.0, lambda_hard=args.lambda_hard)
    print(f'[{args.branch}] KD weights: {defaults}  joint_post={args.joint_post}', flush=True)
    kd = CRFKDLoss(**defaults, joint_post=args.joint_post).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    log_csv = open(os.path.join(args.out_dir, 'train.csv'), 'a' if args.init_ckpt else 'w')
    if not args.init_ckpt:
        log_csv.write('epoch,step,total,post,hard,lr\n')

    t_start = time.time()
    global_step = 0
    nan_skipped = 0
    best_loss = float('inf')
    n_steps_per_epoch = (n_chunks + args.batch_size - 1) // args.batch_size
    total_steps = n_steps_per_epoch * args.epochs
    print(f'[{args.branch}] steps/epoch={n_steps_per_epoch}  total={total_steps}'
          f'  start_epoch={args.start_epoch}', flush=True)

    for epoch_idx in range(args.epochs):
        absolute_epoch = args.start_epoch + epoch_idx
        order = np.random.permutation(n_chunks)
        ep_loss = 0.0; ep_steps = 0
        for b0 in range(0, n_chunks, args.batch_size):
            batch = all_chunks[order[b0:b0 + args.batch_size]]
            sig_t = torch.from_numpy(batch).to(device).unsqueeze(1)

            if args.no_warmup:
                progress = global_step / max(1, total_steps)
                cur_lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
            elif global_step < args.warmup_steps:
                cur_lr = args.lr * (global_step + 1) / args.warmup_steps
            else:
                progress = (global_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
                cur_lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
            for g in opt.param_groups: g['lr'] = cur_lr

            with torch.no_grad():
                teacher_scores = teacher(sig_t.half()).float()
                if not torch.isfinite(teacher_scores).all():
                    nan_skipped += 1; continue
                targets, lengths = teacher_viterbi_targets(teacher_scores, teacher.seqdist)
                targets = targets.to(device); lengths = lengths.to(device)
                min_bases = teacher.seqdist.state_len + 2
                valid = lengths >= min_bases
                if not valid.any():
                    nan_skipped += 1; continue
                if not valid.all():
                    teacher_scores = teacher_scores[:, valid]
                    sig_t = sig_t[valid]
                    targets = targets[valid]
                    lengths = lengths[valid]

            student_scores = student(sig_t.float()).float()
            if not torch.isfinite(student_scores).all():
                nan_skipped += 1; continue

            losses = kd(
                teacher_scores=teacher_scores, student_scores=student_scores,
                teacher_targets=targets, target_lengths=lengths,
                teacher_seqdist=teacher.seqdist, student_seqdist=student.seqdist,
            )
            tot = losses['total']
            if not torch.isfinite(tot):
                nan_skipped += 1; continue

            opt.zero_grad()
            tot.backward()
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            if not torch.isfinite(gn):
                nan_skipped += 1; continue
            opt.step()

            ep_loss += tot.item(); ep_steps += 1; global_step += 1
            log_csv.write(f"{absolute_epoch},{global_step},{tot.item():.4f},"
                          f"{losses['post'].item():.4f},{losses['hard'].item():.4f},"
                          f"{cur_lr:.2e}\n")
            log_csv.flush()
            if global_step % 50 == 0:
                print(f'[{args.branch}] ep {absolute_epoch} step {global_step}/{total_steps} '
                      f'tot={tot.item():.3f} post={losses["post"].item():.3f} '
                      f'hard={losses["hard"].item():.3f} lr={cur_lr:.1e} '
                      f'skip={nan_skipped}  elapsed={time.time()-t_start:.0f}s', flush=True)

        avg = ep_loss / max(ep_steps, 1)
        print(f'[{args.branch}] === epoch {absolute_epoch+1}/{args.start_epoch+args.epochs} done '
              f'avg_loss={avg:.4f} steps={ep_steps} skip={nan_skipped} '
              f'elapsed={time.time()-t_start:.0f}s ===', flush=True)
        ckpt = {'student': student.state_dict(), 'epoch': absolute_epoch,
                'avg_loss': avg, 'preset': args.preset}
        torch.save(ckpt, os.path.join(args.out_dir, f'epoch{absolute_epoch+1}.pt'))
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))

    log_csv.close()

    print(f'[{args.branch}] === FINAL EVAL ===', flush=True)
    student.eval()
    eval_csv = open(os.path.join(args.out_dir, 'eval.csv'), 'w')
    eval_csv.write('isolate,n_reads,median,mean\n')
    overall_ids = []
    for name, ev in eval_per_iso.items():
        if not ev: continue
        ids = run_eval(student, ev, alphabet_arr, device)
        overall_ids.extend(ids)
        if ids:
            med = float(np.median(ids)); mn = float(np.mean(ids))
            print(f'  {name:50s}: n={len(ids):3d}  median={med*100:5.2f}%  mean={mn*100:5.2f}%', flush=True)
            eval_csv.write(f'{name},{len(ids)},{med:.4f},{mn:.4f}\n')
    if overall_ids:
        ov_med = float(np.median(overall_ids))
        ov_mean = float(np.mean(overall_ids))
        print(f'\n  OVERALL: n={len(overall_ids)} median={ov_med*100:.2f}% mean={ov_mean*100:.2f}%', flush=True)
        eval_csv.write(f'OVERALL,{len(overall_ids)},{ov_med:.4f},{ov_mean:.4f}\n')
        with open(os.path.join(args.out_dir, 'eval.txt'), 'w') as f:
            f.write(f'overall_median {ov_med:.4f}\noverall_mean {ov_mean:.4f}\nn_reads {len(overall_ids)}\n')
    eval_csv.close()
    print(f'[{args.branch}] DONE in {time.time()-t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
