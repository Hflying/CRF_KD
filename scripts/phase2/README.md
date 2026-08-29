# Phase 2 — ICASSP'27 main + ablation runs (option B)

This folder contains the scripts that produce **everything still missing for
the ICASSP'27 CRF-KD paper** based on the May 2026 plan:

- **P0a · main runs** — 4 students × 20 epochs on 50-iso × 100-reads
  (resumes 3 of them from R2 epoch 5)
- **P0b · NanoSpecies-50 student eval** — populate `Δ to teacher` in §3.3
- **P1a · loss ablation** — 5 configs × M2_SMALL × +10 epochs
- **P1b · throughput** — reads/sec on a single GH200 for sup, hac, and the
  4 students

Bonito sup teacher reference (`logs/bonito_sup_all.csv`) is **already done**
and not regenerated here.

## Layout

```
scripts/phase2/
├── README.md                       # this file
├── run_all.sh                      # ★ master orchestrator (sequential)
├── run_main.sh                     # P0a · 4 main training stages
├── run_ablation.sh                 # P1a · 5 ablation stages
├── eval_nanospecies50.sh           # eval all 4 main students on NS-50×100
├── eval_ablation50.sh              # eval all 5 ablation students on NS-50×100
├── eval_student_nanospecies50.py   # python core for the eval scripts
├── summarise_main.py               # writes main_table.{md,tex,tsv}
├── summarise_ablation.py           # writes ablation_table.{md,tex,tsv}
├── throughput_benchmark.py         # writes throughput.{md,json}
└── status.sh                       # one-glance progress dashboard
```

Outputs go to `checkpoints_kd/p2_*/` (per-stage ckpts + per-stage 8-iso CSV
from the training script's own eval) and `logs/phase2/` (master orchestrator
log, main eval CSVs on 50-iso × 100-reads, summary tables).

## Budget summary (single GH200)

| Stage | Detail | GPU·h | Wall-clock |
|---|---|---:|---:|
| P0a · M1 | H · Tiny  · resume +15 ep | ~64 | ~2.7 d |
| P0a · M2 | KD · Tiny · resume +15 ep | ~125 | ~5.2 d |
| P0a · M3 | H · Small · 20 ep from scratch | ~120 | ~5.0 d |
| P0a · M4 | KD · Small · resume +15 ep | ~162 | ~6.8 d |
| P0b · main eval | 4 students × NS-50 × 100 reads | ~10 | ~0.4 d |
| P1a · A1–A5 | Small × 10 ep × 5 configs | ~560 | ~23 d |
| P1a · ablation eval | 5 students × NS-50 × 100 reads | ~13 | ~0.5 d |
| P1b · throughput | 6 models × micro-bench | <1 | <1 h |
| **Total** | | **~1055** | **~44 d** |

The 44-day single-card estimate is what option B was promising. If you have
headroom you can:

- Run main + ablation **concurrently** on different GPUs (each stage is
  independent of the others' weights once R2 ep5 is loaded).
- Drop **A4 and A5** to ~700 GPU·h ≈ 30 d wall-clock with mild
  reviewer pushback.

## How to launch

The simplest is one command in the background:

```bash
nohup bash scripts/phase2/run_all.sh > logs/phase2/orch.out 2>&1 &
disown
```

Then monitor with any of:

```bash
bash scripts/phase2/status.sh        # ASCII dashboard
tail -f logs/phase2/main.log         # live training log
ls -lat checkpoints_kd/p2_*          # most recent ckpts
```

## Step-by-step (if you prefer to run manually)

```bash
# 1) Main runs (~470 GPU·h)
bash scripts/phase2/run_main.sh                   # → checkpoints_kd/p2_*

# 2) Eval main runs on NanoSpecies-50 (~10 GPU·h)
bash scripts/phase2/eval_nanospecies50.sh         # → logs/phase2/eval50_*.csv
                                                  #   logs/phase2/main_table.{md,tex,tsv}

# 3) Loss ablation (~560 GPU·h)
bash scripts/phase2/run_ablation.sh               # → checkpoints_kd/p2_abl_*

# 4) Eval ablation on NanoSpecies-50 (~13 GPU·h)
bash scripts/phase2/eval_ablation50.sh            # → logs/phase2/ablation_table.{md,tex,tsv}

# 5) Throughput benchmark (~1 h)
python scripts/phase2/throughput_benchmark.py \
    --out logs/phase2/throughput.json             # → logs/phase2/throughput.{md,json}
```

## Resume / restart semantics

- All training stages save `epochN.pt` after every epoch and `best.pt`
  whenever epoch loss improves. Killing the orchestrator and re-launching
  `bash scripts/phase2/run_all.sh` is **not** a clean resume — each
  per-stage script will see its `out_dir` already exists and start from
  epoch 1 again. To genuinely resume a partially-finished stage you need
  to:
    1. Move the existing `out_dir` aside (keep it as a safety copy).
    2. Update the stage's `--init_ckpt` to point at the latest
       `epochN.pt` and reduce `--epochs` accordingly.
- Once a `logs/phase2/eval50_*.csv` exists, the eval scripts skip its
  re-generation. Delete the CSV to force a re-eval.

## What lands in the paper

After all stages finish:

1. `logs/phase2/main_table.md` → drop into `icassppaper/paper.md` §3.3 to
   replace the placeholder Table 1.
2. `logs/phase2/ablation_table.md` → drop into §3.4 to replace Table 2.
3. `logs/phase2/throughput.md` → drop into §3.3 (or extend Table 1 with
   reads/s columns).
4. `logs/phase2/main_table.tex` and `ablation_table.tex` are direct
   booktabs LaTeX for the camera-ready.

The Phase 1.5 R2 prelim section already in `paper.md` §3.2 stays — Phase 2
results supersede it as the **main** result, but Phase 1.5 remains a
useful "validation that the pipeline works" appendix.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Resume from R2 ep5 with new optimizer state diverges | `--lr 1e-4 --no_warmup`; AdamW restarts from clean state. R2 logs show loss has plateaued, so this is the *less* risky regime. Watch ep6's `tot=` line in main.log: if it spikes >2× the R2 ep5 final loss, abort and rerun with `--lr 5e-5`. |
| H-Small from scratch lands far below KD-Small | Expected — that's the headline. If H-Small overtakes KD-Small the paper has a bigger problem (i.e., KD doesn't help). Check after main eval. |
| Ablation row A1 (hard only) was already trained as p2_h_small | Yes — A1's `best.pt` should match p2_h_small's. We re-train A1 anyway from R2-KD-Small ep5 for an apples-to-apples ablation start point. |
| `eval_student_nanospecies50.py` uses chunk_size=10000 (matches sup) but training also uses 10000 | OK — see comment at top of `eval_student_nanospecies50.py`. |
| Disk pressure (each ckpt ~64 MB Tiny / ~320 MB Small × 20 ep × 9 runs ≈ 25 GB) | Acceptable on this workspace; run `find checkpoints_kd/p2_* -name 'epoch[0-9].pt' -delete` after final ckpts to keep only `epochN.pt` (final) and `best.pt`. |
