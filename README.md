# CRF-KD

Code and checkpoints for **"CRF-KD: Distilling CTC-CRF Nanopore Basecallers via
Joint Transition-Posterior Matching"** (IEEE BIBM submission).

Standard knowledge distillation matches softmax outputs and ignores the
structured CRF lattice that nanopore basecallers actually decode. CRF-KD distils
a frozen Bonito sup teacher into compact Mamba2 students by matching the
teacher's **joint transition posterior** over the CRF lattice, paired with a hard
CTC-CRF anchor on the teacher's Viterbi decode.

The central finding is algorithmic: matching Bonito's readily available **β-only
conditional** posterior *degrades* the student, while matching the **true joint**
posterior improves it — a sign flip that holds at every training scale.

## Layout

```
m2caller/kd/             the method (see m2caller/kd/README.md for loss derivations)
  crf_kd.py              L_post / L_fwd / L_hard and the combined CRFKDLoss; pure-PyTorch target-constrained forward
  student_configs.py     M2_SMALL / M2_TINY presets and the CRFStudent wrapper
  filter.py              chunk-entropy filter and curriculum scheduler
  distill_train.py       CLI entry point, teacher-Viterbi pseudo-labelling, training loop, CPU smoke test
koi/                     audited pure-PyTorch CTC-CRF Viterbi + forward/backward (no proprietary CUDA kernels)
scripts/
  train_r1_scale.py      the training entry point used for all published runs
  eval_kd_student.py     student evaluation via the audited Viterbi decoder
  run_kd_validation.py   single-isolate validation sweep (Table I)
  build_r2_summary.py    R2 summary tables
  phase2/                full-scale P2 training, NanoSpecies-50 evaluation, throughput benchmark
checkpoints_kd/          one best.pt per run, plus per-epoch train.csv / eval.csv / eval.txt
logs/                    evaluation CSVs behind the published tables
figures/                 figure-generation script and the paper PDFs
```

### Checkpoint map

| Directory | Run |
|---|---|
| `v4_main_kdjp`, `v4_main_h` | main CRF-KD vs. hard-only comparison |
| `p2_kdjp_small`, `p2_h_small` | P2 scale (50 iso x 100 reads x 20 ep), M2-Small |
| `p2_kdjp_tiny`, `p2_h_tiny` | P2 scale, M2-Tiny |
| `r1_kdjp`, `r1_h`, `r1_betaonly` | R1 budget: joint KL vs. hard vs. **beta-only** (the sign-flip ablation) |
| `r1_m2small_kdjp` | R1 budget, M2-Small |
| `r2_kdjp_small`, `r2_kdjp_tiny`, `r2_h` | R2 scale (50 x 100 x 5) |
| `arch_cnn_kd`, `arch_cnn_h` | architecture control: CNN encoder instead of Mamba2 (Table III) |
| `sweep_lam{0.25,0.5,1.0,2.0,4.0}` | lambda_post sensitivity |
| `sweep_{KD,H}_seed{1,7,42}` | 3-seed repeats (Table VI) |
| `validation*` | single-isolate loss-formulation ablation (Table I) |

Per-epoch `epochN.pt` files are not published (1.4 GB); each run ships the
`best.pt` that its published number was measured on, plus its full training curve.

## Install

```bash
pip install torch numpy editdistance ont-fast5-api
pip install ont-bonito              # teacher loading (bonito.util / bonito.reader)
pip install mamba-ssm causal-conv1d  # optional, CUDA-only; falls back to a CNN backbone otherwise
export PYTHONPATH="$PWD:$PYTHONPATH" # make the bundled pure-PyTorch koi importable
```

The bundled `koi` replaces ONT's closed-source CUDA package, so training and
evaluation run on CPU, CUDA, and ARM GH200 alike.

## Smoke test

No teacher and no data required — runs on CPU with a simple CNN backbone and
fabricated teacher scores:

```bash
python -m m2caller.kd.distill_train --smoke_test
```

## Train

All published runs go through `scripts/train_r1_scale.py`, which reads the
isolate FAST5 directories directly. `--branch` selects the objective and
`--joint_post` is what makes it CRF-KD rather than beta-only matching:

```bash
# CRF-KD (joint transition posterior + hard anchor)
python scripts/train_r1_scale.py --branch KD --joint_post \
    --preset m2_small_s5 \
    --data_root data/wick/train_extracted \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 20 --batch_size 32 --lr 2e-4 \
    --lambda_post 1.0 --lambda_hard 1.0 \
    --out_dir checkpoints_kd/p2_kdjp_small

# hard-target baseline
python scripts/train_r1_scale.py --branch H \
    --preset m2_small_s5 \
    --data_root data/wick/train_extracted \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 20 --batch_size 32 --lr 2e-4 \
    --out_dir checkpoints_kd/p2_h_small
```

Dropping `--joint_post` from the `KD` branch gives the **beta-only** variant that
the paper shows *degrades* the student. Presets are `m2_tiny_s5` (1.07 M) and
`m2_small_s5` (3.51 M); the published P2 runs resume from R2 epoch 5 via
`--init_ckpt ... --start_epoch 5`.

Full-scale P2 runs and their NanoSpecies-50 evaluation are orchestrated by
`scripts/phase2/` (see `scripts/phase2/README.md` for the stage-by-stage budget);
`run_r1_betaonly.sh`, `run_arch_cnn_baseline.sh` and `lambda_seed_sweep.sh`
reproduce the ablations.

`L_fwd` is disabled in the published configuration — it collapses the student
(Table I). The default `lambda_post = 1.0` is used for all full-scale results.

### Known gaps in this release

Two helper modules that predate the paper were lost from the authors' workspace
and are therefore **not** in this release; only compiled `.pyc` artefacts
survived. They do not affect any published number, because every published run
was produced by `scripts/train_r1_scale.py`, which does its own FAST5 loading:

| Missing module | Affects |
|---|---|
| `m2caller.utils.dataset` | the `run_full_training()` branch of `m2caller/kd/distill_train.py`, which its own docstring describes as a skeleton |
| `m2caller.utils.labeled_dataset` | `scripts/run_kd_validation.py` (the Table I sweep); the losses it exercises are covered by the CPU smoke test |

The loss implementations, the student architectures, the decoder and the
published training path are all complete.

## Data and teacher

Neither the corpus nor the teacher checkpoint is redistributed here:

- **Teacher** — public Bonito sup R9.4.1 `dna_r9.4.1_e8_sup@v3.3` (27 M params,
  frozen), via `bonito download --models`.
- **Corpus** — 50 bacterial isolates from the public Wick *et al.* set. Training
  budgets: R1 (50 iso x 20 reads x 3 ep), R2 (50 x 100 x 5), P2 (50 x 100 x 20).

Evaluation uses NanoSpecies-50 (50 isolates x 100 deterministic reads, per-read
identity by edit distance, median-of-medians) with the audited pure-PyTorch
Viterbi decoder, which is vendored here under `koi/` and released in full with
the companion benchmark paper.

## Main results

Which posterior to match (M2-Tiny, 8-isolate hold-out median identity):

| Training scale | Hard | beta-only KL | Joint KL |
|---|---|---|---|
| 1 iso, 200 reads, 4 ep | 45.2 | 28.1 | **72.7** |
| 50 iso, 20 reads, 3 ep (R1) | 76.4 | 70.0 | **78.9** |

The sign is unchanged across scales: beta-only KL always underperforms the
hard-target baseline, joint KL always beats it.

Architecture control at R1 budget (8-isolate hold-out):

| Encoder | Hard | CRF-KD | delta |
|---|---|---|---|
| Mamba2 (1.07 M) | 76.4 | 78.9 | +2.5 |
| CNN (1.00 M) | 73.3 | 76.8 | +3.5 |

Three seeds at the same slice: hard-only 72.08 +/- 0.55 % vs. CRF-KD
78.39 +/- 0.52 %. The +6.31 pp gap exceeds 11x either standard deviation, so the
joint-posterior gain is not seed noise.

At full P2 scale the soft prior shrinks: the best compact student (M2-Small,
hard, 87.70 %) sits within 3.4 pp of Bonito sup at 7.7x fewer parameters, while
CRF-KD lifts the 1.07 M Tiny frontier point — the regime that fits a portable
device. KD matters most when the student is data- or parameter-starved.

## Reproduce tables and figures

`logs/` and `checkpoints_kd/*/eval.*` contain the numbers behind every published
table, so the analysis can be regenerated without a GPU:

```bash
python scripts/phase2/summarise_main.py
python scripts/phase2/summarise_ablation.py
python scripts/build_r2_summary.py
python figures/make_figures.py    # 10 of the 11 paper figures
python figures/make_pareto.py     # figures/pareto.pdf
```

`make_figures.py` is shared with the companion benchmark paper: it writes this
paper's figures to `figures/` and the NanoSpecies ones to
`figures/companion_nanospecies/` (untracked).

The shell drivers under `scripts/` call `${PY:-python3}`, so point `PY` at the
interpreter you want:

```bash
PY=/path/to/venv/bin/python bash scripts/phase2/run_main.sh
```
