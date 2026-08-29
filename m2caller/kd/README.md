# `m2caller.kd` — CRF-Structured Knowledge Distillation

Scaffolding for **Direction B** of the research proposal
(`docs/research_proposal.md` §B): distilling a frozen Bonito sup
(CTC-CRF) teacher into a small Mamba2-based student (M2Caller).

> "Gap: no one has distilled the CRF transition score distribution
> from a large basecaller into a smaller one."
> — *research_proposal.md* §B.2

## What's here

| File | Purpose |
|------|---------|
| `crf_kd.py`          | Three distillation losses + combined `CRFKDLoss` module. |
| `filter.py`          | `compute_chunk_entropy`, `EntropyFilter`, `CurriculumScheduler`. |
| `student_configs.py` | `M2_SMALL` / `M2_TINY` presets and the `CRFStudent` wrapper (backbone + `LinearCRFEncoder` head). |
| `distill_train.py`   | CLI entry point, teacher-Viterbi pseudo-labeling, full training skeleton, smoke test. |

## The three losses (§B.3.2)

Let `P_t(s, x)` = teacher transition posterior from forward-backward,
`α_t(s)` = normalised forward log-scores, `y^T` = teacher's Viterbi decode.

* **L_post** `KL(P^T_t ‖ P^S_t)` summed over `(s, x)`, mean over `(t, n)`.
* **L_fwd**  `‖ α^T_t − α^S_t ‖²` on per-timestep-normalised forward logs.
* **L_hard** `-log P^S(y^T | x)` — CTC-CRF NLL on the teacher's Viterbi decode.

Total:  `L = λ₁·L_post + λ₂·L_fwd + λ₃·L_hard`  (defaults 1.0 / 0.5 / 1.0).

The forward/backward machinery is borrowed from Bonito's
`CTC_CRF.forward_scores`, `backward_scores`, `compute_transition_probs`
(which in turn call our pure-PyTorch `koi.ctc.{fwd,bwd}_scores_cu_sparse`
stubs).

### `state_len` compatibility fallback

Teacher (Bonito sup, `state_len=5`) and student may not share `state_len`.
In the research proposal we allow the student to run with `state_len=3`
(64 states) for extra speed (§B.6). When `teacher.state_len !=
student.state_len`:

* **L_post** and **L_fwd** are **skipped** (their lattices are not aligned).
* **L_hard** is still evaluated using the student's own `seqdist`.
* A one-time `RuntimeWarning` is emitted at first forward call.

Pass `require_same_state_len=True` to `CRFKDLoss` to make this
configuration an error instead of a warning.

### `l_hard_kd` without `logZ_cu`

The shipped `koi.ctc.logZ_cu` stub raises `NotImplementedError` (it is the
target-constrained forward normally provided by ONT's proprietary CUDA
kernels). We therefore include a small pure-PyTorch
`_logZ_target_forward` in `crf_kd.py` that replicates the inner logic of
`CTC_CRF.ctc_loss` on top of `prepare_ctc_scores`. The stay / move scores
produced by Bonito are used verbatim.

## Smoke test

No teacher / data needed — runs entirely on CPU with a simple CNN backbone
and fabricated teacher scores:

```bash
python -m m2caller.kd.distill_train --smoke_test
```

Expected output (scalar values will differ run-to-run):

```
[smoke] building tiny student (CPU, simple backbone)
[smoke] student output: T=128 N=2 n_score=5120 (state_len=5, n_base=4)
[smoke] teacher targets shape=(2, 50) lengths=[47, 45]
[smoke] losses: total=12.34 post=3.21 fwd=0.98 hard=8.15
[smoke] mismatched state_len fallback: post=0.00 fwd=0.00 hard=8.15
[smoke] entropies=[...] threshold=... keep=[True, True]
[smoke] curriculum: epoch 0 -> 50%, epoch 15 -> 75%, epoch 40 -> 100%
[smoke] final loss: 12.344567
[smoke] OK
```

## Running real distillation

```bash
python -m m2caller.kd.distill_train \
    --teacher_model tools/bonito_models/dna_r9.4.1_e8_sup@v3.3 \
    --student_preset m2_small \
    --data_dir data/wick/train_extracted \
    --out_dir checkpoints_kd/m2_small \
    --lambda_post 1.0 --lambda_fwd 0.5 --lambda_hard 1.0 \
    --use_entropy_filter --use_curriculum \
    --epochs 50 --lr 2e-4
```

Pipeline inside the training loop (cf. §B.3):

```python
with torch.no_grad():
    teacher_scores  = teacher(signal).float()
    teacher_targets, target_lengths = teacher_viterbi_targets(
        teacher_scores, teacher.seqdist,
    )
student_scores = student(signal).float()
loss_dict = kd_loss(
    teacher_scores, student_scores, teacher_targets, target_lengths,
    teacher_seqdist=teacher.seqdist,
    student_seqdist=student.seqdist,
)
loss_dict["total"].backward()
```

## Design notes

* **λ defaults `(1.0, 0.5, 1.0)`** follow the proposal's implicit weighting
  — the posterior KL and hard CTC-CRF anchor are both first-class
  objectives, the forward-score MSE is a softer regulariser so we scale it
  down.
* **Teacher targets** are produced by `koi.ctc.viterbi_decode_with_states`
  (emits base indices 1..n_base, 0 for stay), which we compress to a
  padded `(N, L)` tensor consumed by `prepare_ctc_scores`.
* **Entropy filter** keeps a rolling buffer of per-chunk entropies and
  estimates the cutoff via `torch.quantile`. A `CurriculumScheduler`
  drives that percentile from 50 (easy chunks only) to 100 (all chunks)
  over the first 30 epochs (§B.3.4).
* **Mamba2 is CUDA-only**, so `build_student(..., backbone="auto")` falls
  back to a CNN-only backbone (`_SimpleBackbone`) when no GPU is present.
  This is what makes the CPU smoke test possible.
