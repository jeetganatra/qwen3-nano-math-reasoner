# qwen3-nano-math-reasoner

GRPO-based RLVR (Reinforcement Learning from Verifiable Rewards) training pipeline
for Qwen3-0.6B-base on the MATH dataset.
This is based on learning exercise from Reasoning From Scratch book by Sebastian Raschka

The reward signal is a 0/1 correctness check on `\boxed{...}` answers, plus a small
correctness-conditional format bonus and an unconditional length penalty for
saturating responses. Sampling is sequential (batch=1 per rollout) because of some weird inconsistencies caused when batching sampling as the kernel changes when bs>1. This seems to
have to do with sensitivity of smaller model and less robust inferenece implementation in `reasoning_from_scratch` package. More details
should be available on my blog[ToUpdate]

## Headline result

The training loop ran for 111 steps (interrupted before the planned 500) with held-out
MATH-500 evaluations on a 50-example subset at two checkpoints:

| Step | MATH-500 acc | Correct | Train correctness (last 10) | Train reward (last 10) |
|---|---|---|---|---|
| 50  | **44.0%** | 22/50 | 0.725 | 0.761 |
| 100 | **40.0%** | 20/50 | 0.775 | 0.814 |

Training-rollout correctness on in-distribution problems climbed from 0.725 to 0.775
between the two checkpoints, but held-out eval accuracy moved in the opposite direction.
This is the canonical train/eval gap that shows up in cold-start RLVR on small base models.
The model rapidly masters the training-set problem distribution; that mastery does
not transfer one-for-one to held-out problems.

Full per-step training curves and eval breakdown:

- [`remote_artifacts/training_curves_logp_batched.png`](remote_artifacts/training_curves_logp_batched.png)
- [`remote_artifacts/eval_summary.md`](remote_artifacts/eval_summary.md)

The 500 evaluation problems are explicitly excluded from the training pool (the source
dataset is named `math_full_minus_math500`).

## Files

| File | Purpose |
|---|---|
| [`qwen_grpo.py`](qwen_grpo.py) | Reference GRPO trainer with sequential sampling and per-rollout logp computation |
| [`qwen_grpo_logp_batched.py`](qwen_grpo_logp_batched.py) | **Production version.** Sequential sampling + batched logp scoring.|
| [`qwen_grpo_batched.py`](qwen_grpo_batched.py) | Fully-batched experiment (sampling + logp). Diverges in fp32 due to cuBLAS routing batched vs unbatched matmuls through different kernels — see top-of-file note. Kept for documentation |
| [`diag_batched.py`](diag_batched.py) | Diagnostic that compares batched vs unbatched first-token logits for the same prompt. Confirmed the kernel-level fp32 divergence |
| [`plot_runs.py`](plot_runs.py) | Generates per-run training-curve PNGs and the combined eval table. Extensible to additional runs by appending to the `RUNS` list |

## Reward shape

```python
correctness   = 1.0 if extracted == ground_truth else 0.0
format_bonus  = 0.05 if (correctness and \boxed{} present) else 0.0
length_penalty = 0.1 if gen_len > 0.8 * max_new_tokens else 0.0

reward = correctness + format_bonus - length_penalty
```

Two design choices worth noting:

1. **Format bonus is conditional on correctness.** Independent positive reward for
   emitting `\boxed{}` is exploitable and the policy will learn to emit `\boxed{N}`
   immediately with no reasoning to claim it. Making the bonus contingent on
   actually solving the problem removes the gaming attractor.
2. **Length penalty is unconditional.** It fires whenever a rollout saturates at
   `0.8 × max_new_tokens`, regardless of correctness, so the policy is
   discouraged from rambling whether right or wrong.

## Run command

This is according to the hyperparams that were used for the main run:

```bash
python qwen_grpo_logp_batched.py \
  --steps 500 \
  --num_rollouts 8 \
  --max_new_tokens 768 \
  --clip_eps 0.2 \
  --kl_coeff 0.02 \
  --inner_epochs 1 \
  --format_bonus_weight 0.05 \
  --length_penalty_weight 0.1 \
  --length_penalty_threshold_frac 0.8 \
  --skip-zero-advantage-updates \
  --eval_on_checkpoint 50 \
  --show_eta
```

Key sanity check: `policy_ratio` should print as `1.0000` on every active step (the
gradient is on-policy under `inner_epochs=1`). If it drifts away from 1.0 with no
parameter updates between sampling and loss computation, something is biasing the
log-probability estimator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch reasoning-from-scratch numpy requests sympy tokenizers
```

On a CUDA box you may need to pin a torch build matching your driver, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124` for
CUDA 12.4 drivers.

Model weights, tokenizer, and the MATH train/eval data download lazily on
first run via the `reasoning_from_scratch` package.

## Reproducing the plots

```bash
python plot_runs.py
```

Reads `remote_artifacts/qwen_grpo_logp_batched/logs/qwen_grpo_logp_batched_metrics.csv`
and writes:

- `remote_artifacts/training_curves_logp_batched.png`
- `remote_artifacts/eval_summary.{md,csv}`

To compare against additional runs, append a new entry to the `RUNS` list at the top
of the script and re-run; the plots and table will pick up the new run automatically.
