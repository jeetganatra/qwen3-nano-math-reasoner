# Eval summary

## GRPO training run (`logp_batched`)

50-problem MATH-500 subset evaluated at training checkpoints.

| Run | Step | MATH-500 acc | Correct | Total | Train correct (last 10) | Train reward (last 10) | Entropy (last 10) | Resp len (last 10) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logp_batched | 50 | 44.0% | 22 | 50 | 0.725 | 0.761 | 0.225 | 369 |
| logp_batched | 100 | 40.0% | 20 | 50 | 0.775 | 0.814 | 0.244 | 330 |

## SFT distillation cross-method comparison

Full MATH-500 (n=500), `max_new_tokens=4096`, greedy decoding.

| Model | Accuracy | Correct/N | Avg gen chars | Median gen chars |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B base (no fine-tuning) | 16.0% | 80/500 | 216 | 12 |
| **DeepSeek-SFT (epoch 3, 12K V4 Pro traces, cross-family)** | **34.6%** | **173/500** | **5,835** | **3,817** |
| **Qwen-SFT (best-val, 12K Qwen3-235B-A22B traces, same-family, `max_seq_len=2048`)** | **37.8%** | **189/500** | **34,274** | **41,541** |
| Qwen3-0.6B reasoning (Qwen official) | 60.4% | 302/500 | 5,847 | 4,042 |

The Qwen-SFT row's huge avg/median gen-chars (34K/41K) is post-answer rambling: the
student emits `</think>\boxed{answer}` properly 79.4% of the time but then keeps
generating "Actually, let me double-check..." prose until `max_new_tokens=4096` is hit.
The correct answers are usually at the front of the output; the rambling is cosmetic
and doesn't affect accuracy.

Raw eval JSONLs: [`remote_artifacts/qwen_distill/evals/`](qwen_distill/evals).
Aggregated CSVs: [`comparison.csv`](qwen_distill/comparison.csv), [`comparison_by_level.csv`](qwen_distill/comparison_by_level.csv), [`overlap_matrix.csv`](qwen_distill/overlap_matrix.csv).

### Per-difficulty breakdown

| Level | Distilled | Base | Qwen reasoning |
| --- | --- | --- | --- |
| 1 | 65.1% (28/43) | 25.6% (11/43) | 83.7% (36/43) |
| 2 | 61.1% (55/90) | 27.8% (25/90) | 76.7% (69/90) |
| 3 | 41.0% (43/105) | 23.8% (25/105) | 68.6% (72/105) |
| 4 | 23.4% (30/128) | 8.6% (11/128) | 60.2% (77/128) |
| 5 | 12.7% (17/134) | 6.0% (8/134) | 35.8% (48/134) |

Distillation lift over base decreases monotonically with difficulty: +39.5 pp on Level 1, +6.7 pp on Level 5. The hardest problems require multi-step reasoning that 0.6B can't fully absorb from supervised traces.

### Solution overlap

| Pair | Both correct | Only first | Only second | Neither |
| --- | --- | --- | --- | --- |
| distilled vs base | 49 | 124 | 31 | 296 |
| distilled vs qwen_reasoning | 148 | 25 | 154 | 173 |
| base vs qwen_reasoning | 63 | 17 | 239 | 181 |

The distilled model uniquely solves 25 problems that Qwen reasoning misses — partial independence in predictions. Oracle ensemble (distilled OR qwen_reasoning) would hit 65.4%.
