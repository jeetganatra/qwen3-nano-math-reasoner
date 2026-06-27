# OPD-from-base showcase — a clean capability rise

After Phase 1 showed OPD *from the seed* pins/holds (the seed is saturated), this run
demonstrates OPD where it actually has headroom: **from the base model toward Qwen3-4B**.

## Result

| step | 50-subset @2048 | note |
|---|---|---|
| base | 14% (7/50) | floor |
| 40 | 26% (13/50) | climbing |
| **80** | **30% (15/50)** | **peak** |
| 120 | 20% (10/50) | over-training degrades |

**Full-500 confirmation of the step-80 peak: 28.8% (144/500) @2048** — the subset peak is
real. So OPD-from-base lifts the 0.6B **~16% → 28.8%** on full MATH-500, a ~1.8× rise.

| Model | MATH-500 (full) |
|---|---|
| base | ~16% |
| **OPD-from-base (step 80)** | **28.8%** |
| SFT seed (235B offline distill) | 37.8% |
| Qwen3-0.6B-reasoning (official) | 60.4% |

## Recipe (what worked)

`--student_checkpoint base --teacher_model_id Qwen/Qwen3-4B --distill_coeff 1.0
--kl_type jsd --kl_coeff 0.05 --ent_coef 1e-3 --kl_clamp 10 --warmup_steps 15
--num_rollouts 4 --max_new_tokens 768 --lr 2e-6 --sampling batched`, eval every 40 @2048.
**Early-stop at the peak (step 80).** k3 stays ~0.001 throughout — the climb comes from
gentle logit nudges flipping greedy decisions (eliciting latent capability), not large KL
drift.

## What did NOT work

- **Aggressive tuning backfired:** lr 5e-6 + kl_coeff 0.02 (meant to climb faster) drove the
  fragile base *below* the floor (10% @ step 40). From base, gentle lr is essential.
- **Running it out:** past the step-80 peak the policy over-trains and degrades (30→20%),
  the same pathology as the SFT past best-val (37.8%→21% by epoch 3) and the no-anchor seed
  run. OPD-from-base is a peaked curve, not monotonic — early-stopping is the whole game.

## Honest interpretation

This is **not OPD teaching new math** — it is OPD **eliciting + formatting latent capability
the Qwen3 pretraining already deposited**. Qwen3's pretraining has a dedicated *reasoning
stage* (+5T tokens upsampling STEM/code/reasoning, with synthetic math from Qwen2.5-Math;
[Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)), so the base is "pseudo-
pretrained on math" — it just hasn't learned the `<think>…\boxed{}` chat format (its rollouts
ramble, `capped=1.00`). OPD surfaces and formats that latent skill on-policy.

This also explains the ceiling: the peak (~29%) lands **below** the offline-SFT seed (37.8%)
because on-policy elicitation from base extracts *less* of the same pretrained capability
than 12K curated 235B traces × 3 epochs do. Both surface pretrained math; curated offline
SFT surfaces more. And it's **not representative** of how labs deploy OPD (they run it on a
warm-started reasoner, not base) — it's the from-base headroom that makes the rise visible.

## Artifacts
- Checkpoints: `results/phase1/opd_frombase_v2/checkpoints/` (step 40/80/120; **80 = best**)
- Full-500 eval: `results/phase1/opd_frombase_v2_step80_full500_2048.jsonl`
- Curve log: `results/phase1/opd_frombase_v2.log`
