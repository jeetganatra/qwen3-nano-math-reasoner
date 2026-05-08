"""Batched GRPO trainer for Qwen3-0.6B on MATH.

Same training logic and reward shape as qwen_grpo.py, but with three batched
primitives:

1. sample_response_batched: all `num_rollouts` rollouts share the same prompt,
   so we replicate to batch=N and run a single autoregressive loop. Per-row EOS
   tracking; finished rows still consume forward passes (KV cache shape must
   stay consistent) but their generated tokens are ignored.

2. batched_sequence_logprob: pads `num_rollouts` rollouts to max length and
   runs one fresh forward instead of N sequential forwards. Padding positions
   only contaminate their own (ignored) outputs because attention is causal.

3. compute_grpo_loss uses the batched logprob fn for the with-grad new_logp.

CLI, metrics, eval, and checkpoints are identical to qwen_grpo.py — drop-in.
"""
import argparse
import copy
import time
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    evaluate_math500_stream,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
    load_model_and_tokenizer,
    load_math500_test,
    load_tokenizer_only,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import (
    load_math_train,
)
from reasoning_from_scratch.qwen3 import KVCache, Qwen3Model, QWEN_CONFIG_06_B

SCRIPT_NAME = Path(__file__).stem
LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_outputs.txt"
METRICS_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.txt"
CSV_LOG_PATH = Path(__file__).parent / "logs" / f"{SCRIPT_NAME}_metrics.csv"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / SCRIPT_NAME


@torch.no_grad()
def sample_response_batched(
    model,
    tokenizer,
    prompt,
    device,
    num_rollouts,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
):
    """Generate `num_rollouts` rollouts in one batched autoregressive loop.

    Returns a list of (token_ids, prompt_len, text) tuples — one per rollout —
    matching sample_response's per-rollout output.
    """
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    prompt_len = prompt_ids.numel()
    batch_input = prompt_ids.unsqueeze(0).expand(num_rollouts, -1).contiguous()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(batch_input, cache=cache)[:, -1]  # (N, V)

    finished = torch.zeros(num_rollouts, dtype=torch.bool, device=device)
    generated = [[] for _ in range(num_rollouts)]
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        if temperature and temperature != 1.0:
            logits = logits / temperature
        probas = torch.softmax(logits, dim=-1)
        probas = top_p_filter(probas, top_p)
        next_token = torch.multinomial(probas, num_samples=1)  # (N, 1)

        sampled = next_token.squeeze(-1).tolist()
        for i in range(num_rollouts):
            if finished[i]:
                continue
            tid = sampled[i]
            generated[i].append(tid)
            if eos_id is not None and tid == eos_id:
                finished[i] = True

        if bool(finished.all()):
            break

        # Forward all rows; finished rows produce ignored outputs but keep the
        # KV-cache shape consistent.
        logits = model(next_token, cache=cache)[:, -1]

    out = []
    for i in range(num_rollouts):
        full_ids = torch.cat([
            prompt_ids,
            torch.tensor(generated[i], device=device, dtype=prompt_ids.dtype),
        ])
        text = tokenizer.decode(generated[i])
        out.append((full_ids, prompt_len, text))
    return out


def reward_rlvr(answer_text, ground_truth):
    extracted = extract_final_candidate(answer_text, fallback=None)
    if not extracted:
        return 0.0
    correct = grade_answer(extracted, ground_truth)
    return float(correct)


def has_boxed(answer_text):
    return extract_final_candidate(answer_text, fallback=None) is not None


def length_penalty(gen_len, max_new_tokens, threshold_frac=0.8):
    return 1.0 if gen_len > int(threshold_frac * max_new_tokens) else 0.0


def batched_sequence_logprob(model, token_ids_list, prompt_lens, return_entropy=False, pad_id=0):
    """Sum-of-logprobs (and optionally mean entropy) for each rollout via one padded forward.

    Each row is padded to max_len but only the first `len(token_ids)` positions
    are used when gathering target logprobs. Causal attention means real
    positions never attend to padding, so per-row outputs at real positions are
    bit-for-bit equivalent to running each sequence on its own.
    """
    device = token_ids_list[0].device
    lengths = [t.numel() for t in token_ids_list]
    max_len = max(lengths)
    n = len(token_ids_list)

    batch = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(token_ids_list):
        batch[i, :lengths[i]] = t

    logits = model(batch).float()                   # (N, max_len, V)
    logprobs = torch.log_softmax(logits, dim=-1)

    logp_sums = []
    entropies = []
    for i, length in enumerate(lengths):
        plen = prompt_lens[i]
        targets = batch[i, 1:length]
        selected = logprobs[i, :length - 1].gather(1, targets.unsqueeze(-1)).squeeze(-1)
        logp_sums.append(selected[plen - 1:].sum())
        if return_entropy:
            ans_logprobs = logprobs[i, plen - 1:length - 1]
            if ans_logprobs.numel() == 0:
                entropies.append(logp_sums[-1].new_tensor(0.0))
            else:
                ans_probs = torch.exp(ans_logprobs)
                step_ent = -(ans_probs * ans_logprobs).sum(-1)
                entropies.append(step_ent.mean())

    if return_entropy:
        return torch.stack(logp_sums), torch.stack(entropies)
    return torch.stack(logp_sums)


def collect_rollouts(
    model,
    ref_model,
    tokenizer,
    example,
    device,
    num_rollouts=8,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    kl_coeff=0.001,
    format_bonus_weight=0.05,
    length_penalty_weight=0.1,
    length_penalty_threshold_frac=0.8,
):
    """Sample rollouts in one batch, then compute old_logp / ref_logp in one
    batched forward each. The on-policy ratio fix from qwen_grpo.py is preserved:
    `old_logp` is computed via the same fresh-forward path that `new_logp` uses,
    so policy_ratio is exactly 1.0 at the start of each inner epoch.
    """
    if kl_coeff and ref_model is None:
        raise ValueError("ref_model must be provided when kl_coeff is non-zero.")

    prompt = render_prompt(example["problem"])

    was_training = model.training
    model.eval()

    sampled = sample_response_batched(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        num_rollouts=num_rollouts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    token_ids_list = [s[0] for s in sampled]
    prompt_lens = [s[1] for s in sampled]
    texts = [s[2] for s in sampled]

    with torch.no_grad():
        old_logps, entropies = batched_sequence_logprob(
            model, token_ids_list, prompt_lens, return_entropy=True
        )
        if kl_coeff:
            ref_logps = batched_sequence_logprob(
                ref_model, token_ids_list, prompt_lens, return_entropy=False
            )
        else:
            ref_logps = None

    rollouts = []
    for i in range(num_rollouts):
        token_ids = token_ids_list[i]
        plen = prompt_lens[i]
        text = texts[i]
        gen_len = token_ids.numel() - plen

        correctness = reward_rlvr(text, example["answer"])
        format_bonus = format_bonus_weight if (correctness and has_boxed(text)) else 0.0
        len_pen = length_penalty_weight * length_penalty(
            gen_len, max_new_tokens, length_penalty_threshold_frac
        )
        reward = correctness + format_bonus - len_pen

        rollouts.append({
            "token_ids": token_ids,
            "prompt_len": plen,
            "old_logp": old_logps[i].detach(),
            "ref_logp": ref_logps[i].detach() if ref_logps is not None else None,
            "reward": reward,
            "correctness": correctness,
            "format_bonus": format_bonus,
            "length_penalty": len_pen,
            "entropy": entropies[i].item(),
            "text": text,
            "gen_len": gen_len,
        })

    if was_training:
        model.train()

    return rollouts


def compute_grpo_loss(
    model,
    rollouts,
    device,
    clip_eps=10.0,
    kl_coeff=0.001,
    skip_zero_adv=False,
):
    roll_rewards = [r["reward"] for r in rollouts]
    roll_correctness = [r["correctness"] for r in rollouts]
    roll_format_bonuses = [r["format_bonus"] for r in rollouts]
    roll_length_penalties = [r["length_penalty"] for r in rollouts]
    roll_entropies = [r["entropy"] for r in rollouts]
    samples = [
        {
            "text": r["text"],
            "reward": r["reward"],
            "correctness": r["correctness"],
            "format_bonus": r["format_bonus"],
            "length_penalty": r["length_penalty"],
            "gen_len": r["gen_len"],
        }
        for r in rollouts
    ]

    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    adv = advantages.detach()

    is_zero_adv = torch.allclose(
        advantages,
        torch.zeros_like(advantages),
        atol=1e-8,
        rtol=0.0,
    )

    if skip_zero_adv and is_zero_adv:
        return {
            "loss": 0.0,
            "pg_loss": 0.0,
            "kl_loss": 0.0,
            "policy_ratio": None,
            "rewards": roll_rewards,
            "correctness": roll_correctness,
            "format_bonuses": roll_format_bonuses,
            "length_penalties": roll_length_penalties,
            "entropies": roll_entropies,
            "advantages": advantages.detach().cpu().tolist(),
            "is_zero_adv": True,
            "samples": samples,
            "loss_tensor": None,
        }

    # Single batched with-grad forward for new_logp across all rollouts.
    new_logps = batched_sequence_logprob(
        model,
        [r["token_ids"] for r in rollouts],
        [r["prompt_len"] for r in rollouts],
    )

    old_logps = torch.stack([r["old_logp"] for r in rollouts])
    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    unclipped = ratio * adv
    clipped = clipped_ratio * adv

    obj = torch.where(
        adv >= 0,
        torch.minimum(unclipped, clipped),
        torch.maximum(unclipped, clipped),
    )

    pg_loss = -obj.mean()
    if kl_coeff:
        ref_logps = torch.stack([r["ref_logp"] for r in rollouts])
        kl_loss = kl_coeff * torch.mean(new_logps - ref_logps)
    else:
        kl_loss = torch.tensor(0.0, device=new_logps.device)
    loss = pg_loss + kl_loss

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl_loss": kl_loss.item(),
        "policy_ratio": ratio.mean().item(),
        "rewards": roll_rewards,
        "correctness": roll_correctness,
        "format_bonuses": roll_format_bonuses,
        "length_penalties": roll_length_penalties,
        "entropies": roll_entropies,
        "advantages": advantages.detach().cpu().tolist(),
        "is_zero_adv": is_zero_adv,
        "samples": samples,
        "loss_tensor": loss,
    }


def append_sample_logs(step_idx, samples, max_samples=3):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[Step {step_idx}] sample outputs\n")
        for i, sample in enumerate(samples[:max_samples]):
            text = sample["text"].replace("\n", "\\n")
            f.write(
                f"  {i+1}) reward={sample['reward']:.3f} "
                f"len={sample['gen_len']}: {text}\n"
            )
        f.write("\n")


def append_step_metrics(
    step_idx,
    total_steps,
    loss,
    kl_loss,
    policy_ratio,
    reward_avg,
    correctness_avg,
    format_bonus_avg,
    length_penalty_avg,
    tokens_per_sec,
    avg_response_len,
    adv_avg,
    adv_std,
    entropy_avg,
    eval_acc=None,
):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    policy_ratio_str = (
        "" if policy_ratio is None else f" policy_ratio={policy_ratio:.2f}"
    )
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Step {step_idx}/{total_steps}] "
            f"loss={loss:.2f} reward_avg={reward_avg:.3f} "
            f"correctness={correctness_avg:.3f} fmt_bonus={format_bonus_avg:.3f} "
            f"len_pen={length_penalty_avg:.3f} "
            f"kl={kl_loss:.2f} "
            f"tokens_per_sec={tokens_per_sec:.1f} "
            f"avg_response_len={avg_response_len:.1f}"
            f" adv_avg={adv_avg:.2f}"
            f" adv_std={adv_std:.2f}"
            f" entropy_avg={entropy_avg:.2f}"
            f"{policy_ratio_str}\n"
        )
    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_LOG_PATH.exists():
        CSV_LOG_PATH.write_text(
            "step,total_steps,loss,kl_loss,policy_ratio,reward_avg,correctness_avg,format_bonus_avg,length_penalty_avg,tokens_per_sec,avg_response_len,adv_avg,adv_std,entropy_avg,eval_acc\n",
            encoding="utf-8",
        )
    with CSV_LOG_PATH.open("a", encoding="utf-8") as f:
        eval_acc_str = "" if eval_acc is None else f"{eval_acc:.6f}"
        policy_ratio_str = "" if policy_ratio is None else f"{policy_ratio:.6f}"
        f.write(
            f"{step_idx},{total_steps},{loss:.6f},{kl_loss:.6f},{policy_ratio_str},{reward_avg:.6f},"
            f"{correctness_avg:.6f},{format_bonus_avg:.6f},{length_penalty_avg:.6f},"
            f"{tokens_per_sec:.6f},{avg_response_len:.6f},"
            f"{adv_avg:.6f},{adv_std:.6f},{entropy_avg:.6f},{eval_acc_str}\n"
        )


def append_eval_metrics(step_idx, acc, correct, total):
    METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(
            f"[Eval step {step_idx}] math500_acc={acc:.2f} "
            f"({correct}/{total})\n"
        )


def save_checkpoint(model, checkpoint_dir, step, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    ckpt_path = checkpoint_dir / f"qwen3-0.6B-rlvr-grpo-step{step:05d}{suffix}.pth"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


def train_rlvr_grpo(
    model,
    ref_model,
    tokenizer,
    math_data,
    math500_eval_data,
    device,
    steps=None,
    num_rollouts=8,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.9,
    clip_eps=0.2,
    inner_epochs=1,
    kl_coeff=0.02,
    lr=1e-5,
    checkpoint_every=50,
    checkpoint_dir=CHECKPOINT_DIR,
    eval_max_items=0,
    skip_zero_advantage_updates=False,
    show_eta=False,
    format_bonus_weight=0.05,
    length_penalty_weight=0.1,
    length_penalty_threshold_frac=0.8,
):
    if steps is None:
        steps = len(math_data)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    current_step = 0
    train_start_time = time.time() if show_eta else None

    try:
        for step in range(steps):
            step_start = time.perf_counter()
            current_step = step + 1
            example = math_data[step % len(math_data)]

            rollouts = collect_rollouts(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                example=example,
                device=device,
                num_rollouts=num_rollouts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                kl_coeff=kl_coeff,
                format_bonus_weight=format_bonus_weight,
                length_penalty_weight=length_penalty_weight,
                length_penalty_threshold_frac=length_penalty_threshold_frac,
            )

            stats = None
            for _ in range(inner_epochs):
                stats = compute_grpo_loss(
                    model=model,
                    rollouts=rollouts,
                    device=device,
                    clip_eps=clip_eps,
                    kl_coeff=kl_coeff,
                    skip_zero_adv=skip_zero_advantage_updates,
                )

                if stats["loss_tensor"] is not None:
                    optimizer.zero_grad()
                    stats["loss_tensor"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            reward_avg = torch.tensor(stats["rewards"]).mean().item()
            correctness_avg = torch.tensor(stats["correctness"]).mean().item()
            format_bonus_avg = torch.tensor(stats["format_bonuses"]).mean().item()
            length_penalty_avg = torch.tensor(stats["length_penalties"]).mean().item()
            entropy_avg = torch.tensor(stats["entropies"]).mean().item()
            advantage_tensor = torch.tensor(stats["advantages"])
            adv_avg = advantage_tensor.mean().item()
            adv_std = advantage_tensor.std().item()
            policy_ratio = stats.get("policy_ratio")
            step_time = time.perf_counter() - step_start
            step_tokens = sum(sample["gen_len"] for sample in stats["samples"])
            avg_response_len = (
                step_tokens / len(stats["samples"]) if stats["samples"] else 0.0
            )
            tokens_per_sec = step_tokens / step_time if step_time > 0 else 0.0
            if current_step % 10 == 0:
                append_sample_logs(current_step, stats["samples"])

            eval_acc = None
            if checkpoint_every and current_step % checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    model=model,
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                )
                print(f"Saved checkpoint to {ckpt_path}")

                if eval_max_items and math500_eval_data:
                    was_training = model.training
                    model.eval()
                    subset = (
                        math500_eval_data[:eval_max_items]
                        if eval_max_items
                        else math500_eval_data
                    )
                    out_path = (
                        Path(checkpoint_dir)
                        / f"{SCRIPT_NAME}-step{current_step:05d}-math500.jsonl"
                    )

                    num_correct, num_examples, acc = evaluate_math500_stream(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        math_data=subset,
                        out_path=out_path,
                        max_new_tokens=max_new_tokens,
                        verbose=False,
                    )

                    eval_acc = acc
                    append_eval_metrics(current_step, acc, num_correct, num_examples)
                    print(
                        f"MATH-500 eval @ step {current_step}: "
                        f"acc={acc:.3f} ({num_correct}/{num_examples})"
                    )
                    if was_training:
                        model.train()

            append_step_metrics(
                current_step,
                steps,
                stats["loss"],
                stats["kl_loss"],
                policy_ratio,
                reward_avg,
                correctness_avg,
                format_bonus_avg,
                length_penalty_avg,
                tokens_per_sec,
                avg_response_len,
                adv_avg=adv_avg,
                adv_std=adv_std,
                entropy_avg=entropy_avg,
                eval_acc=eval_acc,
            )

            policy_ratio_str = (
                "" if policy_ratio is None else f"policy_ratio={policy_ratio:.2f} "
            )

            eta_suffix = ""
            if show_eta:
                eta_msg = eta_progress_message(
                    processed=current_step,
                    total=steps,
                    start_time=train_start_time,
                    show_eta=True,
                    label="Step",
                ).rstrip()
                eta_part = eta_msg.split(" | ", 1)[-1]
                eta_suffix = f" | {eta_part}"
            print(
                f"[Step {current_step}/{steps}] "
                f"loss={stats['loss']:.2f} "
                f"kl={stats['kl_loss']:.2f} "
                f"reward_avg={reward_avg:.3f} "
                f"correct={correctness_avg:.3f} "
                f"fmt_bonus={format_bonus_avg:.3f} "
                f"len_pen={length_penalty_avg:.3f} "
                f"tok/sec={tokens_per_sec:.1f} "
                f"avg_resp_len={avg_response_len:.1f} "
                f"adv_avg={adv_avg:.2f} "
                f"adv_std={adv_std:.2f} "
                f"entropy_avg={entropy_avg:.2f} "
                f"{policy_ratio_str}"
                f"{eta_suffix}"
            )

    except KeyboardInterrupt:
        ckpt_path = save_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            step=max(1, current_step),
            suffix="interrupt",
        )
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt_path}")
        return model
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Batched RLVR GRPO trainer for Qwen3-0.6B on MATH.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps.")
    parser.add_argument("--num_rollouts", type=int, default=8, help="Number of rollouts per step.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generated tokens per rollout.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling cutoff.")
    parser.add_argument(
        "--clip_eps", type=float, default=0.2,
        help="GRPO/PPO clip range epsilon (PPO/DeepSeek-Math standard is 0.2).",
    )
    parser.add_argument(
        "--inner_epochs", type=int, default=1,
        help="Inner update iterations per step. Use 1 for on-policy.",
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=0.02,
        help="KL penalty coefficient. Author's stable run uses 0.02.",
    )
    parser.add_argument("--seed", type=str, default="42", help="Random seed (int) or 'none'.")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Optional .pth checkpoint to resume training from.",
    )
    parser.add_argument(
        "--eval_on_checkpoint", type=int, default=0,
        help="MATH-500 examples to eval at each checkpoint (0 disables).",
    )
    parser.add_argument(
        "--skip-zero-advantage-updates", action="store_true",
        help="Skip optimizer step when rollout advantages are all near zero.",
    )
    parser.add_argument("--show_eta", action="store_true", help="Append ETA to step logs.")
    parser.add_argument(
        "--format_bonus_weight", type=float, default=0.05,
        help="Tiny bonus when correct AND has \\boxed{}.",
    )
    parser.add_argument(
        "--length_penalty_weight", type=float, default=0.1,
        help="Flat penalty when gen_len exceeds threshold_frac * max_new_tokens.",
    )
    parser.add_argument(
        "--length_penalty_threshold_frac", type=float, default=0.8,
        help="Fraction of max_new_tokens above which length_penalty kicks in.",
    )
    args = parser.parse_args()

    if args.seed is not None and str(args.seed).strip().lower() != "none":
        torch.manual_seed(int(args.seed))
    device = get_device()

    math_data = load_math_train()
    if args.checkpoint_path:
        tokenizer = load_tokenizer_only(which_model="base")
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model="base", device=device, use_compile=False
        )

    if args.kl_coeff:
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
    else:
        ref_model = None

    trained = train_rlvr_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        math_data=math_data,
        math500_eval_data=load_math500_test(),
        device=device,
        steps=args.steps,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_eps=args.clip_eps,
        inner_epochs=args.inner_epochs,
        kl_coeff=args.kl_coeff,
        eval_max_items=args.eval_on_checkpoint,
        skip_zero_advantage_updates=args.skip_zero_advantage_updates,
        show_eta=args.show_eta,
        format_bonus_weight=args.format_bonus_weight,
        length_penalty_weight=args.length_penalty_weight,
        length_penalty_threshold_frac=args.length_penalty_threshold_frac,
    )

    if torch.cuda.is_available():
        max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Max CUDA memory allocated: {max_mem_gb:.2f} GB")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(trained.state_dict(), CHECKPOINT_DIR / "qwen3-0.6B-rlvr-grpo.pth")
