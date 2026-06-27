"""On-Policy Distillation (OPD) trainer for Qwen3-0.6B with a local same-family teacher.

The student samples its OWN rollouts (on-policy, like GRPO); a frozen local Qwen3
teacher scores every token; the student minimizes a per-token divergence to the
teacher (JSD by default). Runs on the HF/SDPA runtime for both generation and scoring.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    render_prompt,
    load_tokenizer_only,
    load_math500_test,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
)
from reasoning_from_scratch.ch06 import load_math_train

from qwen_hf_runtime import load_hf_qwen_model, convert_hf_state_to_scratch
from evaluate_math500 import evaluate_math500_stream_hf

from transformers import AutoModelForCausalLM


SCRIPT_NAME = Path(__file__).stem
REPO_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "qwen_sft_opd"


# --------------------------------------------------------------------------------------
# Generation (HF runtime)
# --------------------------------------------------------------------------------------
def _trim_generated(row_ids, eos_token_id, pad_token_id):
    """Strip right-padding and cut at the first EOS (inclusive)."""
    out = []
    for tok in row_ids:
        out.append(tok)
        if eos_token_id is not None and tok == eos_token_id:
            return out
    # remove trailing pads if no eos was emitted
    while out and pad_token_id is not None and out[-1] == pad_token_id:
        out.pop()
    return out


@torch.inference_mode()
def sample_rollouts_batched(model, tokenizer, prompt, device,
                            num_rollouts, max_new_tokens, temperature, top_p):
    """N rollouts for one prompt via a single batched generate() (no padding: all
    rows share the identical prompt)."""
    prompt_ids = tokenizer.encode(prompt)
    plen = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device).repeat(num_rollouts, 1)
    attn = torch.ones_like(input_ids)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    gens = [
        _trim_generated(out[r, plen:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id)
        for r in range(num_rollouts)
    ]
    return prompt_ids, gens


@torch.inference_mode()
def sample_rollouts_sequential(model, tokenizer, prompt, device,
                               num_rollouts, max_new_tokens, temperature, top_p):
    """N rollouts for one prompt via N batch=1 generate() calls (proven-stable path)."""
    prompt_ids = tokenizer.encode(prompt)
    plen = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(input_ids)
    gens = []
    for _ in range(num_rollouts):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        gens.append(
            _trim_generated(out[0, plen:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id)
        )
    return prompt_ids, gens


def sample_rollouts(mode, model, tokenizer, prompt, device,
                    num_rollouts, max_new_tokens, temperature, top_p):
    fn = sample_rollouts_batched if mode == "batched" else sample_rollouts_sequential
    return fn(model, tokenizer, prompt, device,
              num_rollouts, max_new_tokens, temperature, top_p)


# --------------------------------------------------------------------------------------
# Divergence gate: is batched sampling safe on this runtime?
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def _prefill_last_logits(model, prompt_ids, device, n):
    ids = torch.tensor([prompt_ids], device=device).repeat(max(n, 1), 1)
    attn = torch.ones_like(ids)
    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, -1, :]
    return logits.float()


def _length_stats(gens):
    lens = [len(g) for g in gens] or [0]
    return {
        "n": len(lens),
        "mean": statistics.mean(lens),
        "median": statistics.median(lens),
        "min": min(lens),
        "max": max(lens),
        "frac_short": sum(1 for x in lens if x < 30) / len(lens),
    }


def run_divergence_gate(model, tokenizer, math_data, device, *,
                        num_probe_prompts=3, samples_per_prompt=16,
                        max_new_tokens=768, temperature=0.8, top_p=0.9,
                        logit_diff_tol=0.15, median_ratio_tol=0.30, short_frac_tol=0.20):
    """Decide whether batched sampling distorts the rollout distribution under this
    runtime. Returns (is_safe: bool, report: dict).

    Two checks:
      1. Prefill last-token logits: batch=1 vs batch=N for the same prompt. Large
         divergence (> logit_diff_tol) means the kernel routes batched matmuls
         differently enough to perturb the policy.
      2. Empirical rollout-length distribution: sequential vs batched over many samples.
         A collapse to short outputs (frac_short spike, median drop) is the symptom of
         perturbed logits on a small bimodal-length model.
    """
    report = {"prompts": [], "max_logit_mean_diff": 0.0, "max_logit_max_diff": 0.0}
    length_ok = True
    for i in range(min(num_probe_prompts, len(math_data))):
        prompt = render_prompt(math_data[i]["problem"])
        prompt_ids = tokenizer.encode(prompt)

        u = _prefill_last_logits(model, prompt_ids, device, n=1)[0]
        b = _prefill_last_logits(model, prompt_ids, device, n=samples_per_prompt)
        diffs = (b - u).abs()
        mean_diff = diffs.mean().item()
        max_diff = diffs.max().item()
        report["max_logit_mean_diff"] = max(report["max_logit_mean_diff"], mean_diff)
        report["max_logit_max_diff"] = max(report["max_logit_max_diff"], max_diff)

        _, seq_gens = sample_rollouts_sequential(
            model, tokenizer, prompt, device, samples_per_prompt,
            max_new_tokens, temperature, top_p)
        _, bat_gens = sample_rollouts_batched(
            model, tokenizer, prompt, device, samples_per_prompt,
            max_new_tokens, temperature, top_p)
        s_stats = _length_stats(seq_gens)
        b_stats = _length_stats(bat_gens)

        denom = max(s_stats["median"], 1.0)
        median_ratio = abs(b_stats["median"] - s_stats["median"]) / denom
        short_excess = b_stats["frac_short"] - s_stats["frac_short"]
        prompt_len_ok = (median_ratio <= median_ratio_tol) and (short_excess <= short_frac_tol)
        length_ok = length_ok and prompt_len_ok

        report["prompts"].append({
            "idx": i, "logit_mean_diff": mean_diff, "logit_max_diff": max_diff,
            "seq_len": s_stats, "batched_len": b_stats,
            "median_ratio": median_ratio, "short_excess": short_excess,
            "prompt_ok": prompt_len_ok,
        })

    logits_ok = report["max_logit_max_diff"] <= logit_diff_tol
    is_safe = bool(logits_ok and length_ok)
    report["logits_ok"] = logits_ok
    report["length_ok"] = length_ok
    report["is_safe"] = is_safe
    return is_safe, report


def print_gate_report(report):
    print("\n" + "=" * 64)
    print("BATCHED-SAMPLING DIVERGENCE GATE (HF/SDPA runtime)")
    print("=" * 64)
    print(f"  max prefill logit |diff|: mean={report['max_logit_mean_diff']:.3e} "
          f"max={report['max_logit_max_diff']:.3e}  (logits_ok={report['logits_ok']})")
    for p in report["prompts"]:
        s, b = p["seq_len"], p["batched_len"]
        print(f"  prompt {p['idx']}: logit_max_diff={p['logit_max_diff']:.3e} | "
              f"len median seq={s['median']:.0f} batched={b['median']:.0f} "
              f"(ratio={p['median_ratio']:.2f}) | "
              f"frac_short seq={s['frac_short']:.2f} batched={b['frac_short']:.2f} "
              f"(excess={p['short_excess']:+.2f}) -> {'OK' if p['prompt_ok'] else 'FAIL'}")
    verdict = "BATCHED SAFE" if report["is_safe"] else "USE SEQUENTIAL"
    print(f"  VERDICT: {verdict}")
    print("=" * 64 + "\n")


# --------------------------------------------------------------------------------------
# Scoring + per-token KL loss
# --------------------------------------------------------------------------------------
def _forward_logits(model, full_ids_list, device, pad_id, with_grad):
    """Forward a right-padded batch of [prompt + rollout] sequences and return the full
    logits [N, L_max, V] plus per-row real lengths. An attention mask blocks padded
    positions; the answer slices are taken by the caller (no redundant copy)."""
    lengths = [len(t) for t in full_ids_list]
    max_len = max(lengths)
    n = len(full_ids_list)
    batch = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((n, max_len), dtype=torch.long, device=device)
    for i, t in enumerate(full_ids_list):
        batch[i, :lengths[i]] = torch.tensor(t, dtype=torch.long, device=device)
        attn[i, :lengths[i]] = 1

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=batch, attention_mask=attn, use_cache=False).logits
    return logits, lengths


def per_token_kl(student_logits, teacher_logits, kl_type):
    """Per-position divergence between teacher and student next-token distributions.
    Inputs are fp32 logits [*, V]. Returns per-position scalar [*]."""
    s_lp = torch.log_softmax(student_logits, dim=-1)
    t_lp = torch.log_softmax(teacher_logits, dim=-1)
    if kl_type == "forward":          # KL(teacher || student) — mass-covering
        t_p = t_lp.exp()
        return (t_p * (t_lp - s_lp)).sum(-1)
    if kl_type == "reverse":          # KL(student || teacher) — mode-seeking
        s_p = s_lp.exp()
        return (s_p * (s_lp - t_lp)).sum(-1)
    if kl_type == "jsd":              # symmetric
        m_lp = torch.log(0.5 * (s_lp.exp() + t_lp.exp()) + 1e-12)
        s_p, t_p = s_lp.exp(), t_lp.exp()
        return 0.5 * (s_p * (s_lp - m_lp)).sum(-1) + 0.5 * (t_p * (t_lp - m_lp)).sum(-1)
    raise ValueError(f"unknown kl_type={kl_type}")


def opd_loss(student, teacher, ref, prompt_ids, gen_ids_list, device, pad_id,
             kl_type, kl_coeff, ent_coef, kl_clamp, distill_coeff=1.0,
             teacher_prompt_ids=None):
    """On-policy distillation loss with a trust-region reference anchor.

    per-token = distill_coeff * D(teacher||student)[clamped]
                  +  kl_coeff * k3(student||ref)  -  ent_coef * H(student)

    - distill_coeff scales the teacher-divergence (distillation) term. Set 0.0 to disable
      distillation while keeping the k3 anchor, entropy floor, clamp, warmup and sampling.
    - D: teacher divergence (JSD default). Direct differentiation is the correct GKD
      gradient for forward KL; pure reverse KL this way needs a policy-gradient term, so
      JSD/forward are the safe choices.
    - k3: low-variance KL(student||ref) on the sampled tokens (GRPO's estimator), the trust
      region toward the frozen init that prevents drift-to-degenerate collapse.
    - H: student entropy floor, opposing the repetition attractor.
    Teacher/ref logits are stop-grad (no_grad forwards); sampled token ids are fixed
    (no backprop through sampling). Logits upcast to fp32 before any divergence."""
    plen = len(prompt_ids)
    # Teacher scores the SAME rollout tokens but in its OWN prompt context (native ChatML
    # when teacher_prompt_ids is given), so the teacher prompt length can differ from the
    # student's — align each model's answer slice at its own prompt length (as in opsd_loss).
    t_prompt = teacher_prompt_ids if teacher_prompt_ids is not None else prompt_ids
    t_plen = len(t_prompt)
    keep = [g for g in gen_ids_list if len(g) > 0]
    empty_stats = {"n_rollouts": 0, "kl_mean": 0.0, "avg_answer_len": 0.0,
                   "kl_only": 0.0, "ref_k3": 0.0, "entropy": 0.0}
    if not keep:
        return None, empty_stats

    student_full = [prompt_ids + g for g in keep]
    teacher_full = [t_prompt + g for g in keep]
    s_logits, lengths = _forward_logits(student, student_full, device, pad_id, with_grad=True)
    t_logits, _ = _forward_logits(teacher, teacher_full, device, pad_id, with_grad=False)
    r_logits, _ = _forward_logits(ref, student_full, device, pad_id, with_grad=False)

    total = s_logits.new_tensor(0.0)
    tok_count = 0
    kl_sum = k3_sum = ent_sum = 0.0
    # Answer positions predict tokens [plen .. length-1] -> logits at [plen-1 .. length-2].
    for i, length in enumerate(lengths):
        a = length - plen
        if a <= 0:
            continue
        s_slice = s_logits[i, plen - 1: plen - 1 + a].float()
        t_slice = t_logits[i, t_plen - 1: t_plen - 1 + a].float()
        r_slice = r_logits[i, plen - 1: plen - 1 + a].float()
        # teacher divergence, clamped per token to drop destructive heavy-tail tokens
        kl = per_token_kl(s_slice, t_slice, kl_type).clamp(0.0, kl_clamp)
        # k3 reference anchor on the sampled tokens: KL(student||ref), low-variance
        tgt = torch.tensor(keep[i], device=device)
        s_lp = torch.log_softmax(s_slice, -1)
        r_lp = torch.log_softmax(r_slice, -1)
        s_tok = s_lp.gather(-1, tgt[:, None]).squeeze(-1)
        r_tok = r_lp.gather(-1, tgt[:, None]).squeeze(-1)
        logr = r_tok - s_tok
        k3 = torch.exp(logr) - logr - 1.0
        # student entropy (floor)
        ent = -(s_lp.exp() * s_lp).sum(-1)
        per_tok = distill_coeff * kl + kl_coeff * k3 - ent_coef * ent
        total = total + per_tok.sum()
        tok_count += a
        kl_sum += kl.sum().item()
        k3_sum += k3.sum().item()
        ent_sum += ent.sum().item()

    loss = total / max(tok_count, 1)
    stats = {
        "n_rollouts": len(keep),
        "kl_mean": loss.item(),
        "avg_answer_len": tok_count / max(len(keep), 1),
        "kl_only": kl_sum / max(tok_count, 1),
        "ref_k3": k3_sum / max(tok_count, 1),
        "entropy": ent_sum / max(tok_count, 1),
    }
    return loss, stats


def render_prompt_chatml(problem):
    """Qwen3 NATIVE chat template for the teacher's scoring context."""
    return f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"


def render_prompt_with_hint(problem, hint):
    """Augment the problem with a privileged-info hint for the self-teacher (OPSD).
    The hint (gold solution/answer) is revealed only to the teacher pass; the student
    pass sees the plain problem. Same chat template as render_prompt."""
    augmented = (
        f"{problem}\n\n"
        f"[Reference solution provided for guidance — work through the reasoning "
        f"yourself rather than copying it:\n{hint}\n]"
    )
    return render_prompt(augmented)


def opsd_loss(student, ref, plain_prompt_ids, hint_prompt_ids, gen_ids_list, device, pad_id,
              kl_type, kl_coeff, ent_coef, kl_clamp, distill_coeff=1.0):
    """On-policy SELF-distillation (OPSD). Teacher = the SAME student weights conditioned on
    the gold hint (no external model). Student samples on the PLAIN prompt; teacher scores the
    same rollout tokens with the hint in context. Per-token KL(teacher||student) over the
    rollout, plus the same trust-region anchor / entropy floor / clamp as opd_loss.

    Token alignment: the rollout occupies positions [h_plen:] under the (longer) hint context
    and [p_plen:] under the plain context — same tokens, different absolute offsets — so the
    answer-logit slices are taken at each context's own prompt length."""
    p_plen, h_plen = len(plain_prompt_ids), len(hint_prompt_ids)
    keep = [g for g in gen_ids_list if len(g) > 0]
    empty_stats = {"n_rollouts": 0, "kl_mean": 0.0, "avg_answer_len": 0.0,
                   "kl_only": 0.0, "ref_k3": 0.0, "entropy": 0.0}
    if not keep:
        return None, empty_stats

    plain_full = [plain_prompt_ids + g for g in keep]
    hint_full = [hint_prompt_ids + g for g in keep]
    s_logits, _ = _forward_logits(student, plain_full, device, pad_id, with_grad=True)
    t_logits, _ = _forward_logits(student, hint_full, device, pad_id, with_grad=False)  # self-teacher + hint
    r_logits, _ = _forward_logits(ref, plain_full, device, pad_id, with_grad=False)

    total = s_logits.new_tensor(0.0)
    tok_count = 0
    kl_sum = k3_sum = ent_sum = 0.0
    for i, g in enumerate(keep):
        a = len(g)
        s_slice = s_logits[i, p_plen - 1: p_plen - 1 + a].float()
        t_slice = t_logits[i, h_plen - 1: h_plen - 1 + a].float()
        r_slice = r_logits[i, p_plen - 1: p_plen - 1 + a].float()
        kl = per_token_kl(s_slice, t_slice, kl_type).clamp(0.0, kl_clamp)
        tgt = torch.tensor(g, device=device)
        s_lp = torch.log_softmax(s_slice, -1)
        r_lp = torch.log_softmax(r_slice, -1)
        s_tok = s_lp.gather(-1, tgt[:, None]).squeeze(-1)
        r_tok = r_lp.gather(-1, tgt[:, None]).squeeze(-1)
        logr = r_tok - s_tok
        k3 = torch.exp(logr) - logr - 1.0
        ent = -(s_lp.exp() * s_lp).sum(-1)
        per_tok = distill_coeff * kl + kl_coeff * k3 - ent_coef * ent
        total = total + per_tok.sum()
        tok_count += a
        kl_sum += kl.sum().item()
        k3_sum += k3.sum().item()
        ent_sum += ent.sum().item()

    loss = total / max(tok_count, 1)
    stats = {
        "n_rollouts": len(keep),
        "kl_mean": loss.item(),
        "avg_answer_len": tok_count / max(len(keep), 1),
        "kl_only": kl_sum / max(tok_count, 1),
        "ref_k3": k3_sum / max(tok_count, 1),
        "entropy": ent_sum / max(tok_count, 1),
    }
    return loss, stats


# --------------------------------------------------------------------------------------
# Checkpoint / logging
# --------------------------------------------------------------------------------------
def save_checkpoint(student, checkpoint_dir, step, prefix, suffix=""):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{suffix}" if suffix else ""
    path = checkpoint_dir / f"{prefix}-step{step:05d}{suffix}.pth"
    # Save in SCRATCH key format so evaluate_math500.py --runtime hf can load it.
    torch.save(convert_hf_state_to_scratch(student.state_dict()), path)
    return path


def append_csv(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["step", "total_steps", "loss", "kl_mean", "kl_only", "ref_k3", "entropy",
              "avg_answer_len", "n_rollouts", "frac_capped", "tokens_per_sec", "eval_acc"]
    if not csv_path.exists():
        csv_path.write_text(",".join(header) + "\n", encoding="utf-8")
    with csv_path.open("a", encoding="utf-8") as f:
        f.write(",".join("" if row.get(k) is None else f"{row[k]}" for k in header) + "\n")


# --------------------------------------------------------------------------------------
# Train loop
# --------------------------------------------------------------------------------------
def train_opd(student, teacher, ref, tokenizer, math_data, eval_data, device, args, sampling_mode):
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    # Linear warmup: the first ~warmup_steps (worst student/teacher overlap) take reduced
    # steps so an early bad gradient can't drag the policy off the SFT manifold.
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    student.train()

    output_root = Path(args.output_dir)
    csv_path = output_root / "logs" / f"{SCRIPT_NAME}_metrics.csv"
    checkpoint_dir = output_root / "checkpoints"
    eval_dir = output_root / "evals"
    for d in (csv_path.parent, checkpoint_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    start_time = time.time()
    current_step = 0
    capped_hist = []   # rolling fraction of rollouts that hit the cap without EOS (collapse signal)
    try:
        for step in range(args.steps):
            current_step = step + 1
            step_start = time.perf_counter()
            example = math_data[step % len(math_data)]
            prompt = render_prompt(example["problem"])

            student.eval()
            prompt_ids, gens = sample_rollouts(
                sampling_mode, student, tokenizer, prompt, device,
                args.num_rollouts, args.max_new_tokens, args.temperature, args.top_p)
            student.train()

            if args.self_distill:
                hint = example["solution"] if args.hint == "solution" else str(example["answer"])
                hint_prompt_ids = tokenizer.encode(render_prompt_with_hint(example["problem"], hint))
                loss, stats = opsd_loss(
                    student, ref, prompt_ids, hint_prompt_ids, gens, device, pad_id,
                    args.kl_type, args.kl_coeff, args.ent_coef, args.kl_clamp,
                    distill_coeff=args.distill_coeff)
            else:
                teacher_prompt_ids = None
                if args.teacher_prompt_format == "chatml":
                    teacher_prompt_ids = tokenizer.encode(
                        render_prompt_chatml(example["problem"]), chat_wrapped=False)
                loss, stats = opd_loss(
                    student, teacher, ref, prompt_ids, gens, device, pad_id,
                    args.kl_type, args.kl_coeff, args.ent_coef, args.kl_clamp,
                    distill_coeff=args.distill_coeff,
                    teacher_prompt_ids=teacher_prompt_ids)

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
                optimizer.step()
                sched.step()

            step_time = time.perf_counter() - step_start
            gen_tokens = sum(len(g) for g in gens)
            tok_per_sec = gen_tokens / step_time if step_time > 0 else 0.0

            # Collapse signal: rollouts that hit the cap without ever emitting EOS.
            capped = sum(1 for g in gens
                         if len(g) >= args.max_new_tokens and (eos_id is None or eos_id not in g))
            frac_capped = capped / max(len(gens), 1)
            capped_hist.append(frac_capped)
            roll_capped = sum(capped_hist[-10:]) / min(len(capped_hist), 10)

            eval_acc = None
            if args.checkpoint_every and current_step % args.checkpoint_every == 0:
                ckpt = save_checkpoint(student, checkpoint_dir, current_step, args.checkpoint_prefix)
                print(f"Saved checkpoint to {ckpt}")
                if args.eval_on_checkpoint and eval_data:
                    student.eval()
                    subset = eval_data[:args.eval_on_checkpoint]
                    out_path = eval_dir / f"{SCRIPT_NAME}-step{current_step:05d}-math500.jsonl"
                    nc, ne, acc = evaluate_math500_stream_hf(
                        model=student, tokenizer=tokenizer, device=device,
                        math_data=subset, out_path=str(out_path),
                        max_new_tokens=(args.eval_max_new_tokens or args.max_new_tokens),
                        batch_size=args.eval_batch_size, verbose=False)
                    eval_acc = acc
                    print(f"MATH-500 eval @ step {current_step}: acc={acc:.3f} ({nc}/{ne})")
                    student.train()

            append_csv(csv_path, {
                "step": current_step, "total_steps": args.steps,
                "loss": f"{stats['kl_mean']:.6f}", "kl_mean": f"{stats['kl_mean']:.6f}",
                "kl_only": f"{stats['kl_only']:.6f}", "ref_k3": f"{stats['ref_k3']:.6f}",
                "entropy": f"{stats['entropy']:.6f}",
                "avg_answer_len": f"{stats['avg_answer_len']:.2f}",
                "n_rollouts": stats["n_rollouts"], "frac_capped": f"{frac_capped:.3f}",
                "tokens_per_sec": f"{tok_per_sec:.2f}",
                "eval_acc": None if eval_acc is None else f"{eval_acc:.6f}",
            })

            eta = eta_progress_message(current_step, args.steps, start_time,
                                       show_eta=True, label="Step").rstrip()
            eta = eta.split(" | ", 1)[-1] if " | " in eta else ""
            print(f"[Step {current_step}/{args.steps}] kl={stats['kl_mean']:.4f} "
                  f"D={stats['kl_only']:.3f} k3={stats['ref_k3']:.3f} H={stats['entropy']:.2f} "
                  f"n_roll={stats['n_rollouts']} ans_len={stats['avg_answer_len']:.0f} "
                  f"capped={roll_capped:.2f} tok/s={tok_per_sec:.0f} | {eta}", flush=True)

            # Opt-in abort on sustained cap-without-EOS (degeneration); disabled by default.
            if args.abort_frac_capped > 0 and current_step >= 10 and roll_capped > args.abort_frac_capped:
                ckpt = save_checkpoint(student, checkpoint_dir, current_step,
                                       args.checkpoint_prefix, suffix="aborted")
                print(f"ABORT: rolling frac_capped={roll_capped:.2f} > "
                      f"{args.abort_frac_capped} — likely degeneration. Saved {ckpt}.", flush=True)
                return student
    except KeyboardInterrupt:
        ckpt = save_checkpoint(student, checkpoint_dir, max(1, current_step),
                               args.checkpoint_prefix, suffix="interrupt")
        print(f"\nKeyboardInterrupt. Saved checkpoint to {ckpt}")
        return student

    save_checkpoint(student, checkpoint_dir, current_step, args.checkpoint_prefix, suffix="final")
    return student


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="On-Policy Distillation (OPD) with a local same-family Qwen3 teacher.")
    p.add_argument("--student_checkpoint", type=str, required=True,
                   help="Scratch-format .pth to init the student, or 'base' for the base model.")
    p.add_argument("--teacher_model_id", type=str, default="Qwen/Qwen3-4B",
                   help="HF model id for the frozen local teacher (e.g. Qwen/Qwen3-4B, Qwen/Qwen3-8B).")
    p.add_argument("--self_distill", action="store_true",
                   help="On-policy SELF-distillation (OPSD): the teacher is the SAME student "
                        "conditioned on the gold hint (no external teacher). Self-distilled-"
                        "reasoner / Cursor Composer-2.5 style.")
    p.add_argument("--hint", type=str, default="solution", choices=["solution", "answer"],
                   help="Privileged-info hint for the self-teacher: full gold solution (richer) or answer only.")
    p.add_argument("--teacher_checkpoint", type=str, default=None,
                   help="Optional local path/dir for the teacher instead of --teacher_model_id.")
    p.add_argument("--teacher_prompt_format", type=str, default="render",
                   choices=["render", "chatml"],
                   help="Prompt context the teacher is scored in. 'render' = the same plain "
                        "render_prompt the student samples on. 'chatml' = the teacher's native "
                        "Qwen3 chat template.")
    p.add_argument("--teacher_fp32", action="store_true",
                   help="Load the frozen teacher in fp32 (removes bf16/padding noise in teacher "
                        "scoring; the teacher is forward-only so the extra memory is just weights "
                        "+ per-step logits).")
    p.add_argument("--kl_type", type=str, default="jsd", choices=["forward", "reverse", "jsd"],
                   help="Teacher divergence. JSD/forward are GKD-correct under direct differentiation; "
                        "pure reverse is a misformulation here (needs a policy-gradient term).")
    p.add_argument("--kl_coeff", type=float, default=0.05,
                   help="Reference-KL anchor strength (trust region toward the frozen SFT init).")
    p.add_argument("--distill_coeff", type=float, default=1.0,
                   help="Weight on the teacher-divergence (distillation) term. Set 0.0 to disable "
                        "distillation while the k3 anchor, entropy floor, clamp, warmup and "
                        "on-policy sampling stay live.")
    p.add_argument("--ent_coef", type=float, default=1e-3, help="Student entropy floor coefficient.")
    p.add_argument("--kl_clamp", type=float, default=10.0, help="Per-token teacher-KL cap (nats).")
    p.add_argument("--warmup_steps", type=int, default=15, help="Linear LR warmup steps.")
    p.add_argument("--abort_frac_capped", type=float, default=0.0,
                   help="Abort if the 10-step rolling fraction of rollouts that hit the cap "
                        "without EOS exceeds this (0 = disabled). Set near 1.0 if the init "
                        "naturally rambles to the cap.")
    p.add_argument("--sampling", type=str, default="auto", choices=["auto", "batched", "sequential"])
    p.add_argument("--num_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--eval_on_checkpoint", type=int, default=50,
                   help="Eval this many MATH-500 problems at each checkpoint (0 = skip).")
    p.add_argument("--eval_max_new_tokens", type=int, default=None,
                   help="Token budget for in-training MATH-500 evals (default: --max_new_tokens). "
                        "Set higher than the rollout budget so a rambling student's boxed answer "
                        "isn't truncated away.")
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--checkpoint_prefix", type=str, default="qwen3-0.6B-opd")
    p.add_argument("--diagnose", action="store_true",
                   help="Run the batched-sampling divergence gate and exit (no training).")
    return p.parse_args()


def load_models(args, device):
    # Student: HF Qwen3-0.6B. Init from a scratch-format SFT checkpoint, or from the BASE
    # model when --student_checkpoint == "base". The ref anchor then anchors to
    # base too, so the trust region is around the actual init.
    ckpt_path = None if args.student_checkpoint == "base" else args.student_checkpoint
    student = load_hf_qwen_model(
        checkpoint_path=ckpt_path, device=device,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    student.train()

    # Teacher: frozen local Qwen3-N, forward-only. SKIPPED in self-distill mode (OPSD), where
    # the teacher is the student itself conditioned on the gold hint — no external model.
    if args.self_distill:
        teacher = None
        print("Self-distillation (OPSD): no external teacher; self-teacher = student + gold hint.")
    else:
        teacher_src = args.teacher_checkpoint or args.teacher_model_id
        teacher_dtype = torch.float32 if args.teacher_fp32 else torch.bfloat16
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_src, torch_dtype=teacher_dtype,
            attn_implementation="sdpa", trust_remote_code=True).to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        print(f"Teacher dtype={teacher_dtype}, prompt_format={args.teacher_prompt_format}.")

    ref_model = load_hf_qwen_model(
        checkpoint_path=ckpt_path, device=device,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    ref_model.eval()
    ref_model.requires_grad_(False)
    return student, teacher, ref_model


def assert_vocab_aligned(student, teacher, tokenizer):
    """Same-family OPD requires identical tokenization; both Qwen3 models share the
    Qwen3 vocab, so feeding rfs-tokenized ids to both is valid. Fail fast if not."""
    s_vocab = student.get_output_embeddings().weight.shape[0]
    t_vocab = teacher.get_output_embeddings().weight.shape[0]
    probe = tokenizer.encode("The answer is \\boxed{42}.", chat_wrapped=False)
    max_id = max(probe) if probe else 0
    if t_vocab < s_vocab or max_id >= t_vocab:
        raise RuntimeError(
            f"Tokenizer/teacher vocab mismatch: student_vocab={s_vocab} "
            f"teacher_vocab={t_vocab} max_probe_id={max_id}. OPD needs a same-tokenizer "
            f"(same-family) teacher.")
    print(f"Vocab alignment OK: student={s_vocab} teacher={t_vocab} (Qwen3 shared vocab).")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")

    print(f"Student checkpoint: {args.student_checkpoint}")
    print(f"Teacher: {args.teacher_checkpoint or args.teacher_model_id}")
    print(f"kl_type={args.kl_type} sampling={args.sampling} num_rollouts={args.num_rollouts}")

    tokenizer = load_tokenizer_only(which_model="reasoning")
    student, teacher, ref_model = load_models(args, device)
    if teacher is not None:
        assert_vocab_aligned(student, teacher, tokenizer)

    math_data = load_math_train()
    sampling_mode = args.sampling
    if args.sampling == "auto" or args.diagnose:
        student.eval()
        is_safe, report = run_divergence_gate(
            student, tokenizer, math_data, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p)
        student.train()
        print_gate_report(report)
        (Path(args.output_dir) / "logs").mkdir(parents=True, exist_ok=True)
        with (Path(args.output_dir) / "logs" / "divergence_gate.json").open("w") as f:
            json.dump(report, f, indent=2)
        if args.diagnose:
            print("Diagnose-only run complete. Exiting before training.")
            return
        sampling_mode = "batched" if is_safe else "sequential"
        print(f"[auto] selected sampling mode: {sampling_mode}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_opd(student, teacher, ref_model, tokenizer, math_data, load_math500_test(),
              device, args, sampling_mode)

    if torch.cuda.is_available():
        print(f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
