import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
RUN_DIR = ROOT / "results" / "sft_deepseek_then_grpo"
OUT_PNG = ROOT / "results" / "training_curves_sft_deepseek_then_grpo.png"
OUT_REPORT = ROOT / "results" / "sft_deepseek_then_grpo_report.md"


def read_metrics(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                elif key in {"step", "total_steps"}:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def read_eval(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    correct = sum(1 for row in rows if row.get("correct"))
    total = len(rows)
    avg_len = sum(len(row.get("generated_text", "").split()) for row in rows) / total
    return {"correct": correct, "total": total, "acc": correct / total, "avg_words": avg_len}


def rolling(values, window=5):
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1): i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def plot(metrics):
    steps = [row["step"] for row in metrics]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("DeepSeek-SFT -> GRPO continuation from epoch-3 SFT checkpoint", fontsize=12)

    panels = [
        ("correctness_avg", "Train rollout correctness", "fraction"),
        ("reward_avg", "Train reward", "reward"),
        ("kl_loss", "KL loss term", "loss"),
        ("entropy_avg", "Entropy", "nats"),
        ("avg_response_len", "Average response length", "tokens"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat[:5], panels):
        vals = [row[key] for row in metrics]
        ax.plot(steps, vals, linewidth=0.8, alpha=0.35, label="raw")
        ax.plot(steps, rolling(vals), linewidth=2, label="rolling mean (w=5)")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes.flat[5]
    eval_points = [row for row in metrics if row["eval_acc"] is not None]
    ax.scatter(
        [row["step"] for row in eval_points],
        [row["eval_acc"] for row in eval_points],
        s=120,
        color="tab:red",
        edgecolor="black",
        zorder=3,
    )
    for row in eval_points:
        ax.annotate(
            f"{row['eval_acc'] * 100:.1f}%",
            (row["step"], row["eval_acc"]),
            xytext=(8, 4),
            textcoords="offset points",
        )
    ax.axvline(50, color="tab:green", linestyle="--", alpha=0.5, label="last good")
    ax.axvline(100, color="tab:red", linestyle="--", alpha=0.5, label="degraded")
    ax.set_title("Checkpoint evals (50 MATH-500 items)")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def maybe_eval(path):
    return read_eval(path) if path.exists() else None


def main():
    metrics_path = RUN_DIR / "logs" / "qwen_grpo_logp_batched_metrics.csv"
    metrics = read_metrics(metrics_path)
    plot(metrics)

    eval50 = read_eval(RUN_DIR / "evals" / "grpo_step00050-math500.jsonl")
    eval100 = read_eval(RUN_DIR / "evals" / "grpo_step00100-math500.jsonl")
    base_grpo50 = maybe_eval(ROOT / "results" / "grpo_from_base" / "evals" / "grpo_step00050-math500.jsonl")
    base_grpo100 = maybe_eval(ROOT / "results" / "grpo_from_base" / "evals" / "grpo_step00100-math500.jsonl")

    rows = [
        ("DeepSeek-SFT seed", "epoch 3", "34.6%", "Full MATH-500 result from prior eval_distilled.jsonl"),
        ("DeepSeek-SFT -> GRPO", "50", f"{eval50['acc'] * 100:.1f}% ({eval50['correct']}/{eval50['total']})", "Best checkpoint in this continuation"),
        ("DeepSeek-SFT -> GRPO", "100", f"{eval100['acc'] * 100:.1f}% ({eval100['correct']}/{eval100['total']})", "Degraded; run stopped"),
    ]
    if base_grpo50:
        rows.append(("Qwen base -> GRPO", "50", f"{base_grpo50['acc'] * 100:.1f}% ({base_grpo50['correct']}/{base_grpo50['total']})", "Existing artifact, same 50-item checkpoint eval protocol"))
    if base_grpo100:
        rows.append(("Qwen base -> GRPO", "100", f"{base_grpo100['acc'] * 100:.1f}% ({base_grpo100['correct']}/{base_grpo100['total']})", "Existing artifact, same 50-item checkpoint eval protocol"))

    last = metrics[-1]
    eval_rows = [row for row in metrics if row["eval_acc"] is not None]
    report = [
        "# DeepSeek-SFT -> GRPO continuation report",
        "",
        "## Decision",
        "",
        "Stopped after the step-100 eval because the continuation degraded from step 50.",
        "The selected checkpoint is `results/sft_deepseek_then_grpo/checkpoints/qwen3-0.6B-rlvr-grpo-step00050.pth`.",
        "",
        "## Checkpoint comparison",
        "",
        "| Run | Step | Accuracy | Notes |",
        "| --- | ---: | ---: | --- |",
    ]
    for run, step, acc, notes in rows:
        report.append(f"| {run} | {step} | {acc} | {notes} |")

    report.extend([
        "",
        "## Safety metrics",
        "",
        f"- Step 50 eval: {eval50['correct']}/{eval50['total']} ({eval50['acc'] * 100:.1f}%).",
        f"- Step 100 eval: {eval100['correct']}/{eval100['total']} ({eval100['acc'] * 100:.1f}%).",
        f"- Last logged row: step {last['step']}, kl_loss={last['kl_loss']:.2f}, entropy_avg={last['entropy_avg']:.2f}, reward_avg={last['reward_avg']:.3f}.",
        "- `policy_ratio` stayed at 1.000000 on active-update rows, so the stop decision is based on eval degradation and KL/entropy drift, not PPO-ratio drift.",
        "",
        "## Artifacts",
        "",
        f"- Plot: `{OUT_PNG.relative_to(ROOT)}`",
        "- Main run logs: `results/sft_deepseek_then_grpo/logs/`",
        "- Per-checkpoint MATH-500 evals: `results/sft_deepseek_then_grpo/evals/grpo_step*-math500.jsonl`",
        "",
        "## Eval Points From CSV",
        "",
        "| Step | Eval accuracy |",
        "| ---: | ---: |",
    ])
    for row in eval_rows:
        report.append(f"| {row['step']} | {row['eval_acc'] * 100:.1f}% |")

    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_REPORT}")


if __name__ == "__main__":
    main()
