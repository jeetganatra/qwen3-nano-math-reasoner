"""Generate per-run training curve plots and a combined eval-results table.

For each registered run, produces:
- results/training_curves_<run_name>.png  (6-panel training metrics)

And once across all runs:
- results/eval_summary.md       (markdown table)
- results/eval_summary.csv      (same data, machine-readable)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).parent.parent
OUT_DIR = REPO / "results"

RUNS = [
    {
        "name": "grpo_from_base",
        "label": "src/qwen_grpo.py (num_rollouts=8, max_new_tokens=768)",
        "csv": REPO / "results/grpo_from_base/logs/qwen_grpo_logp_batched_metrics.csv",
        "color": "tab:orange",
        "max_new_tokens": 768,
    },
]


def load_run(csv_path):
    """Load a metrics CSV, dropping any leading rows from earlier smoke runs.

    The training script appends to the CSV if it already exists, so a single
    file can contain rows from multiple invocations. We dedupe by keeping
    rows from the last occurrence of step==1 onward.
    """
    df = pd.read_csv(csv_path)
    step_one_rows = df.index[df["step"] == 1].tolist()
    if len(step_one_rows) > 1:
        df = df.iloc[step_one_rows[-1]:].reset_index(drop=True)
    return df


def rolling(s, w=5):
    return s.rolling(window=w, min_periods=1).mean()


def plot_one_run(run, df):
    color = run["color"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(run["label"], fontsize=11, y=0.995)

    # 1) correctness_avg
    ax = axes[0, 0]
    ax.plot(df["step"], df["correctness_avg"], color=color, alpha=0.3, linewidth=0.8, label="raw")
    ax.plot(df["step"], rolling(df["correctness_avg"]), color=color, linewidth=2, label="rolling mean (w=5)")
    ax.set_title("Train rollout correctness rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of rollouts correct")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    # 2) avg_response_len
    ax = axes[0, 1]
    ax.plot(df["step"], df["avg_response_len"], color=color, alpha=0.3, linewidth=0.8)
    ax.plot(df["step"], rolling(df["avg_response_len"]), color=color, linewidth=2)
    ax.axhline(run["max_new_tokens"], color="red", linestyle=":", alpha=0.5,
               label=f"max_new_tokens={run['max_new_tokens']}")
    ax.set_title("Average generation length")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # 3) entropy
    ax = axes[0, 2]
    ax.plot(df["step"], df["entropy_avg"], color=color, alpha=0.3, linewidth=0.8)
    ax.plot(df["step"], rolling(df["entropy_avg"]), color=color, linewidth=2)
    ax.axhspan(0, 0.1, color="red", alpha=0.08)
    ax.axhspan(0.1, 0.4, color="orange", alpha=0.06)
    ax.set_title("Per-step entropy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_ylim(0, max(df["entropy_avg"].max() * 1.1, 1.2))
    ax.grid(alpha=0.3)

    # 4) policy_ratio (skip rows where it's NaN/skipped)
    ax = axes[1, 0]
    sub = df.dropna(subset=["policy_ratio"])
    ax.plot(sub["step"], sub["policy_ratio"], color=color, linewidth=1.5,
            marker="o", markersize=3)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4)
    ax.set_title(f"Policy ratio (active steps: {len(sub)}/{len(df)})")
    ax.set_xlabel("Step")
    ax.set_ylabel("exp(new_logp - old_logp)")
    ax.grid(alpha=0.3)

    # 5) reward components
    ax = axes[1, 1]
    ax.plot(df["step"], rolling(df["correctness_avg"]),
            color="tab:green", linewidth=1.8, label="correctness")
    ax.plot(df["step"], rolling(df["format_bonus_avg"]),
            color="tab:purple", linewidth=1.5, label="format_bonus")
    ax.plot(df["step"], rolling(df["length_penalty_avg"]),
            color="tab:red", linewidth=1.5, label="length_penalty")
    ax.set_title("Reward components (smoothed, w=5)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Per-rollout average")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # 6) MATH-500 eval points
    ax = axes[1, 2]
    eval_pts = df.dropna(subset=["eval_acc"])
    if len(eval_pts):
        ax.scatter(eval_pts["step"], eval_pts["eval_acc"], color=color, s=140,
                   zorder=3, edgecolor="black", linewidth=0.8)
        for _, row in eval_pts.iterrows():
            ax.annotate(f"{row['eval_acc'] * 100:.1f}%",
                        (row["step"], row["eval_acc"]),
                        xytext=(10, -3), textcoords="offset points", fontsize=10)
    ax.axhline(0.508, color="gray", linestyle="--", alpha=0.5,
               label="Raschka reasoning baseline (50.8%)")
    ax.set_title("MATH-500 accuracy at checkpoints (50 examples)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(-5, df["step"].max() + 10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = OUT_DIR / f"training_curves_{run['name']}.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def build_eval_table(runs_with_dfs):
    """Build a tidy table of every (run, eval_step) pair with key metrics."""
    rows = []
    for run, df in runs_with_dfs:
        eval_pts = df.dropna(subset=["eval_acc"])
        for _, ev in eval_pts.iterrows():
            step = int(ev["step"])
            # Average over the last 10 active steps up to and including this eval
            window = df[(df["step"] > step - 10) & (df["step"] <= step)]
            rows.append({
                "run": run["name"],
                "step": step,
                "math500_acc": ev["eval_acc"],
                "math500_correct": int(round(ev["eval_acc"] * 50)),
                "math500_total": 50,
                "train_correctness_w10": window["correctness_avg"].mean(),
                "train_reward_w10": window["reward_avg"].mean(),
                "entropy_w10": window["entropy_avg"].mean(),
                "avg_resp_len_w10": window["avg_response_len"].mean(),
            })
    return pd.DataFrame(rows)


def write_eval_outputs(table):
    csv_path = OUT_DIR / "eval_summary.csv"
    md_path = OUT_DIR / "eval_summary.md"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"saved {csv_path}")

    # Markdown
    cols = ["run", "step", "math500_acc", "math500_correct", "math500_total",
            "train_correctness_w10", "train_reward_w10", "entropy_w10", "avg_resp_len_w10"]
    headers = ["Run", "Step", "MATH-500 acc", "Correct", "Total",
               "Train correct (last 10)", "Train reward (last 10)",
               "Entropy (last 10)", "Resp len (last 10)"]
    fmt = {
        "math500_acc": lambda v: f"{v * 100:.1f}%",
        "train_correctness_w10": lambda v: f"{v:.3f}",
        "train_reward_w10": lambda v: f"{v:.3f}",
        "entropy_w10": lambda v: f"{v:.3f}",
        "avg_resp_len_w10": lambda v: f"{v:.0f}",
    }
    lines = ["# Eval summary", "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(fmt[c](v) if c in fmt else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"saved {md_path}")


def main():
    runs_with_dfs = []
    for run in RUNS:
        df = load_run(run["csv"])
        print(f"{run['name']}: {len(df)} steps")
        runs_with_dfs.append((run, df))
        plot_one_run(run, df)

    table = build_eval_table(runs_with_dfs)
    print("\n=== eval table ===")
    print(table.to_string(index=False))
    write_eval_outputs(table)


if __name__ == "__main__":
    main()
