import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "epoch": int(row["epoch"]),
                "step": int(row["total_steps"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--csv",
        type=str,
        default="logs/train_with_distillation_batched_metrics.csv",
        help="Path to the metrics CSV produced by the training script.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="loss_curves.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--log_y",
        action="store_true",
        help="Use log scale on the y axis.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot interactively in addition to saving.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_metrics(csv_path)
    if not rows:
        raise SystemExit(f"CSV has no data rows: {csv_path}")

    steps = [r["step"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    epochs = [r["epoch"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="train_loss", color="tab:blue", marker=".", linewidth=1)
    ax.plot(steps, val_loss, label="val_loss", color="tab:red", marker="o", linewidth=1.5)

    # Mark epoch boundaries (last logged step of each epoch).
    last_step_per_epoch = {}
    for r in rows:
        last_step_per_epoch[r["epoch"]] = r["step"]
    for ep, step in last_step_per_epoch.items():
        ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(step, ax.get_ylim()[1], f"end ep{ep}",
                rotation=90, va="top", ha="right", fontsize=8, color="gray")

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    ax.scatter([steps[best_idx]], [val_loss[best_idx]],
               color="tab:red", s=120, edgecolors="black", zorder=5,
               label=f"best val={val_loss[best_idx]:.4f} @ step {steps[best_idx]}")

    ax.set_xlabel("global step")
    ax.set_ylabel("loss")
    ax.set_title(f"Distillation training curves\n({csv_path.name}, {len(rows)} log points, {max(epochs)} epochs)")
    ax.grid(True, alpha=0.3)
    if args.log_y:
        ax.set_yscale("log")
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path.resolve()}")
    print(f"Best val_loss: {val_loss[best_idx]:.6f} at step {steps[best_idx]} (epoch {epochs[best_idx]})")
    print(f"Final train_loss: {train_loss[-1]:.6f}, val_loss: {val_loss[-1]:.6f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
