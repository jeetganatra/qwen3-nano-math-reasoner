"""Training curve for the OPD-from-base showcase (results/phase1/opd_frombase_v2).

Produces results/training_curves_opd_frombase.png — two panels:
  (left)  MATH-500 50-subset accuracy vs OPD step: the rise from the base floor to the
          step-80 peak and the subsequent over-training degradation, with base/seed
          reference lines and the full-500 confirmation of the peak.
  (right) Training dynamics: k3 trust-region distance (policy barely moves) and student
          entropy — the climb is gentle logit nudging, not large policy drift.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "results/phase1/opd_frombase_v2/logs/qwen_opd_metrics.csv"
OUT = REPO / "results/training_curves_opd_frombase.png"

# Reference points (50-subset @2048, greedy, HF, bs16).
BASE_SUBSET = 0.14       # base floor (7/50)
SEED_SUBSET = 0.38       # SFT seed (19/50)
PEAK_FULL500 = 0.288     # step-80 checkpoint, full-500 @2048 (144/500)


def main():
    rows = list(csv.DictReader(open(CSV)))
    g_step = [int(r["step"]) for r in rows]
    g_k3 = [float(r["ref_k3"]) for r in rows]
    g_ent = [float(r["entropy"]) for r in rows]
    eval_steps = [int(r["step"]) for r in rows if r["eval_acc"]]
    eval_accs = [float(r["eval_acc"]) for r in rows if r["eval_acc"]]
    # prepend the base floor at step 0 (the init, pre-OPD)
    steps = [0] + eval_steps
    accs = [BASE_SUBSET] + eval_accs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- left: the rise ----
    ax1.plot(steps, [a * 100 for a in accs], "o-", color="tab:green",
             lw=2.2, ms=8, label="OPD-from-base (50-subset)")
    peak_i = accs.index(max(accs[1:]))
    ax1.annotate(f"peak {accs[peak_i]*100:.0f}%\n(step {steps[peak_i]})",
                 (steps[peak_i], accs[peak_i] * 100),
                 textcoords="offset points", xytext=(8, 12), fontsize=10,
                 color="tab:green", fontweight="bold")
    ax1.scatter([steps[peak_i]], [PEAK_FULL500 * 100], marker="*", s=240,
                color="darkgreen", zorder=5,
                label=f"step-80 full-500: {PEAK_FULL500*100:.1f}%")
    ax1.axhline(BASE_SUBSET * 100, ls="--", color="gray", lw=1.3,
                label=f"base floor ({BASE_SUBSET*100:.0f}%)")
    ax1.axhline(SEED_SUBSET * 100, ls=":", color="tab:blue", lw=1.3,
                label=f"SFT seed ({SEED_SUBSET*100:.0f}%)")
    ax1.set_xlabel("OPD step")
    ax1.set_ylabel("MATH-500 accuracy (%)")
    ax1.set_title("OPD from base toward Qwen3-4B: a peaked rise\n(early-stop at the peak)")
    ax1.set_ylim(0, 45)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    # ---- right: dynamics ----
    ax2.plot(g_step, g_k3, color="tab:red", lw=1.6, label="k3 (trust-region dist.)")
    ax2.set_xlabel("OPD step")
    ax2.set_ylabel("k3  KL(student‖ref)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_title("Dynamics: policy barely moves (k3≈0), gentle elicitation")
    ax2b = ax2.twinx()
    ax2b.plot(g_step, g_ent, color="tab:purple", lw=1.1, alpha=0.6,
              label="student entropy")
    ax2b.set_ylabel("entropy (nats)", color="tab:purple")
    ax2b.tick_params(axis="y", labelcolor="tab:purple")
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"rise points (step, acc): {list(zip(steps, [round(a,3) for a in accs]))}")


if __name__ == "__main__":
    main()
