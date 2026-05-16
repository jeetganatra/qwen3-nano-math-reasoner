import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def summarize(rows, label):
    total = len(rows)
    if total == 0:
        return {"label": label, "total": 0}

    correct_keys = ["is_correct", "correct", "match"]
    pred_keys = ["predicted_answer", "extracted", "prediction", "model_answer"]
    gen_len_keys = [
        "num_generated_tokens",
        "n_generated_tokens",
        "generated_tokens",
        "response_length",
    ]
    level_keys = ["level", "difficulty"]

    def first_present(row, keys, default=None):
        for k in keys:
            if k in row:
                return row[k]
        return default

    correct = sum(1 for r in rows if bool(first_present(r, correct_keys, False)))

    gen_lens = []
    for r in rows:
        v = first_present(r, gen_len_keys)
        if v is not None:
            try:
                gen_lens.append(int(v))
            except (TypeError, ValueError):
                pass

    level_buckets = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        level = first_present(r, level_keys)
        if level is None:
            continue
        bucket = level_buckets[str(level)]
        bucket["total"] += 1
        if bool(first_present(r, correct_keys, False)):
            bucket["correct"] += 1

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "avg_gen_tokens": statistics.mean(gen_lens) if gen_lens else None,
        "median_gen_tokens": statistics.median(gen_lens) if gen_lens else None,
        "by_level": dict(level_buckets),
    }


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--eval",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeatable: label=path.jsonl. Order determines table column order.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="comparison.csv",
        help="Where to write the comparison summary CSV.",
    )
    return parser.parse_args()


def default_evals():
    """If user doesn't pass --eval, look for common file names."""
    candidates = [
        ("base", "results/sft_deepseek/evals/eval_base.jsonl"),
        ("distilled", "results/sft_deepseek/evals/eval_distilled.jsonl"),
        ("qwen_reasoning", "results/sft_deepseek/evals/eval_qwen_reasoning.jsonl"),
        ("qwen_sft", "results/sft_qwen/evals/eval.jsonl"),
        ("grpo", "results/grpo_from_base/evals/grpo_step00050-math500.jsonl"),
    ]
    return [(label, path) for label, path in candidates if Path(path).exists()]


def main():
    args = parse_args()

    if args.eval:
        evals = []
        for spec in args.eval:
            if "=" not in spec:
                raise SystemExit(f"--eval expects label=path, got: {spec!r}")
            label, _, path = spec.partition("=")
            evals.append((label.strip(), path.strip()))
    else:
        evals = default_evals()
        if not evals:
            raise SystemExit(
                "No eval files found. Pass --eval label=path.jsonl, or place "
                "files under results/<run>/evals/eval_*.jsonl"
            )

    summaries = []
    for label, path in evals:
        if not Path(path).exists():
            print(f"  warn: missing file for {label!r}: {path}")
            continue
        rows = load_jsonl(path)
        s = summarize(rows, label)
        summaries.append(s)

    if not summaries:
        raise SystemExit("No summaries to print.")

    # Headline table
    print()
    print(f"{'Model':<22} {'N':>4} {'Acc':>7} {'AvgTok':>9} {'MedTok':>9}")
    print("-" * 56)
    for s in summaries:
        if s["total"] == 0:
            print(f"{s['label']:<22} {'-':>4} {'-':>7} {'-':>9} {'-':>9}")
            continue
        avg_tok = f"{s['avg_gen_tokens']:.0f}" if s["avg_gen_tokens"] is not None else "-"
        med_tok = f"{s['median_gen_tokens']:.0f}" if s["median_gen_tokens"] is not None else "-"
        print(
            f"{s['label']:<22} {s['total']:>4} "
            f"{100*s['accuracy']:>6.1f}% {avg_tok:>9} {med_tok:>9}"
        )

    # Per-level breakdown if any eval has level info
    has_levels = any(s.get("by_level") for s in summaries)
    if has_levels:
        all_levels = sorted({lvl for s in summaries for lvl in s.get("by_level", {})})
        print()
        header = f"{'Model':<22}" + "".join(f"{lvl[:8]:>10}" for lvl in all_levels)
        print(header)
        print("-" * len(header))
        for s in summaries:
            line = f"{s['label']:<22}"
            for lvl in all_levels:
                bucket = s.get("by_level", {}).get(lvl)
                if bucket and bucket["total"] > 0:
                    line += f"{100*bucket['correct']/bucket['total']:>9.1f}%"
                else:
                    line += f"{'-':>10}"
            print(line)

    # Write CSV
    out_path = Path(args.out_csv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "correct", "accuracy", "avg_gen_tokens", "median_gen_tokens"])
        for s in summaries:
            writer.writerow([
                s["label"],
                s.get("total", 0),
                s.get("correct", 0),
                f"{s.get('accuracy', 0):.4f}" if s["total"] else "",
                f"{s['avg_gen_tokens']:.1f}" if s.get("avg_gen_tokens") is not None else "",
                f"{s['median_gen_tokens']:.1f}" if s.get("median_gen_tokens") is not None else "",
            ])
    print(f"\nWrote summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
