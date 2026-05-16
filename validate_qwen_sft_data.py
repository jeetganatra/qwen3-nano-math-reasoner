import argparse
import json
import re
from pathlib import Path


REQUIRED_KEYS = {
    "problem",
    "gtruth_answer",
    "answer",
    "message_thinking",
    "message_content",
    "provider",
    "teacher_model",
    "usage",
    "finish_reason",
}

BOXED_RE = re.compile(
    r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
    re.DOTALL,
)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", help="Path to generated Qwen SFT JSON data.")
    parser.add_argument("--expected_rows", type=int, default=12000)
    parser.add_argument("--teacher_model", default="qwen/qwen3-235b-a22b")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument(
        "--allow_empty_thinking",
        action="store_true",
        help="Allow empty message_thinking fields if the provider does not expose reasoning.",
    )
    parser.add_argument(
        "--clean_out",
        type=str,
        default=None,
        help=(
            "Optional path to write a cleaned dataset. Cleaning keeps only rows "
            "with finish_reason='stop', non-empty message_content and "
            "message_thinking, and a regex-extractable \\boxed{...} answer in "
            "message_content."
        ),
    )
    return parser.parse_args()


def fail(message):
    raise SystemExit(f"Validation failed: {message}")


def clean_rows(rows):
    kept = []
    drops = {
        "finish_reason_not_stop": 0,
        "empty_message_content": 0,
        "empty_message_thinking": 0,
        "missing_regex_extractable_boxed": 0,
    }

    for row in rows:
        if row.get("finish_reason") != "stop":
            drops["finish_reason_not_stop"] += 1
            continue
        content = row.get("message_content")
        thinking = row.get("message_thinking")
        if not isinstance(content, str) or not content.strip():
            drops["empty_message_content"] += 1
            continue
        if not isinstance(thinking, str) or not thinking.strip():
            drops["empty_message_thinking"] += 1
            continue
        if BOXED_RE.search(content) is None:
            drops["missing_regex_extractable_boxed"] += 1
            continue
        kept.append(row)

    return kept, drops


def main():
    args = parse_args()
    data_path = Path(args.data_path).expanduser()
    with data_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        fail(f"root must be a list, got {type(rows).__name__}")
    if len(rows) != args.expected_rows:
        fail(f"expected {args.expected_rows} rows, got {len(rows)}")

    seen_problems = set()
    total_content_chars = 0
    empty_thinking = 0
    finish_reasons = {}

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            fail(f"row {idx} must be an object, got {type(row).__name__}")

        missing = REQUIRED_KEYS - set(row)
        if missing:
            fail(f"row {idx} missing keys: {sorted(missing)}")

        if row["provider"] != args.provider:
            fail(f"row {idx} provider={row['provider']!r}, expected {args.provider!r}")
        if row["teacher_model"] != args.teacher_model:
            fail(
                f"row {idx} teacher_model={row['teacher_model']!r}, "
                f"expected {args.teacher_model!r}"
            )
        if row["answer"] != row["gtruth_answer"]:
            fail(f"row {idx} answer and gtruth_answer differ")

        problem = row["problem"]
        content = row["message_content"]
        thinking = row["message_thinking"]
        if not isinstance(problem, str) or not problem.strip():
            fail(f"row {idx} has empty/non-string problem")
        if problem in seen_problems:
            fail(f"row {idx} duplicates an earlier problem")
        seen_problems.add(problem)

        if not isinstance(content, str) or not content.strip():
            fail(f"row {idx} has empty/non-string message_content")
        if not isinstance(thinking, str):
            fail(f"row {idx} has non-string message_thinking")
        if not thinking.strip():
            empty_thinking += 1

        finish_reason = row["finish_reason"]
        finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
        total_content_chars += len(content)

    if empty_thinking and not args.allow_empty_thinking:
        fail(
            f"{empty_thinking} rows have empty message_thinking. "
            "Use --allow_empty_thinking only if this is expected for the selected model."
        )

    if args.clean_out:
        cleaned_rows, drops = clean_rows(rows)
        clean_path = Path(args.clean_out).expanduser()
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        with clean_path.open("w", encoding="utf-8") as f:
            json.dump(cleaned_rows, f, ensure_ascii=False, indent=2)
            f.write("\n")

        raw_count = len(rows)
        clean_count = len(cleaned_rows)
        retention = clean_count / raw_count if raw_count else 0.0
        print(f"Clean output: {clean_path}")
        print(f"Clean rows: {clean_count}")
        print(f"Clean retention: {100 * retention:.2f}%")
        print(f"Drops: {json.dumps(drops, sort_keys=True)}")

    avg_content_chars = total_content_chars / len(rows)
    print(f"Validated rows: {len(rows)}")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Provider: {args.provider}")
    print(f"Average message_content chars: {avg_content_chars:.1f}")
    print(f"Empty thinking rows: {empty_thinking}")
    finish_reasons_for_print = {str(key): value for key, value in finish_reasons.items()}
    print(f"Finish reasons: {json.dumps(finish_reasons_for_print, sort_keys=True)}")


if __name__ == "__main__":
    main()
