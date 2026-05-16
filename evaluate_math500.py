# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
from pathlib import Path

import torch

from reasoning_from_scratch.qwen3 import (
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    load_math500_test,
    evaluate_math500_stream,
    load_model_and_tokenizer,
    load_tokenizer_only,
    render_prompt,
    extract_final_candidate,
    grade_answer,
    eta_progress_message,
)
from qwen_hf_runtime import generate_text_hf, generate_texts_hf, load_hf_qwen_model


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Zero-based index of the first MATH-500 example to evaluate.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help=(
            "Zero-based exclusive end index. If omitted, evaluates "
            "--dataset_size examples from --start_index."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional path to a .pth checkpoint to load model weights from.",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="scratch",
        choices=["scratch", "hf"],
        help="Model runtime. 'hf' uses Transformers AutoModelForCausalLM with SDPA.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help=(
            "Generation batch size for --runtime hf. Defaults to serial "
            "generation to preserve previous behavior."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help=(
            "Optional explicit output JSONL path. If omitted, a filename is "
            "derived from the model variant, checkpoint name (if any), and "
            "device — so distinct runs don't overwrite each other."
        ),
    )
    return parser.parse_args()


def evaluate_math500_stream_hf(
    model,
    tokenizer,
    device,
    math_data,
    out_path,
    max_new_tokens=512,
    batch_size=1,
    verbose=False,
):
    import json
    import time

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0
    start_time = time.time()
    if batch_size <= 0:
        raise ValueError("--eval_batch_size must be > 0")

    with open(out_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, num_examples, batch_size):
            batch_rows = math_data[batch_start:batch_start + batch_size]
            prompts = [render_prompt(row["problem"]) for row in batch_rows]
            if batch_size == 1:
                gen_texts = [
                    generate_text_hf(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompts[0],
                        device=device,
                        max_new_tokens=max_new_tokens,
                        verbose=verbose,
                    )
                ]
            else:
                gen_texts = generate_texts_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )

            for offset, (row, gen_text) in enumerate(zip(batch_rows, gen_texts)):
                i = batch_start + offset + 1
                total_len += len(tokenizer.encode(gen_text))
                extracted = extract_final_candidate(gen_text)
                is_correct = grade_answer(extracted, row["answer"])
                num_correct += int(is_correct)

                record = {
                    "index": i,
                    "problem": row["problem"],
                    "gtruth_answer": row["answer"],
                    "generated_text": gen_text,
                    "extracted": extracted,
                    "correct": bool(is_correct),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if verbose:
                    print(
                        f"\n\n{'='*50}\nMATH-500: {i}/{num_examples}\n"
                        f"{'='*50}\nExtracted: {extracted}\n"
                        f"Expected:  {row['answer']}\n"
                        f"Correct so far: {num_correct}\n{'-'*50}"
                    )
            f.flush()

            processed = min(batch_start + len(batch_rows), num_examples)
            progress_msg = eta_progress_message(
                processed=processed,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    avg_len = total_len / num_examples if num_examples else 0.0
    print(f"Average response length: {avg_len:.2f} tokens")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens
    use_compile = args.compile

    print("Model:", which_model)
    print("Runtime:", args.runtime)
    print("Device:", device)
    print("Eval batch size:", args.eval_batch_size)
    dev_name = str(device).replace(":", "-")

    math_data = load_math500_test()
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end_index must be greater than or equal to --start_index")
    end_index = args.end_index
    if end_index is None:
        end_index = args.start_index + dataset_size
    eval_data = math_data[args.start_index:end_index]

    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    if args.runtime == "hf":
        tokenizer = load_tokenizer_only(which_model=which_model)
        model = load_hf_qwen_model(
            checkpoint_path=args.checkpoint_path,
            device=device,
            attn_implementation="sdpa",
        )
    elif args.checkpoint_path:
        # To load the saved RL checkpoint files from chapter 6
        tokenizer = load_tokenizer_only(which_model=which_model)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        if args.compile:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            model = torch.compile(model)
    else:
        model, tokenizer = load_model_and_tokenizer(
            which_model=which_model,
            device=device,
            use_compile=args.compile
        )

    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    if args.out_path:
        out_path = args.out_path
    else:
        if args.checkpoint_path:
            ckpt_tag = Path(args.checkpoint_path).stem
            out_path = (
                f"math500_{which_model}-{args.runtime}-{ckpt_tag}-"
                f"{dev_name}-evaluate-script.jsonl"
            )
        else:
            out_path = (
                f"math500_{which_model}-{args.runtime}-{dev_name}-"
                "evaluate-script.jsonl"
            )

    print("Writing results to:", out_path)

    if args.runtime == "hf":
        num_correct, num_examples, acc = evaluate_math500_stream_hf(
            model=model,
            out_path=out_path,
            tokenizer=tokenizer,
            device=device,
            math_data=eval_data,
            max_new_tokens=max_new_tokens,
            batch_size=args.eval_batch_size,
            verbose=args.verbose,
        )
    else:
        num_correct, num_examples, acc = evaluate_math500_stream(
            model=model,
            out_path=out_path,
            tokenizer=tokenizer,
            device=device,
            math_data=eval_data,
            max_new_tokens=max_new_tokens,
            verbose=args.verbose,
        )
