import argparse

import torch

from qwen_distill import render_prompt
from qwen_hf_runtime import load_hf_qwen_model
from reasoning_from_scratch.ch03 import load_tokenizer_only
from reasoning_from_scratch.qwen3_batched import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    load_model_and_tokenizer,
)


PROMPTS = [
    "What is 2+2?",
    "If x+3=10, what is x?",
    "Find the value of 7 squared minus 5.",
]


def load_scratch_model(checkpoint_path, device):
    if checkpoint_path:
        model = Qwen3Model(QWEN_CONFIG_06_B, float32_upcast=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        return model

    model, _ = load_model_and_tokenizer(
        which_model="base",
        device=device,
        use_compile=False,
        float32_upcast=False,
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare scratch Qwen logits against the HF/SDPA runtime."
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--which_model", type=str, default="reasoning")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer_only(which_model=args.which_model)

    scratch = load_scratch_model(args.checkpoint_path, device=device)
    hf = load_hf_qwen_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        attn_implementation="sdpa",
    )
    scratch.eval()
    hf.eval()

    for idx, problem in enumerate(PROMPTS, start=1):
        prompt = render_prompt(problem)
        input_ids = torch.tensor(
            tokenizer.encode(prompt)[: args.max_prompt_tokens],
            device=device,
        ).unsqueeze(0)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)

        scratch_logits = scratch(input_ids, attn_mask=attn_mask)[:, -1, :].float()
        hf_logits = hf(input_ids=input_ids, attention_mask=attn_mask.long()).logits[
            :, -1, :
        ].float()

        diff = (scratch_logits - hf_logits).abs()
        scratch_top = torch.argmax(scratch_logits, dim=-1).item()
        hf_top = torch.argmax(hf_logits, dim=-1).item()
        print(f"prompt_{idx}")
        print("tokens", input_ids.shape[1])
        print("max_abs_diff", diff.max().item())
        print("mean_abs_diff", diff.mean().item())
        print("scratch_top", scratch_top, tokenizer.decode([scratch_top]))
        print("hf_top", hf_top, tokenizer.decode([hf_top]))
        print("top_match", scratch_top == hf_top)


if __name__ == "__main__":
    main()
