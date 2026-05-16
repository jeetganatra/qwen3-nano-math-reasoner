"""Compare unbatched vs batched first-token logits for the same prompt.

If the batched logits are identical (modulo fp32 noise) to unbatched logits at
the same row, then the divergence in sample_response_batched is pure RNG and
the long-tail length distribution for math problems makes 8 samples noisy.

If the logits differ, there's a real bug in how I'm setting up the batched
forward and we need to fix it before the long run.
"""
import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    render_prompt,
    load_model_and_tokenizer,
)
from reasoning_from_scratch.ch04 import top_p_filter
from reasoning_from_scratch.ch06 import load_math_train
from reasoning_from_scratch.qwen3 import KVCache


def prefill_logits_unbatched(model, prompt_ids, device):
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(prompt_ids.unsqueeze(0), cache=cache)[:, -1]  # (1, V)
    return logits.float().squeeze(0)  # (V,)


def prefill_logits_batched(model, prompt_ids, device, n=8):
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    batch_input = prompt_ids.unsqueeze(0).expand(n, -1).contiguous()
    logits = model(batch_input, cache=cache)[:, -1]  # (N, V)
    return logits.float()  # (N, V)


def main():
    torch.manual_seed(42)
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(
        which_model="base", device=device, use_compile=False
    )
    model.eval()

    math_data = load_math_train()
    prompt = render_prompt(math_data[0]["problem"])
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    print(f"prompt_len = {prompt_ids.numel()}")

    with torch.no_grad():
        u = prefill_logits_unbatched(model, prompt_ids, device)            # (V,)
        b = prefill_logits_batched(model, prompt_ids, device, n=8)         # (8, V)

    # All 8 batched rows should be identical to each other and to the unbatched.
    print("\n=== batched-vs-unbatched logits ===")
    for i in range(8):
        diff = (b[i] - u).abs()
        print(f"  row {i}: max_abs_diff={diff.max().item():.3e} "
              f"mean_abs_diff={diff.mean().item():.3e}")

    print("\n=== batched-vs-batched (rows should be IDENTICAL) ===")
    for i in range(1, 8):
        diff = (b[i] - b[0]).abs()
        print(f"  row {i} vs row 0: max={diff.max().item():.3e}")

    # Top-10 tokens by probability in each
    eos = tokenizer.eos_token_id
    print(f"\n=== EOS token id = {eos} ===")
    p_u = torch.softmax(u, dim=-1)
    p_b0 = torch.softmax(b[0], dim=-1)
    print(f"P(EOS) unbatched = {p_u[eos].item():.6e}")
    print(f"P(EOS) batched row 0 = {p_b0[eos].item():.6e}")
    if eos is not None:
        # also check after temperature/top-p
        for label, raw in [("unbatched", u), ("batched_row0", b[0])]:
            x = raw / 0.8
            p = torch.softmax(x, dim=-1).unsqueeze(0)
            p = top_p_filter(p, 0.9).squeeze(0)
            renorm = p / p.sum()
            print(f"  after temp=0.8/top_p=0.9, {label}: "
                  f"P(EOS)={renorm[eos].item():.6e} "
                  f"top-5 ids={renorm.topk(5).indices.tolist()} "
                  f"top-5 prs={[round(x, 4) for x in renorm.topk(5).values.tolist()]}")


if __name__ == "__main__":
    main()
