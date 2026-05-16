from pathlib import Path

import torch

from reasoning_from_scratch.qwen3 import QWEN_CONFIG_06_B
from reasoning_from_scratch.qwen3_batched import load_model_and_tokenizer
from transformers import AutoConfig, AutoModelForCausalLM


HF_MODEL_ID = "Qwen/Qwen3-0.6B"


def _scratch_to_hf_key(key):
    if key == "tok_emb.weight":
        return "model.embed_tokens.weight"
    if key == "final_norm.scale":
        return "model.norm.weight"
    if key == "out_head.weight":
        return "lm_head.weight"

    prefix = "trf_blocks."
    if not key.startswith(prefix):
        raise KeyError(f"Unrecognized scratch key: {key}")

    rest = key[len(prefix):]
    layer_idx, subkey = rest.split(".", 1)
    layer_prefix = f"model.layers.{layer_idx}."
    replacements = {
        "att.W_query.weight": "self_attn.q_proj.weight",
        "att.W_key.weight": "self_attn.k_proj.weight",
        "att.W_value.weight": "self_attn.v_proj.weight",
        "att.out_proj.weight": "self_attn.o_proj.weight",
        "att.q_norm.scale": "self_attn.q_norm.weight",
        "att.k_norm.scale": "self_attn.k_norm.weight",
        "ff.fc1.weight": "mlp.gate_proj.weight",
        "ff.fc2.weight": "mlp.up_proj.weight",
        "ff.fc3.weight": "mlp.down_proj.weight",
        "norm1.scale": "input_layernorm.weight",
        "norm2.scale": "post_attention_layernorm.weight",
    }
    if subkey not in replacements:
        raise KeyError(f"Unrecognized scratch key: {key}")
    return layer_prefix + replacements[subkey]


def _hf_to_scratch_key(key):
    if key == "model.embed_tokens.weight":
        return "tok_emb.weight"
    if key == "model.norm.weight":
        return "final_norm.scale"
    if key == "lm_head.weight":
        return "out_head.weight"

    prefix = "model.layers."
    if not key.startswith(prefix):
        raise KeyError(f"Unrecognized HF key: {key}")

    rest = key[len(prefix):]
    layer_idx, subkey = rest.split(".", 1)
    layer_prefix = f"trf_blocks.{layer_idx}."
    replacements = {
        "self_attn.q_proj.weight": "att.W_query.weight",
        "self_attn.k_proj.weight": "att.W_key.weight",
        "self_attn.v_proj.weight": "att.W_value.weight",
        "self_attn.o_proj.weight": "att.out_proj.weight",
        "self_attn.q_norm.weight": "att.q_norm.scale",
        "self_attn.k_norm.weight": "att.k_norm.scale",
        "mlp.gate_proj.weight": "ff.fc1.weight",
        "mlp.up_proj.weight": "ff.fc2.weight",
        "mlp.down_proj.weight": "ff.fc3.weight",
        "input_layernorm.weight": "norm1.scale",
        "post_attention_layernorm.weight": "norm2.scale",
    }
    if subkey not in replacements:
        raise KeyError(f"Unrecognized HF key: {key}")
    return layer_prefix + replacements[subkey]


def convert_scratch_state_to_hf(scratch_state):
    return {_scratch_to_hf_key(key): value for key, value in scratch_state.items()}


def convert_hf_state_to_scratch(hf_state):
    return {_hf_to_scratch_key(key): value for key, value in hf_state.items()}


def _load_base_scratch_state():
    model, _ = load_model_and_tokenizer(
        which_model="base",
        device="cpu",
        use_compile=False,
        float32_upcast=False,
    )
    return model.state_dict()


def _load_scratch_state(checkpoint_path=None):
    if checkpoint_path is None:
        return _load_base_scratch_state()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def load_hf_qwen_model(
    checkpoint_path=None,
    device=None,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
):
    config = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    scratch_state = _load_scratch_state(checkpoint_path)
    hf_state = convert_scratch_state_to_hf(scratch_state)
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "HF state load mismatch: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    if device is not None:
        model.to(device)
    return model


class HFQwenScratchCompat(torch.nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.model = load_hf_qwen_model(
            checkpoint_path=checkpoint_path,
            device=device,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        self.cfg = QWEN_CONFIG_06_B

    def forward(self, input_ids, attn_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            use_cache=False,
        )
        return outputs.logits

    def state_dict(self, *args, **kwargs):
        hf_state = self.model.state_dict(*args, **kwargs)
        return convert_hf_state_to_scratch(hf_state)


@torch.inference_mode()
def generate_text_hf(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    if verbose:
        print(tokenizer.decode(generated_ids), end="", flush=True)
    return tokenizer.decode(generated_ids)


@torch.inference_mode()
def generate_texts_hf(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens,
):
    if not prompts:
        return []

    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    max_input_len = max(len(ids) for ids in encoded_prompts)
    pad_id = tokenizer.pad_token_id

    input_rows = []
    mask_rows = []
    for ids in encoded_prompts:
        pad_len = max_input_len - len(ids)
        input_rows.append([pad_id] * pad_len + ids)
        mask_rows.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(input_rows, device=device)
    attention_mask = torch.tensor(mask_rows, device=device, dtype=torch.long)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return [
        tokenizer.decode(output_ids[row_idx, max_input_len:].tolist())
        for row_idx in range(len(prompts))
    ]
