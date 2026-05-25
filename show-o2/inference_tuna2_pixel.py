# coding=utf-8
# Copyright 2025 NUS Show Lab.

import os
from pathlib import Path

import torch
from PIL import Image

from models import Tuna2PixelQwen2_5, omni_attn_mask_naive
from models.jit_utils import JiTSampler
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import denorm, get_config, get_weight_type, load_state_dict, path_to_llm_name


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_type = get_weight_type(config)

    text_tokenizer, token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)
    model = Tuna2PixelQwen2_5(**config.model.showo).to(device).to(weight_type)

    model_path = config.get("model_path", None)
    if model_path:
        state_dict = load_state_dict(model_path)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    prompts = config.get("prompts", None)
    if prompts is None:
        with open(config.dataset.params.validation_prompts_file, "r", encoding="utf-8") as f:
            prompts = f.read().splitlines()[:config.get("batch_size", 1)]
    if isinstance(prompts, str):
        prompts = [prompts]

    num_image_tokens = config.dataset.preprocessing.num_t2i_image_tokens
    max_seq_len = config.dataset.preprocessing.max_seq_length
    max_text_len = max_seq_len - num_image_tokens - 4
    guidance_scale = config.get("guidance_scale", config.transport.guidance_scale)

    batch_text_tokens, batch_text_tokens_null, batch_positions, batch_positions_null = prepare_gen_input(
        prompts,
        text_tokenizer,
        num_image_tokens,
        token_ids["bos_id"],
        token_ids["eos_id"],
        token_ids["boi_id"],
        token_ids["eoi_id"],
        text_tokenizer.pad_token_id,
        token_ids["img_pad_id"],
        max_text_len,
        device,
    )

    resolution = config.dataset.preprocessing.resolution
    z = torch.randn(len(prompts), 3, 1, resolution, resolution, device=device, dtype=weight_type)
    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_positions, batch_positions_null], dim=0)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_positions

    block_mask = omni_attn_mask_naive(
        text_tokens.size(0),
        text_tokens.size(1),
        modality_positions,
        device,
    ).to(weight_type)

    sampler = JiTSampler(device, noise_scale=config.model.showo.get("noise_scale", 1.0))
    sample_fn, _ = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        cfg_interval=config.transport.get("cfg_interval", None),
    )
    samples = sample_fn(
        z,
        model.t2i_generate,
        text_tokens=text_tokens,
        attention_mask=block_mask,
        diffhead_attention_mask=block_mask,
        modality_positions=modality_positions,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale,
    )[-1]
    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]
    samples = samples.squeeze(2).clamp(-1, 1)

    output_dir = Path(config.get("output_dir", "tuna2_pixel_samples"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(denorm(samples)):
        Image.fromarray(image).save(output_dir / f"sample_{i:04d}.png")


if __name__ == "__main__":
    main()
