# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from transformers import AutoConfig

from .jit_utils import jit_x0_prediction_loss
from .misc import next_token_prediction
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig, FinalLayer, ModulatedAttentionBlock, RMSNorm, RotaryEmbedding, TimestepEmbedder
from .qwen2 import Qwen2ForCausalLM


def _patchify_5d(pixel_values, patch_embed, reshape_frame_to_batch_dim=False):
    b, c, t, h, w = pixel_values.shape
    if reshape_frame_to_batch_dim:
        pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w")
        return patch_embed(pixel_values)
    frame_embeddings = []
    for t_idx in range(t):
        frame_embeddings.append(patch_embed(pixel_values[:, :, t_idx]))
    return torch.cat(frame_embeddings, dim=1)


class SimplePixelPatchEmbedding(nn.Module):
    """Raw RGB pixel patch embedding used by the Tuna-2 pixel path."""

    def __init__(self, in_channels=3, hidden_size=2048, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.norm = RMSNorm(hidden_size)

    @property
    def proj(self):
        """Compatibility alias for Show-o2-style initialization helpers."""
        return self.patch_embedding

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class Tuna2PixelQwen2_5(ModelMixin, ConfigMixin):
    """Tuna-2 style pixel-space model built on the Show-o2/Qwen2.5 backbone."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            llm_vocab_size=None,
            llm_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            load_stage1_model=None,
            init_llm_from_config=False,
            image_channels=3,
            image_latent_dim=None,
            image_size=432,
            pixel_patch_size=16,
            patch_size=None,
            hidden_size=1536,
            num_diffusion_layers=10,
            num_attention_heads=24,
            num_key_value_heads=8,
            reshape_frame_to_batch_dim=False,
            add_time_embeds=True,
            add_aspect_ratio_embeds=True,
            use_3d_rope=True,
            use_mask_token=True,
            enable_mask_token=None,
            mask_ratio=0.0,
            masked_image_ratio=None,
            masked_image_ratio_min=0.0,
            use_disp=False,
            **kwargs,
    ):
        super().__init__()
        if image_latent_dim is not None:
            image_channels = image_latent_dim
        if patch_size is not None:
            pixel_patch_size = patch_size
        if enable_mask_token is not None:
            use_mask_token = enable_mask_token
        if masked_image_ratio is not None:
            mask_ratio = masked_image_ratio

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if init_llm_from_config:
            self.tuna = Qwen2ForCausalLM(llm_config)
        else:
            self.tuna = Qwen2ForCausalLM.from_pretrained(llm_model_path, attn_implementation="sdpa")
        if llm_vocab_size is not None:
            self.tuna.resize_token_embeddings(llm_vocab_size)

        self.vision_encoder = SimplePixelPatchEmbedding(
            in_channels=image_channels,
            hidden_size=hidden_size,
            patch_size=pixel_patch_size,
        )

        if add_aspect_ratio_embeds:
            self.aspect_ratio_embed = TimestepEmbedder(hidden_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if use_mask_token else None

        self.use_disp = use_disp
        self.gradient_checkpointing = False
        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=self.tuna.config.hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=self.tuna.config.intermediate_size,
            max_position_embeddings=self.tuna.config.max_position_embeddings,
        )
        self.rotary_emb = RotaryEmbedding(config=self.diffusion_head_config)
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size),
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
            if add_aspect_ratio_embeds:
                self.ar_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in range(num_diffusion_layers)]
        )
        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size,
            pixel_patch_size,
            image_channels,
        )

        self.reset_parameters()
        if load_stage1_model is not None and load_stage1_model != "no":
            self.load_stage1_checkpoint(load_stage1_model)

    @property
    def showo(self):
        """Compatibility alias for older Show-o2 training utilities."""
        return self.tuna

    @property
    def pixel_embedder(self):
        """Compatibility alias for the first implementation of this local Tuna-2 path."""
        return self.vision_encoder

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vision_encoder.proj.weight.view(self.vision_encoder.proj.weight.shape[0], -1))
        nn.init.constant_(self.vision_encoder.proj.bias, 0)
        if self.mask_token is not None:
            scale = self.config.hidden_size ** -0.5
            nn.init.normal_(self.mask_token, std=scale)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.diffusion_head_a.apply(_basic_init)
        if hasattr(self, "diff_proj"):
            self.diff_proj.apply(_basic_init)
        if hasattr(self, "time_embed_proj"):
            _basic_init(self.time_embed_proj)
        if hasattr(self, "ar_embed_proj"):
            _basic_init(self.ar_embed_proj)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if hasattr(self, "aspect_ratio_embed"):
            nn.init.normal_(self.aspect_ratio_embed.mlp[0].weight, std=0.02)
            nn.init.normal_(self.aspect_ratio_embed.mlp[2].weight, std=0.02)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def load_stage1_checkpoint(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        state_dict = OrderedDict()
        own_state = self.state_dict()
        for key, value in checkpoint.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("tuna_model.")
            if key.startswith("showo."):
                key = "tuna." + key[len("showo."):]
            if key.startswith("pixel_embedder."):
                key = "vision_encoder." + key[len("pixel_embedder."):]
            if key in own_state and own_state[key].shape == value.shape:
                state_dict[key] = value
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded Tuna-2 stage checkpoint from {checkpoint_path}; "
            f"matched={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )

    def patchify_pixels(self, pixels):
        p = self.config.pixel_patch_size
        b, c, h, w = pixels.shape
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Image size {(h, w)} must be divisible by pixel_patch_size={p}")
        patches = pixels.reshape(b, c, h // p, p, w // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(b, (h // p) * (w // p), p * p * c)
        return patches

    def unpatchify_pixels(self, patches, height=None, width=None):
        p = self.config.pixel_patch_size
        c = self.config.image_channels
        height = height or self.config.image_size
        width = width or self.config.image_size
        h, w = height // p, width // p
        pixels = patches.reshape(patches.shape[0], h, w, p, p, c)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4).reshape(patches.shape[0], c, height, width)
        return pixels

    def _apply_patch_mask(self, image_embeds):
        if not self.training or self.mask_token is None or self.config.mask_ratio <= 0:
            return image_embeds
        min_ratio = getattr(self.config, "masked_image_ratio_min", 0.0)
        max_ratio = self.config.mask_ratio
        ratios = torch.empty(image_embeds.shape[0], device=image_embeds.device).uniform_(min_ratio, max_ratio)
        mask = torch.rand(image_embeds.shape[:2], device=image_embeds.device) < ratios[:, None]
        if not mask.any():
            return image_embeds
        image_embeds = image_embeds.clone()
        image_embeds[mask] = self.mask_token.to(image_embeds.dtype)
        return image_embeds

    def _prefix_count(self):
        count = 0
        if getattr(self.config, "add_aspect_ratio_embeds", False):
            count += 2
        if getattr(self.config, "add_time_embeds", False):
            count += 1
        return count

    def _patch_axis_ids(self, num_patches, patch_height, patch_width, num_frames, device):
        expected_patches = patch_height * patch_width * num_frames
        num_patches = min(num_patches, expected_patches)
        frame_ids = torch.arange(num_frames, device=device).view(num_frames, 1, 1)
        row_ids = torch.arange(patch_height, device=device).view(1, patch_height, 1)
        col_ids = torch.arange(patch_width, device=device).view(1, 1, patch_width)
        linear_ids = frame_ids * patch_height * patch_width + row_ids * patch_width + col_ids
        return linear_ids.reshape(-1)[:num_patches]

    def build_rope(self, hidden_states, modality_positions, patch_height, patch_width, num_frames=1):
        position_ids = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0).repeat(hidden_states.shape[0], 1)

        if getattr(self.config, "use_3d_rope", True) and modality_positions is not None:
            prefix_count = self._prefix_count()
            for i, modality_batch in enumerate(modality_positions):
                for offset, length in modality_batch:
                    offset = int(offset)
                    length = int(length)
                    if length <= prefix_count:
                        continue
                    start = offset + prefix_count
                    patch_len = length - prefix_count
                    axis_ids = self._patch_axis_ids(
                        patch_len,
                        patch_height,
                        patch_width,
                        num_frames,
                        hidden_states.device,
                    )
                    position_ids[i, start:start + axis_ids.numel()] = axis_ids

        return self.rotary_emb(hidden_states, position_ids)

    def _aspect_ratio_prefix(self, num_images, patch_height, patch_width, dtype, device):
        if not getattr(self.config, "add_aspect_ratio_embeds", False):
            return []
        h_ids = torch.full((num_images,), patch_height, device=device, dtype=dtype)
        w_ids = torch.full((num_images,), patch_width, device=device, dtype=dtype)
        height_embeds = self.aspect_ratio_embed(h_ids, dtype)
        width_embeds = self.aspect_ratio_embed(w_ids, dtype)
        if hasattr(self, "ar_embed_proj"):
            height_embeds = self.ar_embed_proj(height_embeds)
            width_embeds = self.ar_embed_proj(width_embeds)
        return [height_embeds.to(dtype), width_embeds.to(dtype)]

    def _insert_image_embeds(
            self,
            input_embeds,
            image_embeds,
            time_embeds_proj,
            modality_positions,
            image_labels=None,
            image_masks=None,
            max_seq_len=None,
            prefix_embeds=None,
            clean_image_embeds=None,
            only_denoise_last_image=False,
    ):
        dtype = input_embeds.dtype
        if image_labels is not None:
            label_dim = image_labels.shape[-1]
            new_image_labels = torch.zeros(
                input_embeds.shape[0],
                max_seq_len,
                label_dim,
                device=input_embeds.device,
                dtype=dtype,
            )
            image_masks = image_masks[:, :, None].repeat(1, 1, label_dim)
        else:
            new_image_labels = None

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                offset = int(offset)
                length = int(length)
                if length == 0:
                    continue
                idx = i * modality_positions.size(1) + j
                cursor = offset
                if prefix_embeds is not None:
                    for prefix in prefix_embeds:
                        input_embeds[i, cursor] = prefix[idx]
                        if new_image_labels is not None:
                            image_masks[i, cursor] = 0
                        cursor += 1
                if getattr(self.config, "add_time_embeds", False):
                    input_embeds[i, cursor] = time_embeds_proj[idx]
                    if new_image_labels is not None:
                        image_masks[i, cursor] = 0
                    cursor += 1

                image_len = min(max(length - (cursor - offset), 0), image_embeds.shape[1])
                source_embeds = image_embeds[idx]
                if clean_image_embeds is not None:
                    use_clean = not only_denoise_last_image or j < len(modality_batch) - 1
                    if use_clean and idx < clean_image_embeds.shape[0]:
                        source_embeds = clean_image_embeds[idx]
                input_embeds[i, cursor:cursor + image_len] = source_embeds[:image_len]
                if new_image_labels is not None:
                    new_image_labels[i, cursor:cursor + image_len] = image_labels[idx, :image_len]

        return input_embeds, new_image_labels, image_masks

    def forward_und_only(
            self,
            text_tokens=None,
            image_latents=None,
            image_pixels=None,
            t=None,
            attention_mask=None,
            text_labels=None,
            modality_positions=None,
            output_hidden_states=True,
            max_seq_len=None,
            **kwargs,
    ):
        outputs = self.forward(
            text_tokens=text_tokens,
            image_latents=image_latents,
            image_pixels=image_pixels,
            t=t,
            attention_mask=attention_mask,
            text_labels=text_labels,
            modality_positions=modality_positions,
            output_hidden_states=output_hidden_states,
            max_seq_len=max_seq_len,
        )
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def forward(
            self,
            text_tokens=None,
            image_pixels=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            diffhead_attention_mask=None,
            output_hidden_states=True,
            max_seq_len=None,
            device=None,
            **kwargs,
    ):
        if max_seq_len is None:
            max_seq_len = text_tokens.size(1)
        if device is None:
            device = text_tokens.device

        if image_pixels is None and image_latents is not None:
            image_pixels = image_latents

        if image_pixels is None:
            outputs = self.tuna(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
            logits = outputs["logits"]
            if text_labels is None:
                return logits
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp, logits.new_zeros(())

        input_embeds = self.tuna.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if image_pixels.dim() == 4:
            image_pixels = image_pixels.unsqueeze(2)
        b, c, T, h, w = image_pixels.shape
        p = self.config.pixel_patch_size
        h_, w_ = h // p, w // p

        image_embeds = _patchify_5d(
            image_pixels.to(dtype),
            self.vision_encoder,
            getattr(self.config, "reshape_frame_to_batch_dim", False),
        )
        clean_image_embeds = kwargs.get("clean_image_embeds", None)
        if clean_image_embeds is not None:
            if clean_image_embeds.dim() == 4:
                clean_image_embeds = clean_image_embeds.unsqueeze(2)
            if clean_image_embeds.dim() == 5:
                clean_image_embeds = _patchify_5d(
                    clean_image_embeds.to(dtype),
                    self.vision_encoder,
                    getattr(self.config, "reshape_frame_to_batch_dim", False),
                )
            clean_image_embeds = clean_image_embeds.to(dtype)
        image_embeds = self._apply_patch_mask(image_embeds)
        if image_labels is not None:
            if image_labels.dim() == 4:
                image_labels = image_labels.unsqueeze(2)
            image_labels = rearrange(image_labels.to(dtype), "b c t h w -> b (t h w) c")
            image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
            image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))

        if t is None:
            t = torch.ones(image_pixels.shape[0], device=device, dtype=dtype)
        time_embeds = self.time_embed(t.to(dtype), dtype)
        time_embeds_proj = self.time_embed_proj(time_embeds) if hasattr(self, "time_embed_proj") else time_embeds
        prefix_embeds = self._aspect_ratio_prefix(
            image_pixels.shape[0],
            h_,
            w_,
            dtype,
            image_pixels.device,
        )

        input_embeds, new_image_labels, image_masks = self._insert_image_embeds(
            input_embeds,
            image_embeds,
            time_embeds_proj,
            modality_positions,
            image_labels=image_labels,
            image_masks=image_masks,
            max_seq_len=max_seq_len,
            prefix_embeds=prefix_embeds,
            clean_image_embeds=clean_image_embeds,
            only_denoise_last_image=kwargs.get("only_denoise_last_image", False),
        )

        outputs = self.tuna(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        logits, last_hidden_states = outputs["logits"], outputs["hidden_states"][-1]

        if hasattr(self, "diff_proj"):
            last_hidden_states = self.diff_proj(last_hidden_states)
        position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
        position_embeddings = self.build_rope(last_hidden_states, modality_positions, h_, w_, T)
        for layer in self.diffusion_head_a:
            layer_kwargs = {
                "hidden_states": last_hidden_states,
                "adaln_input": time_embeds,
                "attention_mask": diffhead_attention_mask if diffhead_attention_mask is not None else attention_mask,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
                "modality_positions": modality_positions,
            }
            if self.gradient_checkpointing and self.training:
                last_hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda hidden_states, adaln_input: layer(
                        hidden_states=hidden_states,
                        adaln_input=adaln_input,
                        attention_mask=layer_kwargs["attention_mask"],
                        position_ids=layer_kwargs["position_ids"],
                        position_embeddings=layer_kwargs["position_embeddings"],
                        modality_positions=layer_kwargs["modality_positions"],
                    )[0],
                    last_hidden_states,
                    time_embeds,
                    use_reentrant=False,
                )
            else:
                last_hidden_states = layer(**layer_kwargs)[0]
        x_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)

        if text_labels is None and image_labels is None:
            prefix_count = self._prefix_count()
            pred_patches = []
            for i, modality_batch in enumerate(modality_positions):
                for offset, length in modality_batch:
                    offset = int(offset)
                    length = int(length)
                    if length == 0:
                        continue
                    start = offset + prefix_count
                    image_len = max(length - prefix_count, 0)
                    pred_patches.append(x_pred[i, start:start + image_len])
            if len(pred_patches) == 0:
                return logits, None
            pred_patches = torch.stack(pred_patches, dim=0)
            pred_pixels = self.unpatchify_pixels(pred_patches, height=h, width=w)
            if T == 1:
                pred_pixels = pred_pixels.unsqueeze(2)
            return logits, pred_pixels

        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
        else:
            loss_ntp = logits.new_zeros(())

        if new_image_labels is not None and image_masks is not None and image_masks.bool().any():
            if t.shape[0] == x_pred.shape[0]:
                t_for_loss = t
            elif t.shape[0] == x_pred.shape[0] * 2:
                t_for_loss = t[1::2]
            else:
                t_for_loss = None
            loss_flow = jit_x0_prediction_loss(
                x_pred,
                new_image_labels[:x_pred.shape[0]],
                t_for_loss,
                image_masks,
            )
            if getattr(self.config, "use_disp", False):
                valid = image_masks.bool()
                loss_flow = loss_flow + 0.1 * (x_pred[valid] - new_image_labels[:x_pred.shape[0]][valid]).abs().mean()
        else:
            loss_flow = logits.new_zeros(())

        return logits, loss_ntp, loss_flow

    @torch.no_grad()
    def t2i_generate(
            self,
            image_latents=None,
            image_pixels=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            diffhead_attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            second_time=False,
            **kwargs,
    ):
        clean_image_embeds = kwargs.get("clean_image_embeds", None)
        if image_pixels is None:
            image_pixels = image_latents

        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])

            if second_time:
                t_cond, t_uncond = torch.chunk(t, 2)
                text_tokens_cond, text_tokens_uncond = torch.chunk(text_tokens, 2)
                image_pixels_cond, image_pixels_uncond = torch.chunk(image_pixels, 2)
                clean_image_embeds_cond, clean_image_embeds_uncond = None, None
                if clean_image_embeds is not None:
                    clean_image_embeds_cond, clean_image_embeds_uncond = torch.chunk(clean_image_embeds, 2)

                _, x0_cond = self(
                    text_tokens_cond,
                    image_pixels=image_pixels_cond,
                    t=t_cond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds_cond,
                )
                _, x0_uncond = self(
                    text_tokens_uncond,
                    image_pixels=image_pixels_uncond,
                    t=t_uncond,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds_uncond,
                )
            else:
                _, x0 = self(
                    text_tokens,
                    image_pixels=image_pixels,
                    t=t,
                    attention_mask=attention_mask,
                    diffhead_attention_mask=diffhead_attention_mask,
                    modality_positions=modality_positions,
                    first_frame_as_cond=first_frame_as_cond,
                    only_denoise_last_image=only_denoise_last_image,
                    output_hidden_states=True,
                    max_seq_len=max_seq_len,
                    clean_image_embeds=clean_image_embeds,
                )
                x0_cond, x0_uncond = torch.chunk(x0, 2)

            x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            return torch.cat([x0, x0], dim=0)

        _, x0 = self(
            text_tokens=text_tokens,
            image_pixels=image_pixels,
            t=t,
            attention_mask=attention_mask,
            diffhead_attention_mask=diffhead_attention_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            clean_image_embeds=clean_image_embeds,
        )
        return x0

    @torch.no_grad()
    def t2i_generate_edit(
            self,
            image_latents=None,
            image_pixels=None,
            source_pixels=None,
            clean_image_embeds=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            diffhead_attention_mask=None,
            modality_positions=None,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        if image_pixels is None:
            image_pixels = image_latents
        if clean_image_embeds is None and source_pixels is not None:
            clean_image_embeds = source_pixels
        return self.t2i_generate(
            image_pixels=image_pixels,
            t=t,
            text_tokens=text_tokens,
            attention_mask=attention_mask,
            diffhead_attention_mask=diffhead_attention_mask,
            modality_positions=modality_positions,
            only_denoise_last_image=True,
            max_seq_len=max_seq_len,
            guidance_scale=guidance_scale,
            clean_image_embeds=clean_image_embeds,
            **kwargs,
        )
