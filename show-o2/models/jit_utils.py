# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and affiliates.
# Adapted for the Show-o2 code layout.

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class JiTNoiseScheduler:
    """JiT-style noise scheduler with rectified-flow interpolation."""

    def __init__(self, P_mean=-0.8, P_std=0.8, noise_scale=1.0, t_eps=5e-2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps

    def sample_timesteps(self, n, device, dtype=None):
        z = torch.randn(n, device=device, dtype=dtype) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def add_noise(self, x, t=None):
        if t is None:
            t = self.sample_timesteps(x.size(0), device=x.device, dtype=x.dtype)
        t_expanded = t.view(-1, *([1] * (x.ndim - 1)))
        noise = torch.randn_like(x) * self.noise_scale
        z_t = t_expanded * x + (1 - t_expanded) * noise
        return z_t, t


def jit_x0_prediction_loss(x0_pred, x0_target, t=None, mask=None):
    loss = F.mse_loss(x0_pred, x0_target, reduction="none")
    if t is not None and t.shape[0] == loss.shape[0]:
        t_expand = t.view(-1, *([1] * (loss.ndim - 1)))
        weights = 1.0 / (1.0 - t_expand).clamp(min=5e-2) ** 2
        loss = loss * weights
    if mask is not None:
        loss = loss[mask.bool()].mean()
    else:
        loss = loss.mean()
    return loss


def prepare_jit_training_batch(pixel_values, noise_scheduler, max_t0=None):
    t = noise_scheduler.sample_timesteps(pixel_values.size(0), device=pixel_values.device, dtype=pixel_values.dtype)
    if max_t0 is not None:
        t = torch.full_like(t, max_t0)
    z_t, t = noise_scheduler.add_noise(pixel_values, t)
    return z_t, t, pixel_values


class JiTSampler:
    """Tuna-2/JiT x0 sampler with the same sample_ode API as transport.Sampler."""

    def __init__(self, device, noise_scale=1.0):
        self.device = device
        self.noise_scale = noise_scale

    def sample_ode(self, sampling_method="heun", num_steps=50, **kwargs):
        if sampling_method not in ["euler", "heun"]:
            raise NotImplementedError(f"Sampling method '{sampling_method}' not supported.")

        def sample_fn(z, model_fn, **model_kwargs):
            guidance_scale = model_kwargs.get("guidance_scale", 1.0)
            cfg_interval = model_kwargs.get("cfg_interval", None)
            timesteps = torch.linspace(0, 1, num_steps + 1, device=z.device, dtype=z.dtype)

            def get_v_pred(z_in, t_scalar):
                t_val = float(t_scalar)
                t_input = torch.full((z_in.shape[0],), t_val, device=z_in.device, dtype=z_in.dtype)

                eff_scale = guidance_scale
                if cfg_interval is not None:
                    low, high = cfg_interval
                    in_interval = (t_val < high) and ((low == 0) or (t_val > low))
                    if not in_interval:
                        eff_scale = 1.0
                updated_kwargs = {**model_kwargs, "guidance_scale": eff_scale}
                x0_pred = model_fn(z_in, t_input, **updated_kwargs)

                if eff_scale > 0 and z_in.shape[0] % 2 == 0:
                    x0_cond, _x0_uncond = x0_pred.chunk(2)
                    x0_guided = x0_cond
                    x0_pred_used = torch.cat([x0_guided, x0_guided], dim=0)
                    z_half = z_in.chunk(2)[0]
                    z_used = torch.cat([z_half, z_half], dim=0)
                else:
                    x0_pred_used = x0_pred
                    z_used = z_in

                t_denom = max(5e-2, 1.0 - t_val)
                v_pred = (x0_pred_used - z_used) / t_denom
                return v_pred, z_used

            num_main_steps = num_steps - 1
            for i in range(num_main_steps):
                t_curr = timesteps[i]
                t_next = timesteps[i + 1]
                dt = t_next - t_curr
                if sampling_method == "heun":
                    v_pred_t, z_used = get_v_pred(z, t_curr)
                    z_next_euler = z_used + dt * v_pred_t
                    v_pred_t_next, _ = get_v_pred(z_next_euler, t_next)
                    v_pred = 0.5 * (v_pred_t + v_pred_t_next)
                    z = z_used + dt * v_pred
                else:
                    v_pred, z_used = get_v_pred(z, t_curr)
                    z = z_used + dt * v_pred

            t_curr = timesteps[-2]
            t_next = timesteps[-1]
            dt = t_next - t_curr
            v_pred, z_used = get_v_pred(z, t_curr)
            z = z_used + dt * v_pred
            return [z]

        return sample_fn, 0.0
