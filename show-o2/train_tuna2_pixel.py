# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from datasets import MMUDataset, MixedDataLoader, create_imagetext_dataloader, create_tuna2_jsonl_dataloader
from models import Tuna2PixelQwen2_5, omni_attn_mask_naive
from models.jit_utils import JiTNoiseScheduler, prepare_jit_training_batch
from models.lr_schedulers import get_scheduler
from models.misc import get_text_tokenizer, get_weight_type
from models.my_logging import set_verbosity_error, set_verbosity_info
from utils import AverageMeter, _freeze_params, flatten_omega_conf, get_config, path_to_llm_name

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = get_logger(__name__, log_level="INFO")


def _is_real_path(value):
    return value is not None and str(value).strip() != "" and not str(value).startswith("path/to/")


def _get(config, key, default=None):
    return config.get(key, default) if config is not None else default


def _make_loader(dataset, batch_size, accelerator, num_workers):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def build_dataloaders(config, text_tokenizer, showo_token_ids, accelerator):
    params = config.dataset.params
    preproc = config.dataset.preprocessing
    loaders = []
    names = []

    if _is_real_path(_get(params, "train_t2i_shards_path_or_url")):
        loaders.append(
            create_imagetext_dataloader(
                train_shards_path_or_url=params.train_t2i_shards_path_or_url,
                batch_size=config.training.batch_size_t2i,
                text_tokenizer=text_tokenizer,
                image_size=preproc.resolution,
                max_seq_len=preproc.max_seq_length,
                num_image_tokens=preproc.num_t2i_image_tokens,
                latent_width=preproc.patch_grid_width,
                latent_height=preproc.patch_grid_height,
                cond_dropout_prob=config.training.cond_dropout_prob,
                num_workers=params.num_workers,
                drop_last=True,
                shuffle=True,
                min_res=preproc.min_res,
                random_und_or_gen=0.0,
                showo_token_ids=showo_token_ids,
                system=("", "", ""),
                accelerator=accelerator,
            )
        )
        names.append("showo_t2i")

    if _is_real_path(_get(params, "annotation_path")) and _is_real_path(_get(params, "train_mmu_shards_path_or_url")):
        dataset_mmu = MMUDataset(
            params.train_mmu_shards_path_or_url,
            annotation_path=params.annotation_path,
            default_system_prompt=params.default_system_prompt,
            is_clip_encoder=False,
            text_tokenizer=text_tokenizer,
            image_size=preproc.resolution,
            max_seq_len=preproc.max_seq_length,
            num_image_tokens=preproc.num_mmu_image_tokens,
            latent_width=preproc.patch_grid_width,
            latent_height=preproc.patch_grid_height,
            cond_dropout_prob=config.training.cond_dropout_prob,
            stage=config.training.get("stage", "pre-training"),
            clip_processor=None,
            showo_token_ids=showo_token_ids,
        )
        loaders.append(_make_loader(dataset_mmu, config.training.batch_size_mmu, accelerator, params.num_workers))
        names.append("showo_mmu")
    elif _is_real_path(_get(params, "train_mmu_shards_path_or_url")):
        loaders.append(
            create_imagetext_dataloader(
                train_shards_path_or_url=params.train_mmu_shards_path_or_url,
                batch_size=config.training.batch_size_mmu,
                text_tokenizer=text_tokenizer,
                image_size=preproc.resolution,
                max_seq_len=preproc.max_seq_length,
                num_image_tokens=preproc.num_mmu_image_tokens,
                latent_width=preproc.patch_grid_width,
                latent_height=preproc.patch_grid_height,
                cond_dropout_prob=0.0,
                num_workers=params.num_workers,
                drop_last=True,
                shuffle=True,
                min_res=preproc.min_res,
                random_und_or_gen=0.0,
                showo_token_ids=showo_token_ids,
                system=("", "", ""),
                is_captioning=True,
                accelerator=accelerator,
            )
        )
        names.append("showo_caption_mmu")

    tuna2_paths = [
        ("train_tuna2_ti_path", "t2i", config.training.batch_size_t2i),
        ("train_tuna2_mmu_path", "mmu", config.training.batch_size_mmu),
        ("train_tuna2_edit_path", "edit", config.training.get("batch_size_edit", config.training.batch_size_t2i)),
        ("train_text_only_path", "text_only", config.training.get("batch_size_text", config.training.batch_size_mmu)),
    ]
    for key, task, batch_size in tuna2_paths:
        if _is_real_path(_get(params, key)):
            loaders.append(
                create_tuna2_jsonl_dataloader(
                    anno_path=params[key],
                    batch_size=batch_size,
                    text_tokenizer=text_tokenizer,
                    showo_token_ids=showo_token_ids,
                    task=task,
                    accelerator=accelerator,
                    image_root=_get(params, "image_root", ""),
                    image_size=preproc.resolution,
                    max_seq_len=preproc.max_seq_length,
                    num_image_tokens=preproc.num_t2i_image_tokens,
                    max_num_images=preproc.get("max_num_images", 2),
                    num_workers=params.num_workers,
                    shuffle=True,
                    drop_last=True,
                )
            )
            names.append(f"tuna2_{task}")

    if not loaders:
        raise ValueError("No training dataloaders were created. Set at least one real dataset path in dataset.params.")

    logger.info(f"Created Tuna-2 pixel dataloaders: {names}")
    return loaders


def prepare_pixel_flow_targets(
        pixel_values,
        data_type,
        image_masks,
        modality_positions,
        weight_type,
        jit_noise_scheduler=None,
        und_max_t0=1.0,
        mmu_noise_prob=0.0,
        mmu_noise_level=0.0,
):
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(1)

    batch_size, max_num_images = pixel_values.shape[:2]
    flat_pixels = pixel_values.reshape(batch_size * max_num_images, *pixel_values.shape[2:])
    flat_pixels = flat_pixels.to(weight_type)

    t = torch.ones(flat_pixels.shape[0], device=flat_pixels.device, dtype=weight_type)
    noised_pixels = flat_pixels.clone()
    flow_masks = torch.zeros_like(image_masks)
    jit_noise_scheduler = jit_noise_scheduler or JiTNoiseScheduler()

    flow_types = {"t2i", "ti", "edit", "interleaved_data", "mixed_modal"}
    und_types = {"mmu", "mmu_vid", "mmu_interleaved", "mmu_text", "text_only"}
    for i in range(batch_size):
        sample_type = data_type[i] if isinstance(data_type, list) else data_type
        for j, (offset, length) in enumerate(modality_positions[i]):
            offset = int(offset)
            length = int(length)
            if length == 0:
                continue

            is_flow_target = sample_type in flow_types and image_masks[i, offset:offset + length].bool().any()
            flat_idx = i * max_num_images + j
            max_t0 = None
            if sample_type in und_types:
                if mmu_noise_prob > 0 and torch.rand((), device=flat_pixels.device) < mmu_noise_prob:
                    noise_level = torch.empty((), device=flat_pixels.device, dtype=weight_type).uniform_(0.0, mmu_noise_level)
                    max_t0 = 1.0 - float(noise_level)
                else:
                    max_t0 = und_max_t0

            if is_flow_target or max_t0 is not None:
                zt, sampled_t, _ = prepare_jit_training_batch(
                    flat_pixels[flat_idx:flat_idx + 1],
                    jit_noise_scheduler,
                    max_t0,
                )
                noised_pixels[flat_idx] = zt[0]
                t[flat_idx] = sampled_t[0]

            if is_flow_target:
                flow_masks[i, offset:offset + length] = image_masks[i, offset:offset + length]

    return noised_pixels.unsqueeze(2), t, flat_pixels.unsqueeze(2), flow_masks


def main():
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if config.wandb.get("enabled", True) else None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process and config.wandb.get("enabled", True):
        run_id = config.wandb.get("run_id", None) or wandb.util.generate_id()
        config.wandb.run_id = run_id
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={
                "wandb": {
                    "name": config.experiment.name,
                    "id": run_id,
                    "resume": config.wandb.resume,
                    "entity": config.wandb.get("entity", None),
                }
            },
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    weight_type = get_weight_type(config)
    jit_noise_scheduler = JiTNoiseScheduler(
        P_mean=config.model.showo.get("jit_p_mean", -0.8),
        P_std=config.model.showo.get("jit_p_std", 0.8),
        noise_scale=config.model.showo.get("noise_scale", 1.0),
        t_eps=config.model.showo.get("jit_t_eps", 5e-2),
    )
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    model = Tuna2PixelQwen2_5(**config.model.showo).to(accelerator.device)
    if config.model.get("gradient_checkpointing", False):
        model._set_gradient_checkpointing(model, True)
        if hasattr(model.tuna, "gradient_checkpointing_enable"):
            model.tuna.gradient_checkpointing_enable()
    _freeze_params(model, config.model.showo.get("frozen_params", None))

    optimizer_config = config.optimizer.params
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    loaders = build_dataloaders(config, text_tokenizer, showo_token_ids, accelerator)
    if config.dataset.samp_probs is None:
        samp_probs = [1.0 / len(loaders)] * len(loaders)
    else:
        samp_probs = config.dataset.samp_probs
    mixed_loader = MixedDataLoader(
        loader_list=loaders,
        samp_probs=samp_probs,
        accumulation=config.dataset.accumulation,
        mode=config.dataset.mixed_loader_mode,
    )

    steps_per_epoch = max(len(loader) for loader in loaders)
    max_train_steps = config.training.max_train_steps
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    loader_batch_sizes = [getattr(loader, "batch_size", 1) or 1 for loader in loaders]
    if "concat" in config.dataset.mixed_loader_mode:
        total_batch_size_per_gpu = sum(loader_batch_sizes)
    else:
        total_batch_size_per_gpu = max(loader_batch_sizes) * config.dataset.accumulation

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    logger.info("***** Running Tuna-2 pixel training *****")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Dataloaders = {len(loaders)}")

    global_step = 0
    first_epoch = 0
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in mixed_loader:
            data_time_m.update(time.time() - end)

            text_tokens = batch["text_tokens"].to(accelerator.device)
            text_labels = batch["text_labels"].to(accelerator.device)
            pixel_values = batch["images"].to(accelerator.device).to(weight_type)
            text_masks = batch["text_masks"].to(accelerator.device)
            image_masks = batch["image_masks"].to(accelerator.device)
            modality_positions = batch["modality_positions"].to(accelerator.device)

            if all(tp == "mmu_text" for tp in batch["data_type"]):
                image_pixels = None
                t = torch.zeros(text_tokens.shape[0], device=accelerator.device, dtype=weight_type)
                image_labels = None
                flow_masks = None
            else:
                image_pixels, t, image_labels, flow_masks = prepare_pixel_flow_targets(
                    pixel_values,
                    batch["data_type"],
                    image_masks,
                    modality_positions,
                    weight_type,
                    jit_noise_scheduler=jit_noise_scheduler,
                    und_max_t0=config.model.showo.get("und_max_t0", config.training.get("und_max_t0", 1.0)),
                    mmu_noise_prob=config.model.showo.get("mmu_noise_prob", 0.0),
                    mmu_noise_level=config.model.showo.get("mmu_noise_level", 0.0),
                )

            block_mask = omni_attn_mask_naive(
                text_tokens.size(0),
                text_tokens.size(1),
                modality_positions,
                accelerator.device,
            ).to(weight_type)

            _, loss_ntp, loss_flow = model(
                text_tokens=text_tokens,
                image_pixels=image_pixels,
                t=t,
                attention_mask=block_mask,
                diffhead_attention_mask=block_mask,
                text_masks=text_masks,
                image_masks=flow_masks,
                text_labels=text_labels,
                image_labels=image_labels,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=text_tokens.size(1),
                device=accelerator.device,
            )

            loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow
            accelerator.backward(loss / config.training.gradient_accumulation_steps)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()
                global_step += 1

                if global_step % config.experiment.log_every == 0:
                    logs = {
                        "step_loss_ntp": accelerator.gather(loss_ntp.detach().repeat(1)).mean().item(),
                        "step_loss_flow": accelerator.gather(loss_flow.detach().repeat(1)).mean().item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step)
                    logger.info(
                        f"Epoch: {epoch} Step: {global_step} "
                        f"Loss_NTP: {logs['step_loss_ntp']:.4f} "
                        f"Loss_FLOW: {logs['step_loss_flow']:.4f} "
                        f"LR: {logs['lr']:.6f}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                if global_step % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, "final")
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(config.experiment.output_dir, safe_serialization=False)
    accelerator.end_training()


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 10**18)
        if len(checkpoints) >= checkpoints_total_limit:
            for checkpoint in checkpoints[:len(checkpoints) - checkpoints_total_limit + 1]:
                shutil.rmtree(os.path.join(output_dir, checkpoint))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main()
