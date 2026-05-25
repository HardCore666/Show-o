# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import collections
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import image_transform


IGNORE_INDEX = -100


def _first_present(record: Dict[str, Any], keys: List[str], default=None):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


class Tuna2JsonlDataset(Dataset):
    """Flexible jsonl dataset for Tuna-2 style text, image, and edit examples."""

    def __init__(
            self,
            anno_path: str,
            text_tokenizer: Any,
            showo_token_ids: Dict[str, int],
            task: str,
            image_root: str = "",
            image_size: int = 432,
            max_seq_len: int = 1024,
            num_image_tokens: int = 729,
            max_num_images: int = 2,
    ):
        super().__init__()
        self.records = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids["bos_id"]
        self.eos_id = showo_token_ids["eos_id"]
        self.boi_id = showo_token_ids["boi_id"]
        self.eoi_id = showo_token_ids["eoi_id"]
        self.img_pad_id = showo_token_ids["img_pad_id"]
        self.task = task
        self.image_root = image_root
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.num_image_tokens = num_image_tokens
        self.max_num_images = max_num_images
        self.max_text_len = max_seq_len - num_image_tokens - 4

    def __len__(self):
        return len(self.records)

    def _tokenize(self, text, max_length=None):
        return self.text_tokenizer(
            text or "",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length or self.max_text_len,
        ).input_ids

    def _load_image(self, path: Optional[str]):
        if not path:
            return torch.zeros(3, self.image_size, self.image_size)
        full_path = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        image = Image.open(full_path).convert("RGB")
        return image_transform(image, resolution=self.image_size)

    def _pad_sequence(self, text_tokens, text_labels, modality_positions, image_mask):
        text_tokens = text_tokens[:self.max_seq_len]
        text_labels = text_labels[:self.max_seq_len]
        image_mask = image_mask[:self.max_seq_len]

        pad_len = self.max_seq_len - len(text_tokens)
        text_tokens = torch.tensor(text_tokens + [self.pad_id] * pad_len)
        text_labels = torch.tensor(text_labels + [IGNORE_INDEX] * pad_len)
        image_mask = torch.tensor(image_mask + [0] * pad_len)

        if len(modality_positions) < self.max_num_images:
            modality_positions = modality_positions + [(0, 0)] * (self.max_num_images - len(modality_positions))
        modality_positions = torch.tensor(modality_positions[:self.max_num_images])

        text_mask = torch.where(
            (text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
            torch.ones_like(text_tokens),
            torch.zeros_like(text_tokens),
        )
        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def _format_text_only(self, record):
        text = _first_present(record, ["text", "prompt", "content", "caption"], "")
        tokens = self._tokenize(text, max_length=self.max_seq_len - 2)
        text_tokens = [self.bos_id] + tokens + [self.eos_id]
        text_labels = [IGNORE_INDEX] + tokens + [self.eos_id]
        image_mask = [0] * len(text_tokens)
        images = torch.zeros(self.max_num_images, 3, self.image_size, self.image_size)
        return self._finalize(text_tokens, text_labels, [], image_mask, images, text, "mmu_text")

    def _format_t2i(self, record):
        prompt = _first_present(record, ["prompt", "caption", "text", "instruction"], "")
        image_path = _first_present(record, ["path", "image", "image_path", "target", "target_image"])
        tokens = self._tokenize(prompt)
        offset = 1 + len(tokens) + 1
        text_tokens = [self.bos_id] + tokens + [self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [
            self.eoi_id, self.eos_id
        ]
        text_labels = [IGNORE_INDEX] * len(text_tokens)
        image_mask = [0] * (offset) + [1] * self.num_image_tokens + [0, 0]
        images = [self._load_image(image_path)]
        return self._finalize(text_tokens, text_labels, [(offset, self.num_image_tokens)], image_mask, images, prompt, "t2i")

    def _format_mmu(self, record):
        image_path = _first_present(record, ["path", "image", "image_path"])
        question = _first_present(record, ["question", "prompt", "instruction"], "")
        answer = _first_present(record, ["answer", "response", "target", "caption"], "")
        if "conversations" in record:
            humans = [c.get("value", "") for c in record["conversations"] if c.get("from") == "human"]
            assistants = [c.get("value", "") for c in record["conversations"] if c.get("from") != "human"]
            question = humans[0].replace("<image>", "").strip() if humans else question
            answer = assistants[0].strip() if assistants else answer

        q_tokens = self._tokenize(question, max_length=self.max_text_len // 2)
        a_tokens = self._tokenize(answer + self.text_tokenizer.eos_token, max_length=self.max_text_len // 2)
        offset = 2
        text_tokens = [self.bos_id, self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [
            self.eoi_id
        ] + q_tokens + a_tokens
        text_labels = [IGNORE_INDEX] * (2 + self.num_image_tokens + 1 + len(q_tokens)) + a_tokens
        image_mask = [0] * len(text_tokens)
        images = [self._load_image(image_path)]
        return self._finalize(text_tokens, text_labels, [(offset, self.num_image_tokens)], image_mask, images, question, "mmu")

    def _format_edit(self, record):
        instruction = _first_present(record, ["instruction", "prompt", "text"], "")
        source_path = _first_present(record, ["source_image", "source", "input_image", "image"])
        target_path = _first_present(record, ["target_image", "target", "edited_image", "output_image"])
        tokens = self._tokenize(instruction, max_length=self.max_text_len)

        source_offset = 2
        target_offset = source_offset + self.num_image_tokens + 1 + len(tokens) + 1
        text_tokens = (
            [self.bos_id, self.boi_id]
            + [self.img_pad_id] * self.num_image_tokens
            + [self.eoi_id]
            + tokens
            + [self.boi_id]
            + [self.img_pad_id] * self.num_image_tokens
            + [self.eoi_id, self.eos_id]
        )
        text_labels = [IGNORE_INDEX] * len(text_tokens)
        image_mask = (
            [0] * target_offset
            + [1] * self.num_image_tokens
            + [0, 0]
        )
        images = [self._load_image(source_path), self._load_image(target_path)]
        positions = [(source_offset, self.num_image_tokens), (target_offset, self.num_image_tokens)]
        return self._finalize(text_tokens, text_labels, positions, image_mask, images, instruction, "edit")

    def _finalize(self, text_tokens, text_labels, modality_positions, image_mask, images, text, data_type):
        text_tokens, text_labels, modality_positions, text_mask, image_mask = self._pad_sequence(
            text_tokens,
            text_labels,
            modality_positions,
            image_mask,
        )

        if isinstance(images, torch.Tensor):
            image_tensor = images
        else:
            while len(images) < self.max_num_images:
                images.append(torch.zeros(3, self.image_size, self.image_size))
            image_tensor = torch.stack(images[:self.max_num_images], dim=0)

        return {
            "text_tokens": text_tokens,
            "text_labels": text_labels,
            "images": image_tensor,
            "modality_positions": modality_positions,
            "text_masks": text_mask,
            "image_masks": image_mask,
            "texts": text,
            "data_type": data_type,
        }

    def __getitem__(self, idx):
        record = self.records[idx]
        try:
            if self.task == "text_only":
                return self._format_text_only(record)
            if self.task == "mmu":
                return self._format_mmu(record)
            if self.task == "edit":
                return self._format_edit(record)
            return self._format_t2i(record)
        except Exception as exc:
            print(f"Skipping bad Tuna-2 {self.task} sample {idx}: {exc}")
            return self.__getitem__((idx + 1) % len(self.records))

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key in ("texts", "data_type"):
                continue
            batched[key] = torch.stack(value, dim=0)
        return batched


def create_tuna2_jsonl_dataloader(
        anno_path,
        batch_size,
        text_tokenizer,
        showo_token_ids,
        task,
        accelerator=None,
        image_root="",
        image_size=432,
        max_seq_len=1024,
        num_image_tokens=729,
        max_num_images=2,
        num_workers=0,
        shuffle=True,
        drop_last=True,
):
    dataset = Tuna2JsonlDataset(
        anno_path=anno_path,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        task=task,
        image_root=image_root,
        image_size=image_size,
        max_seq_len=max_seq_len,
        num_image_tokens=num_image_tokens,
        max_num_images=max_num_images,
    )

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
