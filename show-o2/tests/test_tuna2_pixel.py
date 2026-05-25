import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.modeling_tuna2_qwen2_5 import SimplePixelPatchEmbedding  # noqa: E402
from models.jit_utils import JiTNoiseScheduler  # noqa: E402
from train_tuna2_pixel import prepare_pixel_flow_targets  # noqa: E402


def test_pixel_patch_embedding_shape():
    embedder = SimplePixelPatchEmbedding(in_channels=3, hidden_size=32, patch_size=16)
    pixels = torch.randn(2, 3, 432, 432)
    embeds = embedder(pixels)
    assert embeds.shape == (2, 729, 32)


def test_prepare_pixel_flow_targets_masks_only_generation_samples():
    pixels = torch.randn(2, 1, 3, 32, 32)
    image_masks = torch.zeros(2, 12, dtype=torch.long)
    image_masks[:, 2:6] = 1
    modality_positions = torch.tensor([[[2, 4]], [[2, 4]]])

    noised, t, labels, flow_masks = prepare_pixel_flow_targets(
        pixels,
        ["t2i", "mmu"],
        image_masks,
        modality_positions,
        torch.float32,
        jit_noise_scheduler=JiTNoiseScheduler(noise_scale=2.0),
    )

    assert noised.shape == labels.shape == (2, 3, 1, 32, 32)
    assert t.shape == (2,)
    assert flow_masks[0, 2:6].sum().item() == 4
    assert flow_masks[1].sum().item() == 0
