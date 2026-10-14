"""Compact 1D seizure CNN (D1 / C-LITE).

Implements the design in ``Grand Development Plan.md``: a small Conv1D stem,
depthwise-separable convolution blocks, global average pooling, and a single
raw logit.  It deliberately reuses the standard CNN's channel configuration
so both models receive the identical 23-channel EEG window.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from eeg_seizure.models.cnn1d import CHANNELS_CONFIG_PATH, num_canonical_channels

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_compact_cnn.yaml"


def load_compact_model_config(config_path: str | Path = MODEL_CONFIG_PATH) -> dict:
    """Load the complete, explicit compact-model architecture configuration."""
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise temporal convolution followed by a pointwise channel mixer."""

    def __init__(self, in_channels: int, block_config: dict) -> None:
        super().__init__()
        out_channels = block_config["out_channels"]
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=block_config["kernel_size"],
                stride=block_config.get("stride", 1),
                padding=block_config.get("padding", 0),
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        pool_size = block_config.get("pool_size")
        self.pool = nn.MaxPool1d(pool_size) if pool_size else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.layers(x))


class CompactCnn1d(nn.Module):
    """Small stem -> depthwise-separable blocks -> GAP -> one seizure logit."""

    def __init__(
        self, in_channels: int, stem: dict, blocks: list[dict], num_classes: int = 1
    ) -> None:
        super().__init__()
        stem_channels = stem["out_channels"]
        stem_layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                stem_channels,
                kernel_size=stem["kernel_size"],
                stride=stem.get("stride", 1),
                padding=stem.get("padding", 0),
                bias=False,
            ),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(),
        ]
        if stem.get("pool_size"):
            stem_layers.append(nn.MaxPool1d(stem["pool_size"]))
        self.stem = nn.Sequential(*stem_layers)

        channels = stem_channels
        compact_blocks: list[nn.Module] = []
        for block_config in blocks:
            compact_blocks.append(DepthwiseSeparableBlock(channels, block_config))
            channels = block_config["out_channels"]
        self.blocks = nn.ModuleList(compact_blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.classifier(x)


def build_compact_cnn1d_from_config(
    model_config_path: str | Path = MODEL_CONFIG_PATH,
    channels_config_path: str | Path = CHANNELS_CONFIG_PATH,
) -> CompactCnn1d:
    config = load_compact_model_config(model_config_path)
    return CompactCnn1d(
        in_channels=num_canonical_channels(channels_config_path),
        stem=config["stem"],
        blocks=config["depthwise_separable_blocks"],
        num_classes=config.get("num_classes", 1),
    )
