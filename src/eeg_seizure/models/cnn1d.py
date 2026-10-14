"""Standard 1D CNN ("Suggested D0", Gate E) architecture (US-001).

Architecture only -- no training code. Input `[batch, C, 1024]` -> three
`Conv1d` blocks (the first two with `BatchNorm1d` + `MaxPool1d`, the last with
neither) -> global average pooling -> `Linear` -> one raw seizure logit per
window (no sigmoid; that belongs to `BCEWithLogitsLoss` in
`src/eeg_seizure/training/losses.py`).

Every architecture hyperparameter (channel counts, kernel sizes, strides, pool
sizes) is loaded from `configs/model_cnn.yaml` at construction time, mirroring
`configs/preprocessing.yaml`'s pattern -- see `build_cnn1d_from_config`. `C`
(in_channels) is derived from `len(canonical_channels)` in
`configs/channels_chbmit.yaml`, not duplicated in `model_cnn.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_cnn.yaml"
CHANNELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "channels_chbmit.yaml"

WINDOW_SECONDS = 4
SAMPLING_RATE_HZ = 256
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLING_RATE_HZ  # 1024


def load_model_config(config_path: str | Path = MODEL_CONFIG_PATH) -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def num_canonical_channels(channels_config_path: str | Path = CHANNELS_CONFIG_PATH) -> int:
    with Path(channels_config_path).open() as f:
        config = yaml.safe_load(f)
    return len(config["canonical_channels"])


class Cnn1d(nn.Module):
    """Standard 1D CNN: `conv_blocks` config -> global average pool -> `Linear(num_classes)`.

    `conv_blocks` is a list of dicts (as loaded from `configs/model_cnn.yaml`),
    each with `out_channels`, `kernel_size`, `stride`, `padding`, `batch_norm`
    (bool), and `pool_size` (int or `None`, meaning no pooling in that block).
    The last block in the list conventionally omits batch norm and pooling per
    the "Suggested D0" spec, but this class does not enforce that -- it simply
    builds whatever sequence of blocks the config describes.
    """

    def __init__(self, in_channels: int, conv_blocks: list[dict], num_classes: int = 1) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        channels = in_channels
        for block_cfg in conv_blocks:
            layers: list[nn.Module] = [
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=block_cfg["out_channels"],
                    kernel_size=block_cfg["kernel_size"],
                    stride=block_cfg.get("stride", 1),
                    padding=block_cfg.get("padding", 0),
                )
            ]
            if block_cfg.get("batch_norm", False):
                layers.append(nn.BatchNorm1d(block_cfg["out_channels"]))
            layers.append(nn.ReLU())
            pool_size = block_cfg.get("pool_size")
            if pool_size:
                layers.append(nn.MaxPool1d(pool_size))
            blocks.append(nn.Sequential(*layers))
            channels = block_cfg["out_channels"]

        self.conv_blocks = nn.ModuleList(blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)  # [batch, channels]
        return self.classifier(x)  # [batch, num_classes]


def build_cnn1d_from_config(
    model_config_path: str | Path = MODEL_CONFIG_PATH,
    channels_config_path: str | Path = CHANNELS_CONFIG_PATH,
) -> Cnn1d:
    config = load_model_config(model_config_path)
    in_channels = num_canonical_channels(channels_config_path)
    return Cnn1d(
        in_channels=in_channels,
        conv_blocks=config["conv_blocks"],
        num_classes=config.get("num_classes", 1),
    )


def _verify_architecture() -> bool:
    """Self-contained VERIFY step (US-001): synthetic batch, param count, backward
    pass, and eval-mode determinism -- see `prompt & markdown/CNN main network
    steps.md` Step 2."""
    torch.manual_seed(2026)
    model = build_cnn1d_from_config()

    batch = torch.randn(8, num_canonical_channels(), WINDOW_SAMPLES)

    model.train()
    output = model(batch)
    shape_ok = tuple(output.shape) == (8, 1)
    status = "OK" if shape_ok else "FAIL"
    print(f"Output shape: {tuple(output.shape)} (expected (8, 1)) -> {status}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params}")

    model.zero_grad()
    output.sum().backward()
    grad_ok = True
    for name, p in model.named_parameters():
        if p.grad is None:
            grad_ok = False
            print(f"  MISSING GRADIENT: {name}")
            continue
        if torch.isnan(p.grad).any():
            grad_ok = False
            print(f"  NaN GRADIENT: {name}")
        if torch.all(p.grad == 0):
            grad_ok = False
            print(f"  ALL-ZERO GRADIENT: {name}")
    print(f"Backward pass gradients (no NaN, no all-zero): {'OK' if grad_ok else 'FAIL'}")

    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    deterministic = bool(torch.equal(out1, out2))
    print(f"eval() determinism (bit-identical repeat run): {'OK' if deterministic else 'FAIL'}")

    all_pass = shape_ok and grad_ok and deterministic
    print(f"\nArchitecture verification: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> int:
    return 0 if _verify_architecture() else 1


if __name__ == "__main__":
    raise SystemExit(main())
