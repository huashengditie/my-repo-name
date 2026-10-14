"""Tests for the Standard 1D CNN architecture (src/eeg_seizure/models/cnn1d.py)."""

from __future__ import annotations

import torch

from eeg_seizure.models.cnn1d import (
    Cnn1d,
    build_cnn1d_from_config,
    num_canonical_channels,
)

WINDOW_SAMPLES = 1024

SMALL_CONV_BLOCKS = [
    {
        "out_channels": 4,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "batch_norm": True,
        "pool_size": 2,
    },
    {
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "batch_norm": False,
        "pool_size": None,
    },
]


def test_output_shape_is_one_logit_per_window() -> None:
    model = Cnn1d(in_channels=3, conv_blocks=SMALL_CONV_BLOCKS)
    batch = torch.randn(8, 3, WINDOW_SAMPLES)
    output = model(batch)
    assert tuple(output.shape) == (8, 1)


def test_build_cnn1d_from_config_matches_real_config_channel_count() -> None:
    model = build_cnn1d_from_config()
    n_channels = num_canonical_channels()
    batch = torch.randn(4, n_channels, WINDOW_SAMPLES)
    output = model(batch)
    assert tuple(output.shape) == (4, 1)


def test_backward_pass_produces_finite_nonzero_gradients() -> None:
    torch.manual_seed(2026)
    model = Cnn1d(in_channels=3, conv_blocks=SMALL_CONV_BLOCKS)
    batch = torch.randn(8, 3, WINDOW_SAMPLES)
    output = model(batch)
    output.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"missing gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"
        assert not torch.all(p.grad == 0), f"all-zero gradient for {name}"


def test_eval_mode_is_deterministic_across_repeated_calls() -> None:
    torch.manual_seed(2026)
    model = Cnn1d(in_channels=3, conv_blocks=SMALL_CONV_BLOCKS)
    batch = torch.randn(4, 3, WINDOW_SAMPLES)
    model.eval()
    with torch.no_grad():
        out1 = model(batch)
        out2 = model(batch)
    assert torch.equal(out1, out2)


def test_no_sigmoid_applied_internally() -> None:
    """Logits are raw (can be negative) -- a sigmoid output is always in (0, 1)."""
    torch.manual_seed(2026)
    model = Cnn1d(in_channels=3, conv_blocks=SMALL_CONV_BLOCKS)
    batch = torch.randn(32, 3, WINDOW_SAMPLES)
    output = model(batch)
    assert (output < 0).any(), "expected at least one negative raw logit (no internal sigmoid)"
