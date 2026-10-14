"""Architecture tests for the D1 compact EEG CNN."""

import torch
from scripts.train_compact_cnn import stratified_subset_indices

from eeg_seizure.models.cnn1d import build_cnn1d_from_config, num_canonical_channels
from eeg_seizure.models.compact_cnn1d import build_compact_cnn1d_from_config


def test_compact_cnn_output_is_one_raw_logit_per_window() -> None:
    model = build_compact_cnn1d_from_config()
    output = model(torch.randn(4, num_canonical_channels(), 1024))
    assert tuple(output.shape) == (4, 1)


def test_compact_cnn_backward_is_finite_and_nonzero() -> None:
    model = build_compact_cnn1d_from_config()
    model(torch.randn(4, num_canonical_channels(), 1024)).sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert not torch.all(parameter.grad == 0)


def test_compact_cnn_has_fewer_trainable_parameters_than_standard_cnn() -> None:
    compact_params = sum(p.numel() for p in build_compact_cnn1d_from_config().parameters())
    standard_params = sum(p.numel() for p in build_cnn1d_from_config().parameters())
    assert compact_params < standard_params


def test_stratified_sample_keeps_twenty_percent_of_each_class() -> None:
    labels = torch.tensor([0] * 50 + [1] * 10).numpy()
    selected = stratified_subset_indices(labels, fraction=0.2, seed=2026)
    sampled = labels[selected]
    assert len(sampled) == 12
    assert int((sampled == 0).sum()) == 10
    assert int((sampled == 1).sum()) == 2
    assert selected.tolist() == stratified_subset_indices(labels, fraction=0.2, seed=2026).tolist()
