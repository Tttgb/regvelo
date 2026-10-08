"""Check Hill-saturated GRN Jacobians against the transcription-rate forward pass."""

import numpy as np
import pytest
import torch

from regvelo._module import velocity_encoder


@pytest.mark.parametrize("activate", ["softplus", "sigmoid"])
@pytest.mark.parametrize("base_alpha", [True, False])
def test_hill_jacobians_match_autograd(activate, base_alpha):
    encoder = velocity_encoder(
        activate=activate,
        base_alpha=base_alpha,
        n_int=3,
        hill_K=np.array([0.5, 2.0, 5.0]),
    )
    encoder.register_buffer("alpha_unconstr_max", torch.tensor(10.0))
    with torch.no_grad():
        encoder.fc1.weight.copy_(torch.tensor([[1.0, -0.3, 0.5], [-0.8, 0.7, 0.2], [0.1, 0.4, -0.6]]))
        encoder.fc1.bias.copy_(torch.tensor([0.2, -0.5, 0.8]))

    s = torch.tensor([[0.4, 2.0, 8.0], [2.0, 0.5, 5.0]])
    expected = torch.stack([torch.autograd.functional.jacobian(encoder.transcription_rate, cell) for cell in s])

    torch.testing.assert_close(encoder.GRN_Jacobian2(s), expected)
    torch.testing.assert_close(encoder.GRN_Jacobian(s), expected.mean(dim=0))
    torch.testing.assert_close(encoder.GRN_Jacobian(s[0]), expected[0])


def test_hill_jacobian_respects_softplus_clamp():
    encoder = velocity_encoder(n_int=2, hill_K=np.array([1.0, 2.0]))
    with torch.no_grad():
        encoder.fc1.weight.fill_(0.5)
        encoder.fc1.bias.fill_(60.0)

    s = torch.tensor([[1.0, 2.0]])
    assert torch.count_nonzero(encoder.GRN_Jacobian2(s)) == 0
    assert torch.count_nonzero(encoder.GRN_Jacobian(s)) == 0


@pytest.mark.parametrize("hill_K", [[0.0, 1.0], [1.0], [float("nan"), 1.0]])
def test_hill_k_must_be_positive_and_match_gene_count(hill_K):
    with pytest.raises(ValueError, match="hill_K must contain"):
        velocity_encoder(n_int=2, hill_K=np.asarray(hill_K))
