import torch
import torch.nn as nn

from rsl_rl.utils.redo import (
    create_mask_helper,
    discover_mlp_recycle_layers,
    estimate_neuron_score,
    leastk_mask,
    weight_reinit_random,
    weight_reinit_zero,
)


def test_leastk_mask_selects_lowest_scores():
    scores = torch.tensor([0.9, 0.1, 0.5, 0.2])
    mask = leastk_mask(scores, ones_fraction=0.5)
    assert mask.sum().item() == 2.0
    assert mask[1].item() == 1.0
    assert mask[3].item() == 1.0


def test_estimate_neuron_score_normalizes():
    activation = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    score = estimate_neuron_score(activation)
    assert torch.allclose(score.sum(), torch.tensor(2.0), atol=1e-5)


def test_create_mask_helper_for_linear_layers():
    current = nn.Linear(4, 3)
    nxt = nn.Linear(3, 2)
    neuron_mask = torch.tensor([1.0, 0.0, 1.0])
    incoming, outgoing = create_mask_helper(neuron_mask, current.weight, nxt.weight)
    assert incoming.shape == current.weight.shape
    assert outgoing.shape == nxt.weight.shape
    assert incoming[0].unique().tolist() == [1.0]
    assert incoming[1].unique().tolist() == [0.0]
    assert outgoing[:, 0].unique().tolist() == [1.0]
    assert outgoing[:, 2].unique().tolist() == [1.0]


def test_discover_mlp_recycle_layers_skips_output_layer():
    mlp = nn.Sequential(
        nn.Linear(8, 4),
        nn.ELU(),
        nn.Linear(4, 2),
        nn.ELU(),
        nn.Linear(2, 1),
    )
    specs = discover_mlp_recycle_layers(mlp, "test")
    assert len(specs) == 2
    assert specs[0].linear.out_features == 4
    assert specs[1].linear.out_features == 2


def test_weight_reinit_zero_and_random():
    param = torch.ones(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    zeroed = weight_reinit_zero(param, mask)
    assert torch.all(zeroed[0] == 0.0)
    assert torch.all(zeroed[1] == 1.0)

    randomed = weight_reinit_random(param, mask, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(randomed[0], torch.ones(3))
    assert torch.all(randomed[1] == 1.0)
