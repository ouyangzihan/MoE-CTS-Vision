import torch
import torch.nn as nn

from rsl_rl.utils.redo import (
    RedoConfig,
    RedoManager,
    _align_neuron_mask,
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
    weight_reinit_random(current.weight.data, incoming)
    weight_reinit_random(nxt.weight.data, outgoing)


def test_create_mask_helper_for_actor_like_layer():
    current = nn.Linear(85, 512)
    nxt = nn.Linear(512, 256)
    neuron_mask = torch.tensor([1.0, 0.0] + [0.5] * 510)
    incoming, outgoing = create_mask_helper(neuron_mask, current.weight, nxt.weight)
    assert incoming.shape == current.weight.shape
    assert outgoing.shape == nxt.weight.shape
    weight_reinit_random(current.weight.data, incoming)
    weight_reinit_random(nxt.weight.data, outgoing)


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


def test_align_neuron_mask_expand_and_reduce():
    narrow = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expanded = _align_neuron_mask(narrow, 16)
    assert expanded.shape == (16,)
    assert expanded[0].item() == 1.0
    assert expanded[4].item() == 0.0

    wide = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    reduced = _align_neuron_mask(wide, 4)
    assert reduced.tolist() == [1.0, 1.0, 0.0, 1.0]


def test_create_mask_helper_catelu_width():
    current = nn.Linear(8, 4)
    nxt = nn.Linear(8, 2)
    neuron_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    incoming, outgoing = create_mask_helper(neuron_mask, current.weight, nxt.weight)
    assert incoming.shape == current.weight.shape
    assert outgoing.shape == nxt.weight.shape
    bias_mask = _align_neuron_mask(neuron_mask, current.bias.shape[0])
    assert bias_mask.shape == current.bias.shape
    torch.where(bias_mask == 1, torch.zeros_like(current.bias), current.bias)


def _shared_elu_actor(in_dim: int = 10, hidden: tuple[int, ...] = (32, 8)) -> nn.Module:
    """Reproduce the training MLP pattern: one ELU instance reused across layers."""
    act = nn.ELU()
    policy = nn.Module()
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden[0]), act]
    for prev, nxt in zip(hidden, hidden[1:]):
        layers.extend([nn.Linear(prev, nxt), act])
    layers.append(nn.Linear(hidden[-1], 2))
    policy.actor = nn.Sequential(*layers)
    return policy


def test_redo_hooks_capture_per_layer_activations_with_shared_elu():
    policy = _shared_elu_actor()
    manager = RedoManager(
        policy=policy,
        optimizers=[],
        cfg=RedoConfig(enabled=True, module_names=("actor",)),
        device="cpu",
    )
    assert len(manager.layer_specs) == 2
    policy.actor(torch.randn(5, 10))
    first = manager._activations[manager.layer_specs[0].name]
    second = manager._activations[manager.layer_specs[1].name]
    assert first.shape[-1] == 32
    assert second.shape[-1] == 8


def test_recycle_neurons_shared_elu_bias_width():
    """Regression for 128-vs-512 bias crash when a shared ELU overwrote activations."""
    policy = _shared_elu_actor(in_dim=16, hidden=(32, 8))
    optimizer = torch.optim.Adam(policy.parameters())
    manager = RedoManager(
        policy=policy,
        optimizers=[optimizer],
        cfg=RedoConfig(enabled=True, module_names=("actor",), recycle_rate=0.5),
        device="cpu",
    )
    policy.actor(torch.randn(16, 16))
    logs = manager._recycle_neurons(update_step=1)
    assert "redo/recycled_total" in logs
