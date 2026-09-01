# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recycling Dormant Neurons (ReDo) for PyTorch policies.

Port of the reference JAX implementation from:
  "The Dormant Neuron Phenomenon in Deep Reinforcement Learning"
  (Sokar et al., 2023) — see Reference/.../redo/weight_recyclers.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
from tensordict import TensorDict


def leastk_mask(scores: torch.Tensor, ones_fraction: float) -> torch.Tensor:
    """Return a binary mask selecting the lowest-scoring fraction of neurons."""
    if ones_fraction is None or ones_fraction <= 0:
        return torch.zeros_like(scores)
    flat_scores = scores.reshape(-1)
    n_ones = max(1, int(round(flat_scores.numel() * ones_fraction)))
    # Top-k on negated scores selects the smallest original scores.
    _, indices = torch.topk(-flat_scores, k=n_ones, largest=True)
    mask = torch.zeros_like(flat_scores)
    mask[indices] = 1.0
    return mask.reshape(scores.shape)


def estimate_neuron_score(activation: torch.Tensor, sub_mean_score: bool = False) -> torch.Tensor:
    """Score neurons/channels by normalized mean absolute activation."""
    reduce_axes = list(range(activation.ndim - 1))
    if sub_mean_score:
        activation = activation - activation.mean(dim=reduce_axes, keepdim=True)
    score = activation.abs().mean(dim=reduce_axes)
    score = score / (score.mean() + 1e-9)
    return score


def _get_norm_per_neuron(param: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    return torch.sqrt(torch.sum(param.pow(2), dim=axes))


def _fill_uniform(
    tensor: torch.Tensor,
    low: float,
    high: float,
    generator: torch.Generator | None,
) -> None:
    """Fill a tensor uniformly; only use *generator* when it matches *tensor*'s device."""
    if generator is not None and tensor.device.type == generator.device.type:
        if tensor.device.type == "cuda" and tensor.device.index != generator.device.index:
            tensor.uniform_(low, high)
        else:
            tensor.uniform_(low, high, generator=generator)
    else:
        tensor.uniform_(low, high)


def weight_reinit_random(
    param: torch.Tensor,
    mask: torch.Tensor,
    *,
    weight_scaling: bool = False,
    scale: float = 1.0,
    weights_type: str = "incoming",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly reinitialize selected weights (Xavier uniform), optionally scaled."""
    if mask is None:
        return param
    new_param = param.clone()
    if mask.any():
        fan_in = param.shape[1] if param.ndim == 2 else param.shape[1] * param.shape[2] * param.shape[3]
        bound = math.sqrt(6.0 / max(fan_in, 1))
        random_values = torch.empty_like(param)
        _fill_uniform(random_values, -bound, bound, generator)
        if weight_scaling:
            if weights_type == "outgoing":
                axes = tuple(i for i in range(param.ndim) if i != param.ndim - 2)
            else:
                axes = tuple(i for i in range(param.ndim) if i != param.ndim - 1)
            neuron_mask = mask.float().mean(dim=axes)
            non_dead_count = max(int((1.0 - neuron_mask).sum().item()), 1)
            norm_per_neuron = _get_norm_per_neuron(param, axes)
            non_recycled_norm = (norm_per_neuron * (1.0 - neuron_mask)).sum() / non_dead_count
            non_recycled_norm = non_recycled_norm * scale
            normalized = random_values / (_get_norm_per_neuron(random_values, axes).unsqueeze(axes) + 1e-9)
            random_values = normalized * non_recycled_norm
        new_param = torch.where(mask == 1, random_values, param)
    return new_param


def weight_reinit_zero(param: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return param
    return torch.where(mask == 1, torch.zeros_like(param), param)


def reset_adam_state(state: dict[str, torch.Tensor], mask: torch.Tensor | None) -> None:
    """Zero Adam moments for recycled weight entries."""
    if mask is None or not mask.any():
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key in state and state[key] is not None:
            state[key] = state[key] * (1.0 - mask)


def create_mask_helper(
    neuron_mask: torch.Tensor,
    current_param: torch.Tensor,
    next_param: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build incoming/outgoing weight masks from a 1D neuron mask."""
    if current_param.ndim == 2:
        in_axes = (0,)
        out_axes = (1,)
        if current_param.shape[0] > neuron_mask.shape[0]:
            repeat = int(current_param.shape[0] / neuron_mask.shape[0])
            neuron_mask = neuron_mask.repeat_interleave(repeat)
    elif current_param.ndim == 4:
        in_axes = (0, 2, 3)
        out_axes = (1, 2, 3)
    else:
        raise ValueError(f"Unsupported parameter rank for ReDo: {current_param.ndim}")

    def _expand(mask: torch.Tensor, param: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        expanded = mask
        for axis in axes:
            expanded = expanded.unsqueeze(axis)
        for axis in axes:
            expanded = expanded.repeat_interleave(param.shape[axis], dim=axis)
        return expanded

    incoming_mask = _expand(neuron_mask, current_param, in_axes)
    outgoing_mask = _expand(neuron_mask, next_param, out_axes)
    return incoming_mask, outgoing_mask


@dataclass
class RecycleLayerSpec:
    """One hidden layer eligible for neuron recycling."""

    name: str
    linear: nn.Linear | nn.Conv2d
    next_linear: nn.Linear | nn.Conv2d
    activation_module: nn.Module | None = None


def _iter_sequential_modules(module: nn.Module) -> list[tuple[int, nn.Module]]:
    if hasattr(module, "network"):
        children = list(module.network.children())
    elif isinstance(module, nn.Sequential):
        children = list(module.children())
    else:
        children = list(module.children())
    return list(enumerate(children))


def discover_mlp_recycle_layers(module: nn.Module, prefix: str) -> list[RecycleLayerSpec]:
    """Discover hidden Linear layers inside an MLP-like module."""
    modules = [module for _, module in _iter_sequential_modules(module)]
    linear_layers: list[tuple[int, nn.Linear]] = [
        (idx, mod) for idx, mod in enumerate(modules) if isinstance(mod, nn.Linear)
    ]
    specs: list[RecycleLayerSpec] = []
    for idx in range(len(linear_layers) - 1):
        layer_idx, linear = linear_layers[idx]
        next_idx, next_linear = linear_layers[idx + 1]
        activation_module = None
        if layer_idx + 1 < len(modules) and not isinstance(modules[layer_idx + 1], nn.Linear):
            activation_module = modules[layer_idx + 1]
        specs.append(
            RecycleLayerSpec(
                name=f"{prefix}/{linear_layers[idx][1].__class__.__name__}_{idx}",
                linear=linear,
                next_linear=next_linear,
                activation_module=activation_module,
            )
        )
    return specs


def discover_conv_recycle_layers(module: nn.Module, prefix: str) -> list[RecycleLayerSpec]:
    """Discover hidden Conv2d layers that feed another Conv2d layer."""
    modules = [module for _, module in _iter_sequential_modules(module)]
    conv_layers: list[tuple[int, nn.Conv2d]] = [
        (idx, mod) for idx, mod in enumerate(modules) if isinstance(mod, nn.Conv2d)
    ]
    specs: list[RecycleLayerSpec] = []
    for idx in range(len(conv_layers) - 1):
        layer_idx, conv = conv_layers[idx]
        next_idx, next_conv = conv_layers[idx + 1]
        activation_module = None
        if layer_idx + 1 < len(modules) and not isinstance(modules[layer_idx + 1], nn.Conv2d):
            activation_module = modules[layer_idx + 1]
        specs.append(
            RecycleLayerSpec(
                name=f"{prefix}/Conv2d_{idx}",
                linear=conv,
                next_linear=next_conv,
                activation_module=activation_module,
            )
        )
    return specs


def discover_policy_recycle_layers(
    policy: nn.Module,
    module_names: Iterable[str],
) -> list[RecycleLayerSpec]:
    """Collect recyclable hidden layers from a MoE-CTS policy."""
    specs: list[RecycleLayerSpec] = []
    for module_name in module_names:
        if not hasattr(policy, module_name):
            continue
        root = getattr(policy, module_name)
        if module_name == "teacher_encoder" and isinstance(root, nn.Sequential):
            root = root[0]
        if module_name == "student_moe_encoder":
            moe = root.moe
            specs.extend(discover_mlp_recycle_layers(moe.gating_mlp, f"{module_name}/gating"))
            specs.extend(discover_mlp_recycle_layers(moe.experts.backbone, f"{module_name}/experts_backbone"))
            continue
        if module_name == "student_cnn_gru":
            specs.extend(discover_conv_recycle_layers(root.cnn, f"{module_name}/cnn"))
            continue
        specs.extend(discover_mlp_recycle_layers(root, module_name))
    return specs


@dataclass
class RedoConfig:
    enabled: bool = False
    reset_period: int = 200_000
    reset_start_step: int = 0
    reset_end_step: int = 2_500_000
    logging_period: int = 20_000
    recycle_rate: float = 0.3
    score_type: str = "redo"
    dead_neurons_threshold: float = 0.0
    init_method_outgoing: str = "zero"
    weight_scaling: bool = False
    incoming_scale: float = 1.0
    outgoing_scale: float = 1.0
    sub_mean_score: bool = False
    batch_size_statistics: int = 256
    module_names: tuple[str, ...] = (
        "actor",
        "critic",
        "teacher_encoder",
        "student_moe_encoder",
        "student_cnn_gru",
    )
    reset_start_layer_idx: int = 0
    seed: int = 0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any] | RedoConfig | None) -> RedoConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, RedoConfig):
            return cfg
        if hasattr(cfg, "to_dict"):
            cfg = cfg.to_dict()
        if not isinstance(cfg, dict):
            raise TypeError(f"Unsupported redo_cfg type: {type(cfg)}")
        data = {**cfg}
        if "module_names" in data and data["module_names"] is not None:
            data["module_names"] = tuple(data["module_names"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _maybe_reset_optimizer_state(
    optimizer: torch.optim.Optimizer | None,
    param: nn.Parameter,
    mask: torch.Tensor,
) -> None:
    if optimizer is None or param not in optimizer.state:
        return
    reset_adam_state(optimizer.state[param], mask)


@dataclass
class RedoManager:
    """Apply ReDo neuron recycling to a MoE-CTS policy."""

    policy: nn.Module
    optimizers: list[torch.optim.Optimizer]
    cfg: RedoConfig
    device: str
    gradient_step: int = 0
    layer_specs: list[RecycleLayerSpec] = field(default_factory=list)
    _hooks: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list, repr=False)
    _activations: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _generator: torch.Generator | None = field(default=None, repr=False)
    _prev_neuron_score: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.cfg.enabled:
            return
        self.layer_specs = discover_policy_recycle_layers(self.policy, self.cfg.module_names)
        if self.cfg.reset_start_layer_idx > 0:
            self.layer_specs = self.layer_specs[self.cfg.reset_start_layer_idx :]
        self._register_activation_hooks()
        gen_device = torch.device(self.device)
        if gen_device.type == "cuda":
            self._generator = torch.Generator(device=gen_device)
        else:
            self._generator = torch.Generator()
        self._generator.manual_seed(self.cfg.seed)

    def _register_activation_hooks(self) -> None:
        for spec in self.layer_specs:
            target = spec.activation_module if spec.activation_module is not None else spec.linear

            def _make_hook(name: str) -> Callable:
                def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                    activation = output[0] if isinstance(output, tuple) else output
                    self._activations[name] = activation.detach()

                return hook

            self._hooks.append(target.register_forward_hook(_make_hook(spec.name)))

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def increment_gradient_steps(self, count: int = 1) -> None:
        self.gradient_step += count

    def is_reset_step(self, step: int | None = None) -> bool:
        step = self.gradient_step if step is None else step
        within = self.cfg.reset_start_step <= step < self.cfg.reset_end_step
        return step > 0 and step % self.cfg.reset_period == 0 and within

    def is_logging_step(self, step: int | None = None) -> bool:
        step = self.gradient_step if step is None else step
        return step > 0 and step % self.cfg.logging_period == 0

    def maybe_step(
        self,
        obs_statistics: TensorDict | None,
        *,
        learning_iteration: int | None = None,
    ) -> dict[str, float]:
        if not self.cfg.enabled:
            return {}
        step = self.gradient_step
        should_log = self.is_logging_step(step)
        should_reset = self.is_reset_step(step)
        if not should_log and not should_reset:
            return {}
        if obs_statistics is None:
            return {}

        self._activations.clear()
        self._forward_for_statistics(obs_statistics)
        log_dict = self._log_dead_neurons() if should_log else {}
        if should_reset:
            reset_logs = self._recycle_neurons(step)
            log_dict.update(reset_logs)
            if learning_iteration is not None:
                log_dict["redo/learning_iteration"] = float(learning_iteration)
        if log_dict:
            print(
                f"[INFO] ReDo {'recycle+log' if should_reset else 'log'} "
                f"at gradient_step={step} iter={learning_iteration} "
                f"({len(log_dict)} metrics)."
            )
        return log_dict

    @torch.no_grad()
    def _forward_for_statistics(self, obs: TensorDict) -> None:
        training = self.policy.training
        self.policy.eval()
        if hasattr(self.policy, "student_cnn_gru"):
            self.policy.student_cnn_gru.reset(None)
        if hasattr(self.policy, "student_latent"):
            latent, _ = self.policy.student_latent(obs, update_memory=False)
            single_obs = self.policy.single_obs_normalizer(obs["single_obs"])
            obs_c = self.policy.critic_obs_normalizer(self.policy.get_critic_obs(obs))
            _ = self.policy.actor(torch.cat([latent, single_obs], dim=-1))
            _ = self.policy.critic(torch.cat([latent.detach(), obs_c], dim=-1))
            _ = self.policy.teacher_latent(obs)
        else:
            obs_a = self.policy.actor_obs_normalizer(self.policy.get_actor_obs(obs))
            obs_c = self.policy.critic_obs_normalizer(self.policy.get_critic_obs(obs))
            _ = self.policy.teacher_encoder(obs_c)
            student_latent, _ = self.policy.student_moe_encoder(obs_a)
            single_obs = self.policy.single_obs_normalizer(obs["single_obs"])
            _ = self.policy.actor(torch.cat([student_latent, single_obs], dim=-1))
            _ = self.policy.critic(torch.cat([student_latent.detach(), obs_c], dim=-1))
        if training:
            self.policy.train()

    def _scheduled_recycle_fraction(self, update_step: int) -> float:
        multiplier = max(0.0, update_step / max(self.cfg.reset_end_step, 1))
        return math.cos(math.pi * 0.5 * multiplier) * self.cfg.recycle_rate

    def _score_to_mask(
        self,
        activation: torch.Tensor,
        *,
        update_step: int,
    ) -> torch.Tensor:
        score = estimate_neuron_score(activation, sub_mean_score=self.cfg.sub_mean_score)
        if self.cfg.score_type == "random":
            perm = torch.randperm(score.numel(), device=score.device)
            if self._generator is not None and score.device.type == self._generator.device.type:
                perm = torch.randperm(score.numel(), generator=self._generator, device=score.device)
            score = score.reshape(-1)[perm].reshape(score.shape)
        elif self.cfg.score_type == "redo_inverted":
            score = -score
        elif self.cfg.score_type == "redo":
            return leastk_mask(score, self._scheduled_recycle_fraction(update_step))
        elif self.cfg.score_type == "threshold":
            return (score <= self.cfg.dead_neurons_threshold).float()
        else:
            raise ValueError(f"Unknown ReDo score_type: {self.cfg.score_type}")
        return leastk_mask(score, self._scheduled_recycle_fraction(update_step))

    def _recycle_neurons(self, update_step: int) -> dict[str, float]:
        recycled_counts: dict[str, float] = {}
        param_to_optim: dict[nn.Parameter, torch.optim.Optimizer] = {}
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param_to_optim[param] = optimizer

        for spec in self.layer_specs:
            activation = self._activations.get(spec.name)
            if activation is None:
                continue
            neuron_mask = self._score_to_mask(activation, update_step=update_step)
            if neuron_mask.numel() == 0 or neuron_mask.sum() == 0:
                continue

            incoming_mask, outgoing_mask = create_mask_helper(
                neuron_mask.to(spec.linear.weight.device),
                spec.linear.weight,
                spec.next_linear.weight,
            )

            spec.linear.weight.data.copy_(
                weight_reinit_random(
                    spec.linear.weight.data,
                    incoming_mask,
                    weight_scaling=self.cfg.weight_scaling,
                    scale=self.cfg.incoming_scale,
                    weights_type="incoming",
                    generator=self._generator,
                )
            )
            _maybe_reset_optimizer_state(param_to_optim.get(spec.linear.weight), spec.linear.weight, incoming_mask)

            if spec.linear.bias is not None:
                bias_mask = neuron_mask.to(spec.linear.bias.device)
                spec.linear.bias.data.copy_(
                    torch.where(bias_mask == 1, torch.zeros_like(spec.linear.bias), spec.linear.bias)
                )
                _maybe_reset_optimizer_state(param_to_optim.get(spec.linear.bias), spec.linear.bias, bias_mask)

            if self.cfg.init_method_outgoing == "random":
                spec.next_linear.weight.data.copy_(
                    weight_reinit_random(
                        spec.next_linear.weight.data,
                        outgoing_mask,
                        weight_scaling=self.cfg.weight_scaling,
                        scale=self.cfg.outgoing_scale,
                        weights_type="outgoing",
                        generator=self._generator,
                    )
                )
            elif self.cfg.init_method_outgoing == "zero":
                spec.next_linear.weight.data.copy_(
                    weight_reinit_zero(spec.next_linear.weight.data, outgoing_mask)
                )
            else:
                raise ValueError(f"Invalid init_method_outgoing: {self.cfg.init_method_outgoing}")
            _maybe_reset_optimizer_state(
                param_to_optim.get(spec.next_linear.weight),
                spec.next_linear.weight,
                outgoing_mask,
            )

            recycled_counts[f"redo/recycled_count/{spec.name}"] = float(neuron_mask.sum().item())

        total_recycled = sum(recycled_counts.values())
        recycled_counts["redo/recycled_total"] = total_recycled
        return recycled_counts

    def _log_dead_neurons(self) -> dict[str, float]:
        log_dict: dict[str, float] = {}
        total_neurons = 0.0
        total_dead = 0.0
        total_dormant = 0.0
        current_scores: dict[str, torch.Tensor] = {}
        recycle_fraction = self._scheduled_recycle_fraction(self.gradient_step)
        log_dict["redo/scheduled_recycle_fraction"] = recycle_fraction
        for spec in self.layer_specs:
            activation = self._activations.get(spec.name)
            if activation is None:
                continue
            score = estimate_neuron_score(activation, sub_mean_score=self.cfg.sub_mean_score)
            current_scores[spec.name] = score
            dead_mask = score <= self.cfg.dead_neurons_threshold
            dormant_mask = leastk_mask(score, recycle_fraction)
            layer_size = float(score.numel())
            dead_count = float(dead_mask.sum().item())
            dormant_count = float(dormant_mask.sum().item())
            total_neurons += layer_size
            total_dead += dead_count
            total_dormant += dormant_count
            # Strict threshold (paper logging; often 0 with ELU + normalized scores).
            log_dict[f"redo/dead_percentage/{spec.name}"] = (dead_count / layer_size) * 100.0 if layer_size else 0.0
            log_dict[f"redo/dead_count/{spec.name}"] = dead_count
            # Relative dormancy: lowest-scoring fraction that ReDo would recycle now.
            log_dict[f"redo/dormant_percentage/{spec.name}"] = (dormant_count / layer_size) * 100.0 if layer_size else 0.0
            log_dict[f"redo/dormant_count/{spec.name}"] = dormant_count
            log_dict[f"redo/score_min/{spec.name}"] = float(score.min().item())
            log_dict[f"redo/score_median/{spec.name}"] = float(score.median().item())

        if total_neurons > 0:
            log_dict["redo/dead_percentage/total"] = (total_dead / total_neurons) * 100.0
            log_dict["redo/dead_count/total"] = total_dead
            log_dict["redo/dormant_percentage/total"] = (total_dormant / total_neurons) * 100.0
            log_dict["redo/dormant_count/total"] = total_dormant

        if self._prev_neuron_score is not None:
            for name, score in current_scores.items():
                prev_score = self._prev_neuron_score.get(name)
                if prev_score is None:
                    continue
                prev_dead = prev_score <= self.cfg.dead_neurons_threshold
                intersected = prev_dead & (score <= self.cfg.dead_neurons_threshold)
                prev_dead_count = float(prev_dead.sum().item())
                intersected_count = float(intersected.sum().item())
                log_dict[f"redo/dead_intersected_percent/{name}"] = (
                    (intersected_count / prev_dead_count) * 100.0 if prev_dead_count else 0.0
                )
        self._prev_neuron_score = current_scores
        return log_dict


def sample_observations_for_redo(
    observations: TensorDict,
    batch_size: int,
    device: str,
) -> TensorDict:
    """Sample flattened observations from rollout storage."""
    num_steps, num_envs = observations.batch_size[:2]
    flat_size = num_steps * num_envs
    batch_size = min(batch_size, flat_size)
    flat_obs = observations.reshape(flat_size)
    indices = torch.randint(0, flat_size, (batch_size,), device=observations.device)
    return flat_obs[indices].to(device)
