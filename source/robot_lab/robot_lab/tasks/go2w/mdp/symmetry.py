"""Symmetry observation and action augmentation for Go2W."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from robot_lab.tasks.go2.mdp.symmetry import Go2MoECTSSymmetry, Go2SymmetryMapper


class Go2WSymmetryMapper(Go2SymmetryMapper):
    """Mirror Go2W observations and actions across the robot sagittal plane."""

    WHEEL_PERM = [1, 0, 3, 2]
    WHEEL_SIGN = [-1, -1, -1, -1]
    LEG_DIM = 12
    WHEEL_DIM = 4
    ACTION_DIM = LEG_DIM + WHEEL_DIM
    VECTOR_SIGNS = {
        "base_lin_vel": [1, -1, 1],
        "base_ang_vel": [-1, 1, -1],
        "projected_gravity": [1, -1, 1],
    }

    def reverse_legs(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right leg joints and flip signed axes."""
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        signs = value.new_tensor(self.JOINT_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_legs(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right leg joints without changing signs."""
        perm = torch.tensor(self.JOINT_PERM, device=value.device)
        return value.index_select(-1, perm)

    def reverse_wheels(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right wheel velocities and flip spin direction."""
        perm = torch.tensor(self.WHEEL_PERM, device=value.device)
        signs = value.new_tensor(self.WHEEL_SIGN)
        return value.index_select(-1, perm) * signs

    def permute_wheels(self, value: torch.Tensor) -> torch.Tensor:
        """Swap left and right wheel joints without changing signs."""
        perm = torch.tensor(self.WHEEL_PERM, device=value.device)
        return value.index_select(-1, perm)

    def reverse_all_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Mirror leg and wheel components in a concatenated joint/action vector."""
        leg = value[..., : self.LEG_DIM]
        wheel = value[..., self.LEG_DIM : self.ACTION_DIM]
        return torch.cat([self.reverse_legs(leg), self.reverse_wheels(wheel)], dim=-1)

    def permute_all_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Permute leg and wheel components without sign flips."""
        leg = value[..., : self.LEG_DIM]
        wheel = value[..., self.LEG_DIM : self.ACTION_DIM]
        return torch.cat([self.permute_legs(leg), self.permute_wheels(wheel)], dim=-1)

    def reverse_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Mirror joint-ordered tensors for Go2W layouts."""
        if value.shape[-1] == self.ACTION_DIM:
            return self.reverse_all_joints(value)
        if value.shape[-1] == self.LEG_DIM:
            return self.reverse_legs(value)
        return super().reverse_joints(value)

    def permute_joints(self, value: torch.Tensor) -> torch.Tensor:
        """Permute joint-ordered tensors for Go2W layouts."""
        if value.shape[-1] == self.ACTION_DIM:
            return self.permute_all_joints(value)
        if value.shape[-1] == self.LEG_DIM:
            return self.permute_legs(value)
        return super().permute_joints(value)

    def reverse_term(self, name: str, cfg, value: torch.Tensor) -> torch.Tensor:
        """Mirror one observation term according to its semantic name."""
        if name == "velocity_commands":
            value = self.restore_history(value, cfg)
            cmd_dim = value.shape[-1]
            if cmd_dim == 2:
                signs = [1, -1]
            elif cmd_dim == 3:
                signs = [1, -1, -1]
            else:
                raise ValueError(f"Unsupported velocity_commands dim {cmd_dim}; expected 2 or 3.")
            mirrored = self.reverse_vector(value, signs)
            return self.flatten_history(mirrored, cfg)
        if name == "joint_pos":
            return self.reverse_legs(value)
        if name in {"joint_vel", "actions"}:
            return self.reverse_all_joints(value)
        if name in {"joint_acc", "joint_torque"}:
            return self.reverse_legs(value)
        return super().reverse_term(name, cfg, value)

    def data_augmentation(self, obs: TensorDict, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor]:
        """Return the original batch followed by its mirrored counterpart."""
        return torch.cat([obs, self.reverse_obs(obs)], dim=0), torch.cat([actions, self.reverse_joints(actions)], dim=0)


class Go2WMoECTSSymmetry(Go2MoECTSSymmetry):
    """Go2W symmetry augmentation helpers for MoECTS segmented mini-batches."""

    def __init__(self, env) -> None:
        self.env = env
        self.mapper = Go2WSymmetryMapper(env)
        self.num_aug = 2
