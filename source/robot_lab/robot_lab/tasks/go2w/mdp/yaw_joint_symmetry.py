"""Canonicalize yaw command magnitude and mirror joint mappings for Go2W.

When ``env.cfg.use_yaw_joint_symmetry`` is True and the signed yaw command is
negative, joint-related network inputs are left-right mirrored and the policy
action is un-mirrored before being applied. The command term itself stays
signed for rewards; only the observation exposes ``|yaw|``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.go2w.mdp.symmetry import Go2WSymmetryMapper

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_LEG_DIM = Go2WSymmetryMapper.LEG_DIM
_ACTION_DIM = Go2WSymmetryMapper.ACTION_DIM


def _enabled(env: ManagerBasedEnv) -> bool:
    return bool(getattr(env.cfg, "use_yaw_joint_symmetry", False))


def yaw_negative_mask(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Boolean mask of envs whose signed yaw command is negative."""
    command = env.command_manager.get_command(command_name)
    return command[:, -1] < 0


def reverse_legs(value: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and flip signed hip axes."""
    perm = torch.tensor(Go2WSymmetryMapper.JOINT_PERM, device=value.device)
    signs = value.new_tensor(Go2WSymmetryMapper.JOINT_SIGN)
    return value.index_select(-1, perm) * signs


def reverse_wheels(value: torch.Tensor) -> torch.Tensor:
    """Swap left/right wheels and flip spin direction."""
    perm = torch.tensor(Go2WSymmetryMapper.WHEEL_PERM, device=value.device)
    signs = value.new_tensor(Go2WSymmetryMapper.WHEEL_SIGN)
    return value.index_select(-1, perm) * signs


def reverse_all_joints(value: torch.Tensor) -> torch.Tensor:
    """Mirror concatenated leg (12) + wheel (4) vectors."""
    leg = value[..., :_LEG_DIM]
    wheel = value[..., _LEG_DIM:_ACTION_DIM]
    return torch.cat([reverse_legs(leg), reverse_wheels(wheel)], dim=-1)


def reverse_joint_vector(value: torch.Tensor) -> torch.Tensor:
    """Mirror a joint-ordered vector by trailing dimension."""
    dim = value.shape[-1]
    if dim == _ACTION_DIM:
        return reverse_all_joints(value)
    if dim == _LEG_DIM:
        return reverse_legs(value)
    if dim == Go2WSymmetryMapper.WHEEL_DIM:
        return reverse_wheels(value)
    raise ValueError(f"Unsupported joint vector dim {dim}; expected {_LEG_DIM}, {Go2WSymmetryMapper.WHEEL_DIM}, or {_ACTION_DIM}.")


def maybe_mirror_joints(
    env: ManagerBasedEnv,
    value: torch.Tensor,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Mirror joint vectors for envs with negative yaw when symmetry is enabled."""
    if not _enabled(env):
        return value
    mask = yaw_negative_mask(env, command_name=command_name)
    if not torch.any(mask):
        return value
    mirrored = reverse_joint_vector(value)
    return torch.where(mask.unsqueeze(-1), mirrored, value)


def generated_commands_abs_yaw(env: ManagerBasedEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Velocity command observation with absolute yaw when symmetry is enabled.

    Command layout is ``(vx, vy, yaw)`` for Go2RLGym or ``(vx, yaw)`` for PoseVelocity.
    """
    command = env.command_manager.get_command(command_name).clone()
    if _enabled(env):
        command[:, -1] = command[:, -1].abs()
    return command


def joint_pos_rel_yaw_sym(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative joint positions, mirrored when yaw < 0."""
    from isaaclab.envs.mdp import joint_pos_rel

    return maybe_mirror_joints(env, joint_pos_rel(env, asset_cfg=asset_cfg))


def joint_vel_rel_yaw_sym(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative joint velocities, mirrored when yaw < 0."""
    from isaaclab.envs.mdp import joint_vel_rel

    return maybe_mirror_joints(env, joint_vel_rel(env, asset_cfg=asset_cfg))


def joint_acc_yaw_sym(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint accelerations, mirrored when yaw < 0."""
    from robot_lab.tasks.go2.mdp.observations import joint_acc

    return maybe_mirror_joints(env, joint_acc(env, asset_cfg=asset_cfg))


def joint_effort_yaw_sym(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint efforts/torques, mirrored when yaw < 0."""
    from isaaclab.envs.mdp import joint_effort

    return maybe_mirror_joints(env, joint_effort(env, asset_cfg=asset_cfg))


def apply_yaw_joint_symmetry_to_actions(
    env: ManagerBasedEnv,
    action: torch.Tensor,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Map canonical policy actions to physical joints when yaw < 0.

    Policy actions stay stored in canonical (positive-yaw) space; this returns
    the robot-frame actions to feed action terms.
    """
    if not _enabled(env):
        return action
    mask = yaw_negative_mask(env, command_name=command_name)
    if not torch.any(mask):
        return action
    mirrored = reverse_joint_vector(action)
    return torch.where(mask.unsqueeze(-1), mirrored, action)
