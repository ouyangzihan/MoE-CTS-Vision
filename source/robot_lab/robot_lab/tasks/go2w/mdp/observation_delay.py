"""Physics-rate proprioceptive observation delay for Go2W.

Unlike actuator command delay (lags motor setpoints) or camera ``max_delay``
(lags depth only), this buffers IMU / joint encoder signals so the policy sees
them late — matching ethernet sensing latency.

Delay is applied every physics step (``sim.dt``), not every policy step, so
``[min_s, max_s] = [0.01, 0.03]`` maps cleanly to 2–6 steps at ``dt=0.005``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class ObservationDelayManager:
    """Ring-buffer delay for root ang-vel, projected gravity, and joint states."""

    def __init__(self, env: ManagerBasedRLEnv, min_delay_s: float, max_delay_s: float):
        self._env = env
        physics_dt = float(env.physics_dt)
        if physics_dt <= 0.0:
            raise ValueError(f"Invalid physics_dt={physics_dt}")
        if min_delay_s < 0.0 or max_delay_s < min_delay_s:
            raise ValueError(f"Invalid delay range [{min_delay_s}, {max_delay_s}]")

        self.min_steps = int(round(min_delay_s / physics_dt))
        self.max_steps = int(round(max_delay_s / physics_dt))
        if self.min_steps < 0 or self.max_steps < self.min_steps:
            raise ValueError(
                f"Delay steps invalid: min={self.min_steps}, max={self.max_steps} "
                f"(from [{min_delay_s}, {max_delay_s}] s @ dt={physics_dt})"
            )
        # DelayBuffer history_length is max lag; need at least 1 slot when max_steps=0.
        history = max(self.max_steps, 0)

        robot = env.scene["robot"]
        num_envs = env.num_envs
        device = env.device
        num_joints = robot.num_joints

        self._ang_vel_buf = DelayBuffer(history, num_envs, device)
        self._gravity_buf = DelayBuffer(history, num_envs, device)
        self._joint_pos_rel_buf = DelayBuffer(history, num_envs, device)
        self._joint_vel_buf = DelayBuffer(history, num_envs, device)

        self.ang_vel = torch.zeros(num_envs, 3, device=device)
        self.projected_gravity = torch.zeros(num_envs, 3, device=device)
        self.joint_pos_rel = torch.zeros(num_envs, num_joints, device=device)
        self.joint_vel = torch.zeros(num_envs, num_joints, device=device)

        self.reset(None)
        # Prime buffers so the first policy step is not all zeros.
        for _ in range(self.max_steps + 1):
            self.advance()

        print(
            f"[ObservationDelayManager] proprio delay "
            f"{min_delay_s*1e3:.0f}-{max_delay_s*1e3:.0f} ms "
            f"({self.min_steps}-{self.max_steps} physics steps @ dt={physics_dt*1e3:.1f} ms)"
        )

    def reset(self, env_ids: Sequence[int] | slice | None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids_idx: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
            n = self._env.num_envs
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_idx = env_ids.to(device=self._env.device, dtype=torch.long)
                n = int(env_ids_idx.numel())
            else:
                env_ids_idx = torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)
                n = int(env_ids_idx.numel())

        # DelayBuffer._time_lags is torch.int; torch.long fails on index put.
        lags = torch.randint(
            self.min_steps,
            self.max_steps + 1,
            (n,),
            device=self._env.device,
            dtype=torch.int,
        )
        for buf in (
            self._ang_vel_buf,
            self._gravity_buf,
            self._joint_pos_rel_buf,
            self._joint_vel_buf,
        ):
            buf.set_time_lag(lags, env_ids_idx)
            buf.reset(env_ids_idx)

    def advance(self):
        """Push current proprio and refresh delayed outputs (call every physics step)."""
        robot = self._env.scene["robot"]
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel

        self.ang_vel = self._ang_vel_buf.compute(robot.data.root_ang_vel_b)
        self.projected_gravity = self._gravity_buf.compute(robot.data.projected_gravity_b)
        self.joint_pos_rel = self._joint_pos_rel_buf.compute(joint_pos_rel)
        self.joint_vel = self._joint_vel_buf.compute(joint_vel)


def _delay_manager(env: ManagerBasedEnv) -> ObservationDelayManager | None:
    return getattr(env, "obs_delay_manager", None)


def base_ang_vel_delayed(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity, optionally delayed."""
    mgr = _delay_manager(env)
    if mgr is None:
        from isaaclab.envs.mdp import base_ang_vel

        return base_ang_vel(env, asset_cfg=asset_cfg)
    return mgr.ang_vel


def projected_gravity_delayed(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projected gravity, optionally delayed."""
    mgr = _delay_manager(env)
    if mgr is None:
        from isaaclab.envs.mdp import projected_gravity

        return projected_gravity(env, asset_cfg=asset_cfg)
    return mgr.projected_gravity


def joint_pos_rel_yaw_sym_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative joint positions (legs), delayed then yaw-mirrored if enabled."""
    from robot_lab.tasks.go2w.mdp.yaw_joint_symmetry import maybe_mirror_joints

    mgr = _delay_manager(env)
    if mgr is None:
        from isaaclab.envs.mdp import joint_pos_rel

        value = joint_pos_rel(env, asset_cfg=asset_cfg)
    else:
        value = mgr.joint_pos_rel[:, asset_cfg.joint_ids]
    return maybe_mirror_joints(env, value)


def joint_vel_rel_yaw_sym_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities, delayed then yaw-mirrored if enabled."""
    from robot_lab.tasks.go2w.mdp.yaw_joint_symmetry import maybe_mirror_joints

    mgr = _delay_manager(env)
    if mgr is None:
        from isaaclab.envs.mdp import joint_vel_rel

        value = joint_vel_rel(env, asset_cfg=asset_cfg)
    else:
        value = mgr.joint_vel[:, asset_cfg.joint_ids]
    return maybe_mirror_joints(env, value)
