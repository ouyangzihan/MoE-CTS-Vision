from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn

from robot_lab.tasks.go2.manager.action_manager import ActionManagerGo2W
from robot_lab.tasks.go2w.mdp.observation_delay import ObservationDelayManager


class Go2WEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    def load_managers(self):
        super().load_managers()
        self.action_manager = ActionManagerGo2W(self.cfg.actions, self)
        print("[Go2WEnv-INFO] Overriding action manager with ActionManagerGo2W: ", self.action_manager)
        print(
            f"[Go2WEnv-INFO] use_yaw_joint_symmetry={getattr(self.cfg, 'use_yaw_joint_symmetry', False)}"
        )

        min_s = float(getattr(self.cfg, "obs_delay_min_s", 0.0))
        max_s = float(getattr(self.cfg, "obs_delay_max_s", 0.0))
        if max_s > 0.0:
            self.obs_delay_manager = ObservationDelayManager(self, min_delay_s=min_s, max_delay_s=max_s)
        else:
            self.obs_delay_manager = None
            print("[Go2WEnv-INFO] proprio observation delay disabled")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Same as ManagerBasedRLEnv.step, plus physics-rate proprio delay updates."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            if self.obs_delay_manager is not None:
                self.obs_delay_manager.advance()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if self.obs_delay_manager is not None:
            self.obs_delay_manager.reset(env_ids)
            # Re-seed delayed outputs for reset envs from the post-reset robot state.
            self.obs_delay_manager.advance()
