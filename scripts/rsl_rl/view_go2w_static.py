# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Load Go2W (D435i) into Isaac Sim at training default joint angles, fixed-base and static.

Default joint angles come from ``GO2W_CFG_UNITREE.init_state.joint_pos`` in:

    source/robot_lab/robot_lab/assets/unitree.py

Edit that ``joint_pos`` dict to change the pose shown here (and used in training).

Usage:
    python scripts/rsl_rl/view_go2w_static.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View Go2W-D435i at default joint angles (fixed base, static).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from robot_lab.assets.unitree import GO2W_CFG_UNITREE_D435I

# Training / asset defaults live here — edit joint_pos to change the static pose:
#   source/robot_lab/robot_lab/assets/unitree.py  (GO2W_CFG_UNITREE.init_state)
_DEFAULT_JOINT_SOURCE = "source/robot_lab/robot_lab/assets/unitree.py"


def design_scene() -> Articulation:
    """Spawn ground, light, and a fixed-base Go2W-D435i."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = GO2W_CFG_UNITREE_D435I.replace(prim_path="/World/Robot")
    robot_cfg.spawn.fix_base = True
    robot = Articulation(robot_cfg)
    return robot


def run_simulator(sim: sim_utils.SimulationContext, robot: Articulation):
    """Hold joints at default angles until the app is closed."""
    sim_dt = sim.get_physics_dt()

    # Snap once to the training default pose.
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    print(f"[INFO]: Default joint angles from {_DEFAULT_JOINT_SOURCE} (GO2W_CFG_UNITREE.init_state.joint_pos)")
    print(f"[INFO]: Joint names: {robot.data.joint_names}")
    print(f"[INFO]: Default joint pos: {joint_pos[0].tolist()}")
    print("[INFO]: Fixed base; holding default pose. Close the window to exit.")

    while simulation_app.is_running():
        # Re-apply defaults each step so legs/wheels stay static.
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.set_joint_velocity_target(joint_vel)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view(eye=[2.0, 2.0, 1.2], target=[0.0, 0.0, 0.4])

    robot = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
