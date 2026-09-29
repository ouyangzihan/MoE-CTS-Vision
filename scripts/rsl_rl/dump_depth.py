"""Print depth-observation statistics and save a few frames as PNG."""

import argparse
import os
import sys

import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="RobotLab-Go2W-D435i-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", type=str, default="logs/depth_dump")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    os.makedirs(args_cli.out, exist_ok=True)
    zero = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for step in range(args_cli.steps):
        obs, *_ = env.step(zero)
    depth = obs["depth"].reshape(args_cli.num_envs, 60, 60).cpu()
    print("\n===== DEPTH OBS =====")
    print(f"global min {depth.min():.4f} max {depth.max():.4f} mean {depth.mean():.4f}")
    print(f"per-image pixel std (mean over envs) {depth.flatten(1).std(1).mean():.4f}")
    print(f"across-env std per pixel (mean)      {depth.std(0).mean():.4f}")
    print(f"fraction of pixels at max            {(depth >= depth.max() - 1e-4).float().mean():.3f}")
    print("row means (top→bottom) env0:", depth[0].mean(1)[::6].numpy().round(3))
    cam = env.unwrapped.scene.sensors["front_depth_camera"]
    raw = cam.data.output["distance_to_image_plane"]
    print(f"raw sensor output shape {tuple(raw.shape)} min {raw.min():.3f} max {raw.max():.3f}")
    for i in range(min(8, args_cli.num_envs)):
        img = depth[i].numpy()
        img = (255 * (img - img.min()) / max(img.max() - img.min(), 1e-6)).astype(np.uint8)
        Image.fromarray(img).resize((240, 240), Image.NEAREST).save(f"{args_cli.out}/env{i}.png")
    print(f"saved PNGs to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
