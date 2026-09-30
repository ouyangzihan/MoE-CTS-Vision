"""Measure whether a Go2W D435i CTS checkpoint actually uses depth / height-scan input.

Envs are split into four groups (env_id % 4):
  0 student_real      student, true depth
  1 student_shuffled  student, depth stream taken from another env (fixed mapping)
  2 teacher_real      teacher latent from privileged critic obs
  3 teacher_blind     teacher latent with height_scan + forward_height_scan from another env

Per step it also computes counterfactual actions (same hidden state, swapped perception)
for the student_real and teacher_real groups, so sensitivity is measured on-policy.

Example:
    python scripts/rsl_rl/diagnose_depth.py --task RobotLab-Go2W-D435i-v0 --headless \
        --num_envs 512 --steps 3000 --checkpoint logs/rsl_rl/.../model_34000.pt
"""

import argparse
import sys

import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from utils import apply_moe_gating_cfg, pin_front_depth_camera_for_play, sync_depth_aux_heads_from_checkpoint

parser = argparse.ArgumentParser(description="Depth-usage diagnostic for CTS CNN-GRU policies.")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", type=str, default="RobotLab-Go2W-D435i-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunnerCTS

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

GROUP_NAMES = ("student_real", "student_shuffled", "teacher_real", "teacher_blind")


def _column_terrain_names(env_cfg) -> list[str]:
    gen = env_cfg.scene.terrain.terrain_generator
    names = list(gen.sub_terrains.keys())
    props = np.array([gen.sub_terrains[n].proportion for n in names], dtype=float)
    cum = np.cumsum(props / props.sum())
    return [names[int(np.min(np.where(col / gen.num_cols + 0.001 < cum)[0]))] for col in range(gen.num_cols)]


def _critic_slices(manager, names: tuple[str, ...]) -> list[slice]:
    term_names = manager.active_terms["critic"]
    term_dims = manager.group_obs_term_dim["critic"]
    slices, offset = [], 0
    for name, dim in zip(term_names, term_dims):
        width = int(np.prod(dim))
        if name in names:
            slices.append(slice(offset, offset + width))
        offset += width
    return slices


def _donor_map(idxs: torch.Tensor) -> torch.Tensor:
    """Map each env in a group to a far-away env of the same group (likely another terrain column)."""
    return torch.roll(idxs, shifts=max(len(idxs) // 2, 1))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    from robot_lab.tasks.go2w.env_cfg import configure_command_delivery

    configure_command_delivery(env_cfg, use_pose_velocity=False)
    env_cfg.apply_velocity_command_mixture()
    env_cfg.apply_mgdp_depth_aux_settings()
    if hasattr(env_cfg, "apply_state_estimator_settings"):
        env_cfg.apply_state_estimator_settings()
    agent_cfg.policy.enable_depth_aux = bool(env_cfg.use_mgdp_depth_aux)
    if hasattr(agent_cfg.policy, "enable_state_estimator"):
        agent_cfg.policy.enable_state_estimator = bool(getattr(env_cfg, "use_state_estimator", False))
    apply_moe_gating_cfg(agent_cfg)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    pin_front_depth_camera_for_play(env_cfg, args_cli.task)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.step_height_range = None

    column_names = _column_terrain_names(env_cfg)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    sync_depth_aux_heads_from_checkpoint(agent_cfg, resume_path)
    runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    pol = runner.alg.policy
    pol.eval()

    n = env.num_envs
    env_ids = torch.arange(n, device=device)
    group = env_ids % 4
    idx = [env_ids[group == g] for g in range(4)]
    donor = [_donor_map(i) for i in idx]
    scan_slices = _critic_slices(env.unwrapped.observation_manager, ("height_scan", "forward_height_scan"))
    depth_key = pol.actor_image_obs_groups[0]
    terrain = env.unwrapped.scene.terrain

    def swap_depth(obs, dst, src):
        out = obs.clone()
        out[depth_key][dst] = obs[depth_key][src]
        return out

    def swap_scans(obs, dst, src):
        out = obs.clone()
        for s in scan_slices:
            out["critic"][dst, s] = obs["critic"][src, s]
        return out

    def teacher_action(obs):
        single = pol.single_obs_normalizer(obs["single_obs"])
        return pol.actor(torch.cat([pol.teacher_latent(obs), single], dim=-1))

    def student_action(obs, h, update):
        single = pol.single_obs_normalizer(obs["single_obs"])
        lat, _ = pol.student_latent(obs, hidden_state=h, update_memory=update)
        return pol.actor(torch.cat([lat, single], dim=-1)), lat

    stats = {
        "student_depth_dA": [], "teacher_scan_dA": [], "student_A": [], "teacher_A": [],
        "latent_cos": [], "latent_cos_shuffled": [],
    }
    per_type_dA = {name: [[], []] for name in set(column_names)}
    ep_ret = torch.zeros(n, device=device)
    falls = torch.zeros(4, device=device)
    episodes = torch.zeros(4, device=device)
    returns = torch.zeros(4, device=device)

    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs_s = swap_depth(obs, idx[1], donor[1])
            obs_t = swap_scans(obs, idx[3], donor[3])

            h_prev = pol.student_cnn_gru.hidden_state
            a_cf, lat_cf = student_action(swap_depth(obs_s, idx[0], donor[0]), h_prev, update=False)
            a_stu, lat_stu = student_action(obs_s, h_prev, update=True)
            a_tea = teacher_action(obs)
            a_tea_blind = teacher_action(obs_t)
            a_tea_cf = teacher_action(swap_scans(obs, idx[2], donor[2]))

            actions = a_stu.clone()
            actions[idx[2]] = a_tea[idx[2]]
            actions[idx[3]] = a_tea_blind[idx[3]]

            lat_tea = pol.teacher_latent(obs)
            g0, g2 = idx[0], idx[2]
            d_stu = (a_stu[g0] - a_cf[g0]).abs().mean(-1)
            d_tea = (a_tea[g2] - a_tea_cf[g2]).abs().mean(-1)
            stats["student_depth_dA"].append(d_stu.mean().item())
            stats["teacher_scan_dA"].append(d_tea.mean().item())
            stats["student_A"].append((a_stu[g0] - a_stu[g0].mean(0)).abs().mean().item())
            stats["teacher_A"].append((a_tea[g2] - a_tea[g2].mean(0)).abs().mean().item())
            stats["latent_cos"].append(torch.nn.functional.cosine_similarity(lat_stu[g0], lat_tea[g0], dim=-1).mean().item())
            stats["latent_cos_shuffled"].append(
                torch.nn.functional.cosine_similarity(lat_stu[g0], lat_tea[donor[0]], dim=-1).mean().item()
            )
            if step % 10 == 0:
                cols = terrain.terrain_types
                for k, e in enumerate(g0.tolist()):
                    per_type_dA[column_names[int(cols[e])]][0].append(d_stu[k].item())
                for k, e in enumerate(g2.tolist()):
                    per_type_dA[column_names[int(cols[e])]][1].append(d_tea[k].item())

            obs, rew, dones, extras = env.step(actions)
            pol.reset(dones)
            ep_ret += rew
            done = dones.bool()
            if done.any():
                time_outs = extras.get("time_outs", torch.zeros_like(done)).bool()
                for g in range(4):
                    m = done & (group == g)
                    episodes[g] += m.sum()
                    falls[g] += (m & ~time_outs).sum()
                    returns[g] += ep_ret[m].sum()
                ep_ret[done] = 0.0

            if step % 500 == 0:
                print(f"[diag] step {step}/{args_cli.steps}", flush=True)

    levels = terrain.terrain_levels.float()
    print("\n================ DEPTH USAGE DIAGNOSTIC ================")
    print(f"checkpoint: {resume_path}")
    print(f"mean |a - batch_mean|   student {np.mean(stats['student_A']):.4f}   teacher {np.mean(stats['teacher_A']):.4f}")
    print(f"mean |dA| swap depth (student): {np.mean(stats['student_depth_dA']):.4f}")
    print(f"mean |dA| swap scans (teacher): {np.mean(stats['teacher_scan_dA']):.4f}")
    print(
        f"cos(student, teacher latent) same env {np.mean(stats['latent_cos']):.3f}"
        f"   vs other env {np.mean(stats['latent_cos_shuffled']):.3f}"
    )
    print("\nper terrain type |dA|:   student(depth swap)   teacher(scan swap)")
    for name, (s, t) in sorted(per_type_dA.items()):
        if s or t:
            print(f"  {name:18s} {np.mean(s) if s else float('nan'):.4f}              {np.mean(t) if t else float('nan'):.4f}")
    print("\ngroup               mean_level  episodes  fall_rate  mean_return")
    for g, name in enumerate(GROUP_NAMES):
        ep = max(episodes[g].item(), 1.0)
        print(
            f"  {name:18s} {levels[idx[g]].mean().item():8.3f}  {int(episodes[g].item()):8d}"
            f"  {falls[g].item() / ep:9.3f}  {returns[g].item() / ep:11.2f}"
        )
    print("\nper terrain type mean_level:  " + "  ".join(GROUP_NAMES))
    cols = terrain.terrain_types
    for name in sorted(set(column_names)):
        row = []
        for g in range(4):
            m = torch.tensor([column_names[int(c)] == name for c in cols[idx[g]].tolist()], device=device)
            row.append(levels[idx[g]][m].mean().item() if m.any() else float("nan"))
        print(f"  {name:18s} " + "  ".join(f"{v:14.3f}" for v in row))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
