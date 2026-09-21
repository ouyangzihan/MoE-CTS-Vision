from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import (
    MoECTSD435iRunnerCfg,
    MoECTSRunnerCfg,
    MoECTSSymmetryRunnerCfg,
    MoeCtsSymmetryCfg,
    RslRlMoeCtsCnnGruActorCriticCfg,
)


@configclass
class Go2WMoeCtsSymmetryCfg(MoeCtsSymmetryCfg):
    symmetry_class = "robot_lab.tasks.go2w.mdp.symmetry:Go2WMoECTSSymmetry"


@configclass
class Go2WMoECTSRunnerCfg(MoECTSRunnerCfg):
    experiment_name = "go2w_moe_cts"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True


@configclass
class Go2WMoeCtsCnnGruActorCriticCfg(RslRlMoeCtsCnnGruActorCriticCfg):
    """Student MoE with 32 experts and dense softmax gating (no top-k)."""

    expert_num = 32
    gating_top_k = None
    gating_noise_std = 1.0
    # Match Go2WD435iSceneCfg.forward_height_scanner (21×25, 0–2 m ahead).
    height_map_shape = (21, 25)


@configclass
class Go2WMoECTSD435iRunnerCfg(MoECTSD435iRunnerCfg):
    experiment_name = "go2w_moe_cts_d435i"
    policy = Go2WMoeCtsCnnGruActorCriticCfg()

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        # Offline MoE CTS L/R data augmentation (batch doubling).
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True
        self.apply_moe_gating()
        # Defaults when Go2WD435iEnvCfg.use_mgdp_depth_aux is enabled via train.py sync.
        # Leave coefs at 0 here so the current pipeline is unchanged until the env switch is on.
        self.policy.enable_depth_aux = False
        self.algorithm.depth_denoise_coef = 0.0
        self.algorithm.height_recon_coef = 0.0
        self.algorithm.depth_align_coef = 0.0
        self.algorithm.depth_align_loss_type = "infonce"
        # ReDo defaults for D435i MoE-CTS (disabled unless --redo or enabled in cfg).
        grad_steps_per_iter = (
            2 * self.algorithm.num_learning_epochs * self.algorithm.num_mini_batches
        )
        self.algorithm.redo_cfg.reset_end_step = self.max_iterations * grad_steps_per_iter

    def apply_moe_gating(self) -> None:
        """Pin 32-expert dense gating after Hydra ``from_dict`` (same pattern as MGDP obs flags)."""
        self.policy.expert_num = 32
        self.policy.gating_top_k = None
        self.policy.gating_noise_std = 1.0
        self.policy.height_map_shape = (21, 25)
        self.algorithm.router_z_loss_coef = 0.001
        self.algorithm.detach_gate_in_student_surrogate = True


@configclass
class Go2WMoECTSD435iRedoRunnerCfg(Go2WMoECTSD435iRunnerCfg):
    """Go2W D435i MoE-CTS with ReDo enabled."""

    experiment_name = "go2w_moe_cts_d435i_redo"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.redo_cfg.enabled = True


@configclass
class Go2WMoECTSSymmetryRunnerCfg(MoECTSSymmetryRunnerCfg):
    experiment_name = "go2w_moe_cts_symmetry"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True
