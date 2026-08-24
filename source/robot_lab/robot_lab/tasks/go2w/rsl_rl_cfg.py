from isaaclab.utils import configclass

from robot_lab.tasks.go2.rsl_rl_cfg import (
    MoECTSD435iRunnerCfg,
    MoECTSRunnerCfg,
    MoECTSSymmetryRunnerCfg,
    MoeCtsSymmetryCfg,
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
class Go2WMoECTSD435iRunnerCfg(MoECTSD435iRunnerCfg):
    experiment_name = "go2w_moe_cts_d435i"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        # Offline MoE CTS L/R data augmentation (batch doubling).
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True
        # Defaults when Go2WD435iEnvCfg.use_mgdp_depth_aux is enabled via train.py sync.
        # Leave coefs at 0 here so the current pipeline is unchanged until the env switch is on.
        self.policy.enable_depth_aux = False
        self.algorithm.depth_denoise_coef = 0.0
        self.algorithm.height_recon_coef = 0.0
        self.algorithm.depth_align_coef = 0.0
        self.algorithm.depth_align_loss_type = "infonce"


@configclass
class Go2WMoECTSSymmetryRunnerCfg(MoECTSSymmetryRunnerCfg):
    experiment_name = "go2w_moe_cts_symmetry"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = Go2WMoeCtsSymmetryCfg()
        self.algorithm.symmetry_cfg.use_symmetric_augmentation = True
