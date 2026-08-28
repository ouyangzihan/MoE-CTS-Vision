# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing Go2W task implementations."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Go2W-v0",
    entry_point="robot_lab.tasks.go2w.env.go2w_env:Go2WEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:Go2WMoECTSRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Go2W-D435i-v0",
    entry_point="robot_lab.tasks.go2w.env.go2w_env:Go2WEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WD435iEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:Go2WMoECTSD435iRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Go2W-D435i-ReDo-v0",
    entry_point="robot_lab.tasks.go2w.env.go2w_env:Go2WEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WD435iEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:Go2WMoECTSD435iRedoRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Go2W-Symmetry-v1",
    entry_point="robot_lab.tasks.go2w.env.go2w_env:Go2WEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2WEnvSymmetryCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:Go2WMoECTSSymmetryRunnerCfg",
    },
)

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
