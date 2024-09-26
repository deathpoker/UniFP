from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.b2.b2_config import B2RoughCfg, B2RoughCfgPPO
from legged_gym.envs.b2.b2z1_config import B2Z1RoughCfg, B2Z1RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .base.legged_robot_b2z1 import LeggedRobotB2Z1
from .base.legged_robot_humanoidgym import LeggedRobotHumanoidGym
from .g1_humanoidgym.g1_humanoidgym_config import G1HumanoidGymCfg, G1HumanoidGymCfgPPO
from .g1_humanoidgym.g1_humanoidgym_env import G1HumanoidGymEnv

from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.task_registry_humanoidgym import task_registry_humanoidgym

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "b2", LeggedRobot, B2RoughCfg(), B2RoughCfgPPO())
task_registry.register( "b2z1", LeggedRobotB2Z1, B2Z1RoughCfg(), B2Z1RoughCfgPPO())
task_registry.register( "h1", LeggedRobot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", LeggedRobot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO())
task_registry_humanoidgym.register( "g1_humanoidgym", G1HumanoidGymEnv, G1HumanoidGymCfg(), G1HumanoidGymCfgPPO())
