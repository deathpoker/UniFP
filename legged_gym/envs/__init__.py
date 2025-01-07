from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.b2.b2_config import B2RoughCfg, B2RoughCfgPPO
from legged_gym.envs.b2.b2_realrobot_config import B2RealRobotRoughCfg, B2RealRobotRoughCfgPPO
from legged_gym.envs.b2.b2z1_config import B2Z1RoughCfg, B2Z1RoughCfgPPO
from legged_gym.envs.b2.b2z1_realrobot_config import B2Z1RealRobotRoughCfg, B2Z1RealRobotRoughCfgPPO
from legged_gym.envs.b2.b2z1_force_realrobot_config import B2Z1ForceRealRobotRoughCfg, B2Z1ForceRealRobotRoughCfgPPO
from legged_gym.envs.b2.b2z1_pos_force_realrobot_config import B2Z1PosForceRealRobotRoughCfg, B2Z1PosForceRealRobotRoughCfgPPO
from legged_gym.envs.b2.b2z1_pos_force_ee_realrobot_config import B2Z1PosForceEERealRobotRoughCfg, B2Z1PosForceEERealRobotRoughCfgPPO
from legged_gym.envs.b2.z1_realrobot_config import Z1RealRobotRoughCfg, Z1RealRobotRoughCfgPPO
from legged_gym.envs.b2.z1_force_realrobot_config import Z1ForceRealRobotRoughCfg, Z1ForceRealRobotRoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .base.legged_robot_realrobot import LeggedRobot_realrobot
from .base.legged_robot_b2z1 import LeggedRobotB2Z1
from .base.legged_robot_b2z1_realrobot import LeggedRobot_b2z1_realrobot
from .base.legged_robot_b2z1_force_realrobot import LeggedRobot_b2z1_force_realrobot
from .base.legged_robot_b2z1_pos_force_realrobot import LeggedRobot_b2z1_pos_force_realrobot
from .base.legged_robot_b2z1_pos_force_ee_realrobot import LeggedRobot_b2z1_pos_force_ee_realrobot
from .base.legged_robot_z1_force_realrobot import LeggedRobot_z1_force_realrobot
from .base.legged_robot_z1_realrobot import LeggedRobot_z1_realrobot
from .base.legged_robot_humanoidgym import LeggedRobotHumanoidGym
from .g1_humanoidgym.g1_humanoidgym_config import G1HumanoidGymCfg, G1HumanoidGymCfgPPO
from .g1_humanoidgym.g1_humanoidgym_with_arm_config import G1HumanoidGymWithArmCfg, G1HumanoidGymWithArmCfgPPO
from .g1_humanoidgym.g1_humanoidgym_upper_fixed_config import G1HumanoidGymUpperFixedCfg, G1HumanoidGymUpperFixedCfgPPO
from .g1_humanoidgym.g1_humanoidgym_env import G1HumanoidGymEnv

from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.task_registry_humanoidgym import task_registry_humanoidgym
from legged_gym.utils.task_registry_b2z1force import task_registry_b2z1force
from legged_gym.utils.task_registry_b2z1posforce import task_registry_b2z1posforce

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "b2", LeggedRobot, B2RoughCfg(), B2RoughCfgPPO())
task_registry.register( "b2_realrobot", LeggedRobot_realrobot, B2RealRobotRoughCfg(), B2RealRobotRoughCfgPPO())
task_registry.register( "b2z1", LeggedRobotB2Z1, B2Z1RoughCfg(), B2Z1RoughCfgPPO())
task_registry.register( "b2z1_realrobot", LeggedRobot_b2z1_realrobot, B2Z1RealRobotRoughCfg(), B2Z1RealRobotRoughCfgPPO())
task_registry_b2z1force.register( "b2z1_force_realrobot", LeggedRobot_b2z1_force_realrobot, B2Z1ForceRealRobotRoughCfg(), B2Z1ForceRealRobotRoughCfgPPO())
task_registry_b2z1posforce.register( "b2z1_pos_force_realrobot", LeggedRobot_b2z1_pos_force_realrobot, B2Z1PosForceRealRobotRoughCfg(), B2Z1PosForceRealRobotRoughCfgPPO())
task_registry_b2z1posforce.register( "b2z1_pos_force_ee_realrobot", LeggedRobot_b2z1_pos_force_ee_realrobot, B2Z1PosForceEERealRobotRoughCfg(), B2Z1PosForceEERealRobotRoughCfgPPO())
task_registry.register( "z1_realrobot", LeggedRobot_z1_realrobot, Z1RealRobotRoughCfg(), Z1RealRobotRoughCfgPPO())
task_registry_b2z1force.register( "z1_force_realrobot", LeggedRobot_z1_force_realrobot, Z1ForceRealRobotRoughCfg(), Z1ForceRealRobotRoughCfgPPO())
task_registry.register( "h1", LeggedRobot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", LeggedRobot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", LeggedRobot, G1RoughCfg(), G1RoughCfgPPO())
task_registry_humanoidgym.register( "g1_humanoidgym", G1HumanoidGymEnv, G1HumanoidGymCfg(), G1HumanoidGymCfgPPO())
task_registry_humanoidgym.register( "g1_humanoidgym_with_arm", G1HumanoidGymEnv, G1HumanoidGymWithArmCfg(), G1HumanoidGymWithArmCfgPPO())
task_registry_humanoidgym.register( "g1_humanoidgym_upper_fixed", G1HumanoidGymEnv, G1HumanoidGymUpperFixedCfg(), G1HumanoidGymUpperFixedCfgPPO())
