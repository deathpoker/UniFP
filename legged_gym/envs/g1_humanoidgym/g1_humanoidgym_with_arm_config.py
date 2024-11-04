# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from legged_gym.envs.base.legged_robot_config_humanoidgym import LeggedRobotHumanoidGymCfg, LeggedRobotHumanoidGymCfgPPO


class G1HumanoidGymWithArmCfg(LeggedRobotHumanoidGymCfg):
    """
    Configuration class for the G1HumanoidGym humanoid robot.
    """
    class env(LeggedRobotHumanoidGymCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 98 # 47# 98
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 141 # 73 #141
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 29
        num_envs = 4096
        episode_length_s = 20     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotHumanoidGymCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_with_hand_fixed.urdf'

        name = "g1_with_arm"
        foot_name = "ankle_roll"
        knee_name = "knee"

        terminate_after_contacts_on = ['head']
        penalize_contacts_on = ["hip", "knee", "torso"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotHumanoidGymCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = True
        # rough terrain only:
        measure_heights = False
        static_friction = 1.0
        dynamic_friction = 1.0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 1.0    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotHumanoidGymCfg.init_state):
        pos = [0.0, 0.0, 0.85] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,
           'waist_yaw_joint' : 0,
           'waist_roll_joint' : 0,
           'waist_pitch_joint' : 0,
           'left_shoulder_pitch_joint' : 0,
           'left_shoulder_roll_joint' : 0,
           'left_shoulder_yaw_joint' : 0,
           'left_elbow_joint' : 0,
           'left_wrist_roll_joint' : 0,
           'left_wrist_pitch_joint' : 0,
           'left_wrist_yaw_joint' : 0,
           'right_shoulder_pitch_joint' : 0,
           'right_shoulder_roll_joint' : 0,
           'right_shoulder_yaw_joint' : 0,
           'right_elbow_joint' : 0,
           'right_wrist_roll_joint' : 0,
           'right_wrist_pitch_joint' : 0,
           'right_wrist_yaw_joint' : 0,
        }

    class control(LeggedRobotHumanoidGymCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 300,
                     'ankle': 40,
                     'waist_yaw': 300,
                     'waist_roll': 300,
                     'waist_pitch': 300,
                     'shoulder': 100,
                     'elbow': 100,
                     'wrist': 100,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist_yaw': 4,
                     'waist_roll': 4,
                     'waist_pitch': 4,
                     'shoulder': 2,
                     'elbow': 2,
                     'wrist': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 50hz

    class sim(LeggedRobotHumanoidGymCfg.sim):
        dt = 0.005  # 200 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotHumanoidGymCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02

    class commands(LeggedRobotHumanoidGymCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.728
        min_dist = 0.2
        max_dist = 0.5
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 4
        max_contact_force = 700  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            # joint_pos = 1.6
            # feet_clearance = 1.
            # feet_contact_number = 1.2
            # gait
            feet_air_time = 1.0
            # feet_air_time_humanoidgym = 1.
            # foot_slip = -0.05
            # feet_distance = 0.2
            # knee_distance = 0.2
            # contact
            # feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5
            # base pos
            # default_joint_pos = 0.5
            orientation = -1.
            # orientation_humanoidgym = 1.
            base_height = -10.0
            # base_height_humanoidgym = 0.2
            # base_acc = 0.2
            # energy
            # action_smoothness = -0.002
            torques = -1e-5
            # dof_vel = -5e-4
            dof_acc = -2.5e-8
            collision = -0.
            action_rate = -0.01

            # others
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_pos_limits = -5.0

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class G1HumanoidGymWithArmCfgPPO(LeggedRobotHumanoidGymCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    # class algorithm(LeggedRobotHumanoidGymCfgPPO.algorithm):
    #     entropy_coef = 0.001
    #     learning_rate = 1e-5
    #     num_learning_epochs = 2
    #     gamma = 0.994
    #     lam = 0.9
    #     num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 15001  # number of policy updates

        # logging
        save_interval = 200  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'g1_humanoidgym_with_arm'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
