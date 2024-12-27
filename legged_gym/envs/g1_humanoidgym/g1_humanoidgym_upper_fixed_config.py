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


class G1HumanoidGymUpperFixedCfg(LeggedRobotHumanoidGymCfg):
    """
    Configuration class for the G1HumanoidGym humanoid robot.
    """
    class env(LeggedRobotHumanoidGymCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 88
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 20     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

        teleop_mode = False
        
    class safety:
        # safety factors
        pos_limit = 0.9
        vel_limit = 0.9
        torque_limit = 0.85

    class asset(LeggedRobotHumanoidGymCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_upper_fixed_rev_1_0.urdf'

        name = "g1_upper_fixed"
        foot_name = "ankle_roll"
        knee_name = "knee"
        hip_roll_name = "hip_roll"
        hip_yaw_name = "hip_yaw"
        waist_yaw_name = "waist_yaw"
        torso_name = "torso"

        terminate_after_contacts_on = ['torso', 'pelvis']
        penalize_contacts_on = ["hip", "knee"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = True
        fix_base_link = False

    class terrain(LeggedRobotHumanoidGymCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
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
        pos = [0.0, 0.0, 0.80] # x,y,z [m]

        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.2,         
           'left_knee_joint' : 0.4,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.2,                                       
           'right_knee_joint' : 0.4,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }

    class control(LeggedRobotHumanoidGymCfg.control):
        # PD Drive parameters:
        stiffness = {'hip_yaw': 80,
                     'hip_roll': 80,
                     'hip_pitch': 80,
                     'knee': 160,
                     'ankle': 20,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2.5,
                     'hip_roll': 2.5,
                     'hip_pitch': 2.5,
                     'knee': 5,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20  # 50hz

    class sim(LeggedRobotHumanoidGymCfg.sim):
        dt = 0.001  # 200 Hz
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
        friction_range = [0.6, 2.0]
        randomize_base_mass = True
        added_mass_range = [-6., 6.]
        randomize_base_com = True
        added_com_range = [-0.06, 0.06]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.8
        max_push_ang_vel = 0.6
        # dynamic randomization
        action_delay = 0.5
        action_noise = 0.02
        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]
        arm_motor_strength_range = [0.8, 1.2]

    class commands(LeggedRobotHumanoidGymCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
            heading = [-3.14, 3.14]
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1
        ang_vel_yaw_clip = 0.1
    class rewards:
        base_height_target = 0.78
        min_dist = 0.2
        max_dist = 0.4
        # put some settings here for LLM parameter tuning
        target_feet_height = 0.12        # m
        cycle_time = 0.64               # sec
        target_joint_pos_scale = 0.5    # rad
        target_joint_pos_thd = 0.5
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 4
        max_contact_force = 300  # Forces above this value are penalized

        class scales:

            # reference motion tracking
            # joint_pos = 1.6
            # feet_clearance = 1.
            feet_height = 10
            # feet_height_high = -50
            feet_contact_number = 1.2
            
            # gait
            feet_air_time = 3.0
            # feet_air_time_humanoidgym = 1.
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            hip_roll_pos = -0.5
            hip_yaw_pos = -0.5

            # contact
            feet_contact_forces = -5e-4

            # vel tracking
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            # vel_mismatch_exp = 0.5  # lin_z; ang x,y
            # low_speed = 0.2
            # track_vel_hard = 0.5
            
            # base pos
            # default_joint_pos = 0.5
            # walking_dof_upper_body = 2.0
            # standing_dof_upper_body = 2.0
            standing_dof_leg = 1.0
            # walking_dof_leg = 0.2
            walking_ref_dof_leg = 1.5
            # alive = 0.5
            # ref_dof_leg = 0.2
            orientation = -1.
            # orientation_humanoidgym = 1.
            base_height = -10.0
            # base_height_humanoidgym = 0.2
            # base_acc = 0.2
            
            # energy
            action_smoothness = -0.002

            energy_square_leg = -5e-8
            torques = -2.5e-5
            dof_vel = -1e-4
            dof_acc = -1.e-7
            delta_torques = -2.0e-6
            collision = -10.
            # action_rate = -0.01

            # others
            lin_vel_z = -1.0
            ang_vel_xy = -0.1
            dof_pos_limits = -20.0

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


class G1HumanoidGymUpperFixedCfgPPO(LeggedRobotHumanoidGymCfgPPO):
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
        num_steps_per_env = 24  # per iteration
        max_iterations = 15001  # number of policy updates

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'g1_upper_fixed'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
    # class algorithm(LeggedRobotHumanoidGymCfgPPO.algorithm):
    #     grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]