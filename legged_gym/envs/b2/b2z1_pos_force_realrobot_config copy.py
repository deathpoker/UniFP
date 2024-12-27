from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class B2Z1RealRobotRoughCfg( LeggedRobotCfg ):

    class goal_ee:
        
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.5, 0.2, -0.05]
        collision_lower_limits = [-0.4, -0.2, -0.7]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
        
        class sphere_center:
            x_offset = -0.05 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.7 # Relative to terrain
        
        class ranges:
            init_pos_start = [0.66, np.pi/4, 0]
            # init_pos_end = [0.66, np.pi/6, 0]
            init_pos_end = [0.66, 0, 0]
            pos_l = [0.4, 0.80]
            pos_p = [-1 * np.pi / 3, 1 * np.pi / 3]
            pos_y = [- 1 * np.pi/ 4, 1 * np.pi /4]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.15,   # [rad]
            'FL_thigh_joint': 0.67,     # [rad]
            'FL_calf_joint': -1.3,   # [rad]

            'RL_hip_joint': 0.15,   # [rad]
            'RL_thigh_joint': 0.67,   # [rad]
            'RL_calf_joint': -1.3,    # [rad]

            'FR_hip_joint': -0.15,  # [rad]
            'FR_thigh_joint': 0.67,     # [rad]
            'FR_calf_joint': -1.3,  # [rad]

            'RR_hip_joint': -0.15,   # [rad]
            'RR_thigh_joint': 0.67,   # [rad]
            'RR_calf_joint': -1.3,    # [rad]

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -1.5, # -0.63,
            'z1_wrist_angle': 0, # -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57, # 0.0,
            'z1_jointGripper': -0.785,
        }
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class domain_rand:
        randomize_lag_timesteps = True
        lag_timesteps = 6
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = False
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        
        randomize_rigids_after_start = False # True
        randomize_restitution = False # True
        restitution_range = [0.0, 1.0]

        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        # randomize_arm_friction = True
        # arm_friction_range = [0.0, 0.2]
        # randomize_arm_ema = True
        # arm_ema_range = [0.05, 0.25]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

        gripper_forced_prob = 0.8
        max_push_force_xyz_gripper = [-70, 70]
        push_gripper_interval_s = [3.5, 9.0]
        push_gripper_duration_s = [1.0, 3.0]
        max_push_force_xyz_gripper = [-70, 70]
        max_push_vel_xyz_gripper = [0.0, 30.0]
        max_push_force_xyz_gripper_freed = [-120, 120]

    class env( LeggedRobotCfg.env ):

        num_gripper_joints = 1
        num_actions = 18
        num_torques = 18
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 68 #62
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 15 #66
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = False
        frequencies = 2.0

        action_delay = 3 # Not used, assigned in code
        teleop_mode = False

    class commands:
        settling_time_force_gripper_s = 1.0
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            # lin_vel_x = [-0, 0]
            # lin_vel_y = [-0, 0]
            # ang_vel_yaw = [-0, 0]
            heading = [-3.14, 3.14]
        ang_vel_yaw_clip = 0.2
        ang_vel_pitch_clip = 0.5
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.1] # [0.04, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 300., 'thigh': 300, 'calf': 500, 'z1': 200}  # [N*m/rad]
        damping = {'hip': 8.0, 'thigh': 8.0, 'calf': 15.0, 'z1': 6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        arm_scale_reduction = 2.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08

    class asset( LeggedRobotCfg.asset ):
        file = '/home/peiyang/project/unitree/b2z1/WBC/resources/robots/b2z1/b2z1.urdf'
        name = "b2z1"
        foot_name = "foot"
        thigh_name = "thigh"
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["thigh", "calf", "base_link"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        gait_vel_sigma = 2.0
        gait_force_sigma = 2.0
        kappa_gait_probs = 0.07

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 0.8 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.55
        max_contact_force = 80. # forces above this value are penalized

        class scales: # ( ManipLocoCfg.rewards.scales ):

            tracking_contacts_shaped_force = -2.0 # Only works when `observing_gait_commands` is true # 步态
            tracking_contacts_shaped_vel = -2.0 # Only works when `observing_gait_commands` is true   # 步态
            # tracking_contacts_shaped_pos = -0.1
            tracking_lin_vel = 2.0 # 1.5  # track x轴方向速度
            # tracking_lin_vel_x_l1 = 0.
            # tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 1.0 # just for yaw # track 旋转速度
            # delta_torques = -1.0e-7/4.0 # 惩罚力量大小变化
            # work = 0
            # energy = -1e-6
            # energy_square = -1e-5
            torques = -1.e-5 # -1e-5 # 惩罚力量大小
            stand_still = 0.0 #1.5 #走路指令是0的时候，dof pose尽可能和default pos一样
            walking_dof = 0.0 # 和上面一样
            # dof_default_pos = 0.0
            # dof_error = 0.0 # -0.06 # -0.04
            alive = 1.5
            lin_vel_z = -1.5 #b2沿着z轴的速度越小越好
            # roll = -2.0 #惩罚b2侧身旋转
            # pitch = -2.0
            
            # # tracking_ang_pitch_vel = 0.5 # New reward, only useful when pitch_control = True

            # # common rewards
            feet_air_time = 1.0 # 奖励脚腾空时间
            feet_height = 1.5 # 奖励脚腾空高度
            # feet_hind_height = 1.0 # 奖励后腿脚腾空高度
            ang_vel_xy = -0.2 # -0.1 # 惩罚过快的转弯速度
            dof_acc = -2.5e-7 #-2.5e-7 # -0.1 # 惩罚过快的joint 加速度
            dof_acc_arm = -4.0e-7 #-2.5e-7 # -0.1 # 惩罚过快的arm joint 加速度
            collision = -5. # 惩罚大腿小腿躯干触地
            action_rate = -0.015 # 惩罚action 变化速度
            action_rate_arm = -0.025 # 惩罚arm action 变化速度
            dof_pos_limits = -10.0 #惩罚 超过限位角度
            hip_pos = -0.5  # 惩罚髋关节与default pos的差别
            # feet_jerk = -0.0002 # 惩罚关节力抽抽
            # feet_drag = -0.08 # 惩罚脚拖地滑行
            feet_contact_forces = -0.001 # 惩罚大于 max_contact_force的关节力量
            # orientation = 0.0
            # orientation_walking = 0.0
            # orientation_standing = 0.0
            base_height = 0.0 #惩罚不是  
            action_smoothness_1_leg = -0.0
            action_smoothness_2_leg = 0.0


            # torques_walking = 0.0
            # torques_standing = 0.0
            # energy_square_walking = 0.0
            # energy_square_standing = 0.0
            # base_height_walking = 0.0
            # base_height_standing = 0.0
            # penalty_lin_vel_y = 0.#-10.
            
            # symmetry
            # feet_pos_xy = -0.5
            # feet_height_symmetry = -0.05
            # feet_height_high = -10

            # arm_scales:
            arm_termination = 0.
            tracking_ee_sphere = 0.
            tracking_ee_world = 1.#0.45
            tracking_ee_sphere_walking = 0.
            tracking_ee_sphere_standing = 0.
            tracking_ee_cart = 0.
            arm_orientation = 0.
            arm_energy_abs_sum = 0.
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = 0.
            action_smoothness_1_arm = -0.0
            action_smoothness_2_arm = -0.0

class B2Z1RealRobotRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'b2z1_realrobot'

  