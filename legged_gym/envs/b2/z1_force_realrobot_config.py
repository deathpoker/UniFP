from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class Z1ForceRealRobotRoughCfg( LeggedRobotCfg ):

    class goal_ee:
        
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.25, 0.2, -0.15]
        collision_lower_limits = [-0.7, -0.2, -0.8]
        underground_limit = -0.7
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
        
        class sphere_center:
            x_offset = 0.2 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.8 # Relative to terrain
        
        class ranges:
            init_pos_start = [0.66, np.pi/4, 0]
            init_pos_end = [0.66, np.pi/6, 0]
            # init_pos_end = [0.66, 0, 0]
            pos_l = [0.4, 0.8]
            pos_p = [0 * np.pi / 6, 1 * np.pi / 2.5]
            pos_y = [-3 * np.pi/ 4, 3 * np.pi /4]

            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        default_joint_angles = {
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
        observe_priv = True
        randomize_friction = False
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = False
        added_mass_range = [0., 20.]
        randomize_base_com = False
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_leg_mass = False
        leg_mass_scale_range = [-0.20, 0.20]
        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]
        arm_motor_strength_range = [0.9, 1.1]
        
        randomize_rigids_after_start = False # True
        randomize_restitution = False # True
        restitution_range = [0.0, 1.0]

        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        # randomize_arm_friction = True
        # arm_friction_range = [0.0, 0.2]
        # randomize_arm_ema = True
        # arm_ema_range = [0.05, 0.25]

        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 1.0

        push_robot_base = False
        push_robot_interval_s = 5.0
        push_robot_duration_s = [1.0, 2.0] 

        # Push gripper 
        push_gripper_stators = False
        push_gripper_interval_s = [3.5, 9.0]
        push_gripper_duration_s = [1.0, 3.0]
        gripper_forced_prob = 1.0
        randomize_gripper_force_gains = True
        gripper_force_kp_range = [300., 300.]
        gripper_force_kd_range = [3.0, 3.0]
        prop_kd = 0.1

        max_push_force_xyz_gripper = [-70, 70]
        max_push_force_xyz_gripper_freed = [-70, 70]

    class env( LeggedRobotCfg.env ):

        num_gripper_joints = 1
        num_actions = 6
        num_torques = 6
        frame_stack = 15
        c_frame_stack = 1
        num_single_obs = 27
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 12
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = False
        frequencies = 1.0

        action_delay = 3 # Not used, assigned in code
        teleop_mode = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 9 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]    # min max [rad/s]
            heading = [-3.14, 3.14]
        ang_vel_yaw_clip = 0.2
        ang_vel_pitch_clip = 0.5
        lin_vel_x_clip = 0.1
        lin_vel_y_clip = 0.1

        # gripper
        settling_time_force_gripper_s = 1.0

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.00] # [0.04, 0.1]
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
        stiffness = {'hip': 300., 'thigh': 300, 'calf': 500, 'z1_waist': 64., 'z1_shoulder': 128., 'z1_elbow': 64., 'z1_wrist_angle': 64., 'z1_forearm_roll': 64., 'z1_wrist_rotate': 64., 'z1_jointGripper': 64., }  # [N*m/rad]
        damping = {'hip': 9.0, 'thigh': 9.0, 'calf': 15.0, 'z1_waist': 1.5, 'z1_shoulder': 3.0, 'z1_elbow': 1.5, 'z1_wrist_angle': 1.5, 'z1_forearm_roll': 1.5, 'z1_wrist_rotate': 1.5, 'z1_jointGripper': 1.5, }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class sim( LeggedRobotCfg.sim):
        dt =  0.005

    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08

    class asset( LeggedRobotCfg.asset ):
        file = '/home/zhipy/project/unitree_rl_gym/resources/robots/b2z1/b2z1_fixedb2_new.urdf'
        name = "z1"
        gripper_name = "ee_gripper_link"
        penalize_contacts_on = ["ee_gripper", "base_link"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        fix_base_link = True
  
    class rewards( LeggedRobotCfg.rewards ):
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 0.8 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.55

        cycle_time = 0.8
        target_joint_pos_scale = 0.17
        target_joint_pos_thd = 0.5
        class scales: # ( ManipLocoCfg.rewards.scales ):
            
            # delta_torques = -1.0e-6 # 惩罚力量大小变化
            # # work = 0
            # # energy = -1e-6
            # energy_square = -5e-8
            # torques = -5.e-6 # -1e-5 # 惩罚力量大小
            # stand_still = 1.0 #1.5 #走路指令是0的时候，dof pose尽可能和default pos一样
            # walking_dof = 1.0 # 和上面一样
            # walking_ref_dof = 2.0
            dof_acc = -5.0e-7 #-2.5e-7 # -0.1 # 惩罚过快的joint 加速度
            dof_vel = -8.e-4
            collision = -5. # 惩罚大腿小腿躯干触地
            # action_smoothness = -0.02
            action_rate = -0.02 # 惩罚action 变化速度
            dof_pos_limits = -10.0 #惩罚 超过限位角度

            # arm_scales:
            arm_termination = 0.
            tracking_ee_sphere = 0.
            # tracking_ee_world = 0.8
            tracking_ee_force_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = 0.
            arm_orientation = 0.
            arm_energy_abs_sum = 0.
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = 0.

class Z1ForceRealRobotRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'z1_force_realrobot'

  
