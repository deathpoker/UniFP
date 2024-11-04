from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class B2RealRobotRoughCfg( LeggedRobotCfg ):
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
        }
    class env( LeggedRobotCfg.env ):
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 44
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 48
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

        observe_gait_commands = False
        frequencies = 2.0
        teleop_mode = True

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

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
        max_init_terrain_level = 3 # starting curriculum state
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
        stiffness = {'joint': 500.}  # [N*m/rad]
        damping = {'joint': 8.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '/home/zhipy/project/visual_wholebody/low-level/resources/robots/b2z1/b2.urdf'
        name = "b2"
        foot_name = "foot"
        thigh_name = "thigh"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        gait_vel_sigma = 2.0
        gait_force_sigma = 2.0
        kappa_gait_probs = 0.07

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.55
        max_contact_force = 80. # forces above this value are penalized

        class scales: # ( ManipLocoCfg.rewards.scales ):
            tracking_contacts_shaped_force = -2.0 # Only works when `observing_gait_commands` is true # 步态
            tracking_contacts_shaped_vel = -2.0 # Only works when `observing_gait_commands` is true   # 步态
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            
            # energy_square = 0.0
            torques = -1.5e-5 # -1e-5 # 惩罚力量大小
            stand_still = 1.0 #1.5 #走路指令是0的时候，dof pose尽可能和default pos一样
            walking_dof = 1.0

            hip_pos = -0.5
            # dof_default_pos = 0.0
            # dof_error = 0.0 # -0.06 # -0.04
            alive = 1.0
            lin_vel_z = -1.5 #b2沿着z轴的速度越小越好
            roll = -2.0 #惩罚b2侧身旋转
            pitch = -2.0

            # # common rewards
            feet_air_time = 1.0 # 奖励脚腾空时间
            feet_height = 1.5 # 奖励脚腾空高度
            feet_hind_height = 1.0 # 奖励后腿脚腾空高度
            ang_vel_xy = -0.2 # -0.1 # 惩罚过快的转弯速度
            dof_acc = -5.0e-7 #-2.5e-7 # -0.1 # 惩罚过快的joint 加速度
            collision = -5. # 惩罚大腿小腿躯干触地
            action_rate = -0.015 # 惩罚action 变化速度
            dof_pos_limits = -10.0 #惩罚 超过限位角度
            # hip_pos = -0.3  # 惩罚髋关节与default pos的差别
            base_height = -5.0  
            
            feet_contact_forces = -0.001 # 惩罚大于 max_contact_force的关节力量
            # feet_vel_xy = -0.3
            feet_pos_xy = -0.5
            # symmetry
            feet_height_symmetry = -0.05
            feet_height_high = -10

class B2RealRobotRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'b2_realrobot'

  
