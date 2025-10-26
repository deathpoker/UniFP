# UniFP - Unified Force and Position Control


<div align="center">
Conference on Robot Learning (CoRL) 2025 Best paper

[[Website]](https://unified-force.github.io/)
[[Arxiv]](https://arxiv.org/pdf/2505.20829)
[[Oral Talk]](https://youtu.be/9lzFVQoc4Do?t=2652)

<p align="center">
    <img src="docs/teaser.jpg" height=400px"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
</p>

[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3/whatsnew/3.8.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

</div>

## Overview

This project implements a reinforcement learning-based whole body control framework for B2Z1 robots, supporting unified policy learning for both position and force control. The framework uses Isaac Gym for simulation training and supports deployment from simulation to real robots.

**Key Features**:
- Support for B2Z1 robot whole body control
- Unified policy learning for position and force control
- Reinforcement learning training based on PPO algorithm
- Support for multiple robot configurations (B2Z1, G1, etc.)
- Complete simulation-to-real deployment pipeline

## TODO
- [x] Release UniFP training pipeline
- [ ] Release sim2real with ROS2
- [ ] Release sim2sim in MuJoCo
- [ ] Release imitation learing data collection pipeline

## Installation

### System Requirements
- Ubuntu 20.04/22.04
- Python 3.8
- CUDA 11.2+
- Isaac Gym Preview 4 (requires NVIDIA developer account)

### Installation Steps

1. **Set up the environment**
   ```bash
   conda create -n b1z1 python=3.8 
   # isaacgym requires python <=3.8
   conda activate b1z1
   # Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
   cd isaacgym/python && pip install -e .
   ```

2. **Clone this project**
   ```bash
   git clone https://github.com/deathpoker/UniFP.git
   cd UniFP
   ```

3. **Install Python dependencies**
   ```bash
   
   # Install PyTorch (select based on CUDA version)
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
   
   # Install other dependencies
   pip install numpy matplotlib wandb
   ```


## Usage

### Policy Training

#### B2Z1 Position-Force Control Training
```bash
cd legged_gym/scripts
python train_b2z1posforce.py --task=b2z1_pos_force_ee_realrobot --headless
```

### Policy Evaluation and Testing

#### Run Trained Policies
```bash
# B2Z1 position-force control testing
python play_b2z1posforce.py --task=b2z1_pos_force_ee_realrobot --load_run=<run_name>

# B2Z1 force control testing
python play_b2z1force.py --task=b2z1_force_realrobot --load_run=<run_name>
```

#### Visualize Prediction Results
```bash
# Enable visualization prediction
python play_b2z1posforce.py --task=b2z1_pos_force_ee_realrobot --load_run=<run_name>
# Set VISUAL_PRED = True in the script
```

### Parameter Configuration

#### Training Parameters
- `--task`: Task name (b2z1_pos_force_ee_realrobot, b2z1_force_realrobot, h1, g1_humanoidgym, etc.)
- `--headless`: Run in headless mode
- `--num_envs`: Number of parallel environments
- `--max_iterations`: Maximum training iterations

#### Environment Parameters
- `--flat_terrain`: Use flat terrain
- `--physics_engine`: Physics engine (physx)
- `--sim_device`: Simulation device (cuda:0)


### Core Components

- **Environment Configuration** (`legged_gym/envs/b2/b2z1_pos_force_ee_realrobot_config.py`)
  - Robot initial state configuration
  - Reward function parameters
  - Observation space definition
  - Action space definition

- **Environment Implementation** (`legged_gym/envs/b2/legged_robot_b2z1_pos_force_ee_realrobot.py`)
  - Simulation environment logic
  - Reward calculation
  - Observation space construction
  - Action execution

- **Training Algorithm** (`legged_gym/b2_gym_learn/ppo_cse_pf/`)
  - PPO algorithm implementation
  - Policy network structure
  - Value network structure

- **Task Registration** (`legged_gym/utils/task_registry_b2z1posforce.py`)
  - Task registration management
  - Environment creation
  - Trainer creation

