# UniFP - Unified Force and Position Control

[![IsaacGym](https://img.shields.io/badge/IsaacGym-1.0.0-silver.svg)](https://developer.nvidia.com/isaac-gym)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://docs.python.org/3/whatsnew/3.8.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-red.svg)](https://pytorch.org/)

## Overview

This project implements a reinforcement learning-based whole body control framework for B2Z1 robots, supporting unified policy learning for both position and force control. The framework uses Isaac Gym for simulation training and supports deployment from simulation to real robots.

**Key Features**:
- Support for B2Z1 robot whole body control
- Unified policy learning for position and force control
- Reinforcement learning training based on PPO algorithm
- Support for multiple robot configurations (B2, H1, G1, etc.)
- Complete simulation-to-real deployment pipeline

## Installation

### System Requirements
- Ubuntu 20.04/22.04
- Python 3.8
- CUDA 11.2+
- Isaac Gym Preview 4 (requires NVIDIA developer account)

### Installation Steps

1. **Install Isaac Gym**
   ```bash
   # Download Isaac Gym Preview 4 from NVIDIA developer website
   # Extract to specified directory, e.g.: /home/username/isaacgym
   cd /home/username/isaacgym
   python setup.py develop
   ```

2. **Clone this project**
   ```bash
   git clone <your-repo-url>
   cd WBC
   ```

3. **Install Python dependencies**
   ```bash
   # Create conda environment
   conda create -n b2z1_wbc python=3.8
   conda activate b2z1_wbc
   
   # Install PyTorch (select based on CUDA version)
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
   
   # Install other dependencies
   pip install numpy matplotlib wandb
   ```

4. **Set environment variables**
   ```bash
   export ISAACGYM_PATH=/home/username/isaacgym
   export PYTHONPATH=$PYTHONPATH:$ISAACGYM_PATH
   ```

## Usage

### Policy Training

#### B2Z1 Position-Force Control Training
```bash
cd legged_gym/scripts
python train_b2z1posforce.py --task=b2z1_pos_force_ee_realrobot --headless
```

#### B2Z1 Force Control Training
```bash
python train_b2z1force.py --task=b2z1_force_realrobot --headless
```

#### Other Robot Training
```bash
# H1 robot training
python train.py --task=h1 --headless

# G1 robot training  
python train_humanoidgym.py --task=g1_humanoidgym --headless
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

## Project Structure

```
WBC/
├── legged_gym/                    # Main code directory
│   ├── envs/                      # Environment definitions
│   │   ├── b2/                    # B2 robot related
│   │   │   ├── b2z1_pos_force_ee_realrobot_config.py  # B2Z1 position-force control config
│   │   │   ├── legged_robot_b2z1_pos_force_ee_realrobot.py  # B2Z1 environment implementation
│   │   │   └── ...
│   │   ├── h1/                    # H1 robot related
│   │   ├── g1/                    # G1 robot related
│   │   └── base/                  # Base environment classes
│   ├── utils/                     # Utility functions
│   │   ├── task_registry_b2z1posforce.py  # Task registration
│   │   ├── helpers.py             # Helper functions
│   │   └── ...
│   ├── b2_gym_learn/              # Reinforcement learning algorithms
│   │   └── ppo_cse_pf/            # PPO algorithm implementation
│   └── scripts/                   # Training and testing scripts
│       ├── train_b2z1posforce.py  # B2Z1 position-force control training
│       ├── play_b2z1posforce.py   # B2Z1 position-force control testing
│       ├── train_b2z1force.py     # B2Z1 force control training
│       └── ...
├── ckpt/                          # Trained model checkpoints
├── logs/                          # Training logs
├── resources/                     # Robot resource files
└── deploy/                        # Deployment related files
```

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

## Troubleshooting

### Common Issues

1. **Task Not Registered Error**
   ```
   ValueError: Task with name: b2z1_pos_force_ee_realrobot was not registered
   ```
   **Solution**: Ensure the task is properly registered in `legged_gym/envs/__init__.py`

2. **Module Import Error**
   ```
   ModuleNotFoundError: No module named 'legged_gym.envs.b2.legged_robot_b2z1_pos_force_ee_realrobot'
   ```
   **Solution**: Check if file paths and class names are correct

3. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce the number of parallel environments `--num_envs` or use smaller batch sizes

### Debugging Tips

- Use `--headless` parameter for headless training
- Set `VISUAL_PRED = True` to visualize prediction results
- Check log files to understand training progress
- Use `wandb` for experiment tracking