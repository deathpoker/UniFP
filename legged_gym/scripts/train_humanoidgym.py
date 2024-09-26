import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import numpy as np
from datetime import datetime
import sys
sys.path.append(unitree_rl_gym_path)

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry_humanoidgym

def train(args):
    env, env_cfg = task_registry_humanoidgym.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry_humanoidgym.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.headless = True
    train(args)
