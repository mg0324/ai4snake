import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnv
import gym
import os
import argparse

class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(observation_space.shape), 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq=100, render=False, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.render = render

    def _on_step(self) -> bool:
        if self.render and self.n_calls % self.render_freq == 0:
            real_env = self.env.venv.envs[0]
            real_env.render()
        return True

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train a DQN agent on the Snake environment.')
parser.add_argument('--render', action='store_true', help='Enable rendering during training')
args = parser.parse_args()

# 确保网格大小一致
env = DummyVecEnv([lambda: SnakeEnv(grid_size=20)])
env = VecMonitor(env)

policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    features_extractor_kwargs=dict(features_dim=256),
)

model_path = "dqn_snake"
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env, verbose=1)
    model.policy_kwargs = policy_kwargs
    print("继续训练已存在的模型...")
else:
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    print("训练新模型...")

# 根据参数决定是否启用渲染回调
render_callback = RenderCallback(env=env, render_freq=50, render=args.render)
model.learn(total_timesteps=50000, callback=render_callback)
model.save(model_path)
