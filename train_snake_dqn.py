import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnv, a_star_search
import gym
import os

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
    def __init__(self, env, render_freq=100, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.render_freq = render_freq
        self.real_env = self.env.venv.envs[0]  # 从 DummyVecEnv 中提取实际的环境实例

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.real_env.render()
        return True

class CustomSnakeEnv(SnakeEnv):
    def step(self, action):
        # 获取A*算法建议的路径
        grid = np.zeros((self.grid_size, self.grid_size))
        for s in self.snake:
            grid[s] = 1
        path = a_star_search(grid, self.snake[0], self.food)

        # 如果存在路径，获取A*算法建议的下一个位置
        if path:
            next_position = path[0]
            if next_position[0] < self.snake[0][0]:
                a_star_action = 0  # 上
            elif next_position[0] > self.snake[0][0]:
                a_star_action = 1  # 下
            elif next_position[1] < self.snake[0][1]:
                a_star_action = 2  # 左
            elif next_position[1] > self.snake[0][1]:
                a_star_action = 3  # 右
        else:
            a_star_action = action  # 如果没有路径，使用DQN的动作

        observation, reward, done, info = super().step(action)

        # 增强奖励：当DQN动作与A*建议的动作一致时，给予额外奖励
        if action == a_star_action:
            reward += 1

        return observation, reward, done, info

env = DummyVecEnv([lambda: CustomSnakeEnv(grid_size=20)])
env = VecMonitor(env)  # 增加监控装饰器，以便于收集额外信息

policy_kwargs = dict(
    features_extractor_class=CustomNetwork,
    features_extractor_kwargs=dict(features_dim=256),
)

# 检查是否存在已保存的模型
model_path = "dqn_snake"
if os.path.exists(model_path + ".zip"):
    model = DQN.load(model_path, env=env, verbose=1)
    # 更新模型的policy_kwargs
    model.policy_kwargs = policy_kwargs
    print("继续训练已存在的模型...")
else:
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    print("训练新模型...")

# 创建回调对象，设定渲染频率为每100个时间步渲染一次
#render_callback = RenderCallback(env=env, render_freq=100)

# 训练模型并渲染
model.learn(total_timesteps=10000)
model.save(model_path)
