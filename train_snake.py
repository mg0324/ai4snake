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

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            # 从 DummyVecEnv 中提取实际的环境实例
            real_env = self.env.venv.envs[0]
            real_env.render()
        return True

env = DummyVecEnv([lambda: SnakeEnv()])
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
render_callback = RenderCallback(env=env, render_freq=50)

# 训练模型并渲染
model.learn(total_timesteps=10000, callback=render_callback)
model.save(model_path)
