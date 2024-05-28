import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
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

def main():
    env = SnakeEnv(grid_size=20, size=800)

    policy_kwargs = dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model_path = "ppo_snake"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, verbose=1)
        model.policy_kwargs = policy_kwargs
        print("继续训练已存在的模型...")
    else:
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
        print("训练新模型...")

    total_timesteps = 10
    for i in range(total_timesteps):
        print(f"i:{i}")
        obs = env.reset()
        action = env.a_star_action()  # 使用A*算法选择动作
        obs, reward, done, _ = env.step(action)
        print(f"done:{done}") 
        model.learn(total_timesteps=1)  # 在每个步骤都进行一次强化学习
        print(f"学习中...{i}")
    print("更新模型...")    
    model.save(model_path)

if __name__ == "__main__":
    main()
