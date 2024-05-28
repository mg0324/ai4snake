from stable_baselines3 import PPO
from snake_env import SnakeEnv
import logging

env = SnakeEnv(grid_size=20)
model = PPO.load("ppo_snake")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        logging.info(f"step: {i}")
        obs = env.reset()


