from stable_baselines3 import DQN
from snake_env import CustomSnakeEnv
import logging

env = CustomSnakeEnv(grid_size=20)
model = DQN.load("dqn_snake")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        logging.info(f"step: {i}")
        obs = env.reset()


