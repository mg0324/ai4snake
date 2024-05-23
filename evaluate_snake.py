from stable_baselines3 import DQN
from snake_env import SnakeEnv
import logging

env = SnakeEnv()
model = DQN.load("dqn_snake")

obs = env.reset()
for i in range(1000):
    logging.info(f"step: {i}")
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()


