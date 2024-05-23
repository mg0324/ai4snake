import gym
from gym import spaces
import numpy as np
import pygame
import sys
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10, 3), dtype=np.uint8)
        self.reset()

        # 初始化 pygame
        pygame.init()
        self.size = 500  # 窗口大小
        self.block_size = self.size // 10  # 每个方块的大小
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = [(5, 5)]
        self.food = (np.random.randint(10), np.random.randint(10))
        self.done = False
        self.score = 0
        self.snake_length = 1  # 重置蛇的长度
        return self._get_observation()

    def _get_observation(self):
        state = np.zeros((10, 10, 3), dtype=np.uint8)
        for s in self.snake:
            state[s] = (0, 255, 0)
        state[self.food] = (255, 0, 0)
        return state

    def step(self, action):
        if action == 0:  # 上
            new_head = (self.snake[0][0] - 1, self.snake[0][1])
        elif action == 1:  # 下
            new_head = (self.snake[0][0] + 1, self.snake[0][1])
        elif action == 2:  # 左
            new_head = (self.snake[0][0], self.snake[0][1] - 1)
        elif action == 3:  # 右
            new_head = (self.snake[0][0], self.snake[0][1] + 1)

        if self._is_collision(new_head):
            self.done = True
            reward = -10  # 增加碰撞的惩罚
            logging.info(f"Snake died. Length was: {len(self.snake)}")  # 打印蛇的长度
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 20 * len(self.snake)  # 奖励与蛇的长度成比例
                self.score += 1
                self.snake_length += 1  # 增加蛇的长度
                self.food = (np.random.randint(10), np.random.randint(10))
                logging.info(f"Snake ate food at {self.food}")
            else:
                reward = -0.01  # 每一步稍微有点惩罚，以鼓励尽快找到食物
                self.snake.pop()

                # 鼓励靠近食物
                distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
                reward += (10 - distance) * 0.1

        return self._get_observation(), reward, self.done, {}

    def _is_collision(self, position):
        if (position[0] < 0 or position[0] >= 10 or
                position[1] < 0 or position[1] >= 10 or
                position in self.snake):
            return True
        return False

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        for s in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(s[1] * self.block_size, s[0] * self.block_size, self.block_size, self.block_size))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.food[1] * self.block_size, self.food[0] * self.block_size, self.block_size, self.block_size))
        pygame.display.flip()
        self.clock.tick(10)  # 控制游戏速度

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

