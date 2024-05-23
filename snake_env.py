import gym
from gym import spaces
import numpy as np
import pygame
import sys
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.reset()

        # 初始化 pygame
        pygame.init()
        self.size = 500  # 窗口大小
        self.block_size = self.size // self.grid_size  # 每个方块的大小
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

        # 初始化字体
        self.font = pygame.font.SysFont('Arial', 18)

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.food = self._generate_food()
        self.done = False
        self.score = 0
        self.snake_length = 1  # 重置蛇的长度
        self.direction = 3  # 初始方向为右
        self.prev_direction = self.direction
        return self._get_observation()

    def _generate_food(self):
        while True:
            food = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if food not in self.snake:
                return food

    def _get_observation(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for s in self.snake:
            state[s] = (0, 255, 0)
        state[self.food] = (255, 0, 0)
        return state

    def step(self, action):
        # 如果蛇的长度大于1，并且动作是反方向的，则忽略该动作并给惩罚
        if len(self.snake) > 1:
            if (self.direction == 0 and action == 1) or (self.direction == 1 and action == 0) or \
               (self.direction == 2 and action == 3) or (self.direction == 3 and action == 2):
                action = self.direction  # 保持当前方向
                reward = -1  # 给予惩罚
                return self._get_observation(), reward, self.done, {}

        self.prev_direction = self.direction
        self.direction = action

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
                self.food = self._generate_food()
                logging.info(f"Snake ate food at {self.food}")
            else:
                reward = -0.01  # 每一步稍微有点惩罚，以鼓励尽快找到食物
                self.snake.pop()

                # 鼓励靠近食物
                distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
                reward += (self.grid_size - distance) * 0.1

                # 检查是否在抖动
                if (self.prev_direction == 0 and self.direction == 2) or (self.prev_direction == 2 and self.direction == 0) or \
                   (self.prev_direction == 1 and self.direction == 3) or (self.prev_direction == 3 and self.direction == 1):
                    reward -= 0.5  # 如果出现抖动行为，给予负奖励

        return self._get_observation(), reward, self.done, {}

    def _is_collision(self, position):
        if (position[0] < 0 or position[0] >= self.grid_size or
                position[1] < 0 or position[1] >= self.grid_size or
                position in self.snake):
            return True
        return False

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        for s in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(s[1] * self.block_size, s[0] * self.block_size, self.block_size, self.block_size))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.food[1] * self.block_size, self.food[0] * self.block_size, self.block_size, self.block_size))

        # 显示分数在右上角
        score_text = self.font.render(f'Score: {self.snake_length}', True, (255, 255, 255))
        text_rect = score_text.get_rect(topright=(self.size - 10, 10))  # 设置右上角位置
        self.screen.blit(score_text, text_rect)

        pygame.display.flip()
        self.clock.tick(10)  # 控制游戏速度

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
