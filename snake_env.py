import heapq
import numpy as np
import random
import gym
from gym import spaces
import pygame
import sys
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)

def a_star_search(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False


class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10, size=800):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size, 3), dtype=np.uint8)
        self.reset()

        # 初始化 pygame
        pygame.init()
        self.size = size  # 窗口大小
        self.block_size = self.size // self.grid_size  # 每个方块的大小
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.clock = pygame.time.Clock()

        # 初始化字体
        self.font = pygame.font.SysFont('Arial', 24)

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.food = self._generate_food()
        self.done = False
        self.score = 0
        self.snake_length = 1  # 重置蛇的长度
        self.steps = 0
        self.old_distance = self._get_distance_to_food()
        return self._get_observation()

    def _generate_food(self):
        while True:
            food = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if food not in self.snake:
                return food

    def _get_distance_to_food(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

    def _get_observation(self):
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for i, s in enumerate(self.snake):
            state[s] = (0, 255, 0) if i != 0 else (0, 0, 255)  # 蛇头用蓝色，其余用绿色
        state[self.food] = (255, 0, 0)
        return state

    def step(self, action):
        self.steps += 1

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
            reward = -10  # 碰撞惩罚
            logging.info(f"Snake died. Length was: {len(self.snake)}")  # 打印蛇的长度
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                base_reward = 10  # 吃到食物的基础奖励
                length_bonus = self.snake_length  # 根据蛇长度增加的奖励
                reward = base_reward + length_bonus
                self.score += 1
                self.food = self._generate_food()
                self.old_distance = self._get_distance_to_food()  # 重置距离
                #logging.info(f"Snake ate food at {self.food}")
            else:
                new_distance = self._get_distance_to_food()
                if new_distance < self.old_distance:
                    reward = 1  # 接近食物的奖励
                else:
                    reward = -5  # 每一步稍微有点惩罚，以鼓励尽快找到食物
                self.old_distance = new_distance
                self.snake.pop()

        return self._get_observation(), reward, self.done, {}

    def _is_collision(self, position):
        if (position[0] < 0 or position[0] >= self.grid_size or
                position[1] < 0 or position[1] >= self.grid_size or
                position in self.snake):
            return True
        return False

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        for i, s in enumerate(self.snake):
            color = (0, 255, 0) if i != 0 else (0, 0, 255)  # 蛇头用蓝色，其余用绿色
            pygame.draw.rect(self.screen, color, pygame.Rect(s[1] * self.block_size, s[0] * self.block_size, self.block_size, self.block_size))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.food[1] * self.block_size, self.food[0] * self.block_size, self.block_size, self.block_size))

        # 显示分数在右上角
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        text_rect = score_text.get_rect(topright=(self.size - 10, 10))  # 设置右上角位置
        self.screen.blit(score_text, text_rect)

        pygame.display.flip()
        self.clock.tick(10)  # 控制游戏速度

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def get_direction(self, head, goal):
        if goal[0] < head[0]:
            return 0  # 上
        elif goal[0] > head[0]:
            return 1  # 下
        elif goal[1] < head[1]:
            return 2  # 左
        elif goal[1] > head[1]:
            return 3  # 右

# 测试环境
if __name__ == "__main__":
    env = SnakeEnv(grid_size=40)
    obs = env.reset()
    total_reward = 0

    while True:
        action = env.a_star_action()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print("Game Over. Total Reward:", total_reward)
            obs = env.reset()
            total_reward = 0
