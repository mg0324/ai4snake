import pygame
import sys
from snake_env import SnakeEnv

# 初始化游戏环境
env = SnakeEnv(grid_size=30)  # 使用20x20的网格
obs = env.reset()
env.render()

# 映射按键到动作
key_action_mapping = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3
}

# 游戏循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key in key_action_mapping:
                action = key_action_mapping[event.key]
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    print(f"Game over! Your final length was {len(env.snake)}.")
                    obs = env.reset()
                    env.render()
