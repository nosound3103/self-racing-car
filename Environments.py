import pygame
import yaml
import random
import cv2
import numpy as np
from shapely.geometry import LineString

import utils
from Car import Car
from Agent import Agent
import time

random.seed(0)

with open("config.yml", "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


class Environment:
    def __init__(self):
        pygame.init()

        self.agent = Agent()

        self.screen = pygame.display.set_mode(
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))
        self.map = cv2.imread("images/test_map_2.png")
        map_height, map_width, _ = self.map.shape

        self.boundary = utils.find_contours(self.map).reshape(-1, 2)
        self.boundary = utils.resize_points(
            self.boundary,
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.middle_line = utils.find_middle_line(self.map).reshape(-1, 2)
        self.middle_line = utils.resize_points(
            self.middle_line,
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.game_map = pygame.image.load("images/test_map_2.png").convert()
        self.game_map = pygame.transform.scale(
            self.game_map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        pygame.display.set_caption("Car Racing")

        self.all_car_sprites = pygame.sprite.Group()
        for _ in range(cfg["Env"]["NUM_CARS"]):
            self.all_car_sprites.add(Car())

        self.clock = pygame.time.Clock()

        self.reset()

    def draw_background(self):
        self.screen.blit(self.game_map, (0, 0))

    def draw_middle_line(self):
        for i in range(len(self.middle_line) - 1):
            p1 = self.middle_line[i]
            p2 = self.middle_line[i+1]

            pygame.draw.line(self.screen, (255, 0, 0), p1, p2, 5)

    def draw_sensors(self, car):
        sensors = car.calc_sensors(boundary=self.boundary)

        if len(sensors):
            for _, vector in sensors.items():
                try:
                    pygame.draw.line(self.screen, (0, 255, 0),
                                     car.rect.center, vector, 1)
                except Exception:
                    continue

    def reset(self):
        self.all_car_sprites = pygame.sprite.Group()
        for _ in range(cfg["Env"]["NUM_CARS"]):
            self.all_car_sprites.add(Car())

        self.total_rewards = 0
        self.speed_rewards = []
        self.middle_line_rewards = []

    def step(self, car, action):
        bonus_reward = 0

        if action == 0:
            car.accelerate()
        # Turn left
        elif action == 1:
            car.turn(-car.turn_angle)
        # Turn right
        elif action == 2:
            car.turn(car.turn_angle)
        elif action == 3:
            car.backward()

        car.calc_corners()
        car.died = car.is_collided(self.boundary)

        if car.died:
            reward = -100000
        else:
            reward = self.calc_total_rewards(car)

        state = self.get_state(car)
        return state, reward

    def get_state(self, car):
        state = []
        sensors = car.calc_sensors(boundary=self.boundary)

        for intersection in sensors.values():
            dist = utils.calc_distance(intersection, car.rect.center)
            state.append(dist)

        return state

    def calc_middle_line_reward(self, car, max_distance=None):
        if not max_distance:
            max_distance = car.horizontal_size

        distance = utils.calc_distance_to_middle(
            car, LineString(self.middle_line))

        if distance > max_distance:
            return -100
        return 1

    def calc_speed_maintainance_reward(self, car):
        if car.speed < car.max_speed * 0.8:
            return -50
        return 10

    def calc_progress_reward(self, car):
        current_position = np.array(car.rect.center)
        closest_point = utils.find_closest_point(
            current_position, self.middle_line)

    def calc_total_rewards(self, car):
        middle_line_reward = self.calc_middle_line_reward(car)
        speed_reward = self.calc_speed_maintainance_reward(car)

        self.middle_line_rewards.append(middle_line_reward)
        self.speed_rewards.append(speed_reward)

        self.total_rewards += (middle_line_reward * 0.2 + speed_reward * 0.8)

        return self.total_rewards

    def run(self, num_episodes=1000):

        for episode in range(num_episodes):
            episode_start_time = time.time()
            dones = [False] * len(self.all_car_sprites)

            while not all(dones) and time.time() - episode_start_time < 30:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.draw_background()
                self.draw_middle_line()

                for i, car in enumerate(self.all_car_sprites):
                    if car.died:
                        continue

                    state = self.get_state(car)
                    action = self.agent.choose_action(state)

                    next_state, reward = self.step(car, action)
                    dones[i] = car.died

                    self.agent.store_transition(
                        state, action, reward, next_state, car.died)
                    self.agent.learn()

                    self.screen.blit(car.rotated_image, car.rect)
                    self.draw_sensors(car)
                    print(reward)

                pygame.display.flip()
                self.clock.tick(cfg["Env"]["FPS"])

            self.total_rewards -= 100000
            print("All cars are done. Moving to next episode...")
            self.reset()

            print(f"Episode {episode + 1}/{num_episodes} completed.")

    # def run(self):
    #     running = True
    #     while running:
    #         self.draw_background()
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False

    #         keys = pygame.key.get_pressed()
    #         for car in self.all_car_sprites:
    #             # if car.is_collided(boundary):
    #             #     continue
    #             car.update(keys)
    #             self.screen.blit(car.rotated_image, car.rect)

    #             # pygame.draw.polygon(self.screen, (255, 0, 0), car.corners, 1)
    #             sensors = car.calc_sensors(boundary=self.boundary)
    #             if len(sensors):
    #                 for _, vector in sensors.items():
    #                     try:
    #                         pygame.draw.line(self.screen, (0, 255, 0),
    #                                          car.rect.center, vector, 1)
    #                     except Exception:
    #                         continue

    #         pygame.draw.polygon(self.screen, (0, 0, 255), self.boundary, 1)
    #         pygame.display.update()
    #         pygame.display.flip()
    #         self.clock.tick(cfg["Env"]["FPS"])

    #     pygame.quit()


env = Environment()
env.run()
