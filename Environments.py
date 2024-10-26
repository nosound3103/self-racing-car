import pygame
import yaml
import random
import torch
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
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        self.agent = Agent()

        self.screen = pygame.display.set_mode(
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.map = cv2.imread(cfg["Map"]["ROAD_3"]["IMAGE"])
        map_height, map_width, _ = self.map.shape

        self.boundary = utils.find_contours(self.map)
        self.boundary = [utils.resize_points(
            contour,
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))
            for contour in self.boundary]

        self.middle_line = utils.find_middle_line(self.map).reshape(-1, 2)
        self.middle_line = utils.resize_points(
            self.middle_line,
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.distance = utils.calculate_total_distance(self.middle_line)

        self.start_point = [cfg["Map"]["ROAD_3"]
                            ["START_X"], cfg["Map"]["ROAD_3"]["START_Y"]]
        self.start_point = utils.resize_points(
            [self.start_point],
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))[0]

        self.end_point = [cfg["Map"]["ROAD_3"]
                          ["END_X"], cfg["Map"]["ROAD_3"]["END_Y"]]

        self.end_point = utils.resize_points(
            [self.end_point],
            (map_width, map_height),
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))[0]

        self.angle_position = cfg["Map"]["ROAD_3"]["ANGLE"]

        self.game_map = pygame.image.load(
            cfg["Map"]["ROAD_3"]["IMAGE"]).convert()
        self.game_map = pygame.transform.scale(
            self.game_map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.total_rewards = []

        pygame.display.set_caption("Car Racing")

        self.clock = pygame.time.Clock()

        self.reset()

    def draw_background(self):
        self.screen.blit(self.game_map, (0, 0))

    def draw_middle_line(self):
        for i in range(len(self.middle_line) - 1):
            p1 = self.middle_line[i]
            p2 = self.middle_line[i+1]

            pygame.draw.line(self.screen, (255, 0, 0), p1, p2, 1)

    def draw_boundaries(self):
        for contour in self.boundary:
            pygame.draw.polygon(self.screen, (0, 0, 255), contour, 2)

    def draw_sensors(self, car):
        sensors = car.calc_sensors(boundaries=self.boundary)

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
            self.all_car_sprites.add(
                Car(position=self.start_point,
                    angle_position=self.angle_position))

        self.total_rewards = 0
        self.speed_rewards = []
        self.middle_line_rewards = []
        self.distance_rewards = []

    def step(self, car, action):

        if action == 0:
            car.accelerate()
        # Turn left
        elif action == 1:
            car.turn(-car.turn_angle)
        # Turn right
        elif action == 2:
            car.turn(car.turn_angle)
        elif action == 3:
            car.brake()

        car.calc_corners()

        car.died = car.is_collided(self.boundary)

        if car.died:
            reward = -100000
        else:
            reward = self.calc_total_rewards(
                car)

        state = self.get_state(car)
        return state, reward

    def get_state(self, car):
        state = []
        state.extend(car.corners.flatten())
        state.append(car.speed)
        sensors = car.calc_sensors(boundaries=self.boundary)

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
            return -0.5
        return 0.1

    def calc_speed_maintainance_reward(self, car):
        if car.speed < car.max_speed * 0.6:
            return -0.5
        return 0.1

    def calc_distance_reward(self, car):
        current_position = np.array(car.rect.center)
        previous_position = np.array(car.previous_position)

        distance = utils.calculate_distance_travelled(
            current_position, previous_position, self.middle_line)

        car.total_distance_travelled += distance

        car.previous_position = current_position

        return distance * 0.1 if distance > 0 else -1

    def calc_total_rewards(self, car):
        middle_line_reward = self.calc_middle_line_reward(car)
        speed_reward = self.calc_speed_maintainance_reward(car)
        distance_reward = self.calc_distance_reward(car)

        self.middle_line_rewards.append(middle_line_reward)
        self.speed_rewards.append(speed_reward)
        self.distance_rewards.append(distance_reward)

        self.total_rewards += (middle_line_reward +
                               speed_reward + distance_reward)

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

                if episode >= 0:
                    self.draw_background()
                    self.draw_middle_line()
                    self.draw_boundaries()

                alive_cars = [
                    car for car in self.all_car_sprites if not car.died]

                if not len(alive_cars):
                    break

                for car in alive_cars:
                    state = self.get_state(car)
                    action = self.agent.choose_action(state)

                    next_state, reward = self.step(car, action)

                    self.agent.store_transition(
                        state, action, reward, next_state, car.died)

                    if episode >= 0:
                        self.screen.blit(car.rotated_image, car.rect)
                        self.draw_sensors(car)

                self.agent.train()
                pygame.display.flip()
                self.clock.tick(cfg["Env"]["FPS"])

            print("All cars are done. Moving to next episode...")
            self.reset()

            print(f"Episode {episode + 1}/{num_episodes} completed.")

            if episode % 10 == 0:
                torch.save(self.agent.Q_eval.state_dict(), 'dqn.pth')

        pygame.quit()
