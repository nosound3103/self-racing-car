import pygame
import yaml
import random
import torch
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

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

        pygame.display.set_caption("Car Racing")

        self.total_rewards = []

        self.clock = pygame.time.Clock()

    def map_init(self, map_name):
        self.map = cv2.imread(cfg["Map"][map_name]["IMAGE"])
        map_height, map_width, _ = self.map.shape
        screen_height, screen_width = cfg["Env"]["HEIGHT"], cfg["Env"]["WIDTH"]
        width_ratio, height_ratio = screen_width / map_width, screen_height / map_height

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

        self.start_pos = cfg["Map"][map_name]["START_POS"]
        self.start_pos = [[pos["X"] * width_ratio,
                           pos["Y"] * height_ratio,
                           pos["ANGLE"]]
                          for pos in self.start_pos]

        self.game_map = pygame.image.load(
            cfg["Map"][map_name]["SCENERY"]).convert()
        self.game_map = pygame.transform.scale(
            self.game_map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        self.middle_screen = (350 * width_ratio, 300 * height_ratio)

        try:
            self.min_speed_sign_line = cfg["Map"][map_name]["MIN_SPEED_SIGN_POS"]
            self.min_speed_sign_line = [LineString([[pos["X1"] * width_ratio,
                                                     pos["Y1"] * height_ratio],
                                                    [pos["X2"] * width_ratio,
                                                     pos["Y2"] * height_ratio]])
                                        for pos in self.min_speed_sign_line]
            self.max_speed_sign_line = cfg["Map"][map_name]["MAX_SPEED_SIGN_POS"]
            self.max_speed_sign_line = [LineString([[pos["X1"] * width_ratio,
                                                     pos["Y1"] * height_ratio],
                                                    [pos["X2"] * width_ratio,
                                                     pos["Y2"] * height_ratio]])
                                        for pos in self.max_speed_sign_line]

            self.speed_bump_box = cfg["Map"][map_name]["SPEED_BUMP_POS"]
            self.speed_bump_box = [Polygon([[box["X1"] * width_ratio,
                                             box["Y1"] * height_ratio],
                                            [box["X2"] * width_ratio,
                                             box["Y2"] * height_ratio],
                                            [box["X3"] * width_ratio,
                                             box["Y3"] * height_ratio],
                                            [box["X4"] * width_ratio,
                                             box["Y4"] * height_ratio]])
                                   for box in self.speed_bump_box]
        except Exception as e:
            self.min_speed_sign_line = None
            self.max_speed_sign_line = None
            self.speed_bump_box = None
            return

    def draw_background(self):
        self.screen.blit(self.game_map, (0, 0))

    def draw_middle_line(self):
        for i in range(len(self.middle_line) - 1):
            p1 = self.middle_line[i]
            p2 = self.middle_line[i+1]

            pygame.draw.line(self.screen, (255, 255, 255), p1, p2, 1)

    def draw_boundaries(self):
        for contour in self.boundary:
            pygame.draw.polygon(self.screen, (255, 255, 255), contour, 2)

    def draw_sensors(self, car):
        sensors = car.calc_sensors(boundaries=self.boundary)

        if len(sensors):
            for _, vector in sensors.items():
                try:
                    pygame.draw.line(self.screen, (0, 255, 0),
                                     car.rect.center, vector, 1)
                except Exception:
                    continue

    def draw_car_boundary(self, car):
        pygame.draw.polygon(self.screen, (0, 0, 255), car.corners, 2)

    def are_cars_collided(self, cars):
        if len(cars) < 2:
            return False

        car_1 = cars[0]
        car_2 = cars[1]

        car_1_box = Polygon(car_1.corners)
        car_2_box = Polygon(car_2.corners)

        is_collided = car_1_box.intersects(car_2_box)
        if is_collided:
            car_1.died = True
            car_2.died = True
            return True
        return False

    def display_speed(self, cars):
        font_size = 25
        font = pygame.font.Font(None, font_size)
        line_break = 0
        for i, car in enumerate(cars):
            if car.died:
                text = f"{i + 1}: Dead"
            else:
                text = f"{i + 1}: {car.speed}"

            text = font.render(text, True, (0, 0, 0))
            self.screen.blit(text, [self.middle_screen[0],
                                    self.middle_screen[1] + line_break])
            line_break += font_size

    def reset(self):
        self.cars = [
            Car(position=pos[:2],
                angle_position=pos[2])
            for pos in self.start_pos]

        self.total_rewards = 0
        # self.speed_rewards = []
        # self.middle_line_rewards = []
        # self.distance_rewards = []

    def step(self, car, action, start_time, eval=False):

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
        elif action == 4:
            car.maintain_speed()

        car.calc_corners()

        car.died = car.is_collided(self.boundary)

        if eval:
            return

        if car.died:
            reward = -10000
        else:
            reward = self.calc_total_rewards(
                car, start_time)

        state = self.get_state(car)
        return state, reward

    def get_state(self, car):
        state = []
        state.extend(np.array([
            car.sensors_directions["f_l"],
            car.sensors_directions["f_r"],
            car.sensors_directions["b_r"],
            car.sensors_directions["b_l"]]).flatten())
        state.append(car.min_speed_allowed / car.max_speed_allowed)
        state.append(car.speed / car.max_speed_allowed)
        state.append(car.max_speed_allowed / car.max_speed_allowed)
        sensors = car.calc_sensors(boundaries=self.boundary)
        sign_interaction = self.interact_with_signs(car)
        speed_bump_interaction = self.interact_with_speed_bumps(car)

        for intersection in sensors.values():
            dist = utils.calc_distance(intersection, car.rect.center)
            state.append(dist / np.sqrt(1200 ** 2 + 900 ** 2))

        return state + sign_interaction + speed_bump_interaction

    def calc_middle_line_reward(self, car, max_distance=None):
        if not max_distance:
            max_distance = car.horizontal_size

        distance = utils.calc_distance_to_middle(
            car, LineString(self.middle_line))

        if distance > max_distance:
            return -0.1
        return 0.01

    def calc_speed_maintainance_reward(self, car):

        if car.max_speed_temporal and car.speed > car.max_speed_temporal:
            return -0.1

        if car.speed < car.min_speed_allowed or \
                car.speed > car.max_speed_allowed:
            return -0.1
        return 0.01

    def calc_distance_reward(self, car):
        current_position = np.array(car.rect.center)
        previous_position = np.array(car.previous_position)

        distance, _, _ = utils.calculate_distance_travelled(
            current_position, previous_position, self.middle_line)

        car.total_distance_travelled += distance

        car.previous_position = current_position

        return distance * 0.01 if distance > 0 else 0.01

    def calc_total_rewards(self, car, start_time):
        middle_line_reward = self.calc_middle_line_reward(car)
        speed_reward = self.calc_speed_maintainance_reward(car)
        distance_reward = self.calc_distance_reward(car)
        existance_reward = self.reward_existance(start_time)

        # self.middle_line_rewards.append(middle_line_reward)
        # self.speed_rewards.append(speed_reward)
        # self.distance_rewards.append(distance_reward)

        self.total_rewards += (middle_line_reward +
                               speed_reward +
                               distance_reward +
                               existance_reward)

        return self.total_rewards

    def reward_existance(self, start_time):
        return (time.time() - start_time) * 0.01

    def interact_with_signs(self, car):
        if not self.min_speed_sign_line and not self.max_speed_sign_line:
            return [0, 0]

        car_box = Polygon(car.corners)

        for sign_line in self.min_speed_sign_line:
            if car_box.intersects(sign_line):
                car.min_speed_allowed = 15
                car.max_speed_allowed = car.max_speed
                car.sign_history = [1, 0]
                # print("Min speed sign")
                return car.sign_history

        for sign_line in self.max_speed_sign_line:
            if car_box.intersects(sign_line):
                car.min_speed_allowed = car.min_speed
                car.max_speed_allowed = 10
                car.sign_history = [0, 1]
                # print("Max speed sign")
                return car.sign_history

        # if car.sign_history[0] == 1:
        #     print("Min speed sign")
        # elif car.sign_history[1] == 1:
        #     print("Max speed sign")
        return car.sign_history

    def interact_with_speed_bumps(self, car):
        if not self.speed_bump_box:
            return [0]

        car_box = Polygon(car.corners)

        for box in self.speed_bump_box:
            if car_box.intersects(box):
                car.max_speed_temporal = 10
                # print("Speed bump")
                return [1]

        car.max_speed_temporal = None
        return [0]

    def run(self,
            map_name="ROAD_3",
            algo="bellman",
            num_episodes=1000,
            time_per_episode=30):

        reward_per_episode = []

        self.map_init(map_name)

        self.reset()

        for episode in range(num_episodes):
            total_rewards = 0
            episode_start_time = time.time()

            while time.time() - episode_start_time < time_per_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.draw_background()
                self.draw_middle_line()
                self.draw_boundaries()

                alive_cars = [
                    car for car in self.cars if not car.died]

                if not len(alive_cars):
                    break

                for car in alive_cars:
                    state = self.get_state(car)

                    action = self.agent.choose_action(state)

                    next_state, reward = self.step(
                        car, action, episode_start_time)

                    total_rewards += reward

                    self.agent.store_transition(
                        state, action, reward, next_state, car.died)

                    self.screen.blit(car.rotated_image, car.rect)
                    self.draw_car_boundary(car)
                    self.draw_sensors(car)

                self.are_cars_collided(alive_cars)
                self.display_speed(self.cars)
                self.agent.train(algorithm=algo)
                pygame.display.flip()
                self.clock.tick(cfg["Env"]["FPS"])

            reward_per_episode.append(total_rewards / len(self.cars) + 10000)
            utils.plot_reward(reward_per_episode)
            print("All cars are done. Moving to next episode...")
            self.reset()

            print(f"Episode {episode + 1}/{num_episodes} completed.")

            if episode % 10 == 0:
                torch.save(self.agent.Q_eval.state_dict(), f'{algo}.pth')

        pygame.quit()

    def play(self,
             map_name="ROAD_3",
             weights_path="bellman.pth",
             num_episodes=50,
             time_per_episode=30):

        self.map_init(map_name)

        self.reset()

        self.agent.load_model(weights_path)

        for episode in range(num_episodes):
            episode_start_time = time.time()
            dones = [False] * len(self.cars)

            while not all(dones) and time.time() \
                    - episode_start_time < time_per_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.draw_background()
                self.draw_middle_line()
                self.draw_boundaries()

                alive_cars = [
                    car for car in self.cars if not car.died]

                if not len(alive_cars):
                    break

                for car in alive_cars:
                    state = self.get_state(car)
                    action = self.agent.choose_action(state)

                    self.step(
                        car, action, start_time=episode_start_time, eval=True)

                    self.screen.blit(car.rotated_image, car.rect)
                    self.draw_sensors(car)

                self.display_speed(self.cars)
                pygame.display.flip()
                self.clock.tick(cfg["Env"]["FPS"])

            print("Car is done. Moving to next episode...")
            self.reset()

            print(f"Episode {episode + 1}/{num_episodes} completed.")

        pygame.quit()
