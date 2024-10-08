import pygame
import yaml
import math
import numpy as np
import cv2

from shapely.geometry import Polygon, LineString
from shapely.affinity import scale

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class Car(pygame.sprite.Sprite):
    def __init__(self, cfg=config["Car"], speed=3):
        super().__init__()
        self.cfg = cfg
        self.image = pygame.image.load(self.cfg["IMAGE"]).convert_alpha()
        self.image = pygame.transform.scale(
            self.image, (self.cfg["WIDTH"], self.cfg["HEIGHT"]))

        self.diagonal = math.sqrt(
            (self.cfg["WIDTH"]/2) ** 2 + (self.cfg["HEIGHT"]/2) ** 2)

        self.rotated_image = self.image
        self.rect = self.rotated_image.get_rect()
        self.rect.x = 400
        self.rect.y = 300
        self.angle = 0
        self.speed = speed
        self.max_speed = self.cfg["MAX_SPEED"]
        self.min_speed = self.cfg["MIN_SPEED"]
        self.speed_up = self.cfg["SPEED_UP"]
        self.slow_down = self.cfg["SLOW_DOWN"]
        self.turn_speed = self.cfg["TURN_SPEED"]

        self.sensors_directions = {
            "front": [a - b for (a, b) in zip(self.rect.midtop, self.rect.center)],
            "front_right": [a - b for (a, b) in zip(self.rect.topright, self.rect.center)],
            "right": [a - b for (a, b) in zip(self.rect.midright, self.rect.center)],
            "front_left": [a - b for (a, b) in zip(self.rect.topleft, self.rect.center)],
            "left": [a - b for (a, b) in zip(self.rect.midleft, self.rect.center)],
            "back": [a - b for (a, b) in zip(self.rect.midbottom, self.rect.center)],
            "back_right": [a - b for (a, b) in zip(self.rect.bottomright, self.rect.center)],
            "back_left": [a - b for (a, b) in zip(self.rect.bottomleft, self.rect.center)]
        }

        self.sensors_directions = {
            direction: [
                coords[0] / np.linalg.norm(coords),
                coords[1] / np.linalg.norm(coords)]
            for (direction, coords) in self.sensors_directions.items()}

        self.calc_corners()

    def rotate_vector(self, angle_change, vector, points=[0, 0]):
        cos_angle = math.cos(math.radians(angle_change))
        sin_angle = math.sin(math.radians(angle_change))
        x = vector[0]
        y = vector[1]
        o_x = points[0]
        o_y = points[1]

        new_x = (x - o_x) * cos_angle - \
            (y - o_y) * sin_angle + o_x
        new_y = (x - o_x) * sin_angle + \
            (y - o_y) * cos_angle + o_y
        return [new_x, new_y]

    def turn(self, angle_change):
        self.angle += angle_change
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=self.rect.center)

        for direction, vector in self.sensors_directions.items():
            self.sensors_directions[direction] = self.rotate_vector(
                angle_change=angle_change, vector=vector)

        if angle_change > 0:
            print("Turn right: ", self.angle, "with speed: ", self.speed)
        else:
            print("Turn left: ", self.angle, "with speed: ", self.speed)

    def accelerate(self):
        self.speed += self.speed_up
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        x_speed = self.sensors_directions["front"][0] * self.speed
        y_speed = self.sensors_directions["front"][1] * self.speed

        self.rect.move_ip(x_speed, y_speed)

        print("Speed up: ", self.speed)

    def brake(self):
        # self.speed += self.slow_down
        # if self.speed < self.min_speed:
        #     self.speed = self.min_speed

        # slow_down_speed = self.slow_down if self.speed > self.min_speed \
        #     else self.speed

        # self.rect = self.rect.move(
        #     self.sensors_directions["front"][0] * slow_down_speed,
        #     self.sensors_directions["front"][1] * slow_down_speed)

        x_slow_down = self.sensors_directions["front"][0] * self.slow_down
        y_slow_down = self.sensors_directions["front"][1] * self.slow_down
        self.rect.move_ip(x_slow_down, y_slow_down)

        print("Slow down:", self.speed)

    def momentum(self):
        self.speed += -0.2
        if self.speed < 0:
            self.speed = 0

        x_speed = self.sensors_directions["front"][0] * self.speed
        y_speed = self.sensors_directions["front"][1] * self.speed
        self.rect.move_ip(x_speed, y_speed)

    def update(self, keys_pressed):
        if keys_pressed[pygame.K_RIGHT]:
            self.turn(3)

        if keys_pressed[pygame.K_LEFT]:
            self.turn(-3)

        if keys_pressed[pygame.K_UP]:
            self.accelerate()

        if keys_pressed[pygame.K_DOWN]:
            self.brake()

        if not keys_pressed[pygame.K_UP]:
            self.momentum()

        self.calc_corners()

    def reset(self):
        pass

    def draw(self, screen):
        pass

    def calc_corners(self):
        self.corners = np.array([
            self.sensors_directions["front_left"],
            self.sensors_directions["front_right"],
            self.sensors_directions["back_right"],
            self.sensors_directions["back_left"]])
        self.corners *= self.diagonal
        self.corners += np.array(self.rect.center)

    def calc_sensors(self, boundary=None):
        sensors = {}

        if not boundary.size:
            return sensors

        for (direction, vector) in self.sensors_directions.items():
            if direction not in ["front", "front_right", "right", "front_left", "left"]:
                continue

            intersection = self.find_intersection(np.array(vector), boundary)

            sensors[direction] = intersection

        return sensors

    def find_intersection(self, d, boundary):
        closest_intersection = None
        min_t = float('inf')

        for i in range(len(boundary)):
            A_i = np.array(boundary[i])
            A_next = np.array(boundary[(i+1) % len(boundary)])

            segment_direction = A_next - A_i

            matrix = np.array([d, -segment_direction]).T
            rhs = A_i - self.rect.center

            try:
                ts = np.linalg.solve(matrix, rhs)
                t, s = ts

                if 0 <= s <= 1 and t >= 0:
                    if t < min_t:
                        min_t = t
                        closest_intersection = self.rect.center + t * d
            except np.linalg.LinAlgError:

                continue

        return closest_intersection

    def is_collided(self, boundary):
        car_boundary = Polygon(self.corners)
        boundary = Polygon(boundary)

        intersection = car_boundary.intersection(boundary).area
        union = car_boundary.area

        iou = intersection / union

        return iou < 0.95


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[1] if len(contours) == 2 else contours[0] \
        if len(contours) == 1 else None


pygame.init()

screen = pygame.display.set_mode(
    (config["Env"]["WIDTH"], config["Env"]["HEIGHT"]))

pygame.display.set_caption("Car Racing")

map = cv2.imread(
    "images/test_map_2.png")
map = cv2.resize(map, (config["Env"]["WIDTH"], config["Env"]
                 ["HEIGHT"]), interpolation=cv2.INTER_AREA)
game_map = pygame.image.load("images/test_map_2.png").convert()
game_map = pygame.transform.scale(
    game_map, (config["Env"]["WIDTH"], config["Env"]["HEIGHT"]))

boundary = find_contours(map).reshape(-1, 2)

car = Car(speed=2)

running = True
clock = pygame.time.Clock()
angle = 0

while running:
    # if car.is_collided(boundary):
    #     print("Died")
    #     break

    screen.blit(game_map, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    car.update(keys)

    screen.blit(car.rotated_image, car.rect)

    sensors = car.calc_sensors(boundary=boundary)

    if len(sensors):
        for direction, vector in sensors.items():
            try:
                pygame.draw.line(screen, (0, 255, 0),
                                 car.rect.center, vector, 1)
            except:
                continue

    pygame.draw.polygon(screen, (255, 0, 0), car.corners, 1)
    pygame.draw.polygon(screen, (0, 0, 255), boundary, 1)
    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
