import pygame
import yaml
import math
import numpy as np
import shapely

from shapely.geometry import Polygon, LineString, Point

import utils

from Timer import Timer

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class Car:
    def __init__(self,
                 cfg=config["Car"],
                 position=[108, 720],
                 angle_position=0,
                 width_ratio=config["Env"]["WIDTH"] /
                 config["Map"]["ROAD_3"]["WIDTH"],
                 height_ratio=config["Env"]["HEIGHT"] /
                 config["Map"]["ROAD_3"]["HEIGHT"]):

        self.cfg = cfg
        self.resize_width = int(self.cfg["WIDTH"] * width_ratio)
        self.resize_height = int(self.cfg["HEIGHT"] * height_ratio)

        self.image = pygame.image.load(self.cfg["IMAGE"]).convert_alpha()
        self.image = pygame.transform.scale(
            self.image, (self.resize_width,
                         self.resize_height))

        self.diagonal = math.sqrt(
            (self.resize_width / 2) ** 2 + (self.resize_height/2) ** 2)

        self.horizontal_size = self.resize_width
        self.vertical_size = self.resize_height

        self.rotated_image = self.image
        self.rect = self.rotated_image.get_rect()
        self.rect.centerx, self.rect.centery = position
        self.angle = 0
        self.speed_up = self.cfg["SPEED_UP"]
        self.min_speed = self.cfg["MIN_SPEED"]
        self.max_speed = self.cfg["MAX_SPEED"]
        self.slow_down = self.cfg["SLOW_DOWN"]
        self.speed = self.min_speed

        self.turn_angle = self.cfg["TURN_ANGLE"]
        self.total_distance_travelled = 0
        self.previous_position = self.rect.center
        self.died = False

        self.timers = {
            "forward": Timer(0.1),
            "left": Timer(0.1),
            "right": Timer(0.1),
            "backward": Timer(0.1),
            "momentum": Timer(0.1)
        }

        self.sensors_directions = {
            "f": utils.subtract(self.rect.midtop, self.rect.center),
            "f_r":  utils.subtract(self.rect.topright, self.rect.center),
            "r":  utils.subtract(self.rect.midright, self.rect.center),
            "f_l":  utils.subtract(self.rect.topleft, self.rect.center),
            "l":  utils.subtract(self.rect.midleft, self.rect.center),
            "b":  utils.subtract(self.rect.midbottom, self.rect.center),
            "b_r":  utils.subtract(self.rect.bottomright, self.rect.center),
            "b_l":  utils.subtract(self.rect.bottomleft, self.rect.center)
        }

        self.turn(angle_change=angle_position, init=True)

        self.sensors_directions = {
            direction: [
                coords[0] / np.linalg.norm(coords),
                coords[1] / np.linalg.norm(coords)]
            for (direction, coords) in self.sensors_directions.items()}

        self.calc_corners()
        self.min_speed_allowed = 20 * 0.6
        self.max_speed_allowed = 20
        self.max_speed_temporal = None
        self.sign_history = [0, 0]

    def rotate_transform(self, angle_change, vector, points=[0, 0]):
        cos_angle = math.cos(math.radians(angle_change))
        sin_angle = math.sin(math.radians(angle_change))
        x = vector[0]
        y = vector[1]
        o_x = points[0]
        o_y = points[1]

        new_x = (x - o_x) * cos_angle - (y - o_y) * sin_angle + o_x
        new_y = (x - o_x) * sin_angle + (y - o_y) * cos_angle + o_y
        return [new_x, new_y]

    def turn(self, angle_change, init=False):
        self.angle += angle_change
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=self.rect.center)

        for direction, vector in self.sensors_directions.items():
            self.sensors_directions[direction] = self.rotate_transform(
                angle_change=angle_change, vector=vector)

        if init:
            return

        self.speed = self.min_speed
        self.translation_transform(self.speed)

        # if angle_change > 0:
        #     print("Turn right: ", self.angle, "with speed: ", self.speed)
        # else:
        #     print("Turn left: ", self.angle, "with speed: ", self.speed)

    def translation_transform(self, amount_speed):
        x_speed = round(self.sensors_directions["f"][0] * amount_speed)
        y_speed = round(self.sensors_directions["f"][1] * amount_speed)
        new_x = self.rect.x + x_speed
        new_y = self.rect.y + y_speed

        screen_width = pygame.display.get_surface().get_width()
        screen_height = pygame.display.get_surface().get_height()

        if new_x >= 0 and new_x + self.rect.width <= screen_width:
            self.rect.x = new_x
        else:
            self.speed = 0

        if new_y >= 0 and new_y + self.rect.height <= screen_height:
            self.rect.y = new_y
        else:
            self.speed = 0

    def maintain_speed(self):
        if self.speed < self.min_speed_allowed:
            self.speed = self.min_speed_allowed

        elif self.speed > self.max_speed_allowed:
            self.speed = self.max_speed_allowed

        self.translation_transform(self.speed)

    def accelerate(self):
        self.speed += self.speed_up
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.translation_transform(self.speed)

    def brake(self):
        self.speed += self.slow_down

        if self.speed < self.min_speed:
            self.speed = self.min_speed

        self.translation_transform(self.speed)

    def update(self, keys_pressed):
        if self.died:
            return

        if not self.timers["forward"].active:
            if keys_pressed[pygame.K_UP]:
                self.accelerate()
                self.timers["forward"].activate()

        if not self.timers["left"].active:
            if keys_pressed[pygame.K_LEFT]:
                self.turn(-self.turn_angle)
                self.timers["left"].activate()

        if not self.timers["right"].active:
            if keys_pressed[pygame.K_RIGHT]:
                self.turn(self.turn_angle)
                self.timers["right"].activate()

        if not self.timers["backward"].active:
            if keys_pressed[pygame.K_DOWN]:
                self.backward()
                self.timers["backward"].activate()

        [self.timers[action].update() for action in self.timers.keys()]
        self.calc_corners()

    def reset(self):
        self.died = True

    def calc_corners(self):
        self.corners = np.array([
            self.sensors_directions["f_l"],
            self.sensors_directions["f_r"],
            self.sensors_directions["b_r"],
            self.sensors_directions["b_l"]])
        self.corners *= self.diagonal
        self.corners += np.array(self.rect.center)

    def calc_sensors(self, boundaries=None):
        sensors = {}

        if boundaries is None or len(boundaries) == 0:
            return sensors

        for (direction, vector) in self.sensors_directions.items():
            if direction not in ["f", "f_r", "r", "f_l", "l"]:
                continue

            intersections = []

            for boundary in boundaries:
                intersection = self.find_intersection(
                    np.array(vector), boundary)
                if intersection is not None:
                    intersections.append(intersection)

            if intersections:
                sensors[direction] = min(
                    intersections,
                    key=lambda p: utils.calc_distance(p, self.rect.center))

        return sensors

    def find_intersection(self, vector, boundary):
        boundary = Polygon(boundary)

        extended_point = Point(
            self.rect.centerx + vector[0] * 10000,
            self.rect.centery + vector[1] * 10000)
        line = LineString([self.rect.center, extended_point])

        intersection = line.intersection(boundary)

        return shapely.get_coordinates(intersection)[1] \
            if shapely.get_coordinates(intersection).size >= 4 else None

    def is_collided(self, boundaries):
        car_boundary = Polygon(self.corners)

        for boundary in boundaries:
            track_boundary = Polygon(boundary)

            intersection = car_boundary.intersection(track_boundary).area
            union = car_boundary.area

            iou = intersection / union

            if iou < 0.95:
                return True

        return False
