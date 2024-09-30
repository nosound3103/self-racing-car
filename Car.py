import pygame
import yaml
import math
import numpy as np

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class Car(pygame.sprite.Sprite):
    def __init__(self, cfg=config["Car"], speed=3):
        super().__init__()
        self.cfg = cfg
        self.image = pygame.image.load(self.cfg["IMAGE"]).convert_alpha()
        self.rotated_image = self.image
        self.rect = self.rotated_image.get_rect()
        self.rect.x = 50
        self.rect.y = 50
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
        }

        self.sensors_directions = {direction: [coords[0] / np.linalg.norm(coords), coords[1] / np.linalg.norm(
            coords)] for (direction, coords) in self.sensors_directions.items()}

    def turn(self, angle_change):
        self.angle += angle_change
        self.rotated_image = pygame.transform.rotate(self.image, -self.angle)
        self.rect = self.rotated_image.get_rect(center=self.rect.center)

        cos_angle = math.cos(math.radians(angle_change))
        sin_angle = math.sin(math.radians(angle_change))

        for direction, vector in self.sensors_directions.items():
            x = vector[0]
            y = vector[1]
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            self.sensors_directions[direction] = [new_x, new_y]

        if angle_change > 0:
            print("Turn right: ", self.angle, "with speed: ", self.speed)
        else:
            print("Turn left: ", self.angle, "with speed: ", self.speed)

    def accelerate(self):
        self.speed += self.speed_up
        if self.speed > self.max_speed:
            self.speed = self.max_speed

        self.rect = self.rect.move(
            self.sensors_directions["front"][0] * self.speed,
            self.sensors_directions["front"][1] * self.speed)

        print("Speed up: ", self.speed)

    def brake(self):
        self.speed += self.slow_down
        if self.speed < self.min_speed:
            self.speed = self.min_speed

        slow_down_speed = self.slow_down if self.speed > self.min_speed \
            else self.speed

        self.rect = self.rect.move(
            self.sensors_directions["front"][0] * slow_down_speed,
            self.sensors_directions["front"][1] * slow_down_speed)
        print("Slow down:", self.speed)

    def update(self, keys_pressed):
        if keys_pressed[pygame.K_RIGHT]:
            self.turn(5)
        if keys_pressed[pygame.K_LEFT]:
            self.turn(-5)

        if keys_pressed[pygame.K_UP]:
            self.accelerate()

        if keys_pressed[pygame.K_DOWN]:
            self.brake()

    def reset(self):
        pass

    def draw(self, screen):
        pass

    def calc_sensors(self):
        return {direction: [coord * 100 for coord in vector]
                for (direction, vector) in self.sensors_directions.items()}

    def is_collided(self, game_map):
        pass

    def is_alive(self):
        pass


pygame.init()

screen = pygame.display.set_mode(
    (config["Env"]["WIDTH"], config["Env"]["HEIGHT"]))
pygame.display.set_caption("Car Racing")

game_map = pygame.image.load("ref/map.png").convert()
game_map = pygame.transform.scale(
    game_map, (config["Env"]["WIDTH"], config["Env"]["HEIGHT"]))

car = Car(speed=2)

running = True
clock = pygame.time.Clock()
angle = 0

while running:
    screen.blit(game_map, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    car.update(keys)

    screen.blit(car.rotated_image, car.rect)
    sensors = car.calc_sensors()
    for direction, vector in sensors.items():
        position = (car.rect.center[0] + vector[0],
                    car.rect.center[1] + vector[1])
        pygame.draw.line(screen, (0, 255, 0), car.rect.center, position, 1)

    pygame.display.update()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
