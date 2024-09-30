import pygame
import yaml
import random
from Car import Car

random.seed(0)

with open("config.yml", "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


class Environment:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))
        self.game_map = pygame.image.load("ref/map.png").convert()
        self.game_map = pygame.transform.scale(
            self.game_map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        pygame.display.set_caption("Car Racing")

        self.all_car_sprites = pygame.sprite.Group()
        for _ in range(cfg["Env"]["NUM_CARS"]):
            self.all_car_sprites.add(Car(speed=random.randint(1, 6)))

        self.clock = pygame.time.Clock()

    def draw_background(self):
        self.screen.blit(self.game_map, (0, 0))

    def run(self):
        running = True
        while running:
            self.draw_background()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            for car in self.all_car_sprites:
                car.update(keys)
                self.screen.blit(car.rotated_image, car.rect)

            pygame.display.flip()
            pygame.display.update()
            self.clock.tick(cfg["Env"]["FPS"])

        pygame.quit()


env = Environment()
env.run()
