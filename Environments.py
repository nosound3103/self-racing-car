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

        pygame.display.set_caption("Car Racing")

        self.all_car_sprites = pygame.sprite.Group()
        for _ in range(cfg["Env"]["NUM_CARS"]):
            self.all_car_sprites.add(Car(speed=random.randint(1, 6)))

        self.clock = pygame.time.Clock()

    def draw_background(self):
        self.screen.fill((255, 255, 255))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                pass
            if keys[pygame.K_DOWN]:
                pass

            self.draw_background()
            self.all_car_sprites.update()
            self.all_car_sprites.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(cfg["Env"]["FPS"])

        pygame.quit()
