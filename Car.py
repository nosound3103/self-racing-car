import pygame
import yaml

with open("config.yml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class Car(pygame.sprite.Sprite):
    def __init__(self, cfg=config["Car"], speed=10):
        super().__init__()
        self.cfg = cfg
        self.image = pygame.image.load(self.cfg["IMAGE"]).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.x = 50
        self.rect.y = 50
        self.speed = speed

    def update(self):
        pass

    # def reset(self):
    #     pass

    # def draw(self, screen):
    #     pass

    # def draw_radar(self, screen):
    #     pass

    # def is_collided(self, game_map):
    #     pass

    # def is_alive(self):
    #     pass

    def accelerate(self):
        pass

    # def brake(self):
    #     pass

    # def steer_left(self):
    #     pass

    # def steer_right(self):
    #     pass


# pygame.init()

# screen = pygame.display.set_mode(
#     (config["Env"]["WIDTH"], config["Env"]["HEIGHT"]))
# pygame.display.set_caption("Car Racing")

# car = Car()

# all_car_sprites = pygame.sprite.Group()
# all_car_sprites.add(car)

# running = True
# clock = pygame.time.Clock()

# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     all_car_sprites.update(1)

#     screen.fill((255, 255, 255))
#     all_car_sprites.draw(screen)
#     pygame.display.flip()
#     clock.tick(60)

# pygame.quit()
