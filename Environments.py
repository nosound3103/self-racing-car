import pygame
import yaml
import random
import cv2

from Car import Car

random.seed(0)

with open("config.yml", "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


class Environment:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode(
            (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))
        self.map = cv2.imread("images/test_map_2.png")
        self.map = cv2.resize(
            self.map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]),
            interpolation=cv2.INTER_AREA)

        self.game_map = pygame.image.load("images/test_map_2.png").convert()
        self.game_map = pygame.transform.scale(
            self.game_map, (cfg["Env"]["WIDTH"], cfg["Env"]["HEIGHT"]))

        pygame.display.set_caption("Car Racing")

        self.all_car_sprites = pygame.sprite.Group()
        for _ in range(cfg["Env"]["NUM_CARS"]):
            self.all_car_sprites.add(Car(speed=random.randint(1, 6)))

        self.clock = pygame.time.Clock()

    def draw_background(self):
        self.screen.blit(self.game_map, (0, 0))

    def find_contours(self):
        gray = cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours[1] if len(contours) == 2 else contours[0] \
            if len(contours) == 1 else None

    def run(self):
        boundary = self.find_contours().reshape(-1, 2)

        running = True
        while running:
            self.draw_background()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            for car in self.all_car_sprites:
                # if car.is_collided(boundary):
                #     continue
                car.update(keys)
                self.screen.blit(car.rotated_image, car.rect)

                # pygame.draw.polygon(self.screen, (255, 0, 0), car.corners, 1)
                sensors = car.calc_sensors(boundary=boundary)
                if len(sensors):
                    for direction, vector in sensors.items():
                        try:
                            pygame.draw.line(self.screen, (0, 255, 0),
                                             car.rect.center, vector, 1)
                        except:
                            continue

            pygame.draw.polygon(self.screen, (0, 0, 255), boundary, 1)
            pygame.display.update()
            pygame.display.flip()
            self.clock.tick(cfg["Env"]["FPS"])

        pygame.quit()

    def segment(self):
        pass


env = Environment()
env.run()
