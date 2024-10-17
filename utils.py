import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

from shapely.geometry import Point, LineString


def subtract(arr1, arr2):
    return [a - b for a, b in zip(arr1, arr2)]


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[1] if len(contours) == 2 else contours[0] \
        if len(contours) == 1 else None


def find_middle_line(road):
    gray = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    skeleton = morphology.skeletonize(binary // 255)

    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    contours, _ = cv2.findContours(
        skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0] if contours else []


def resize_points(points, old_size, new_size):
    width_ratio = new_size[0] / old_size[0]
    height_ratio = new_size[1] / old_size[1]

    return np.array([[int(point[0] * width_ratio), int(point[1] * height_ratio)] for point in points])


def calc_distance(p1, p2):
    try:
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    except Exception:
        dist = -1

    return dist


def calc_distance_to_middle(car, middle_line):
    car_position = Point(car.rect.center)
    distance = car_position.distance(middle_line)
    return distance
