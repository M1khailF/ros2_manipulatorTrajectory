import numpy as np
import math
from math import *

class Plane():
    def __init__(self) -> None:
        pass

    def centerCoord(self, point1, point2, point3):
        center = []
        for i in range(3):
            center = np.append(center, [(point1[i] + point2[i] + point3[i]) / 3])

        return center

    def find_plane_angles(self, point1, point2, point3):
        center = np.array(self.centerCoord(point1, point2, point3))
        points = np.array([point1, point2, point3])

        points = np.subtract(center, points)
        vectors = np.diff(points, axis=0)

        # Вычисляем нормаль плоскости, используя векторное произведение
        plane_normal = np.cross(vectors[0], vectors[1])

        ox_vector = np.array([1, 0, 0])
        oy_vector = np.array([0, 1, 0])
        oz_vector = np.array([0, 0, 1])

        angle_x = np.arccos(np.dot(plane_normal, ox_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(ox_vector)))
        angle_y = np.arccos(np.dot(plane_normal, oy_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(oy_vector)))
        angle_z = np.arccos(np.dot(plane_normal, oz_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(oz_vector)))

        angles = [angle_y - (pi / 2), angle_x - (pi / 2), angle_z - (pi / 4)]
        # # Нормализуем нормаль плоскости
        # normalized_normal = normal / np.linalg.norm(normal)

        # # Вычисляем углы наклона плоскости относительно осей
        # # angles = np.degrees(np.arccos(normalized_normal))
        # angles = np.arccos(normalized_normal)

        # # angles[0], angles[2] = angles[2], angles[0]

        # angles[0] = angles[0] - (pi / 2)
        # angles[1] = angles[1] - (pi / 2)
        # angles[2] = angles[2] - (pi)

        angles = np.degrees(angles)

        return angles, center

if __name__ == "__main__":
    # Пример использования функции
    point1 = [0, 1, 2]
    point2 = [-2, -1, 0]
    point3 = [2, -1, 0]
    plane = Plane()
    angles, center = plane.find_plane_angles(point1, point2, point3)
    print("Угол наклона плоскости относительно Ox: ", angles[0])
    print("Угол наклона плоскости относительно Oy: ", angles[1])
    print("Угол наклона плоскости относительно Oz: ", angles[2])