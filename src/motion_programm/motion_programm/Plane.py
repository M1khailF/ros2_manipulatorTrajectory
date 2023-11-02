import numpy as np

class Plane():
    def __init__(self) -> None:
        pass

    def angle_between(self, v1, v2):
        v1_normalized = v1 / np.linalg.norm(v1)
        v2_normalized = v2 / np.linalg.norm(v2)
        
        dot_product = np.dot(v1_normalized, v2_normalized)

        angle_rad = np.arccos(dot_product)
        if (round(v1[1], 1) == 0):
            if v1[0] < 0:
                angle_rad = np.arccos(dot_product)
            else:
                angle_rad = -np.arccos(dot_product)

        if (round(v1[0], 1) == 0):
            if v1[1] < 0:
                angle_rad = -np.arccos(dot_product)
            else:
                angle_rad = np.arccos(dot_product)
        return angle_rad

    def calculate_normal(self, matrix):
        p1 = np.array(matrix[0])
        p2 = np.array(matrix[1])
        p3 = np.array(matrix[2])

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)
        return normal
    
    def centerCoord(self, matrix):
        center = []
        for i in range(3):
            center = np.append(center, [(matrix[0][i] + matrix[1][i] + matrix[2][i]) / 3])
        center = np.append(center, 1.0)
        return center


if __name__ == "__main__":
    # point1 = [0, 1, 0]
    # point2 = [-0.866, -0.5, 0.866]
    # point3 = [0.866, -0.5, -0.866]

    point1 = [0, 1, 0]
    point2 = [-0.866, -0.5, 1.5]
    point3 = [-0.866, 0.0, 1.5]


    point_proverka = [0.001, -0.0, -1.5]

    plane = Plane()

    # normal_vector = plane.calculate_normal(point1, point2, point3)

    # angle_x = plane.angle_between(normal_vector, np.array([1, 0, 0]))
    # angle_y = plane.angle_between(normal_vector, np.array([0, 1, 0]))
    angle_z = plane.angle_between(point_proverka, np.array([0, 0, 1]))

    # print("Угол между нормалью и осью X:", angle_x)
    # print("Угол между нормалью и осью Y:", angle_y)
    print("Угол между нормалью и осью Z:", angle_z)