import numpy as np
import math
from math import *

class Plane():
    def __init__(self) -> None:
        pass

    def centerCoord(self, matrix):
        center = []
        for i in range(3):
            center = np.append(center, [(matrix[0][i] + matrix[1][i] + matrix[2][i]) / 3])
        center = np.append(center, 1.0)
        return center

    def find_plane_angles(self, point1, point2, point3):
        center = np.array(self.centerCoord(point1, point2, point3))
        points = np.array([point1, point2, point3])

        points = np.subtract(center, points)
        vectors = np.diff(points, axis=0)

        normal = np.cross(vectors[0], vectors[1])

        # Вычисляем нормаль плоскости, используя векторное произведение
        # plane_normal = np.cross(vectors[0], vectors[1])

        # ox_vector = np.array([1, 0, 0])
        # oy_vector = np.array([0, 1, 0])
        # oz_vector = np.array([0, 0, 1])

        # angle_x = np.arccos(np.dot(plane_normal, ox_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(ox_vector)))
        # angle_y = np.arccos(np.dot(plane_normal, oy_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(oy_vector)))
        # angle_z = np.arccos(np.dot(plane_normal, oz_vector) / (np.linalg.norm(plane_normal) * np.linalg.norm(oz_vector)))

        # angles = [angle_y - (pi / 2), angle_x - (pi / 2), angle_z - (pi / 4)]
        # # Нормализуем нормаль плоскости
        normalized_normal = normal / np.linalg.norm(normal)

        # Вычисляем углы наклона плоскости относительно осей
        # angles = np.degrees(np.arccos(normalized_normal))
        angles = np.arccos(normalized_normal)

        # angles[0], angles[1] = -angles[1], angles[0]

        # angles[0] = -angles[0]
        # angles[1] = -angles[1]
        # angles[2] = -angles[2]

        # angles = np.degrees(angles)

        return angles, center
    
    def align_points(self, a, b, c, d):
        # Создание вектора нормали плоскости
        normal_vector = np.array([a, b, c])
        
        # Векторы, направленные вдоль осей координат
        x_axis_vector = np.array([1, 0, 0])
        y_axis_vector = np.array([0, 1, 0])
        z_axis_vector = np.array([0, 0, 1])
        
        # Нахождение углов
        angle_with_x = np.arccos(np.dot(normal_vector, x_axis_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(x_axis_vector)))
        angle_with_y = np.arccos(np.dot(normal_vector, y_axis_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(y_axis_vector)))
        angle_with_z = np.arccos(np.dot(normal_vector, z_axis_vector) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis_vector)))
        print(np.degrees(angle_with_x), np.degrees(angle_with_y), np.degrees(angle_with_z))
        # return np.degrees(angle_with_x), np.degrees(angle_with_y), np.degrees(angle_with_z)
        return angle_with_x, angle_with_y, angle_with_z
    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def equation_of_plane(self, matrix):
        x1, y1, z1 = matrix[0][:3]
        x2, y2, z2 = matrix[1][:3]
        x3, y3, z3 = matrix[2][:3]
        
        A = (y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1)
        B = (z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1)
        C = (x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1)
        D = -A*x1 - B*y1 - C*z1
        
        return np.array([float(A), float(B), float(C), float(D)])
    
    # def angle_between_planes(self, plane1, plane2):
    #     # Нормализация нормалей плоскостей
    #     normal1 = np.array(plane1[:3])
    #     normal1 /= np.linalg.norm(normal1)

    #     normal2 = np.array(plane2[:3])
    #     normal2 /= np.linalg.norm(normal2)

    #     # Вычисление косинуса угла между нормалями плоскостей
    #     cos_angle = np.dot(normal1, normal2)

    #     # Вычисление угла в радианах
    #     rad_angle = np.arccos(cos_angle)

    #     # Перевод угла из радиан в градусы
    #     deg_angle = np.degrees(rad_angle)

    #     return deg_angle
    
    def angle_between_planes(self, plane1, plane2):
        cos_angle = (plane1[0] * plane2[0]) + (plane1[1] * plane2[1]) + (plane1[2] * plane2[2]) / (sqrt((plane1[0]**2 + plane1[1]**2 + plane1[2]**2)) * sqrt((plane2[0]**2 + plane2[1]**2 + plane2[2]**2)))
        print(cos_angle)
        return np.arccos(cos_angle)
    
    def z_rotation(self, vector, angle):
        z_rot = np.array([[cos(angle), -sin(angle), 0, 0],
                        [sin(angle), cos(angle), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        res = z_rot@np.array(vector)
        return res.tolist()

    def y_rotation(self, vector, angle):
        y_rot = np.array([[cos(angle), 0, sin(angle), 0],
                        [0, 1, 0, 0],
                        [-sin(angle), 0, cos(angle), 0],
                        [0, 0, 0, 1]])

        res = y_rot@np.array(vector)
        return res.tolist()
    
    def x_rotation(self, vector, angle):
        x_rot = np.array([[1, 0, 0, 0],
                        [0, cos(angle), -sin(angle), 0],
                        [0, sin(angle), cos(angle), 0],
                        [0, 0, 0, 1]])

        res = x_rot@np.array(vector)
        return res.tolist()
    
    def shift_system_minus(self, vector, offset):
        for i in range(len(vector)):
            vector[i][0] = vector[i][0] - offset[0]
            vector[i][1] = vector[i][1] - offset[1]
            vector[i][2] = vector[i][2] - offset[2]
        return vector
    
    def shift_system_plus(self, vector, offset):
        vec = np.array([[1, 0 , 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [offset[0], offset[1], offset[2], 1]])

        res = np.matmul(vector, vec)
        return res    
    
    def transformation_matrix(self, matrix):
        center = plane.centerCoord(matrix)
        matrix = plane.shift_system_minus(matrix, center)
        # for i in range(len(matrix)):
        #     matrix[i].append(1.0)
        #     matrix[i] = plane.shift_system_minus(matrix[i], center)

        # print(matrix)

        a1, b1, c1, d1 = plane.equation_of_plane(matrix)
        # matrixXY = [[0,0,0], [1,0,0], [0,1,0]]
        # matrixZY = [[0,0,0], [0,0,1], [0,1,0]]
        # matrixXZ = [[0,0,0], [0,0,1], [1,0,0]]

        # a2, b2, c2, d2 = plane.equation_of_plane(matrixXY)
        # a3, b3, c3, d3 = plane.equation_of_plane(matrixZY)
        # a4, b4, c4, d4 = plane.equation_of_plane(matrixXZ)

        # print([a4, b4, c4, d4])

        angleX = np.degrees(self.angle_between_planes([a1, b1, c1, d1], [0.0, 0.0, 1.0, 1.0]))
        # angleY = np.degrees(self.angle_between_planes([a1, b1, c1, d1], [1.0, 0.0, 0.0, 1.0]))
        # angleZ = np.degrees(self.angle_between_planes([a1, b1, c1, d1], [0.0, 1.0, 0.0, 1.0]))

        print("Angle XY:",angleX)
        # print("Angle ZY:",angleY)
        # print("Angle XZ:",angleZ)
    

if __name__ == "__main__":
    # Пример использования функции
    # point1 = [0, 1, 2]
    # point2 = [-0.866, -0.5, 2]
    # point3 = [0.866, -0.5, 2]

    # point1 = [0.0, 1.0, 0.0]
    # point2 = [2.0, -1.0, -2.0]
    # point3 = [-2.0, -1.0, 2.0]

    point1 = [0, 1, 0]
    point2 = [-0.866, -0.5, 0]
    point3 = [0.866, -0.5, 0]

    # point1 = [1, 2, 3]
    # point2 = [4, 5, 6]
    # point3 = [7, 8, 9]

    matrix = [point1, point2, point3]

    A = np.column_stack((point1, point2, np.ones_like(point1)))

    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, point3, rcond=None)

    print("coeff",coefficients)

    # v1 = np.subtract(point2, point1)
    # v2 = np.subtract(point3, point1)

    plane = Plane()
    # angles, center = plane.find_plane_angles(point1, point2, point3)
    # agl = plane.angle_between(v1, [1,0,0])
    # agl1 = plane.angle_between(v2, [0,1,0])
    # print(np.degrees(agl), np.degrees(agl1))
    # print("Угол наклона плоскости относительно Ox: ", np.degrees(angles[0]))
    # print("Угол наклона плоскости относительно Oy: ", np.degrees(angles[1]))
    # print("Угол наклона плоскости относительно Oz: ", np.degrees(angles[2]))


    # pt = np.array([(0, 1, 0), (2, -1, 0), (-2, -1, 0)])
    # a, b, c, d = plane.equation_of_plane(matrix)
    # angles, center = plane.find_plane_angles(point1, point2, point3)
    print(matrix)
    plane.transformation_matrix(matrix)
    # a1, a2, a3 = plane.align_points(a,b,c,d)
    # angleX = plane.angle_between_planes([a,b,c,d], [0.0, 0.0, 1.0, 0.0])
    # angleY = plane.angle_between_planes([a,b,c,d], [0.0, 1.0, 0.0, 0.0])
    # angleZ = plane.angle_between_planes([a,b,c,d], [1.0, 0.0, 0.0, 0.0])
    # print(angleX, angleY, angleZ)