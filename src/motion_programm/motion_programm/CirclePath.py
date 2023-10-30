import numpy as np
from math import *

class Circle3Points:
    def __init__(self):
        pass

    def calculate(self, inputMatrix):
        
        d1 = inputMatrix[0][0]**2 + inputMatrix[0][1]**2 + inputMatrix[0][2]**2 - (inputMatrix[1][0]**2 + inputMatrix[1][1]**2 + inputMatrix[1][2]**2)
        d2 = inputMatrix[0][0]**2 + inputMatrix[0][1]**2 + inputMatrix[0][2]**2 - (inputMatrix[2][0]**2 + inputMatrix[2][1]**2 + inputMatrix[2][2]**2)
        d3 = inputMatrix[1][0]**2 + inputMatrix[1][1]**2 + inputMatrix[1][2]**2 - (inputMatrix[2][0]**2 + inputMatrix[2][1]**2 + inputMatrix[2][2]**2)

        A = np.array([[(inputMatrix[0][0] - inputMatrix[1][0]), (inputMatrix[0][1] - inputMatrix[1][1]), (inputMatrix[0][2] - inputMatrix[1][2])],
                        [(inputMatrix[0][0] - inputMatrix[2][0]), (inputMatrix[0][1] - inputMatrix[2][1]), (inputMatrix[0][2] - inputMatrix[2][2])],
                        [(inputMatrix[1][0] - inputMatrix[2][0]), (inputMatrix[1][1] - inputMatrix[2][1]), (inputMatrix[1][2] - inputMatrix[2][2])]])
        
        B = np.array([[d1/2],
                        [d2/2],
                        [d3/2]])

        x, residuals, rank, singular_values = np.linalg.lstsq(A, B, rcond=None)
        print(f"Промежуточная матрица {A}")
        print(f"Промежуточные выходы {B}")
        # print(round(x[0][0], 2), round(x[1][0], 2), round(x[2][0], 2))
        radious = sqrt((inputMatrix[0][0] - x[0][0])**2 + (inputMatrix[0][1] - x[1][0])**2 + (inputMatrix[0][2] - x[2][0])**2)
        return x, radious
    
    def angles(self, inputMatrix):
        # Координаты трех точек
        point1 = np.array([inputMatrix[0][0], inputMatrix[0][1], inputMatrix[0][2]])
        point2 = np.array([inputMatrix[1][0], inputMatrix[1][1], inputMatrix[1][2]])
        point3 = np.array([inputMatrix[2][0], inputMatrix[2][1], inputMatrix[2][2]])

        vector1 = point2 - point1
        vector2 = point3 - point1

        normal_vector = np.cross(vector1, vector2)
        normalized_normal = normal_vector / np.linalg.norm(normal_vector)

        normalized_origin_x = np.array([1, 0, 0])
        normalized_origin_y = np.array([0, 1, 0])
        normalized_origin_z = np.array([0, 0, 1])

        alpha = np.arccos(np.dot(normalized_origin_x, normalized_normal)) - pi/2
        beta = np.arccos(np.dot(normalized_origin_y, normalized_normal)) - pi/2
        gamma = np.arccos(np.dot(normalized_origin_z, normalized_normal))

        # return [np.degrees(alpha), np.degrees(beta), np.degrees(gamma)]
        return [alpha, beta, gamma]


if __name__ == "__main__":
    inputMat = np.array([[2.0, 1.0, 0.0],
                    [2.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0]])
    
    print("Входная матрица", inputMat)
    
    c = Circle3Points()
    centre, r = c.calculate(inputMat)
    angles = c.angles(inputMat)
    print("Круг", centre, r)
    print(angles)