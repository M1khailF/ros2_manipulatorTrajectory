import numpy as np
from math import *

class Circle3Points:
    def __init__(self):
        pass

    def circle_from_points(self, matrix):
        x1, y1 = matrix[0][0], matrix[0][1]
        x2, y2 = matrix[1][0], matrix[1][1]
        x3, y3 = matrix[2][0], matrix[2][1]

        x = ((y2 - y1) * (y3**2 - y1**2 + x3**2 - x1**2) - (y3 - y1) * (y2**2 - y1**2 + x2**2 - x1**2)) / (2 * ((x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1)))
        y = ((x2 - x1) * (x3**2 - x1**2 + y3**2 - y1**2) - (x3 - x1) * (x2**2 - x1**2 + y2**2 - y1**2)) / (2 * ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1)))
        
        radius = ((x1 - x)**2 + (y1 - y)**2) ** 0.5
        
        return [round(x, 2), round(y, 2)], round(radius, 2)

if __name__ == "__main__":
    inputMat = np.array([[0.0, 1.0, 0.0],
                    [1.866, -0.5, 1.0],
                    [-0.866, -0.5, 1.0]])
    c = Circle3Points()
    c, r = c.circle_from_points(inputMat)
    print(c,r)
