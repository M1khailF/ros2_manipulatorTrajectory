import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from Plane import Plane


if __name__ == "__main__":
    plane = Plane()

    point1 = [0, 1, 0]
    point2 = [-0.866, -0.5, 1.5]
    point3 = [0.866, -0.5, 1.5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z)
    center = centerCoord(point1, point2, point3)
    print(center)

    print(normal)
    ax.scatter(point1[0], point1[1], point1[2], color='red')
    ax.scatter(point2[0], point2[1], point2[2], color='red')
    ax.scatter(point3[0], point3[1], point3[2], color='red')
    ax.scatter(normal[0], normal[1], normal[2], color='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
