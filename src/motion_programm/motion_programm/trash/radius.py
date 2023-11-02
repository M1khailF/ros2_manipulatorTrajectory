import numpy as np
from scipy.linalg import solve

def find_sphere_center(point1, point2, point3):
    # Преобразование точек в массивы NumPy
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Формирование матрицы A и вектора b для решения линейной системы Ax = b
    A = np.vstack((p2-p1, p3-p1)).T
    b = 0.5 * np.dot(A, p1)

    # Решение системы линейных уравнений для нахождения центра сферы
    center = solve(A, b)

    return center.tolist()

# Пример использования
point1 = [1.0, 2.0, 3.0]
point2 = [4.0, 5.0, 6.0]
point3 = [7.0, 8.0, 9.0]

center = find_sphere_center(point1, point2, point3)
print("Координаты центра сферы:", center)