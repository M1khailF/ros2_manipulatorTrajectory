import time
import numpy as np
from math import *
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # Коэффициент пропорциональной части
        self.Ki = Ki  # Коэффициент интегральной части
        self.Kd = Kd  # Коэффициент дифференциальной части
        
        self.last_error = 0
        self.integral = 0
        self.dt = 1000


    def update(self, feedback, setpoint):
        error = setpoint - feedback

        proportional = self.Kp * error

        # Интегральная часть
        self.integral += error * self.dt
        integral = self.Ki * self.integral

        # Дифференциальная часть
        derivative = self.Kd * ((error - self.last_error) / self.dt)
        self.last_error = error

        # Вычисление выходного сигнала
        output = proportional + integral + derivative

        return output

    def calcAnlgeOffset(inputStart, inputEnd):
        distance = dist(inputStart, inputEnd) / 100

        if inputStart < inputEnd:
            return distance
        else:
            return -distance

if __name__ == "__main__":
    controller = PIDController(0.000001, 0.0, 0.0)
    setpoint = 1.54
    feedback = 0
    # dt = 0.1
    error = setpoint - feedback
    
    while True:
        # if not np.around(feedback, 2) == np.around(setpoint, 2) or output == None:
            # output = controller.update(error)
            # error = feedback - output
            # print(output, error)
            # time.sleep(0.5)

            output = controller.update(feedback, setpoint)
            feedback += output
            print(feedback)
            # time.sleep(0.1)
        # else:
        #     break