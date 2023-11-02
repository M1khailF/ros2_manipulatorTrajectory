import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point, Vector3, PoseArray, Quaternion, Pose, PoseStamped
from nav_msgs.msg import Path
# from sub import MinimalSubscriber
from visualization_msgs.msg import Marker
import random
from math import *
import numpy as np
import tf_transformations
from PID_test import PIDController
from CirclePath import Circle3Points
from Plane import Plane
# from radius import FindRadius


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('main_program')
        self.mode = 3
        self.tf_broadcaster = TransformBroadcaster(self)
        # self.x, self.y, self.z = 3.02, 1.20, 2.95
        # self.pointEnd = np.array([0,0,0])
        self.pointMove = []
        self.numPoint = 0
        # self.x, self.y, self.z = 1.0, 2.0, 0.0
        self.x, self.y, self.z = random.uniform(-0.5, 1.9), random.uniform(-0.5, 1.9), random.uniform(-0.5, 1.9)

        self.speed = 0
        self.maxSpeed = 0.005
        self.stepSpeed = 0.2 / 10000

        # self.msgPoint = False

        self.pointManipulator = np.array([self.x, self.y, self.z, 1.0])

        kp = 0.02

        self.controllerX = PIDController(kp, 0.0, 0.0)
        self.controllerY = PIDController(kp, 0.0, 0.0)
        self.controllerZ = PIDController(kp, 0.0, 0.0)
        # self.controllerRad = FindRadius(1.0, 10)

        self.circlePath = Circle3Points()

        self.publisher = self.create_publisher(TransformStamped, 'tf2_topic1', 10)
        self.publisher_path = self.create_publisher(Path, 'path_topic', 10)
        self.publisher_path_circle = self.create_publisher(Path, 'path_circle', 10)
        # self.publisherVector = self.create_publisher(Marker, "vector_marker", 10)
        self.publisherVector = self.create_publisher(PoseArray, 'vector_manipulator', 10)
        self.timer = self.create_timer(0.01, self.publish_tf)

        self.subscription = self.create_subscription(PoseArray, 'point_topic', self.listener_callback,10)
        

        
        self.subscription 

    def get_quaternion_from_euler(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
    
    def z_rotation(self, vector, angle):
        z_rot = np.array([[cos(angle), -sin(angle), 0],
                        [sin(angle), cos(angle), 0],
                        [0, 0, 1]])

        res = z_rot@np.array(vector)
        return res.tolist()

    def y_rotation(self, vector, angle):
        y_rot = np.array([[cos(angle), 0, sin(angle)],
                        [0, 1, 0],
                        [-sin(angle), 0, cos(angle)]])

        res = y_rot@np.array(vector)
        return res.tolist()
    
    def x_rotation(self, vector, angle):
        x_rot = np.array([[1, 0, 0],
                        [0, cos(angle), -sin(angle)],
                        [0, sin(angle), cos(angle)]])

        res = x_rot@np.array(vector)
        return res.tolist()
    
    def shift_system_minus(self, vector, offset):
        for i in range(len(vector)):
            vector[i][0] = vector[i][0] - offset[0]
            vector[i][1] = vector[i][1] - offset[1]
            vector[i][2] = vector[i][2] - offset[2]
        return vector
    
    def shift_system_plus(self, vector, offset):
        for i in range(len(vector)):
            vector[i][0] = vector[i][0] + offset[0]
            vector[i][1] = vector[i][1] + offset[1]
            vector[i][2] = vector[i][2] + offset[2]
        return vector
    
    # def shift_system_minus(self, vector, offset):
    #     vec = np.array([[1, 0 , 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [-offset[0], -offset[1], -offset[2], 1]])

    #     res = np.matmul(vector, vec)
    #     return res
    
    # def shift_system_plus(self, vector, offset):
    #     vec = np.array([[1, 0 , 0, 0],
    #                     [0, 1, 0, 0],
    #                     [0, 0, 1, 0],
    #                     [offset[0], offset[1], offset[2], 1]])

    #     res = np.matmul(vector, vec)
    #     return res
    
    def listener_callback(self, msg):
        self.pointMove = []
        self.angelsMove = []
        self.distMove = []
        self.countPoint = len(msg.poses)
        self.speed = 0
        self.numPoint = 0
        # self.transform = msg.poses
        for i in range(self.countPoint):
            self.transform = msg.poses[i]
            self.point = [self.transform.position.x, self.transform.position.y, self.transform.position.z, 1.0]
            self.pointMove.append(self.point)
            print(self.point)
            if i == 0:
                self.distMove.append(dist(self.point, self.pointManipulator))
                self.oldPoint = self.point

                self.theta, self.phi = self.find_direction_angles(self.pointManipulator, self.pointMove[i])
                self.phi = self.phi - (pi / 2)

                self.angelsMove.append([0, self.phi, self.theta])

            else:
                self.distMove.append(dist(self.point, self.oldPoint))
                self.oldPoint = self.point

                self.theta, self.phi = self.find_direction_angles(self.pointMove[i - 1], self.pointMove[i])
                self.phi = self.phi - (pi / 2)

                self.angelsMove.append([0, self.phi, self.theta])

        self.localAngle = [self.angelsMove[0][0], self.angelsMove[0][1], self.angelsMove[0][2]]

    def publish_pose_array(self, poses, q):
        pose = Pose()
        pose.position.x = float(poses[0])
        pose.position.y = float(poses[1])
        pose.position.z = float(poses[2])

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

    def find_direction_angles(self, vector, point):
        target_vector_x = point[0] - vector[0]
        target_vector_y = point[1] - vector[1]
        target_vector_z = point[2] - vector[2]

        length = sqrt(target_vector_x**2 + target_vector_y**2 + target_vector_z**2)

        theta = atan2(target_vector_y, target_vector_x)
        phi = acos(target_vector_z / length)

        return theta, phi
    
    def publish_tf(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"
        pose_array.header.stamp = self.get_clock().now().to_msg()

        points = []
        points.append(self.tf_pose("pt1", self.x, self.y, self.z))
        points.append(self.tf_pose("pt2", self.x, self.y + 1, self.z))

        if len(self.pointMove) > 0:
            self.pointManipulator = np.array([self.x, self.y, self.z, 1.0])
            self.get_logger().info(f"{dist(self.pointMove[self.numPoint], self.pointManipulator)}")

            # if (self.mode == 1):
            #     for i in range(self.countPoint - 1):
            #         if i == 0:
            #             self.theta, self.phi = self.find_direction_angles(self.pointManipulator[0], self.pointManipulator[1], self.pointManipulator[2], self.pointMove[i][0], self.pointMove[i][1], self.pointMove[i][2])
            #         else:
            #             self.theta, self.phi = self.find_direction_angles(self.pointMove[i][0], self.pointMove[i][1], self.pointMove[i][2], self.pointMove[i+1][0], self.pointMove[i+1][1], self.pointMove[i+1][2])
            
            #         self.phi = self.phi - (pi / 2)
            #         # self.angelsMove.append([0, self.phi, self.theta])
            #         self.q = self.get_quaternion_from_euler(0, self.phi, self.theta)
            #         self.angelsMove.append(self.q)
            # # else:
            # #     self.theta, self.phi = self.find_direction_angles(self.pointManipulator[0], self.pointManipulator[1], self.pointManipulator[2], self.pointMove[self.numPoint][0], self.pointMove[self.numPoint][1], self.pointMove[self.numPoint][2])
            # #     self.phi = self.phi - (pi / 2)
            # #     print(self.phi, self.theta)

            # self.theta, self.phi = self.find_direction_angles(self.pointManipulator[0], self.pointManipulator[1], self.pointManipulator[2], self.pointMove[self.numPoint][0], self.pointMove[self.numPoint][1], self.pointMove[self.numPoint][2])
            # self.phi = self.phi - (pi / 2)

            # self.q = self.get_quaternion_from_euler(self.angelsMove[self.numPoint][0], self.angelsMove[self.numPoint][1], self.angelsMove[self.numPoint][2])
            if self.numPoint < self.countPoint:
                if self.mode == 1:
                    if not dist(self.pointMove[self.numPoint], self.pointManipulator) < 0.001:
                            self.q = self.get_quaternion_from_euler(self.angelsMove[self.numPoint][0], self.angelsMove[self.numPoint][1], self.angelsMove[self.numPoint][2])
                            
                            if not dist(self.pointMove[self.numPoint], self.pointManipulator) <= (self.distMove[self.numPoint] / 2) - 0.01:
                                # if self.speed < self.maxSpeed:
                                    self.speed += self.stepSpeed
                            else:
                                # if self.speed > 0:
                                    self.speed -= self.stepSpeed 

                            print(self.speed)
                            print("Distance", self.distMove)

                            self.move = [self.speed, 0, 0, 1]
                            self.move = self.y_rotation(self.move, self.angelsMove[self.numPoint][1])
                            self.move = self.z_rotation(self.move, self.angelsMove[self.numPoint][2])
                            self.move = self.shift_system_plus(self.move, self.pointManipulator)

                            self.x = self.move[0]
                            self.y = self.move[1]
                            self.z = self.move[2]

                            pose_array.poses.append(self.publish_pose_array(self.pointManipulator, self.q))

                            for i in range(self.countPoint):
                                print(self.pointMove[i])

                    else:
                        if self.numPoint + 1 < self.countPoint:
                            self.numPoint += 1
                            self.speed = 0

                elif self.mode == 2:
                    if not dist(self.pointMove[self.numPoint], self.pointManipulator) < 0.01: 
                        self.theta, self.phi = self.find_direction_angles(self.pointManipulator, self.pointMove[self.numPoint])
                        self.phi = self.phi - (pi / 2)

                        # for i in range(3):
                        #     if self.localAngle[i] > pi * 2:
                        #         self.localAngle[i] -= pi * 2
                        #     elif self.localAngle[i] < -pi * 2:
                        #         self.localAngle[i] += -pi * 2


                        output1 = self.controllerX.update(self.localAngle[0], 0)
                        output2 = self.controllerY.update(self.localAngle[1], self.phi)
                        output3 = self.controllerZ.update(self.localAngle[2], self.theta)
                        self.localAngle[0] += output1
                        self.localAngle[1] += output2
                        self.localAngle[2] += output3

                        print("AnglesMove:", self.localAngle)
                        print("AnglesPoint:", self.angelsMove[self.numPoint])
                        print(self.numPoint)

                        # self.q = self.get_quaternion_from_euler(self.angelsMove[self.numPoint][0], self.angelsMove[self.numPoint][1], self.angelsMove[self.numPoint][2])
                        self.q = self.get_quaternion_from_euler(self.localAngle[0], self.localAngle[1], self.localAngle[2])
                        
                        if self.numPoint == self.countPoint - 1:
                            if not dist(self.pointMove[self.numPoint], self.pointManipulator) <= (self.distMove[self.numPoint] / 2) - 0.01:
                                if self.speed < self.maxSpeed:
                                    self.speed += self.stepSpeed
                            else:
                                if self.speed > 0:
                                    self.speed -= self.stepSpeed 
                        else:
                            if self.speed < self.maxSpeed:
                                self.speed += self.stepSpeed


                        # if not dist(self.pointMove[self.countPoint - 1], self.pointManipulator) <= (self.distMove[self.countPoint - 1] / 2) - 0.01:
                        #     if self.speed < self.maxSpeed:
                        #         self.speed += self.stepSpeed
                        # else:
                        #     if self.speed > 0:
                        #         self.speed -= self.stepSpeed 

                        print(self.speed)
                        print("Distance", self.distMove)

                        self.move = [self.speed, 0, 0, 1]
                        self.move = self.y_rotation(self.move, self.localAngle[1])
                        self.move = self.z_rotation(self.move, self.localAngle[2])
                        self.move = self.shift_system_plus(self.move, self.pointManipulator)

                        self.x = self.move[0]
                        self.y = self.move[1]
                        self.z = self.move[2]

                        pose_array.poses.append(self.publish_pose_array(self.pointManipulator, self.q))

                    else:
                        if self.numPoint + 1 < self.countPoint:
                            self.numPoint += 1

                elif self.mode == 3:
                    if self.countPoint == 3:
                        matrix = [[self.pointMove[0][0], self.pointMove[0][1], self.pointMove[0][2]],[self.pointMove[1][0], self.pointMove[1][1], self.pointMove[1][2]], [self.pointMove[2][0], self.pointMove[2][1], self.pointMove[2][2]]]
                        plane = Plane()
                    
                        self.center3Points = plane.centerCoord(matrix)
                        matrix = self.shift_system_minus(matrix, self.center3Points)

                        normal_vector = plane.calculate_normal(matrix)
                        projection_oyz = [normal_vector[0], 0.0, normal_vector[2]]
                        self.angle_oyz = plane.angle_between(projection_oyz, np.array([0, 0, 1]))

                        for i in range(len(matrix)):
                            matrix[i] = self.y_rotation(matrix[i], self.angle_oyz)
                            
                        normal_vector = plane.calculate_normal(matrix)
                        self.angle_oxz = plane.angle_between(normal_vector, np.array([0, 0, 1]))

                        for i in range(len(matrix)):
                            matrix[i] = self.x_rotation(matrix[i], self.angle_oxz)
                            points.append(self.tf_pose_with_angles(f"newpt_{i}", matrix[i][0], matrix[i][1], matrix[i][2], [0,0,0]))

                        normal_vector = plane.calculate_normal(matrix)

                        cCircle, radious = self.circlePath.circle_from_points(matrix)

                        self.matrixCircle = []
                        for i in range(36):
                            pointCircle = [cCircle[0] + (radious*cos(np.radians(i * 10))), cCircle[1] + (radious*sin(np.radians(i * 10))), 0.0]
                            # print(pointCircle)
                            pointCircle = self.x_rotation(pointCircle, -self.angle_oxz)
                            pointCircle = self.y_rotation(pointCircle, -self.angle_oyz)
                            self.matrixCircle.append(pointCircle)
                        
                        self.matrixCircle = self.shift_system_plus(self.matrixCircle, self.center3Points)

                        self.CirclePath(self.matrixCircle)
                        points.append(self.tf_pose_with_angles(f"normal_plane_ROT", normal_vector[0], normal_vector[1], normal_vector[2], [0,0,0]))
                        points.append(self.tf_pose_with_angles("circle", self.center3Points[0], self.center3Points[1], self.center3Points[2], [0,0,0]))
                        self.publisher.publish(points[0])
            else:
                self.speed = 0

            self.publisherVector.publish(pose_array)
            self.publish_path()                    

        self.tf_broadcaster.sendTransform(points)
        for i in range(len(points)):
            self.publisher.publish(points[i])

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        point1 = PoseStamped()
        point1.pose.position.x = self.x
        point1.pose.position.y = self.y
        point1.pose.position.z = self.z
        point1.pose.orientation.x = 0.0
        point1.pose.orientation.y = 0.0
        point1.pose.orientation.z = 0.0
        point1.pose.orientation.w = 1.0
        path_msg.poses.append(point1)

        point2 = PoseStamped()
        point2.pose.position.x = self.pointMove[self.numPoint][0]
        point2.pose.position.y = self.pointMove[self.numPoint][1]
        point2.pose.position.z = self.pointMove[self.numPoint][2]
        point2.pose.orientation.x = 0.0
        point2.pose.orientation.y = 0.0
        point2.pose.orientation.z = 0.0
        point2.pose.orientation.w = 1.0
        path_msg.poses.append(point2)

        self.publisher_path.publish(path_msg)
        self.get_logger().info('Published path')

    def CirclePath(self, matrix):
        path_msg = Path()
        path_msg.header.frame_id = 'world'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for i in range(len(matrix) - 1):
            point1 = PoseStamped()
            point1.pose.position.x = matrix[i][0]
            point1.pose.position.y = matrix[i][1]
            point1.pose.position.z = matrix[i][2]
            point1.pose.orientation.x = 0.0
            point1.pose.orientation.y = 0.0
            point1.pose.orientation.z = 0.0
            point1.pose.orientation.w = 1.0
            path_msg.poses.append(point1)

            point2 = PoseStamped()
            point2.pose.position.x = matrix[i + 1][0]
            point2.pose.position.y = matrix[i + 1][1]
            point2.pose.position.z = matrix[i + 1][2]
            point2.pose.orientation.x = 0.0
            point2.pose.orientation.y = 0.0
            point2.pose.orientation.z = 0.0
            point2.pose.orientation.w = 1.0
            path_msg.poses.append(point2)

            self.publisher_path_circle.publish(path_msg)

    def tf_pose(self, name, x, y, z):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = name
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0

        return tf_msg
    
    def tf_pose_with_angles(self, name, x, y, z, angles):
        q = self.get_quaternion_from_euler(angles[0], angles[1], angles[2])
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = name
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]

        return tf_msg

class TF2Publisher(Node):
    def __init__(self):
        super().__init__('tf2_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.x, self.y, self.z = 3.02, 1.20, 2.95
        self.publisher = self.create_publisher(TransformStamped, 'tf2_topic1', 10)
        self.timer = self.create_timer(0.1, self.publish_tf)

    def publish_tf(self):
        points = []
        points.append(self.tf_pose("pt1", self.x, self.y, self.z))
        self.tf_broadcaster.sendTransform(points)
        self.publisher.publish(points[0])

    def tf_pose(self, name, x, y, z):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = name
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0

        return tf_msg

def main(args=None):
    rclpy.init(args=args)

    # tf2_publisher = TF2Publisher()
    # rclpy.spin(tf2_publisher)
    # tf2_publisher.destroy_node()

    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
