import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, PoseArray, Pose
import random

class TF2Publisher(Node):
    def __init__(self):
        super().__init__('tf2_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.x = []
        self.y = []
        self.z = []

        self.countPoints = 3
        # for i in range(self.countPoints):
        #     self.x.append(random.uniform(-0.5, 1.9))
        #     self.y.append(random.uniform(-0.5, 1.9))
        #     self.z.append(random.uniform(-0.5, 1.9))
        for i in range(self.countPoints):
            self.x.append(random.randint(0, 2))
            self.y.append(random.randint(0, 2))
            self.z.append(random.randint(0, 2))

        # for i in range(self.countPoints):
        #     if i == 0:
        #         self.x.append(0)
        #         self.y.append(1)
        #         self.z.append(2)

        #     if i == 1:
        #         self.x.append(-0.866)
        #         self.y.append(-0.5)
        #         self.z.append(1)

        #     if i == 2:
        #         self.x.append(0.866)
        #         self.y.append(-0.5)
        #         self.z.append(1)

        self.timer = self.create_timer(1, self.publish_tf)
        self.publisher = self.create_publisher(PoseArray, 'point_topic', 10)
        

    def publish_tf(self):
        # Create a transform message
        # self.x, self.y, self.z = random.uniform(-0.5, 1.9), random.uniform(-0.5, 1.9), random.uniform(-0.5, 1.9)
        # print(self.x, self.y, self.z)
        self.points = PoseArray()
        self.points.header.frame_id = "world"
        self.points.header.stamp = self.get_clock().now().to_msg()
        # self.points.append(self.tf_pose("pt2", self.x, self.y, self.z))
        for i in range(self.countPoints):
            self.points.poses.append(self.publish_pose_array([self.x[i], self.y[i], self.z[i]], [0.0, 0.0, 0.0, 1.0]))
            print([self.x[i], self.y[i], self.z[i]])

        # self.tf_broadcaster.sendTransform(self.points)
        # for i in range(len(self.points)):
        
        self.publisher.publish(self.points)
        # self.tf_broadcaster.sendTransform(self.points)

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

def main(args=None):
    rclpy.init(args=args)
    tf2_publisher = TF2Publisher()
    rclpy.spin_once(tf2_publisher)
    tf2_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
