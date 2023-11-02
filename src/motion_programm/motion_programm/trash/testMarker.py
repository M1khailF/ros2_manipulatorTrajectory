import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class PoseArrayPublisher(Node):
    def __init__(self):
        super().__init__('pose_array_publisher')
        self.publisher_ = self.create_publisher(PoseArray, 'pose_array', 10)
        self.timer = self.create_timer(0.1, self.publish_pose_array)

    def publish_pose_array(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"  # Замените "base_link" на имя вашего фрейма
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        # Create and add poses to the PoseArray
        for i in range(0, 10):
            pose = Pose()
            pose.position.x = float(i)
            pose.position.y = float(i)
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = -1.0
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        
        self.publisher_.publish(pose_array)
        self.get_logger().info('PoseArray published')

class MarkerArraySubscriber(Node):
    def __init__(self):
        super().__init__('marker_array_subscriber')
        self.subscription = self.create_subscription(
            Marker, 'marker_array', self.marker_array_callback, 10)
        self.subscription

    def marker_array_callback(self, msg):
        # Process the received marker array
        # For simplicity, we will just print the positions
        for marker in msg.markers:
            print(f"Position: {marker.pose.position.x, marker.pose.position.y, marker.pose.position.z}")

def main(args=None):
    rclpy.init(args=args)
    pose_array_publisher = PoseArrayPublisher()
    marker_array_subscriber = MarkerArraySubscriber()

    rclpy.spin(pose_array_publisher)
    rclpy.spin(marker_array_subscriber)

    pose_array_publisher.destroy_node()
    marker_array_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()