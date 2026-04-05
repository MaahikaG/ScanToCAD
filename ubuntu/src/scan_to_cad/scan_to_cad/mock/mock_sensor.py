import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Bool


class MockSensor(Node):
    def __init__(self):
        super().__init__('mock_sensor')
        self.pub = self.create_publisher(Range, '/tof/range', 10)
        self.timer = self.create_timer(0.1, self.publish_range)
        self.get_logger().info('Mock sensor ready')
        self.create_subscription(Bool, '/scan_complete', self._on_scan_complete, 10)
        self.shutdown_flag = False


    def _on_scan_complete(self, msg):
        if msg.data:
            self.get_logger().info('Scan complete — shutting down sensor')
            self.timer.cancel()
            self.shutdown_flag = True

    def publish_range(self):
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'tof_sensor'
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = 0.44
        msg.min_range = 0.03
        msg.max_range = 2.0
        msg.range = 0.5
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = MockSensor()
    while rclpy.ok() and not node.shutdown_flag:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()