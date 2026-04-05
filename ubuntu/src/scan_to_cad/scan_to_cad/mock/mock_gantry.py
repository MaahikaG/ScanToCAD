import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
import math

from scan_to_cad.odom_tf_pubs import ARC_RADIUS_M, THETA_HOME_DEG, PHI_HOME_DEG

PHI_MIN_DEG  = 0.0
PHI_MAX_DEG  = 180.0
PHI_SPEED    = 30.0
THETA_STEP   = 30.0
TIMER_PERIOD = 0.1
THETA_MAX    = 360.0   # full rotation = scan complete


class MockGantry(Node):
    def __init__(self):
        super().__init__('mock_gantry')

        self.theta_deg   = THETA_HOME_DEG
        self.phi_deg     = PHI_HOME_DEG
        self.phi_dir     = 1.0
        self.scan_done   = False
        self.theta_steps = 0
        self.total_steps = int(THETA_MAX / THETA_STEP)
        self.shutdown_flag = False


        self.pos_pub        = self.create_publisher(Float32MultiArray, '/gantry/position', 10)
        self.odom_pub       = self.create_publisher(Odometry, '/odom', 10)
        self.done_pub       = self.create_publisher(Bool, '/scan_complete', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(
            Float32MultiArray, '/gantry/cmd', self._cmd_callback, 10)

        self.timer = self.create_timer(TIMER_PERIOD, self._publish)
        self.get_logger().info(
            f'Mock gantry ready — {self.total_steps} theta steps to full scan')

    def _cmd_callback(self, msg):
        self.theta_deg = msg.data[0]
        self.phi_deg   = msg.data[1]

    def _update_position(self):
        if self.scan_done:
            return

        self.phi_deg += PHI_SPEED * TIMER_PERIOD * self.phi_dir

        if self.phi_deg >= PHI_MAX_DEG:
            self.phi_deg  = PHI_MAX_DEG
            self.phi_dir  = -1.0
            self.theta_deg = (self.theta_deg + THETA_STEP) % THETA_MAX
            self.theta_steps += 1
            self.get_logger().info(
                f'Step {self.theta_steps}/{self.total_steps} — '
                f'θ={self.theta_deg:.1f}°')

        elif self.phi_deg <= PHI_MIN_DEG:
            self.phi_deg  = PHI_MIN_DEG
            self.phi_dir  = 1.0
            self.theta_deg = (self.theta_deg + THETA_STEP) % THETA_MAX
            self.theta_steps += 1
            self.get_logger().info(
                f'Step {self.theta_steps}/{self.total_steps} — '
                f'θ={self.theta_deg:.1f}°')

        if self.theta_steps >= self.total_steps:
            self.scan_done = True
            self.get_logger().info('Scan complete — shutting down')
            done_msg = Bool()
            done_msg.data = True
            self.done_pub.publish(done_msg)
            # Give the message time to send before shutting down
            self.create_timer(1.0, self._shutdown)

    def _shutdown(self):
        self.shutdown_flag = True

    def _publish(self):
        self._update_position()

        if self.scan_done:
            return

        now = self.get_clock().now().to_msg()
        x, y, z = self._spherical_to_cartesian(
            self.theta_deg, self.phi_deg, ARC_RADIUS_M)
        qw, qx, qy, qz = self._inward_quaternion(
            self.theta_deg, self.phi_deg)

        pos_msg = Float32MultiArray()
        pos_msg.data = [self.theta_deg, self.phi_deg]
        self.pos_pub.publish(pos_msg)

        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'scanner_head'
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = z
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.twist.twist.linear.x = self.theta_deg
        odom.twist.twist.linear.y = self.phi_deg
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'scanner_head'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def _spherical_to_cartesian(theta_deg, phi_deg, radius):
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        return (
            radius * math.cos(phi) * math.cos(theta),
            radius * math.cos(phi) * math.sin(theta),
            radius * math.sin(phi)
        )

    @staticmethod
    def _inward_quaternion(theta_deg, phi_deg):
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        dx = -math.cos(phi) * math.cos(theta)
        dy = -math.cos(phi) * math.sin(theta)
        dz = -math.sin(phi)
        forward = [1.0, 0.0, 0.0]
        target  = [dx, dy, dz]
        axis = [
            forward[1]*target[2] - forward[2]*target[1],
            forward[2]*target[0] - forward[0]*target[2],
            forward[0]*target[1] - forward[1]*target[0]
        ]
        dot   = sum(forward[i]*target[i] for i in range(3))
        angle = math.acos(max(-1.0, min(1.0, dot)))
        norm  = math.sqrt(sum(a**2 for a in axis))
        if norm < 1e-6:
            return (1.0, 0.0, 0.0, 0.0) if dot > 0 else (0.0, 0.0, 1.0, 0.0)
        axis = [a / norm for a in axis]
        s = math.sin(angle / 2)
        return (math.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s)


def main():
    rclpy.init()
    node = MockGantry()
    while rclpy.ok() and not node.shutdown_flag:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()