"""
point_cloud_pub.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range, PointCloud2, PointField
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math
import struct
import numpy as np

from scan_to_cad.odom_tf_pubs import ARC_RADIUS_M

MAX_RANGE_M = 1.8
MIN_RANGE_M = 0.05


class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('point_cloud_pub')

        self.points = []
        self.accumulated_points = []
        self.theta_deg = 0.0
        self.phi_deg   = 0.0
        self.shutdown_flag = False


        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.create_subscription(Range, '/tof/range', self._range_callback, 10)
        self.create_subscription(Bool, '/scan_complete', self._on_scan_complete, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 10)

        self.get_logger().info(
            f'PointCloud publisher ready | '
            f'R={ARC_RADIUS_M} m | '
            f'range filter: [{MIN_RANGE_M}, {MAX_RANGE_M}] m'
        )

    def _odom_callback(self, msg: Odometry):
        self.theta_deg = msg.twist.twist.linear.x
        self.phi_deg   = msg.twist.twist.linear.y

    def _range_callback(self, msg: Range):
        d = msg.range
        if not (MIN_RANGE_M <= d <= MAX_RANGE_M):
            return

        point = self._compute_point(d, self.theta_deg, self.phi_deg)
        self.points.append(point)
        self.accumulated_points.append(list(point))  # ← fixed, uses point

        self.cloud_pub.publish(self._build_cloud_msg())

        self.get_logger().debug(
            f'Point #{len(self.points):05d} | '
            f'θ={self.theta_deg:.1f}° φ={self.phi_deg:.1f}° | '
            f'd={d:.3f} m | '
            f'xyz=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})'
        )

    def _on_scan_complete(self, msg):
        if msg.data:
            self.get_logger().info('Saving point cloud to disk...')
            self._save_pcd(self.accumulated_points)
            self.get_logger().info('Shutting down point cloud publisher')
            self.shutdown_flag = True

    def _save_pcd(self, points):
        """Save points as ASCII PCD file without open3d."""
        n = len(points)
        path = '/ros2_ws/scans/latest.pcd'
        with open(path, 'w') as f:
            f.write(f"""# .PCD v0.7
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA ascii
""")
            for p in points:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        self.get_logger().info(f'Saved {n} points to {path}')

    @staticmethod
    def _compute_point(d: float, theta_deg: float, phi_deg: float):
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        r     = ARC_RADIUS_M - d
        x = r * math.cos(phi) * math.cos(theta)
        y = r * math.cos(phi) * math.sin(theta)
        z = r * math.sin(phi)
        return (x, y, z)

    def _build_cloud_msg(self):
        msg = PointCloud2()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step   = 12
        msg.is_bigendian = False
        msg.is_dense     = True

        buf = bytearray()
        for (x, y, z) in self.points:
            buf += struct.pack('fff', x, y, z)

        msg.data     = bytes(buf)
        msg.width    = len(self.points)
        msg.height   = 1
        msg.row_step = msg.point_step * msg.width
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    while rclpy.ok() and not node.shutdown_flag:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()