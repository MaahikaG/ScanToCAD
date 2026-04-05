import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import subprocess
import os
import struct
import numpy as np
from scipy.cluster.vq import kmeans2

class ParseNetTrigger(Node):
    def __init__(self):
        super().__init__('parsenet_trigger')
        self.create_subscription(Bool, '/run_parsenet', self._on_trigger, 10)
        self.get_logger().info('ParseNet trigger node ready')

    def _read_pcd(self, pcd_path):
        """Read ASCII PCD file into numpy array."""
        points = []
        data_started = False
        with open(pcd_path) as f:
            for line in f:
                if line.startswith('DATA'):
                    data_started = True
                    continue
                if data_started:
                    pts = list(map(float, line.strip().split()))
                    if len(pts) >= 3:
                        points.append(pts[:3])
        return np.array(points)

    def _segment_points(self, points, n_clusters=10):
        """Segment points into surface clusters using kmeans."""
        n_clusters = min(n_clusters, len(points))
        _, labels = kmeans2(points, n_clusters, minit='points', iter=20)
        return labels

    def _on_trigger(self, msg):
        if not msg.data:
            return

        input_pcd    = '/ros2_ws/scans/latest.pcd'
        annotated    = '/ros2_ws/scans/annotated.txt'
        output_dir   = '/ros2_ws/scans/point2cad_out'
        point2cad_dir = '/ros2_ws/models/point2cad'

        os.makedirs(output_dir, exist_ok=True)

        # ── Check input exists ────────────────────────────────────────────────
        if not os.path.exists(input_pcd):
            self.get_logger().error(f'No scan found at {input_pcd}')
            return

        # ── Step 1: Read PCD ──────────────────────────────────────────────────
        self.get_logger().info('Reading point cloud...')
        points = self._read_pcd(input_pcd)
        self.get_logger().info(f'Loaded {len(points)} points')

        # ── Step 2: Segment with kmeans ───────────────────────────────────────
        self.get_logger().info('Segmenting point cloud...')
        labels = self._segment_points(points)
        n_segments = len(set(labels))
        self.get_logger().info(f'Found {n_segments} surface segments')

        # Save as (x, y, z, s) for Point2CAD
        xyzs = np.column_stack([points, labels])
        np.savetxt(annotated, xyzs, fmt='%.6f %.6f %.6f %d')
        self.get_logger().info(f'Saved annotated cloud to {annotated}')

        # ── Step 3: Run Point2CAD via Docker ──────────────────────────────────
        self.get_logger().info('Running Point2CAD...')
        result = subprocess.run([
            'docker', 'run', '--rm',
            '-v', f'{point2cad_dir}:/work/point2cad',
            'toshas/point2cad:v1',
            'python', '-m', 'point2cad.main',
            '--input', annotated,
            '--output', output_dir
        ], capture_output=True, text=True)

        if result.returncode == 0:
            self.get_logger().info(f'CAD export complete → {output_dir}')
        else:
            self.get_logger().error(f'Point2CAD failed:\n{result.stderr}')

def main():
    rclpy.init()
    node = ParseNetTrigger()
    rclpy.spin(node)
    rclpy.shutdown()