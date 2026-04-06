"""
parsenet_trigger.py
───────────────────
Listens for a trigger from Unity, uploads the completed scan to transfer.sh
(no authentication required), and logs the download URL.

Flow:
  Unity publishes True on /run_parsenet
    → uploads /home/mie_g28/scan_to_cad/ubuntu/scans/latest.pcd to transfer.sh
    → logs a URL, e.g. https://transfer.sh/abc123/latest.pcd
    → paste that URL into the Colab notebook when prompted

Prerequisites on the Pi:
  pip install requests

Topics subscribed:
  /run_parsenet  (std_msgs/Bool)  — published True by Unity when ready

Topics published:
  /scan_url  (std_msgs/String)  — transfer.sh download URL of the latest scan
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import requests
import os
import time

SCAN_PATH    = '/home/mie_g28/scan_to_cad/ubuntu/scans/latest.pcd'
FILE_TIMEOUT = 15.0   # seconds to wait for point_cloud_pub to finish writing


class ParseNetTrigger(Node):
    def __init__(self):
        super().__init__('parsenet_trigger')
        self.create_subscription(Bool, '/run_parsenet', self._on_trigger, 10)
        self.url_pub = self.create_publisher(String, '/scan_url', 10)
        self.get_logger().info('ParseNet trigger ready')

    def _wait_for_file(self, path):
        """Block until the file exists and its size has stopped changing."""
        deadline  = time.time() + FILE_TIMEOUT
        last_size = -1
        while time.time() < deadline:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > 0 and size == last_size:
                    return True   # stable — writing complete
                last_size = size
            time.sleep(0.3)
        return False

    def _on_trigger(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info('Waiting for latest.pcd to be written...')
        if not self._wait_for_file(SCAN_PATH):
            self.get_logger().error(
                f'Timed out waiting for {SCAN_PATH} — '
                f'did point_cloud_pub save it?'
            )
            return

        self.get_logger().info('Uploading scan to transfer.sh...')

        try:
            with open(SCAN_PATH, 'rb') as f:
                response = requests.put(
                    'https://transfer.sh/latest.pcd',
                    data=f,
                    timeout=60
                )
            response.raise_for_status()
        except requests.RequestException as e:
            self.get_logger().error(f'Upload failed: {e}')
            return

        url = response.text.strip()

        # Publish URL so Unity can display it
        url_msg = String()
        url_msg.data = url
        self.url_pub.publish(url_msg)

        self.get_logger().info(
            f'\n'
            f'  Scan uploaded successfully.\n'
            f'  URL: {url}\n'
            f'  Paste this into the Colab notebook when prompted.'
        )


def main():
    rclpy.init()
    node = ParseNetTrigger()
    rclpy.spin(node)
    rclpy.shutdown()
