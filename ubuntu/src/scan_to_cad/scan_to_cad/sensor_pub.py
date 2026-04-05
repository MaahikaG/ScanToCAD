"""
sensor_pub.py
─────────────
Reads the VL53L0X time-of-flight sensor over I2C and publishes distance
measurements as ROS 2 Range messages.

Topics published:
  /tof/range  (sensor_msgs/Range)  — distance in metres

Wiring (I2C):
  SDA → Pi Pin 3  (BCM 2)
  SCL → Pi Pin 5  (BCM 3)
  VIN → Pi 3.3V
  GND → Pi GND

Install library:
  pip install VL53L0X
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import VL53L0X


class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_pub')

        # ── Initialise VL53L0X over I2C ───────────────────────────────────────
        self.tof = VL53L0X.VL53L0X()
        self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
        # Accuracy modes:
        #   GOOD    — ~30ms/reading,  fastest
        #   BETTER  — ~66ms/reading,  balanced  ← default
        #   BEST    — ~200ms/reading, most accurate

        # ── Publisher ─────────────────────────────────────────────────────────
        self.pub = self.create_publisher(Range, '/tof/range', 10)

        # Publish at 10 Hz (BETTER mode supports up to ~15 Hz)
        self.create_timer(0.1, self.publish_range)

        self.get_logger().info('TOF sensor publisher ready (VL53L0X, BETTER mode)')

    # ── Publish loop ──────────────────────────────────────────────────────────

    def publish_range(self):
        distance_mm = self.tof.get_distance()

        if distance_mm <= 0:
            self.get_logger().warn(
                'TOF invalid reading — skipping', throttle_duration_sec=5.0
            )
            return

        msg = Range()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'scanner_head'   # sensor lives on the scanner head
        msg.radiation_type  = Range.INFRARED
        msg.field_of_view   = 0.436            # ~25° in radians (VL53L0X spec)
        msg.min_range       = 0.03             # 30 mm
        msg.max_range       = 2.0              # 2000 mm
        msg.range           = distance_mm / 1000.0  # mm → metres

        self.pub.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.tof.stop_ranging()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()