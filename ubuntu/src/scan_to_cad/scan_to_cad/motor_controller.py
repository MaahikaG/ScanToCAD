"""
motor_controller.py
───────────────────
Controls the two gantry stepper motors via A4988 drivers.

COORDINATE SYSTEM:
  Motor 1 → θ (theta): rotation of the entire semicircular arm around the Z axis.
             Positive θ = counterclockwise when viewed from above.
             Negative θ = clockwise when viewed from above.

  Motor 2 → φ (phi): angular position of the pan-tilt head along the
             semicircle arc. The diameter of the semicircle lies flat in
             the XY plane (perpendicular to Z), with the arc bulging upward.
             φ = 0°   → home end of the diameter (horizontal)
             φ = 90°  → top of the arc (directly above center, along +Z)
             φ = 180° → far end of the diameter (horizontal, opposite side)

Topics subscribed:
  /motor/theta_steps  (std_msgs/Float32)
      Steps to move Motor 1 (Z rotation).
      Positive = counterclockwise from above, negative = clockwise.

  /motor/phi_steps    (std_msgs/Float32)
      Steps to move Motor 2 (arc position).
      Positive = move from φ=0° toward φ=180° (through the top).
      Negative = move back toward φ=0°.

GPIO (BCM numbering):
  Motor 1 (theta):  STEP=17  DIR=27  EN=22
  Motor 2 (phi):    STEP=23  DIR=24  EN=25
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import RPi.GPIO as GPIO
import time

# ── GPIO pin assignments (BCM numbering) ──────────────────────────────────────
M1_STEP = 17;  M1_DIR = 27;  M1_EN = 22    # Motor 1 — theta (Z rotation)
M2_STEP = 23;  M2_DIR = 24;  M2_EN = 25    # Motor 2 — phi   (arc position)

STEP_DELAY = 0.001      # seconds per pulse edge → period = 2×STEP_DELAY
                        # 0.001 s → 500 steps/sec. Reduce to go faster.


class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller')

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setmode(GPIO.BCM)
        for pin in [M1_STEP, M1_DIR, M1_EN, M2_STEP, M2_DIR, M2_EN]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # Enable both drivers (EN is active LOW on A4988)
        GPIO.output(M1_EN, GPIO.LOW)
        GPIO.output(M2_EN, GPIO.LOW)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(
            Float32, '/motor/theta_steps', self.theta_cb, 10
        )
        self.create_subscription(
            Float32, '/motor/phi_steps', self.phi_cb, 10
        )

        self.get_logger().info(
            'Motor controller ready | '
            '/motor/theta_steps → Motor 1 (Z rotation) | '
            '/motor/phi_steps   → Motor 2 (arc position)'
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _move(self, step_pin: int, dir_pin: int, steps: int):
        """
        Send `steps` pulses to one motor.
        Positive steps → DIR HIGH (counterclockwise / toward φ=180°).
        Negative steps → DIR LOW  (clockwise / toward φ=0°).
        """
        if steps == 0:
            return
        GPIO.output(dir_pin, GPIO.HIGH if steps > 0 else GPIO.LOW)
        for _ in range(abs(steps)):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(STEP_DELAY)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def theta_cb(self, msg: Float32):
        """Move Motor 1 — Z-axis rotation (theta)."""
        self._move(M1_STEP, M1_DIR, int(msg.data))

    def phi_cb(self, msg: Float32):
        """Move Motor 2 — arc position along semicircle (phi)."""
        self._move(M2_STEP, M2_DIR, int(msg.data))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        GPIO.output(M1_EN, GPIO.HIGH)
        GPIO.output(M2_EN, GPIO.HIGH)
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()