"""
pan_tilt_controller.py
──────────────────────
Controls two NEMA 8 stepper motors via TB6600 drivers.
Motor A and Motor B are mechanically independent — each directly
rotates its own axis at a 1:1 ratio with no gearing.

COORDINATE SYSTEM (local to scanner_head frame):
  Motor A → α (pan):  rotation around scanner_head +z (e_φ, up along arc).
             α = 0°  → sensor points directly inward (toward scan target).
             Positive α = pan counterclockwise when viewed from above.

  Motor B → β (tilt): rotation around scanner_head +y (-e_θ, lateral).
             β = 0°  → sensor points directly inward.
             Positive β = tilt upward (+Z direction).

Topics subscribed:
  /pan_tilt/alpha_steps  (std_msgs/Float32)
      Steps to move Motor A (pan).
      Positive = counterclockwise from above, negative = clockwise.

  /pan_tilt/beta_steps   (std_msgs/Float32)
      Steps to move Motor B (tilt).
      Positive = tilt upward, negative = tilt downward.

Topics published:
  /pan_tilt/angles  (geometry_msgs/Vector3)
      x = α (pan)  in degrees — accumulated from home position
      y = β (tilt) in degrees — accumulated from home position
      z = 0 (unused)
      Published after every move so odom_tf_pubs can update the TF.

GPIO (BCM numbering):
  Motor A (pan):  STEP=16  DIR=20  EN=21
  Motor B (tilt): STEP=12  DIR=7   EN=8
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3
import RPi.GPIO as GPIO
import time

# ── Steps-per-revolution — TUNE TO YOUR DRIVER MICROSTEPPING SETTING ─────────
# NEMA 8 = 1.8°/step = 200 full steps/rev.
# Multiply by microstep divisor (e.g. 1600 for 1/8 microstepping).
STEPS_PER_REV    = 200
DEGREES_PER_STEP = 360.0 / STEPS_PER_REV

# ── GPIO pin assignments (BCM numbering) ──────────────────────────────────────
PA_STEP = 16;  PA_DIR = 20;  PA_EN = 21    # Motor A — pan  (α)
PB_STEP = 12;  PB_DIR = 7;   PB_EN = 8    # Motor B — tilt (β)

STEP_DELAY = 0.001      # seconds per pulse edge → period = 2×STEP_DELAY
                        # 0.001 s → 500 steps/sec. Reduce to go faster.


class PanTiltController(Node):
    def __init__(self):
        super().__init__('pan_tilt_controller')

        # ── Accumulated step counts (from home) ───────────────────────────────
        self._alpha_steps = 0
        self._beta_steps  = 0

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setmode(GPIO.BCM)
        for pin in [PA_STEP, PA_DIR, PA_EN, PB_STEP, PB_DIR, PB_EN]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        # Enable both drivers (EN is active LOW on TB6600)
        GPIO.output(PA_EN, GPIO.LOW)
        GPIO.output(PB_EN, GPIO.LOW)

        # ── Publisher ─────────────────────────────────────────────────────────
        self.angles_pub = self.create_publisher(Vector3, '/pan_tilt/angles', 10)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Float32, '/pan_tilt/alpha_steps', self.alpha_cb, 10)
        self.create_subscription(Float32, '/pan_tilt/beta_steps',  self.beta_cb,  10)

        # Publish zero angles at startup so odom_tf_pubs has an initial value
        self._publish_angles()

        self.get_logger().info(
            'Pan-tilt controller ready | '
            '/pan_tilt/alpha_steps → Motor A (pan) | '
            '/pan_tilt/beta_steps  → Motor B (tilt) | '
            f'{DEGREES_PER_STEP:.4f} deg/step'
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _move(self, step_pin: int, dir_pin: int, steps: int):
        """
        Send `steps` pulses to one motor.
        Positive steps → DIR HIGH.
        Negative steps → DIR LOW.
        """
        if steps == 0:
            return
        GPIO.output(dir_pin, GPIO.HIGH if steps > 0 else GPIO.LOW)
        for _ in range(abs(steps)):
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(STEP_DELAY)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(STEP_DELAY)

    def _publish_angles(self):
        msg = Vector3()
        msg.x = self._alpha_steps * DEGREES_PER_STEP   # pan  α in degrees
        msg.y = self._beta_steps  * DEGREES_PER_STEP   # tilt β in degrees
        msg.z = 0.0
        self.angles_pub.publish(msg)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def alpha_cb(self, msg: Float32):
        """Move Motor A — pan (α)."""
        steps = int(msg.data)
        self._move(PA_STEP, PA_DIR, steps)
        self._alpha_steps += steps
        self._publish_angles()

    def beta_cb(self, msg: Float32):
        """Move Motor B — tilt (β)."""
        steps = int(msg.data)
        self._move(PB_STEP, PB_DIR, steps)
        self._beta_steps += steps
        self._publish_angles()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        GPIO.output(PA_EN, GPIO.HIGH)
        GPIO.output(PB_EN, GPIO.HIGH)
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PanTiltController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()