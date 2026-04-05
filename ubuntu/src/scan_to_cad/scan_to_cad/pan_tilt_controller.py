"""
pan_tilt_controller.py
──────────────────────
Controls two NEMA 17 stepper motors wired as a DIFFERENTIAL DRIVE pan-tilt
mechanism. Ported from the C# G120 implementation (panTilt/Program.cs).

DIFFERENTIAL DRIVE MIXING:
  Motor A command  = azimuth delta
  Motor B command  = azimuth delta + elevation delta
  Both motors participate in every move. The difference between their
  displacements produces tilt; the common component produces pan.

TRAPEZOIDAL VELOCITY PROFILE:
  Motors accelerate from START_SPEED to max speed, cruise, then decelerate
  symmetrically. Prevents missed steps. Matches C# createMotionBuffer() logic.

Topics subscribed:
  /pan_tilt/move  (geometry_msgs/Vector3)
    x = azimuth delta in degrees   (positive = clockwise when viewed from above)
    y = elevation delta in degrees (positive = tilt up)
    z = unused

Topics published:
  /pan_tilt/position  (geometry_msgs/Vector3)
    x = current azimuth (degrees from home)
    y = current elevation (degrees from home)
    z = unused

GPIO (BCM numbering):
  Motor A:  STEP=16  DIR=20  EN=21
  Motor B:  STEP=12  DIR=7   EN=8

Example usage from command line:
  ros2 topic pub --once /pan_tilt/move geometry_msgs/msg/Vector3 "{x: 45.0, y: 0.0, z: 0.0}"

Example usage from another node:
  msg = Vector3(); msg.x = 30.0; msg.y = -15.0
  self.pan_tilt_pub.publish(msg)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import RPi.GPIO as GPIO
import time
import math
import threading

# ── GPIO pin assignments (BCM numbering) ──────────────────────────────────────
PA_STEP = 16;  PA_DIR = 20;  PA_EN = 21     # Pan-Tilt Motor A
PB_STEP = 12;  PB_DIR = 7;   PB_EN = 8      # Pan-Tilt Motor B

# ── Mechanical constants ───────────────────────────────────────────────────────
STEP_ANGLE  = 1.8           # degrees per full step (standard NEMA 17)
MICROSTEP   = 1 / 16        # A4988 max = 1/16; wire MS1+MS2+MS3 all HIGH
PINION_GEAR = 12            # tooth count of small (motor) gear
MAIN_GEAR   = 81            # tooth count of large (output) gear

# Steps required to rotate the output shaft by 1 degree
CONVERSION  = (MAIN_GEAR / PINION_GEAR) / (MICROSTEP * STEP_ANGLE)

# ── Motion parameters ─────────────────────────────────────────────────────────
START_SPEED    = 10.0       # deg/s — safe speed to start from rest
DEFAULT_SPEED  = 240.0      # deg/s — cruise speed
DEFAULT_ACCEL  = 960.0      # deg/s² — acceleration/deceleration rate
MAX_ELEVATION  =  80.0      # degrees — upper elevation hard limit
MIN_ELEVATION  = -80.0      # degrees — lower elevation hard limit
PULSE_WIDTH_S  = 100e-6     # 100 µs HIGH pulse (A4988 minimum is 1 µs)


def degrees_to_steps(degrees: float) -> int:
    return int(round(abs(degrees) * CONVERSION))

def steps_to_degrees(steps: int) -> float:
    return steps / CONVERSION


class PanTiltController(Node):
    def __init__(self):
        super().__init__('pan_tilt_controller')

        self.current_azimuth   = 0.0
        self.current_elevation = 0.0
        self.speed = DEFAULT_SPEED
        self.accel = DEFAULT_ACCEL
        self._move_lock = threading.Lock()

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setmode(GPIO.BCM)
        for pin in [PA_STEP, PA_DIR, PA_EN, PB_STEP, PB_DIR, PB_EN]:
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(PA_EN, GPIO.LOW)
        GPIO.output(PB_EN, GPIO.LOW)

        # ── ROS ───────────────────────────────────────────────────────────────
        self.create_subscription(Vector3, '/pan_tilt/move', self.move_cb, 10)
        self.position_pub = self.create_publisher(Vector3, '/pan_tilt/position', 10)
        self.create_timer(0.1, self.publish_position)

        self.get_logger().info(
            f'Pan-tilt controller ready | '
            f'conversion={CONVERSION:.1f} steps/deg | '
            f'speed={self.speed} deg/s | accel={self.accel} deg/s²'
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def move_cb(self, msg: Vector3):
        threading.Thread(
            target=self._move_xy,
            args=(msg.x, msg.y, self.speed),
            daemon=True
        ).start()

    def publish_position(self):
        msg = Vector3()
        msg.x = self.current_azimuth
        msg.y = self.current_elevation
        msg.z = 0.0
        self.position_pub.publish(msg)

    # ── Differential drive move ───────────────────────────────────────────────

    def _move_xy(self, delta_az: float, delta_el: float, max_speed: float):
        with self._move_lock:
            # Clamp elevation
            delta_el = max(
                MIN_ELEVATION - self.current_elevation,
                min(MAX_ELEVATION - self.current_elevation, delta_el)
            )

            total_dist = math.sqrt(delta_az ** 2 + delta_el ** 2)
            if total_dist == 0:
                return

            # Differential mixing
            motor_a_deg = delta_az
            motor_b_deg = delta_az + delta_el

            speed_a = max(abs(max_speed  * delta_az / total_dist), START_SPEED)
            speed_b = max(abs(max_speed  * delta_el / total_dist), START_SPEED)
            accel_a = max(abs(self.accel * delta_az / total_dist), START_SPEED)
            accel_b = max(abs(self.accel * delta_el / total_dist), START_SPEED)

            steps_a = degrees_to_steps(motor_a_deg)
            steps_b = degrees_to_steps(motor_b_deg)

            GPIO.output(PA_DIR, GPIO.HIGH if motor_a_deg >= 0 else GPIO.LOW)
            GPIO.output(PB_DIR, GPIO.HIGH if motor_b_deg >= 0 else GPIO.LOW)

            profile_a = self._build_profile(steps_a, speed_a, accel_a)
            profile_b = self._build_profile(steps_b, speed_b, accel_b)

            exact_a = steps_to_degrees(steps_a) * (1 if motor_a_deg >= 0 else -1)
            exact_b = steps_to_degrees(steps_b) * (1 if motor_b_deg >= 0 else -1)
            self.current_azimuth   += exact_a
            self.current_elevation += exact_b - exact_a

            self.get_logger().info(
                f'Pan-tilt → Az:{self.current_azimuth:.2f}°  '
                f'El:{self.current_elevation:.2f}°'
            )

            t_a = threading.Thread(target=self._run_motor, args=(PA_STEP, profile_a))
            t_b = threading.Thread(target=self._run_motor, args=(PB_STEP, profile_b))
            t_a.start(); t_b.start()
            t_a.join();  t_b.join()

    # ── Motion profile builder ────────────────────────────────────────────────

    def _build_profile(self, steps: int, max_speed: float, accel_rate: float) -> list:
        """Build a trapezoidal velocity profile as a list of (high_s, low_s) tuples."""
        if steps == 0:
            return []

        accel_steps = int((max_speed - START_SPEED) / accel_rate * CONVERSION) \
                      if accel_rate > 0 else 0
        accel_steps = min(accel_steps, steps // 2)

        profile = []
        for i in range(steps):
            if i < accel_steps:
                current_speed = START_SPEED + (accel_rate * i / CONVERSION)
            elif i >= steps - accel_steps:
                decel_step    = steps - i - 1
                current_speed = START_SPEED + (accel_rate * decel_step / CONVERSION)
            else:
                current_speed = max_speed

            current_speed = max(current_speed, START_SPEED)
            period_s = 1.0 / (current_speed * CONVERSION)
            low_s    = max(period_s - PULSE_WIDTH_S, PULSE_WIDTH_S)
            profile.append((PULSE_WIDTH_S, low_s))

        return profile

    def _run_motor(self, step_pin: int, profile: list):
        for (high_s, low_s) in profile:
            GPIO.output(step_pin, GPIO.HIGH)
            time.sleep(high_s)
            GPIO.output(step_pin, GPIO.LOW)
            time.sleep(low_s)

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