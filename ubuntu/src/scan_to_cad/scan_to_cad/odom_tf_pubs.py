"""
odom_tf_pubs.py
───────────────
Reads both rotary encoders and publishes the scanner mount's position along
the gantry arc. The pan-tilt angles come from /pan_tilt/angles and are folded
into a two-step TF chain so that the 'sensor' frame always reflects the true
beam direction.

GEOMETRY:
  The semicircle's diameter lies flat in the XY plane, passing through
  the Z axis. The arc bulges upward (+Z). Motor 1 rotates the entire
  semicircle around the Z axis (θ). Motor 2 drives the scanner mount
  along the arc (φ).

  φ = 0°   → home end of diameter, horizontal  → (R, 0, 0) rotated by θ
  φ = 90°  → top of arc, directly above center → (0, 0, R)
  φ = 180° → opposite end of diameter          → (-R, 0, 0) rotated by θ

CARTESIAN CONVERSION (gantry position):
  x = R · cos(φ) · cos(θ)
  y = R · cos(φ) · sin(θ)
  z = R · sin(φ)

SCANNER HEAD LOCAL FRAME:
  The frame 'scanner_head' is oriented so that:
    x-axis = -e_r  (inward toward scan target; sensor zero direction)
    y-axis = -e_θ  (lateral)
    z-axis =  e_φ  (upward along the arc)
  This is the parent frame for the pan-tilt rotation.

TF CHAIN:
  odom → scanner_head : gantry position + mount orientation (from θ, φ)
  scanner_head → sensor : pan-tilt rotation (α pan around z, β tilt around y)

Topics published:
  /odom  (nav_msgs/Odometry)
      pose.pose.position.{x,y,z}     = 3D position of scanner mount (metres)
      pose.pose.orientation          = scanner head orientation (from θ, φ)
      twist.twist.linear.x           = θ in degrees  (raw, for point_cloud_pub)
      twist.twist.linear.y           = φ in degrees  (raw, for point_cloud_pub)

  /tf   (via tf2_ros.TransformBroadcaster)
      'odom'         → 'scanner_head'  (gantry position + mount orientation)
      'scanner_head' → 'sensor'        (pan-tilt rotation)

Topics subscribed:
  /pan_tilt/angles  (geometry_msgs/Vector3)
      x = α (pan)  in degrees
      y = β (tilt) in degrees

GPIO (BCM numbering):
  Encoder 1 (Motor 1, θ):  A=5   B=6
  Encoder 2 (Motor 2, φ):  A=13  B=19

Constants to set before running:
  PULSES_PER_REV  — encoder PPR from datasheet
  ARC_RADIUS_M    — distance from Z axis to scanner mount along arm (metres)
  PHI_HOME_DEG    — φ at startup; 0.0 if homed to the φ=0 end of diameter
  THETA_HOME_DEG  — θ at startup; almost always 0.0
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Vector3
import tf2_ros
import RPi.GPIO as GPIO
import math

# ── GPIO pin assignments (BCM numbering) ──────────────────────────────────────
ENC1_A = 5;   ENC1_B = 6    # Encoder 1 — Motor 1 (θ, Z rotation)
ENC2_A = 13;  ENC2_B = 19   # Encoder 2 — Motor 2 (φ, arc position)

# ── Physical constants — TUNE THESE TO YOUR HARDWARE ──────────────────────────
PULSES_PER_REV = 600        # Encoder pulses per full shaft revolution (PPR).
                            # From encoder datasheet. Common: 100, 200, 400, 600.
                            # x2 quadrature decoding applied in code →
                            # effective resolution = PULSES_PER_REV * 2.

ARC_RADIUS_M   = 0.30       # Distance from the Z axis to the scanner mount
                            # point, measured along the arm (metres).
                            # This is the radius of the semicircle.
                            # Imported by point_cloud_pub.py — only set it here.

PHI_HOME_DEG   = 0.0        # φ angle at startup (degrees).
                            # 0.0 = mount starts at home end of diameter.

THETA_HOME_DEG = 0.0        # θ angle at startup. Almost always 0.0.

# ── Derived constant ──────────────────────────────────────────────────────────
DEGREES_PER_COUNT = 360.0 / (PULSES_PER_REV * 2)

# Exported so point_cloud_pub.py can import ARC_RADIUS_M directly,
# keeping the value defined in exactly one place.
__all__ = ['ARC_RADIUS_M']


class OdomTfPublisher(Node):
    def __init__(self):
        super().__init__('odom_tf_pubs')

        # ── Encoder state ─────────────────────────────────────────────────────
        self.enc1_count = 0
        self.enc2_count = 0

        # ── Pan-tilt state (populated by /pan_tilt/angles subscription) ───────
        self.alpha_deg = 0.0    # pan  (α)
        self.beta_deg  = 0.0    # tilt (β)

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setmode(GPIO.BCM)
        for pin in [ENC1_A, ENC1_B, ENC2_A, ENC2_B]:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        GPIO.add_event_detect(ENC1_A, GPIO.BOTH, callback=self._enc1_cb)
        GPIO.add_event_detect(ENC2_A, GPIO.BOTH, callback=self._enc2_cb)

        # ── ROS subscribers ───────────────────────────────────────────────────
        self.create_subscription(Vector3, '/pan_tilt/angles', self._pan_tilt_cb, 10)

        # ── ROS publishers ────────────────────────────────────────────────────
        self.odom_pub       = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publish at 20 Hz
        self.create_timer(0.05, self.publish_odometry)

        self.get_logger().info(
            f'Odom/TF publisher ready | '
            f'R={ARC_RADIUS_M} m | '
            f'{PULSES_PER_REV} PPR | '
            f'{DEGREES_PER_COUNT:.4f} deg/count'
        )

    # ── Quadrature decoding ───────────────────────────────────────────────────

    def _enc1_cb(self, channel):
        """x2 quadrature decode — Motor 1 (θ). A leads B → count up."""
        a = GPIO.input(ENC1_A)
        b = GPIO.input(ENC1_B)
        self.enc1_count += 1 if a != b else -1

    def _enc2_cb(self, channel):
        """x2 quadrature decode — Motor 2 (φ). A leads B → count up."""
        a = GPIO.input(ENC2_A)
        b = GPIO.input(ENC2_B)
        self.enc2_count += 1 if a != b else -1

    # ── Pan-tilt angle subscription ───────────────────────────────────────────

    def _pan_tilt_cb(self, msg: Vector3):
        """Receive pan (α) and tilt (β) angles in degrees from pan-tilt controller."""
        self.alpha_deg = msg.x
        self.beta_deg  = msg.y

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def _counts_to_angles(self):
        """Convert raw encoder counts to (theta_deg, phi_deg)."""
        theta_deg = THETA_HOME_DEG + self.enc1_count * DEGREES_PER_COUNT
        phi_deg   = PHI_HOME_DEG   + self.enc2_count * DEGREES_PER_COUNT
        return theta_deg, phi_deg

    @staticmethod
    def spherical_to_cartesian(theta_deg: float, phi_deg: float, radius: float):
        """
        Convert (θ, φ) to 3D Cartesian (x, y, z).

          x = R · cos(φ) · cos(θ)
          y = R · cos(φ) · sin(θ)
          z = R · sin(φ)
        """
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        x = radius * math.cos(phi) * math.cos(theta)
        y = radius * math.cos(phi) * math.sin(theta)
        z = radius * math.sin(phi)
        return x, y, z

    @staticmethod
    def _mount_orientation_quat(theta_deg: float, phi_deg: float):
        """
        Quaternion for the 'scanner_head' frame orientation in the 'odom' frame.

        The scanner_head frame is defined as:
          x-axis = -e_r  = (-cosφ cosθ, -cosφ sinθ, -sinφ)   inward/forward
          y-axis = -e_θ  = ( sinθ,       -cosθ,       0)      lateral
          z-axis =  e_φ  = (-sinφ cosθ, -sinφ sinθ,  cosφ)   up along arc

        This is a right-handed frame: x × y = z.

        The rotation matrix R (columns = frame axes in odom) is converted to a
        quaternion using Shepperd's method.

        Returns (qx, qy, qz, qw).
        """
        θ = math.radians(theta_deg)
        φ = math.radians(phi_deg)

        # Rotation matrix, column-major (R[:,j] = j-th axis of scanner_head in odom)
        R = [
            # row 0
            [-math.cos(φ)*math.cos(θ),  math.sin(θ), -math.sin(φ)*math.cos(θ)],
            # row 1
            [-math.cos(φ)*math.sin(θ), -math.cos(θ), -math.sin(φ)*math.sin(θ)],
            # row 2
            [-math.sin(φ),              0.0,           math.cos(φ)             ],
        ]

        trace = R[0][0] + R[1][1] + R[2][2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2][1] - R[1][2]) * s
            y = (R[0][2] - R[2][0]) * s
            z = (R[1][0] - R[0][1]) * s
        elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
            s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
            w = (R[2][1] - R[1][2]) / s
            x = 0.25 * s
            y = (R[0][1] + R[1][0]) / s
            z = (R[0][2] + R[2][0]) / s
        elif R[1][1] > R[2][2]:
            s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
            w = (R[0][2] - R[2][0]) / s
            x = (R[0][1] + R[1][0]) / s
            y = 0.25 * s
            z = (R[1][2] + R[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
            w = (R[1][0] - R[0][1]) / s
            x = (R[0][2] + R[2][0]) / s
            y = (R[1][2] + R[2][1]) / s
            z = 0.25 * s
        return x, y, z, w

    @staticmethod
    def _pan_tilt_quat(alpha_deg: float, beta_deg: float):
        """
        Quaternion for the 'sensor' frame relative to 'scanner_head'.

        Pan  α: rotation around scanner_head +z  (left/right sweep).
        Tilt β: rotation around scanner_head +y  (up/down tilt).

        Composition: q_z(α) * q_y(β)  — β applied first in the mount frame,
        then α.  At (α=0, β=0) the sensor's +x axis aligns with mount's +x
        (pointing inward toward the scan target).

        Returns (qx, qy, qz, qw).
        """
        a = math.radians(alpha_deg)
        b = math.radians(beta_deg)
        sa, ca = math.sin(a / 2), math.cos(a / 2)
        sb, cb = math.sin(b / 2), math.cos(b / 2)
        # Closed-form product of q_z(α) * q_y(β):
        return (-sa * sb,   # qx
                 ca * sb,   # qy
                 sa * cb,   # qz
                 ca * cb)   # qw

    # ── Publish loop ──────────────────────────────────────────────────────────

    def publish_odometry(self):
        now = self.get_clock().now().to_msg()

        theta_deg, phi_deg = self._counts_to_angles()
        x, y, z = self.spherical_to_cartesian(theta_deg, phi_deg, ARC_RADIUS_M)
        qx, qy, qz, qw = self._mount_orientation_quat(theta_deg, phi_deg)

        # ── Odometry: gantry position + mount orientation ─────────────────────
        odom = Odometry()
        odom.header.stamp    = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'scanner_head'
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        # Raw gantry angles for point_cloud_pub.py
        odom.twist.twist.linear.x = theta_deg
        odom.twist.twist.linear.y = phi_deg
        self.odom_pub.publish(odom)

        # ── TF 1: odom → scanner_head (gantry position + mount orientation) ───
        t1 = TransformStamped()
        t1.header.stamp       = now
        t1.header.frame_id    = 'odom'
        t1.child_frame_id     = 'scanner_head'
        t1.transform.translation.x = x
        t1.transform.translation.y = y
        t1.transform.translation.z = z
        t1.transform.rotation.x = qx
        t1.transform.rotation.y = qy
        t1.transform.rotation.z = qz
        t1.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t1)

        # ── TF 2: scanner_head → sensor (pan-tilt rotation only) ──────────────
        pqx, pqy, pqz, pqw = self._pan_tilt_quat(self.alpha_deg, self.beta_deg)

        t2 = TransformStamped()
        t2.header.stamp       = now
        t2.header.frame_id    = 'scanner_head'
        t2.child_frame_id     = 'sensor'
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = pqx
        t2.transform.rotation.y = pqy
        t2.transform.rotation.z = pqz
        t2.transform.rotation.w = pqw
        self.tf_broadcaster.sendTransform(t2)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        GPIO.remove_event_detect(ENC1_A)
        GPIO.remove_event_detect(ENC2_A)
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
