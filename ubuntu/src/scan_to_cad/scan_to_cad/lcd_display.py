"""
lcd_display.py
──────────────
Drives the 16x2 LCD in 4-bit PARALLEL mode (no I2C backpack).

WIRING:
  LCD Pin 1  VSS        → Pi GND
  LCD Pin 2  VDD        → Pi 5V (Pin 2)
  LCD Pin 3  V0         → middle wiper of 10kΩ contrast potentiometer
                          (pot outer legs between 5V and GND)
  LCD Pin 4  RS         → Pi BCM 26 (Physical Pin 37)
  LCD Pin 5  RW         → Pi GND  (always write mode)
  LCD Pin 6  E          → Pi BCM 18 (Physical Pin 12)
  LCD Pin 7  D0         → not connected
  LCD Pin 8  D1         → not connected
  LCD Pin 9  D2         → not connected
  LCD Pin 10 D3         → not connected
  LCD Pin 11 D4         → Pi BCM 4  (Physical Pin 7)
  LCD Pin 12 D5         → Pi BCM 9  (Physical Pin 21)
  LCD Pin 13 D6         → Pi BCM 10 (Physical Pin 19)
  LCD Pin 14 D7         → Pi BCM 11 (Physical Pin 23)
  LCD Pin 15 A (bklt+)  → Pi 3.3V via 220Ω resistor
  LCD Pin 16 K (bklt-)  → Pi GND

CONTRAST:
  If the screen is blank or shows solid black squares, adjust the
  potentiometer on pin 3 slowly until characters appear.

DISPLAY LAYOUT (16x2):
  Row 0:  T:000.0 P:000.0    (theta and phi in degrees)
  Row 1:  TOF:  0.000 m      (TOF distance)

Topics subscribed:
  /odom       (nav_msgs/Odometry)  — twist.linear.x=θ, twist.linear.y=φ
  /tof/range  (sensor_msgs/Range)  — distance in metres

Install library:
  pip install RPLCD
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from RPLCD.gpio import CharLCD
import RPi.GPIO as GPIO

# ── GPIO pin assignments (BCM numbering) ──────────────────────────────────────
LCD_RS = 26
LCD_E  = 18
LCD_D4 = 4
LCD_D5 = 9
LCD_D6 = 10
LCD_D7 = 11

LCD_COLS = 16
LCD_ROWS = 2


class LCDDisplay(Node):
    def __init__(self):
        super().__init__('lcd_display')

        # ── Initialise LCD in 4-bit parallel mode ─────────────────────────────
        # RPLCD handles all the HD44780 initialisation sequence automatically.
        self.lcd = CharLCD(
            numbering_mode=GPIO.BCM,
            cols=LCD_COLS,
            rows=LCD_ROWS,
            pin_rs=LCD_RS,
            pin_e=LCD_E,
            pins_data=[LCD_D4, LCD_D5, LCD_D6, LCD_D7],
            compat_mode=True,
        )
        self.lcd.clear()

        # ── Cached display values ─────────────────────────────────────────────
        self.theta_deg = 0.0
        self.phi_deg   = 0.0
        self.dist      = 0.0

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(Odometry, '/odom',      self.odom_cb,  10)
        self.create_subscription(Range,    '/tof/range', self.range_cb, 10)

        # Refresh at 2 Hz — LCD writes are slow, going faster causes glitches
        self.create_timer(0.5, self.update_display)

        self.get_logger().info('LCD display ready (parallel 4-bit mode)')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        # θ and φ are packed into twist.linear by odom_tf_pubs.py
        self.theta_deg = msg.twist.twist.linear.x
        self.phi_deg   = msg.twist.twist.linear.y

    def range_cb(self, msg: Range):
        self.dist = msg.range

    # ── Display update ────────────────────────────────────────────────────────

    def update_display(self):
        line1 = f'T:{self.theta_deg:6.1f} P:{self.phi_deg:6.1f}'
        line2 = f'TOF: {self.dist:6.3f} m'

        self.lcd.cursor_pos = (0, 0)
        self.lcd.write_string(line1[:LCD_COLS].ljust(LCD_COLS))
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(line2[:LCD_COLS].ljust(LCD_COLS))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.lcd.clear()
        self.lcd.close(clear=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LCDDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()