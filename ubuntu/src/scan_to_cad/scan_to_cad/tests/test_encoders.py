"""
test_encoders.py
────────────────
Standalone encoder test — NO ROS 2 required.

Prints live counts for both encoders as you manually rotate each shaft.
Press Ctrl+C to exit.

Usage:
  python3 tests/test_encoders.py

Pin assignments must match odom_tf_pubs.py.

Direction conventions:
  Encoder 1 (Motor 1 / theta):
    Count UP   = counterclockwise rotation of arm (positive θ)
    Count DOWN = clockwise (negative θ)

  Encoder 2 (Motor 2 / phi):
    Count UP   = head moving from φ=0° toward φ=180° (through the top)
    Count DOWN = head moving back toward φ=0°

If either encoder counts the wrong direction:
  Swap the A and B signal wires on that encoder, OR flip the sign in the
  corresponding callback in odom_tf_pubs.py.
"""

import RPi.GPIO as GPIO
import time

# ── Pin assignments — must match odom_tf_pubs.py ──────────────────────────────
ENC1_A = 5;   ENC1_B = 6    # Encoder 1 — Motor 1 (theta)
ENC2_A = 13;  ENC2_B = 19   # Encoder 2 — Motor 2 (phi)

enc1_count = 0
enc2_count = 0


def enc1_cb(channel):
    global enc1_count
    a = GPIO.input(ENC1_A)
    b = GPIO.input(ENC1_B)
    enc1_count += 1 if a != b else -1


def enc2_cb(channel):
    global enc2_count
    a = GPIO.input(ENC2_A)
    b = GPIO.input(ENC2_B)
    enc2_count += 1 if a != b else -1


def main():
    GPIO.setmode(GPIO.BCM)
    for pin in [ENC1_A, ENC1_B, ENC2_A, ENC2_B]:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    GPIO.add_event_detect(ENC1_A, GPIO.BOTH, callback=enc1_cb)
    GPIO.add_event_detect(ENC2_A, GPIO.BOTH, callback=enc2_cb)

    print("Encoder test — rotate each shaft manually. Ctrl+C to exit.\n")
    print(f"{'Enc1 theta':>16}  {'Enc2 phi':>16}")
    print("-" * 36)

    try:
        prev1, prev2 = None, None
        while True:
            if enc1_count != prev1 or enc2_count != prev2:
                print(f"{enc1_count:>16}  {enc2_count:>16}", end='\r')
                prev1, prev2 = enc1_count, enc2_count
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n\nFinal — Enc1: {enc1_count}  Enc2: {enc2_count}")
    finally:
        GPIO.cleanup()
        print("GPIO cleaned up.")


if __name__ == '__main__':
    main()