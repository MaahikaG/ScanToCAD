"""
test_motors.py
──────────────
Standalone gantry motor test — NO ROS 2 required.

Run directly on the Pi to verify Motor 1 (theta) and Motor 2 (phi)
are wired and working before launching any ROS 2 nodes.

Usage:
  python3 tests/test_motors.py

What it does:
  1. Motor 1 (theta) — 200 steps forward, pause, 200 steps back.
  2. Motor 2 (phi)   — same.
  3. Both simultaneously — 200 steps forward, then back.

Troubleshooting:
  Motor doesn't move      → Check EN pin is LOW, VMOT on 12V,
                             SLEEP+RESET tied together on A4988.
  Wrong direction         → Swap one coil pair (1A↔1B or 2A↔2B on A4988).
  Skips steps / stalls    → Increase STEP_DELAY, or turn up A4988 current
                             limit potentiometer (clockwise = more current).
  One motor slower        → Different current limits on the two A4988 boards.
"""

import RPi.GPIO as GPIO
import time
import threading

# ── Pin assignments — must match motor_controller.py ──────────────────────────
M1_STEP = 17;  M1_DIR = 27;  M1_EN = 22    # Motor 1 (theta)
M2_STEP = 23;  M2_DIR = 24;  M2_EN = 25    # Motor 2 (phi)

STEP_DELAY = 0.002      # slower than production — safer for first test
TEST_STEPS = 200
PAUSE_S    = 1.0


def setup():
    GPIO.setmode(GPIO.BCM)
    for pin in [M1_STEP, M1_DIR, M1_EN, M2_STEP, M2_DIR, M2_EN]:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(M1_EN, GPIO.LOW)
    GPIO.output(M2_EN, GPIO.LOW)
    print("GPIO initialised. Both drivers enabled.")


def move(step_pin, dir_pin, steps, label=""):
    direction = "forward" if steps > 0 else "backward"
    print(f"  {label}: {abs(steps)} steps {direction}...")
    GPIO.output(dir_pin, GPIO.HIGH if steps > 0 else GPIO.LOW)
    for _ in range(abs(steps)):
        GPIO.output(step_pin, GPIO.HIGH)
        time.sleep(STEP_DELAY)
        GPIO.output(step_pin, GPIO.LOW)
        time.sleep(STEP_DELAY)
    print(f"  {label}: done.")


def move_both(steps_m1, steps_m2):
    t1 = threading.Thread(target=move, args=(M1_STEP, M1_DIR, steps_m1, "Motor 1"))
    t2 = threading.Thread(target=move, args=(M2_STEP, M2_DIR, steps_m2, "Motor 2"))
    t1.start(); t2.start()
    t1.join();  t2.join()


def main():
    print("═" * 60)
    print("  scan_to_cad — Gantry Motor Test")
    print(f"  Steps: {TEST_STEPS}  |  Delay: {STEP_DELAY*1000:.1f} ms/edge")
    print("═" * 60)

    try:
        setup()
        time.sleep(0.5)

        print("\n── Test 1: Motor 1 (theta — Z rotation) ─────────────────────")
        print("  Expected: arm rotates CCW, pauses, returns CW.")
        move(M1_STEP, M1_DIR,  TEST_STEPS, "Motor 1 (theta)")
        time.sleep(PAUSE_S)
        move(M1_STEP, M1_DIR, -TEST_STEPS, "Motor 1 (theta)")
        time.sleep(PAUSE_S)

        print("\n── Test 2: Motor 2 (phi — arc position) ─────────────────────")
        print("  Expected: head moves along arc toward phi=90, pauses, returns.")
        move(M2_STEP, M2_DIR,  TEST_STEPS, "Motor 2 (phi)")
        time.sleep(PAUSE_S)
        move(M2_STEP, M2_DIR, -TEST_STEPS, "Motor 2 (phi)")
        time.sleep(PAUSE_S)

        print("\n── Test 3: Both motors simultaneously ────────────────────────")
        print("  Expected: both move forward together, then both return.")
        move_both( TEST_STEPS,  TEST_STEPS)
        time.sleep(PAUSE_S)
        move_both(-TEST_STEPS, -TEST_STEPS)

        print("\n✓ All motor tests complete.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        GPIO.output(M1_EN, GPIO.HIGH)
        GPIO.output(M2_EN, GPIO.HIGH)
        GPIO.cleanup()
        print("GPIO cleaned up.")


if __name__ == '__main__':
    main()