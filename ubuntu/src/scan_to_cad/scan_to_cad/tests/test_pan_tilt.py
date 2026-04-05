"""
test_pan_tilt.py
────────────────
Standalone pan-tilt motor test — NO ROS 2 required.

Tests Motor A and Motor B independently, then tests differential moves
(pure pan = both same direction, pure tilt = opposite directions).

Usage:
  python3 tests/test_pan_tilt.py

Pin assignments must match pan_tilt_controller.py.
"""

import RPi.GPIO as GPIO
import time
import threading

# ── Pin assignments — must match pan_tilt_controller.py ──────────────────────
PA_STEP = 16;  PA_DIR = 20;  PA_EN = 21
PB_STEP = 12;  PB_DIR = 7;   PB_EN = 8

STEP_DELAY = 0.002
TEST_STEPS = 200
PAUSE_S    = 1.0


def setup():
    GPIO.setmode(GPIO.BCM)
    for pin in [PA_STEP, PA_DIR, PA_EN, PB_STEP, PB_DIR, PB_EN]:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(PA_EN, GPIO.LOW)
    GPIO.output(PB_EN, GPIO.LOW)
    print("GPIO initialised. Both pan-tilt drivers enabled.")


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


def move_both(steps_a, steps_b):
    t_a = threading.Thread(target=move, args=(PA_STEP, PA_DIR, steps_a, "Motor A"))
    t_b = threading.Thread(target=move, args=(PB_STEP, PB_DIR, steps_b, "Motor B"))
    t_a.start(); t_b.start()
    t_a.join();  t_b.join()


def main():
    print("═" * 60)
    print("  scan_to_cad — Pan-Tilt Motor Test")
    print(f"  Steps: {TEST_STEPS}  |  Delay: {STEP_DELAY*1000:.1f} ms/edge")
    print("═" * 60)

    try:
        setup()
        time.sleep(0.5)

        print("\n── Test 1: Motor A only ──────────────────────────────────────")
        move(PA_STEP, PA_DIR,  TEST_STEPS, "Motor A")
        time.sleep(PAUSE_S)
        move(PA_STEP, PA_DIR, -TEST_STEPS, "Motor A")
        time.sleep(PAUSE_S)

        print("\n── Test 2: Motor B only ──────────────────────────────────────")
        move(PB_STEP, PB_DIR,  TEST_STEPS, "Motor B")
        time.sleep(PAUSE_S)
        move(PB_STEP, PB_DIR, -TEST_STEPS, "Motor B")
        time.sleep(PAUSE_S)

        print("\n── Test 3: Both same direction (pure pan) ────────────────────")
        print("  Expected: head pans, no tilt change.")
        move_both( TEST_STEPS,  TEST_STEPS)
        time.sleep(PAUSE_S)
        move_both(-TEST_STEPS, -TEST_STEPS)
        time.sleep(PAUSE_S)

        print("\n── Test 4: Opposite directions (pure tilt) ───────────────────")
        print("  Expected: head tilts, no pan change.")
        move_both( TEST_STEPS, -TEST_STEPS)
        time.sleep(PAUSE_S)
        move_both(-TEST_STEPS,  TEST_STEPS)

        print("\n✓ All pan-tilt tests complete.")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        GPIO.output(PA_EN, GPIO.HIGH)
        GPIO.output(PB_EN, GPIO.HIGH)
        GPIO.cleanup()
        print("GPIO cleaned up.")


if __name__ == '__main__':
    main()