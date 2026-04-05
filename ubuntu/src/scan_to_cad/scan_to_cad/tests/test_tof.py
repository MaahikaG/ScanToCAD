"""
test_tof.py
───────────
Standalone VL53L0X TOF sensor test — NO ROS 2 required.

Prints live distance readings at ~10 Hz.

Usage:
  python3 tests/test_tof.py

Install library if needed:
  pip install VL53L0X
"""

import VL53L0X
import time


def main():
    print("Initialising VL53L0X...")
    tof = VL53L0X.VL53L0X()
    tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
    print("Ready. Hold objects at varying distances. Ctrl+C to exit.\n")
    print(f"{'#':>6}  {'mm':>10}  {'metres':>10}")
    print("-" * 32)

    n = 0
    try:
        while True:
            d = tof.get_distance()
            n += 1
            if d > 0:
                print(f"{n:>6}  {d:>10.1f}  {d/1000:>10.4f}")
            else:
                print(f"{n:>6}  {'OUT OF RANGE':>10}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tof.stop_ranging()
        print("Sensor stopped.")


if __name__ == '__main__':
    main()