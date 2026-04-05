"""
test_lcd.py
───────────
Standalone LCD test — NO ROS 2 required.

Writes test text to the display to verify wiring and contrast before
launching any ROS 2 nodes.

Usage:
  python3 tests/test_lcd.py

If you see nothing:
  - Adjust the contrast potentiometer on LCD pin 3 slowly.
  - Check 5V on pin 2 and GND on pin 1.

If you see solid black squares on one row:
  - The display is powered but the contrast is too high — turn the pot.

If characters are garbled:
  - Check D4–D7 wiring. A swapped data pin causes scrambled characters.

Pin assignments must match lcd_display.py.
"""

import RPi.GPIO as GPIO
from RPLCD.gpio import CharLCD
import time

LCD_RS = 26
LCD_E  = 18
LCD_D4 = 4
LCD_D5 = 9
LCD_D6 = 10
LCD_D7 = 11


def main():
    lcd = CharLCD(
        numbering_mode=GPIO.BCM,
        cols=16,
        rows=2,
        pin_rs=LCD_RS,
        pin_e=LCD_E,
        pins_data=[LCD_D4, LCD_D5, LCD_D6, LCD_D7],
        compat_mode=True,
    )

    print("Writing to LCD...")

    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string('scan_to_cad Ready')
    lcd.cursor_pos = (1, 0)
    lcd.write_string('Testing LCD...')
    time.sleep(3)

    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string('T:  45.0 P:  90.0')
    lcd.cursor_pos = (1, 0)
    lcd.write_string('TOF:  0.523 m')
    time.sleep(3)

    lcd.clear()
    lcd.write_string('Test complete.')
    time.sleep(2)

    lcd.clear()
    lcd.close(clear=True)
    GPIO.cleanup()
    print("Done.")


if __name__ == '__main__':
    main()