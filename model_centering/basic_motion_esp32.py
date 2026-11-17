#!/usr/bin/env python3
"""Run the JetBot basic-motion examples through the ESP32 serial protocol.

Stop the existing Jetson car service before using this script: only one
process may own the ESP32 serial port at a time.
"""
import argparse
import time

import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('forward', 'backward', 'left', 'right', 'motors', 'stop'))
    parser.add_argument('--port', default='/dev/ttyACM0')
    parser.add_argument('--speed', type=float, default=0.3)
    parser.add_argument('--left', type=float, default=0.3)
    parser.add_argument('--right', type=float, default=0.6)
    parser.add_argument('--seconds', type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.speed <= 1.0:
        parser.error('--speed must be between 0 and 1')
    if not -1.0 <= args.left <= 1.0 or not -1.0 <= args.right <= 1.0:
        parser.error('--left and --right must be between -1 and 1')
    if not 0.0 < args.seconds <= 5.0:
        parser.error('--seconds must be greater than 0 and at most 5')

    if args.action == 'motors':
        command = 'SET_MOTORS,{:.3f},{:.3f}'.format(args.left, args.right)
    elif args.action == 'stop':
        command = 'STOP'
    else:
        command = '{},{}'.format(args.action.upper(), args.speed)

    with serial.Serial(args.port, 115200, timeout=0.1, write_timeout=0.2) as port:
        time.sleep(1.2)  # ESP32 commonly restarts when the serial port opens.
        print('Sending: {}'.format(command), flush=True)
        try:
            if command != 'STOP':
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    port.write((command + '\n').encode('ascii'))
                    time.sleep(0.1)  # Refresh the ESP32's 500 ms watchdog.
        finally:
            port.write(b'STOP\n')
            port.flush()
            print('STOP', flush=True)


if __name__ == '__main__':
    main()
