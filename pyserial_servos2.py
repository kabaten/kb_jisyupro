# -*- coding: utf-8 -*-
import serial
import numpy as np
import time

cmds_np = np.array([[45,  0,  0],
                    [90,  0,  0],
                    [45,  0,  0],
                    [ 0,  0,  0],
                    [ 0, 45,  0],
                    [ 0, 90,  0],
                    [ 0, 45,  0],
                    [ 0,  0,  0],
                    [ 0,  0, 45],
                    [ 0,  0, 90],
                    [ 0,  0, 45],
                    [ 0,  0,  0]])

def main():
    cmds_list = cmds_np.tolist()
    # シリアル通信の設定(
    ser = serial.Serial("/dev/cu.usbmodem14401", 9600, timeout=1)
    time.sleep(1)
    for cmd in cmds_list:
        angle1, angle2, angle3 = cmd
        ser.write(str.encode(f"{angle1},{angle2},{angle3},\0"))
        print(f"{angle1},{angle2},{angle3},\0")
        time.sleep(1)

if __name__ == '__main__':
    main()