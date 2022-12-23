# -*- coding: utf-8 -*-
import serial
import numpy as np
import time

# cmds_np = np.array([[45,  5,  25],# サーボ1, サーボ2は0度までは動けない.
#                     [90,  5,  25],
#                     [45,  5,  25],
#                     [ 5,  5,  25],
#                     [ 5, 45,  25],
#                     [ 5, 90,  25],
#                     [ 5, 45,  25],
#                     [ 5,  5,  25],
#                     [ 5,  5,  0]])

# cmds_np = np.array([[ 5,  5,  0],
#                     [ 5,  5, 25],
#                     [ 5,  5,  0],
#                     [ 5,  5, 25],
#                     [ 5,  5,  0],
#                     [ 5,  5, 25],
#                     [ 5,  5,  0]])

def send(cmds_np):
    cmds_list = cmds_np.tolist()
    # シリアル通信の設定(
    ser = serial.Serial("/dev/cu.usbmodem14401", 9600, timeout=1)
    time.sleep(1)
    for cmd in cmds_list:
        angle1, angle2, angle3 = cmd
        ser.write(str.encode(f"{angle1},{angle2},{angle3},\0"))
        print(f"{angle1},{angle2},{angle3},\0")
        time.sleep(0.1)

if __name__ == '__main__':
    cmds_np = np.array([[45,  5,  25],# サーボ1, サーボ2は0度までは動けない.
                    [90,  5,  25],
                    [45,  5,  25],
                    [ 5,  5,  25],
                    [ 5, 45,  25],
                    [ 5, 90,  25],
                    [ 5, 45,  25],
                    [ 5,  5,  25],
                    [ 5,  5,  0]])
    send(cmds_np)