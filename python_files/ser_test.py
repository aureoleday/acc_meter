# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:05:47 2019

@author: Administrator
"""


import serial

def hexsend(string_data=''):
    hex_data = string_data.decode("hex")
    return hex_data

if __name__ == '__main__':
    ser = serial.Serial("com16",115200)
    print(ser.name)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
    try:        
        while True:
            count = ser.inWaiting() 
            if count > 0:
                data = ser.read(count) 
                print("receive:", data)
#                if data != b'': 
#                    print("receive:", data) 
#                else: 
#                    ser.write(hexsend(data)) 
    except KeyboardInterrupt: 
        if serial != None: 
            ser.close()