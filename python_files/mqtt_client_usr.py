# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
from functools import reduce
import paho.mqtt.client as mqtt

HOST = "127.0.0.1"
PORT = 1883
 
def on_message_callback(client, userdata, message):
    if ( not hasattr(on_message_callback,'x')):   #hasattr函数的第一个变量为当前函数名，第二个为变量名，加单引号
        on_message_callback.x = 0 
    # print(message.topic+" " + ":" + str(message.payload))下一页
    # print(message.payload.hex())
    on_message_callback.x += 1
    temp = on_message_callback.x
    if(temp%10 == 0):
        print(on_message_callback.x)
    # sys.exit()
 
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("chanel_01")
 
 
def write_reg(addr,data):
    tmp = bytearray([0x1b,0xdf,0x20,0x05,0x04,0x00,0x00,0x00,0x00])
    tmp[4] = addr
    tmp[5:8] = (data).to_bytes(4,byteorder='little')
    tmp[9] = reduce(lambda x,y:x^y, tmp[4:-1])
    return tmp

def read_reg(addr):
    tmp = bytearray([0x1b,0xdf,0x10,0x01,0x04,0x00])
    tmp[4] = addr
    tmp[5] = addr
    return tmp

 
def wr_reg(addr,data):
    tmp = write_reg(addr,data)
    client = mqtt.Client('test')
    client.connect(HOST, PORT, 60)
    client.username_pw_set('admin', 'password')
    client.on_connect = on_connect
    client.publish("/svr/data", tmp, 1)
    # client.subscribe('gg')
    
def rd_reg(addr):
    tmp = read_reg(addr)
    client = mqtt.Client('test')
    client.connect(HOST, PORT, 60)
    client.username_pw_set('admin', 'password')
    client.on_connect = on_connect
    client.publish("/svr/data", tmp, 1)
    # client.subscribe('gg')

def main():
    # tmp = bytearray([0x1b,0xdf,0x10,0x01,0x04,0x00])
    rd_reg(6)
    # wr_reg(4,32)
    client = mqtt.Client('test')
    client.connect(HOST, PORT, 60)
    client.username_pw_set('admin', 'password')
    client.on_connect = on_connect
    # client.publish("/svr/data", tmp, 1)
    client.subscribe('/clt/data')
    client.on_message = on_message_callback
    client.loop_forever()

if __name__ == '__main__':
    main()
