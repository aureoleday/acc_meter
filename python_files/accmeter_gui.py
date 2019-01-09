# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:52:21 2018

@author: Administrator
"""

import socket
import threading
import time
import struct
import queue
import serial

import numpy as np
from scipy.fftpack import fft 
from ringbuf import RingBuffer
import matplotlib.pyplot as plt 
from matplotlib import animation

SYNC_HEAD = b'\xdf\x1b\xdf\x9b'

# 4 250Hz
# 5 125Hz
# 6 62.5Hz
# 7 31.25Hz
# 8 15.625Hz
# 9 7.813Hz
# 10 3.906Hz

FS = 31.25
WINDOW_SIZE = 256

rb = RingBuffer(WINDOW_SIZE,3)
in_buf = []
inb_q = queue.Queue(0)
gain = 3.9e-6

def func(a):
    if a&(1<<23) > 0:
        temp = a | 0xff000000
    else:
        temp = a
    temp = struct.unpack('i',struct.pack('L',temp))
    return temp[0]>>4


def pkg_resolve():
#    while True:
        arr = []
        xyz1 = []
        j = 0
        flag = 0
        axis = 0
        q_len = inb_q.qsize()
        if q_len != 0:
            for i in range(q_len):
                temp = inb_q.get(block=False)
                arr.append(temp)
                if i>=4:
                    if bytes(arr[i-4:i]) == SYNC_HEAD:
                        flag = 1
                if flag > 0:
                    j = j+1
                    if j == 5:                
                        CMD = int.from_bytes(bytes(arr[i-4:i]),byteorder='little', signed=False)
                        pkg_len = CMD&0x0ffff
                        xyz1 = []
                    elif j%4 == 1:
                        buf = int.from_bytes(bytes(arr[i-4:i]),byteorder='little', signed=False)
                        if buf&1 == 1:
                            axis = 1
                            xyz1 = []
                            xyz1.append(buf)
                        elif axis == 1:
                            xyz1.append(buf)
                            axis = axis + 1
                        elif axis == 2:
                            xyz1.append(buf)
                            ret = np.array(list(map(func,xyz1)))*gain
                            try:
                                rb.append(ret)
                            except:
                                break;
                            else:
                                continue
                            axis = 0                
                    if j>5 and j >= 5+pkg_len*4:
                        break                    
                    
def t_resolve():
    while True:
        if inb_q.qsize() > 0:
            pkg_resolve()
        time.sleep(0.001)    
                    

def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    while True:
        data = sock.recv(4096)
        if data != '':
#            in_buf.extend(data)
            inb_q.queue.extend(data)
        time.sleep(0.001)
    sock.close()
    print('Connection from %s:%s closed.' % addr)

def sock_init(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', port))    
    s.listen(5)
    print('Waiting for connection...')
    sock, addr = s.accept()
    threads = []
    t1 = threading.Thread(target=tcplink, args=(sock, addr))
    threads.append(t1)
    t2 = threading.Thread(target=t_resolve)
    threads.append(t2)
    for t in threads:
        t.setDaemon(True)
        t.start()
#    tr.start()


#    tr.start()

def hexsend(string_data=''):
    hex_data = string_data.decode("hex")
    return hex_data

def ser_init():
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
                inb_q.queue.extend(data)
#                print("receive:", data)
#                if data != b'': 
#                    print("receive:", data) 
#                else: 
#                    ser.write(hexsend(data)) 
                time.sleep(0.001)
    except KeyboardInterrupt: 
        if serial != None: 
            ser.close()

def sys_init():

    threads = []
    t1 = threading.Thread(target=ser_init)
    threads.append(t1)
    t2 = threading.Thread(target=t_resolve)
    threads.append(t2)
    for t in threads:
        t.setDaemon(True)
        t.start()

#sock_init(8888)
sys_init()

fig = plt.figure()
ax = fig.add_subplot(321)
af = fig.add_subplot(322)
bx = fig.add_subplot(323)
bf = fig.add_subplot(324)
cx = fig.add_subplot(325)
cf = fig.add_subplot(326)
 
x = np.arange(0,WINDOW_SIZE)/FS
xh = np.arange(0,WINDOW_SIZE/2)*FS/WINDOW_SIZE
y = np.random.randn(len(x),3)

linex, = ax.plot(x,np.sin(x),color='r')
linexf, = af.plot(xh,np.sin(xh),color='r')
liney, = bx.plot(x,np.sin(x),color='b')
lineyf, = bf.plot(xh,np.sin(xh),color='b')
linez, = cx.plot(x,np.sin(x),color='purple')
linezf, = cf.plot(xh,np.sin(xh),color='purple')

def gen_frames():
    yield 0

def my_fft(din):
    fftx = fft(din)
    abs_x = abs(fftx)/WINDOW_SIZE
    return abs_x[int(WINDOW_SIZE/2):]

def update(i):

    temp = rb.view
#    print(temp)
    habx,haby,habz = map(my_fft,[temp[:,0],temp[:,1],temp[:,2]])

    linex.set_ydata(temp[:,0])
    ax.set_ylim(np.min(temp[:,0]),np.max(temp[:,0]))
    linexf.set_ydata(habx)
    af.set_ylim(np.min(habx),np.max(habx))
    
    liney.set_ydata(temp[:,1])
    bx.set_ylim(np.min(temp[:,1]),np.max(temp[:,1]))
    lineyf.set_ydata(haby)
    bf.set_ylim(np.min(haby),np.max(haby))
    
    linez.set_ydata(temp[:,2])
    cx.set_ylim(np.min(temp[:,2]),np.max(temp[:,2]))
    linezf.set_ydata(habz)
    cf.set_ylim(np.min(habz),np.max(habz))
        
#    return linex,

def initial():
    linex.set_ydata(np.sin(x))
    linexf.set_ydata(np.zeros(int(WINDOW_SIZE/2)))
    liney.set_ydata(np.sin(x))
    lineyf.set_ydata(np.zeros(int(WINDOW_SIZE/2)))
    linez.set_ydata(np.sin(x))
    linezf.set_ydata(np.zeros(int(WINDOW_SIZE/2)))
    ax.set_ylim(-3000,3000)
#    ax.set_xlabel("time")
    ax.set_ylabel("x(g)")
    af.set_ylim(-3000,3000)
#    af.set_xlabel("freq")
    af.set_ylabel("Amp-x")    
    bx.set_ylim(-3000,3000)
#    bx.set_xlabel("time")
    bx.set_ylabel("y(g)")
    bf.set_ylim(-3000,3000)
#    bf.set_xlabel("freq")
    bf.set_ylabel("Amp-y")      
    cx.set_ylim(-3000,3000)
    cx.set_ylabel("z(g)")
    cx.set_xlabel("time(s)")
    cf.set_ylim(-3000,3000)
    cf.set_xlabel("freq(Hz)")
    cf.set_ylabel("Amp-z")      
    return linex,

ani = animation.FuncAnimation(fig=fig,func=update,frames=gen_frames,init_func=initial,interval=10,blit=False)

plt.show()

