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
from scipy import signal
from ringbuf import RingBuffer
import matplotlib.pyplot as plt 
from matplotlib import animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

FSM_IDLE = 0
FSM_SYNC = 1
FSM_DATA = 2


SYNC_HEAD = b'\xdf\x1b\xdf\x9b'

# 0 4000Hz
# 1 2000Hz
# 2 1000Hz
# 3 500Hz
# 4 250Hz
# 5 125Hz
# 6 62.5Hz
# 7 31.25Hz
# 8 15.625Hz
# 9 7.813Hz
# 10 3.906Hz

FILTER_REG = 64

FS = 4000>>(FILTER_REG&0x0f)

#FS = 4000
WINDOW_SIZE = 2**14
FFT_MAV_LEN = 64
#WINDOW_SIZE = 1024

fft_size = WINDOW_SIZE

rb = RingBuffer(WINDOW_SIZE,1)
in_buf = []
inb_q = queue.Queue(0)
#gain = 3.9e-6

def calc_ord(reg_val):
    fs = 4000>>(reg_val&0x0f)
    lpf = fs/4
    lpf_reg = reg_val>>4
    if(lpf_reg == 0 ):
        hpf = 0
    else:
        hpf = 0.247*fs/(4**(lpf_reg-1))
    return fs,lpf,hpf


def func(a):
    temp = struct.unpack('f',struct.pack('L',a))
    return temp    

def checksum(arr_in):
    xsum = 0
    for item in arr_in:
        xsum ^=item
    return xsum

class pkg_fsm(object):
    def __init__(self):
        self.cstate = FSM_IDLE
        self.i_cnt = 0
        self.arr = []
        self.frame = []
    
    def resolve(self,din):
        self.arr.append(din)
        if self.cstate == FSM_IDLE:
            if(bytes(self.arr[-4:]) == SYNC_HEAD):
                self.frame = []
                self.frame.append(int.from_bytes(bytes(SYNC_HEAD),byteorder='little', signed=False))
                self.cstate = FSM_SYNC
                self.i_cnt = 0
            else:
                if(self.i_cnt >4):
                    print("drop\n")
                self.i_cnt += 1
                self.cstate = FSM_IDLE
        elif self.cstate == FSM_SYNC:
            if(self.i_cnt >= 3):
                CMD = int.from_bytes(bytes(self.arr[-4:]),byteorder='little', signed=False)
                self.frame.append(CMD)
                self.cstate = FSM_DATA
                self.i_cnt = 0
            else:
                self.i_cnt += 1
                self.cstate = FSM_SYNC
        elif self.cstate == FSM_DATA:
            off = (self.i_cnt>>2)
            if(self.i_cnt&0x03 == 3):
                if(off >= ((self.frame[1]&0x0ffff))):         
                    buf = int.from_bytes(bytes(self.arr[-4:]),byteorder='little', signed=False)
                    self.frame.append(buf)
                    if(checksum(self.frame) != 0):
                        print("chk erro")
                        for item in self.frame:
                            print(" %x " % item)
                    self.arr = []
                    self.frame = []
                    self.i_cnt = 0
                    self.cstate = FSM_IDLE        
                else:
                    buf = int.from_bytes(bytes(self.arr[-4:]),byteorder='little', signed=False)
                    self.frame.append(buf)
                    buf = func(buf)
                    rb.append(buf) 
                    self.cstate = FSM_DATA                    
                    self.i_cnt += 1
            else:
                self.i_cnt += 1

            
pfsm = pkg_fsm()
                
class my_mav(object):
    def __init__(self,row,col):
        self.mav_buf = np.zeros((int(row),int(col)))
        self.acc_buf = np.zeros(int(col))
        self.row_ind = 0
        self.mav_cnt = 0
        self.acc_cnt = 0
        self.row_max = row
        
    def acc_insert(self,din):
        self.acc_buf += din
        self.acc_cnt += 1
        return self.acc_buf/self.acc_cnt
        
    def mav_insert(self,din):
        self.mav_buf[self.row_ind] = din
        self.row_ind += 1
        if(self.row_ind >= self.row_max):
            self.row_ind = 0
            
        if(self.mav_cnt < self.row_max):
            self.mav_cnt += 1
        else:
            self.mav_cnt = self.mav_cnt
            
        return self.mav_buf.sum(axis=0)/self.mav_cnt

mav_inst = my_mav(FFT_MAV_LEN,(WINDOW_SIZE/2)+1)
    
def t_resolve():
    while True:
        lenq = inb_q.qsize()
        if lenq > 0:
            for i in range(lenq):
                buf = inb_q.get(block=False)
                pfsm.resolve(buf)
        else:
            time.sleep(0.01)                   

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
                time.sleep(0.001)
    except KeyboardInterrupt: 
        if serial != None: 
            ser.close()

def tcp_client_init(ip,port):
    ser_ip = ip
    ser_port = port
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp_client.connect((ser_ip,ser_port))
        print('connected')
        while True:
            data = tcp_client.recv(4096)
            if data != '':
                inb_q.queue.extend(data)                
            time.sleep(0.02)
        tcp_client.close()
        print('Connection closed.')
    except socket.error:
        print("fail to setup socket connection")
    tcp_client.close()
        

def sys_init(mode,ip,port):    
    threads = []
    if mode == 1:
        t1 = threading.Thread(target=tcp_client_init,args=(ip,port))
    elif mode == 2:
        t1 = threading.Thread(target=ser_init)
    threads.append(t1)
    t2 = threading.Thread(target=t_resolve)
    threads.append(t2)
    for t in threads:
        t.setDaemon(True)
        t.start()

class my_filter:
    def __init__(self,N,filt_zone=[0.2],filt_type='lowpass'):
        self.b,self.a = signal.butter(N, filt_zone, filt_type)
        self.z = np.zeros(max(len(self.a),len(self.b))-1,dtype=np.float)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout


fig = plt.figure()
#ax = fig.add_subplot(211)
#af = fig.add_subplot(212)
ax = plt.subplot2grid((4,1),(0,0))
af = plt.subplot2grid((4,1),(1,0),rowspan=3)
 
x = np.arange(0,WINDOW_SIZE)/FS
xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)

linex, = ax.plot(x,np.sin(x),'r')
linexf, = af.plot(xh,np.sin(xh),color = 'g',linestyle='-', marker=',')

filt_inst = my_filter(3,[0.22,0.25],'bandpass')

def gen_frames():
    yield 0

def choose_windows(name='Hanning', N=20): # Rect/Hanning/Hamming 
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Rect':
        window = np.ones(N) 
    return window

def my_fft(din):
    temp = din[:fft_size]*choose_windows(name='Hamming',N=fft_size)
#    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp


def update(i):
    temp = rb.view
    temp[:,0] = filt_inst.filt(temp[:,0])
    habx_t = my_fft(temp[:,0])
#    habx = mav_inst.acc_insert(habx_t)
    habx = mav_inst.mav_insert(habx_t)

    linex.set_ydata(temp[:,0])
    ax.set_ylim(np.min(temp[:,0]),np.max(temp[:,0]))
    linexf.set_ydata(habx)
#    af.set_ylim(np.min(habx),np.max(habx))  
    af.set_ylim(0,0.0001)        

def initial():
    linex.set_ydata(np.sin(x))
    linexf.set_ydata(np.zeros(int(WINDOW_SIZE/2 + 1)))
    ax.set_ylim(-3000,3000)
#    ax.set_xlabel("time")
    ax.set_ylabel("x(g)")    
    
    ax.grid(True, linestyle='-.')

    af.set_ylim(-3000,3000)
    af.grid(True, linestyle='-.')
#    af.set_xlabel("freq")
    af.set_ylabel("Amp-x")
    return linex,

try:    
    FS,LPF,HPF = calc_ord(FILTER_REG)
    print("FS:%.3f,LPF:%.3f,HPF:%.3f\n" % (FS,LPF,HPF))
    sys_init(mode=1,ip="192.168.1.100",port=9996)
#    sys_init(mode=1,ip="192.168.4.1",port=9996) 
    ani = animation.FuncAnimation(fig=fig,func=update,frames=gen_frames,init_func=initial,interval=50,blit=False)
    plt.show()
except KeyboardInterrupt:
    pass

