#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:52:59 2020

@author: aureoleday
"""

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
import ctypes


import numpy as np
from scipy import signal
from ringbuf import RingBuffer
import matplotlib.pyplot as plt 
from matplotlib import animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

FSM_IDLE = 0
FSM_SYNC = 1
FSM_DATA = 2

SYNC_HEAD = b'\x9b\xdf'

# FS = 4000
FS = 4096
TARGET_FREQ = 470
FREQ_SPAN = 30
#FS = 4000
WINDOW_SIZE = 2**12
FFT_MAV_LEN = 32
#WINDOW_SIZE = 1024

fft_size = WINDOW_SIZE

rb = RingBuffer(WINDOW_SIZE,1)
in_buf = []
inb_q = queue.Queue(0)
#gain = 3.9e-6

def calc_ord(reg_val):
    fs = reg_val
    lpf = fs/4	
    lpf_reg = reg_val>>4
    if(lpf_reg == 0 ):
        hpf = 0
    else:
        hpf = 0.247*fs/(4**(lpf_reg-1))
    return fs,lpf,hpf


def func(a):
    temp = struct.unpack('f',struct.pack('I',a))
    return temp    

def checksum(arr_in):
    xsum = 0
    for item in arr_in[2:]:
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
        # print(bytes(self.arr[-2:]))
        if self.cstate == FSM_IDLE:
            if(bytes(self.arr[-2:]) == SYNC_HEAD):
                # print("OK")
                self.frame = []
                self.frame.append(int.from_bytes(bytes(SYNC_HEAD),byteorder='big', signed=False))
                self.cstate = FSM_SYNC
                self.i_cnt = 0
            else:
                if(self.i_cnt >0):
                    print("drop\n")
                self.i_cnt += 1
                self.cstate = FSM_IDLE
        elif self.cstate == FSM_SYNC:
            if(self.i_cnt >= 1):
                CMD = int.from_bytes(bytes(self.arr[-2:]),byteorder='big', signed=False)
                self.frame.append(CMD)
                self.cstate = FSM_DATA
                self.i_cnt = 0
            else:
                self.i_cnt += 1
                self.cstate = FSM_SYNC
        elif self.cstate == FSM_DATA:
            # off = (self.i_cnt>>2)
            # print(bytes(self.frame[0]))
            if(self.i_cnt&0x0003 == 0):
                if(self.i_cnt == 0):
                    # print(self.arr[-1])
                    self.i_cnt += 1 
                else:
                    buf = int.from_bytes(bytes(self.arr[-4:]),byteorder='little', signed=True)
                    self.frame.append(buf)
                    # fbuf = buf*0.00000009933*5        #gain 1
                    fbuf = buf*0.00000009933          #gain 2
                    # fbuf = buf*0.00000009933/5        #gain 3
                    # print(fbuf)
                    rb.append(fbuf) 
                    self.cstate = FSM_DATA
                    self.i_cnt += 1
            else:
                if(self.i_cnt >= ((self.frame[1]&0x0fff)-1)):
                    self.arr = []
                    self.frame = []
                    self.i_cnt = 0
                    self.cstate = FSM_IDLE
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
    
    def get(self,mtype='acc'):
        if mtype=='mac':
            return self.mav_buf.sum(axis=0)/self.mav_cnt
        else:
            return self.acc_buf/self.acc_cnt

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
    
class iirpeak_filter:
    def __init__(self,fs,f0,Q):
        self.b,self.a = signal.iirpeak(f0,Q,fs)
        self.z = np.zeros(max(len(self.a),len(self.b))-1,dtype=np.float)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout

class iirnotch_filter:
    def __init__(self,fs,f0,Q):
        self.b,self.a = signal.iirnotch(f0,Q,fs)
        self.z = np.zeros(max(len(self.a),len(self.b))-1,dtype=np.float)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout

fig = plt.figure()
ax = plt.subplot2grid((7,1),(0,0),rowspan=2)
af = plt.subplot2grid((7,1),(2,0),rowspan=4)
afs = plt.subplot2grid((7,1),(6,0),rowspan=2)
 
x = np.arange(0,WINDOW_SIZE)/FS
xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)

linex, = ax.plot(x,np.sin(x),'g')
linexf, = af.plot(xh,np.sin(xh),color = 'r',linestyle='-', marker=',')
linexfs, = afs.plot(xh,np.sin(xh),color = 'b',linestyle='-', marker=',')

#filt_inst = my_filter(3,[0.22,0.25],'bandpass')
# filt_inst = iirpeak_filter(FS,473,40)
filt_inst = iirnotch_filter(FS,50,40)


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
    temp = din[:fft_size]*choose_windows(name='Rect',N=fft_size)
#    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

def goertzel(din,k,N):
    win = choose_windows('Hanning',N)
    w = 2*np.pi*k/N
    coef = 2*np.cos(w)
    print("w:%f,coef:%f\n"%(w,coef))
    q1=0
    q2=0
    for i in range(N):
        x = din[i]*win[i]
        q0 = coef*q1 - q2 + x
        q2 = q1
        q1 = q0
    return np.sqrt(q1**2 + q2**2 - q1*q2*coef)*2/N

def update(i):
    temp = rb.view
    temp[:,0] = filt_inst.filt(temp[:,0])
    linex.set_ydata(temp[:,0])
    ax.set_ylim(np.min(temp[:,0]),np.max(temp[:,0]))       
      
    habx_t = my_fft(temp[:,0])
    # habx_t[:400] = 0.00000012
    # habx_t[2500:] = 0.000005
    habx = mav_inst.mav_insert(habx_t)
    linexf.set_ydata(habx)
    af.set_ylim(np.min(habx),np.max(habx))  
    
    if rb.flag == 1:
        habx_acc = mav_inst.acc_insert(habx_t)
        linexfs.set_ydata(habx_acc)
        afs.set_ylim(np.min(habx_acc),np.max(habx_acc))
        rb.reset_flag()
    else:
        habx_acc = mav_inst.get('acc')
        linexfs.set_ydata(habx_acc)
        afs.set_ylim(np.min(habx_acc),np.max(habx_acc))    

def initial():
    linex.set_ydata(np.sin(x))
    linexf.set_ydata(np.zeros(int(WINDOW_SIZE/2 + 1)))
    ax.set_ylim(-3,3)
    ax.set_xlabel("time")
    ax.set_ylabel("x(g)")    
    
    ax.grid(True, linestyle='-.')

    af.set_ylim(-1,1)
    af.grid(True, linestyle='-.')
    af.set_xlabel("Freq(Hz)")
    af.set_ylabel("Amp-z")
    
    afs.set_ylim(-1,1)
    afs.grid(True, linestyle='-.')
    afs.set_xlabel("Freq(Hz)")
    afs.set_ylabel("Amp-zs")    
    return linex,

try:    
    # FS,LPF,HPF = calc_ord(FS)
    # print("FS:%.3f,LPF:%.3f,HPF:%.3f\n" % (FS,LPF,HPF)).
    print("FS:%.3f\n" % (FS))
    # sys_init(mode=1,ip="192.168.1.105",port=9996)
    sys_init(mode=1,ip="192.168.3.4",port=9996)
    # sys_init(mode=1,ip="192.168.4.1",port=9996) 
    ani = animation.FuncAnimation(fig=fig,func=update,frames=gen_frames,init_func=initial,interval=100,blit=False)
    plt.show()
except KeyboardInterrupt:
    pass

