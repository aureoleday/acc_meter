#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:51:17 2020

@author: aureoleday
"""
#from gnuradio import gr
#from gnuradio import audio,analog

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from functools import reduce

b, a = signal.butter(3, [0.1], 'lowpass')

#def am_mod(mt,fc,fs,ofs_ca,ofs_f):

def sig_src(fs,wave_form,fc,ampl,phi,t):
    Ts = 1/fs
    n = t/Ts
    n = np.arange(n)
    if(wave_form == "sin"):
        sig = ampl*np.sin(2*np.pi*fc*n*Ts + phi*(np.pi/180))
    else:
        sig = ampl*np.cos(2*np.pi*fc*n*Ts + phi*(np.pi/180))
    return sig
        
# def orthognal_demod(fs,sig_in,fc):
#     Ts = 1/fs
#     sig_len = len(sig_in)
#     n = np.arange(sig_len)
#     q = np.sin(2*np.pi*fc*n*Ts)
#     i = np.cos(2*np.pi*fc*n*Ts)
    
def diff_demod(fs,fc,fr,i_cd,q_cd,i_rd,q_rd):
    Ts = 1/fs
    c_len = len(i_cd)
    n = np.arange(c_len)
    cq = np.sin(2*np.pi*fc*n*Ts)
    ci = np.cos(2*np.pi*fc*n*Ts)
    rq = np.sin(2*np.pi*fr*n*Ts)
    ri = np.cos(2*np.pi*fr*n*Ts)

# def demod(din,f0,fs):
#     d_cnt = int(fs/f0)
#     din_delay = np.zeros(din.size)
#     din_delay[d_cnt:] = din[:-d_cnt]
#     dcross = din*din_delay
#     zi = signal.filtfilt(b,a,dcross)
#     return zi#sig_fs

#osc = np.cos(2*np.pi*f0*np.arange(fs/f0)/fs)

#ms = np.kron((np.random.randint(0,2,N)-0.5)*2,osc)

def AWGN(sin,snr):
    SNR = 10**(snr/10)
    print(SNR)
    Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    print(Ps)
    Pn = Ps/SNR
    print(Pn)
    s_wn = sin + np.random.randn(sin.size)*(Pn**0.5)
    return s_wn

#f0 = 200
fs = 1000
fc = 100
f0 = 10
t = 0.3
SNR=50

e = sig_src(fs,'sin',fc,1,0,t)
s = sig_src(fs,'sin',f0,1,0,t)
c = sig_src(fs,'cos',f0,1,0,t)

es = e*s
ec = e*c

esn = AWGN(es,SNR)
ecn = AWGN(ec,SNR)

arctan = np.arctan(esn/ecn)
arctan_n = np.arctan(es/ec)

#msn = AWGN(ms,SNR)

#dout = demod(msn,f0,fs)

# plt.figure()
# plt.plot(ec)
# plt.figure()
# # plt.plot(arctan)
# plt.plot(e)
# fig = plt.figure()
ax = plt.subplot(411)
ax.set_title('sin(w),SNR = 30')
ax.plot(esn,color='g')
ax = plt.subplot(412)
ax.set_title('cos(w),SNR = 30')
ax.plot(ecn,color='g')
ax = plt.subplot(413)
ax.set_title('actan(x) with SNR = 30')
ax.plot(arctan,color='r')
bx = plt.subplot(414)
bx.set_title('actan(x) without noise')
bx.plot(arctan_n,color='b')