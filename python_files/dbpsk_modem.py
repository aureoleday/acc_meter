#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:51:17 2020

@author: aureoleday
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from functools import reduce

b, a = signal.butter(3, [0.1], 'lowpass')

def demod(din,f0,fs):
    d_cnt = int(fs/f0)
    din_delay = np.zeros(din.size)
    din_delay[d_cnt:] = din[:-d_cnt]
    dcross = din*din_delay
    zi = signal.filtfilt(b,a,dcross)
    return zi

def AWGN(sin,snr):
    SNR = 10**(snr/10)
    print(SNR)
    Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    print(Ps)
    Pn = Ps/SNR
    print(Pn)
    s_wn = sin + np.random.randn(sin.size)*(Pn**0.5)
    return s_wn

f0 = 200
fs = 3200
N = 20

osc = np.cos(2*np.pi*f0*np.arange(fs/f0)/fs)

ms = np.kron((np.random.randint(0,2,N)-0.5)*2,osc)
msn = AWGN(ms,3)

dout = demod(msn,f0,fs)

# plt.figure()
plt.plot(ms)
# plt.figure()
plt.plot(msn)
# plt.plot(dout)