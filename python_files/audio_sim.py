#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 17:17:32 2019

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt


FS = 4800
WINDOW_SIZE = 2**14

fft_size = WINDOW_SIZE

Ts = 100

f1 = 400
f2 = f1*3

delta = np.pi*2

t = np.arange(FS*Ts)/FS

tspan = int(t.size/80)

def choose_windows(name='Hanning', N=20): # Rect/Hanning/Hamming 
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Rect':
        window = np.ones(N) 
    return window

def my_fft(din):
    temp = din[:fft_size]*choose_windows(name='Hanning',N=fft_size)
    fftx = np.fft.rfft(temp)*2/fft_size
    ampl = np.abs(fftx)
    ph = np.angle(fftx)
    return ampl,ph


sig_a = np.sin(2*np.pi*f1*t)
sig_b = np.sin(2*np.pi*f2*t + delta)
noise = np.random.randn(sig_a.size)

sig_sum = sig_a + sig_b + noise
#sig_sum = sig_a

xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(sig_sum)
x = np.arange(0,WINDOW_SIZE)/FS
plt.subplot(211)
plt.plot(xh,habx_t)
plt.subplot(212)
plt.plot(xh,ph)
plt.show()
#plt.plot(t[:tspan],sig_sum[:tspan])

