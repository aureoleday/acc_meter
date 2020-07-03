#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:08:40 2020

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt 

FS = 2**12
T = 1
t_span = np.arange(0,T,1/FS)

N = 1
fft_size = N*FS
WINDOW_SIZE = N*FS

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
    fftx = np.fft.rfft(temp)*2/fft_size
    ampl = np.abs(fftx)
    ph = np.angle(fftx)
    return ampl,ph


f0 = 100
f1 = 105


noise = np.random.randn(t_span.size)
sig = np.cos(2*np.pi*f0*t_span)+np.cos(2*np.pi*f1*t_span) + 5*noise
sig = np.kron(np.ones(N),sig)

sig_exp = np.kron(sig,[1,0,0,0])
#plt.plot(t_span,sig)


xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(sig)
x = np.arange(0,WINDOW_SIZE)/FS
plt.plot(habx_t)
