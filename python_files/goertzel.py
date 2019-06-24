# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 11:17:51 2019

@author: Administrator
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

sig_freq = 470
noise_freq = 360
target_freq = 470
sample_freq= 4000

N = 4000
k = (N*target_freq)/sample_freq


def windows(name='Hanning', N=20): # Rect/Hanning/Hamming 
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Rect':
        window = np.ones(N) 
    return window
    
def goertzel(din,k,N):
    win = windows('Hanning',N)
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

m = np.cos(2*np.pi*sig_freq*np.arange(N*2)/sample_freq)
cn = np.cos(2*np.pi*noise_freq*np.arange(N*2)/sample_freq)
wn = np.random.randn(2*N)
cs = m*1+cn+wn*0
x = np.arange(m.shape[0])/sample_freq
#fig = plt.figure()
#ax = fig.add_subplot(211)
#ax.plot(x[:N],m[:N])
print("k:%d,N:%d"%(k,N))
print(goertzel(cs[:N],k,N))

