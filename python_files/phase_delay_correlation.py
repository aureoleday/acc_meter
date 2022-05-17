#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:55:33 2022

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt

fs = 1000
fc = 40
fm = 4
theta = 0
delta = 0.0
N = 80

def waveform(fs,fc,N,phi):
    return np.sin(2*np.pi*(fc/fs*np.arange(N)+phi))


q = waveform(fs,fc,N,0)
# i = waveform(fs,fc,N,0)*np.cos(2*np.pi*theta)
c = waveform(fs,fc,N,delta)

x = np.correlate(q,c,'same')
# y = np.correlate(i,c,'same')
# print("x:%f,y:%f\n" %(np.sign(x),np.sign(y)))

plt.close('all')
plt.plot(q,label='q')
plt.plot(c,label='c')
plt.plot(x,label='x')
# plt.plot(y,label='y')
# plt.plot(c)
plt.legend()

for i in np.arange(N):
    print("i:", i)
    q = waveform(fs,fc,N,0)
    c = waveform(fs,fc,N,i/N)
    x = np.correlate(q,c,'same')
    print(x)
