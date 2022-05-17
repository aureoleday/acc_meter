#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 13:50:12 2022

@author: aureoleday
"""

import numpy as np
from scipy import signal
from functools import reduce
import matplotlib.pyplot as plt

alpha = 6
b = [1,-1]
a = [1,(2**(2*alpha-16)-1)]

# b,a= signal.butter(3,[0.01],'highpass')
# print(b)
# print(a)

w,h = signal.freqz(b,a)
wgd,gd = signal.group_delay((b,a))

fig, ax1 = plt.subplots()
w=w/2/np.pi
ax1.plot(w,20*np.log10(abs(h)),color='r')
ax1.set_xlabel("normalized frequency")
ax1.set_ylabel('Amplitude [dB]',color='r')

ax2 = ax1.twinx()

ax2.plot(w,gd,color='g')
ax2.set_ylabel('group delay[samples]',color='g')
# ax2.plot(w,np.unwrap(np.angle(h)),color='g')
# ax2.set_ylabel('angle[rad]',color='g')
# plt.show()