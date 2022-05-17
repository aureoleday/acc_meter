#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 10:12:29 2022

@author: aureoleday
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class iir_filter_s(object):
    def __init__(self,k):
        self.k = k
        self.x = np.exp(0j)
        self.y = np.exp(0j)
    
    def filt(self,IQ):
        y = IQ - self.x+ self.k * self.y
        self.x = IQ
        self.y = y 
        return y
    
fm = 20
fs = 1000
N = 2000
dc = 5

K = 0.98

my_iir = iir_filter_s(K)

dout = []
din = []
for i in np.arange(N):
    iq = np.exp(1j*(2*np.pi*fm/fs*i)) +dc
    din.append(iq)
    dout.append(my_iir.filt(iq))

plt.close('all')
sin = list(map(lambda x:x.real,din))
sout = list(map(lambda x:x.real,dout))

fig, [ax1,ax2] = plt.subplots(2,sharex=True)
ax1.plot(sin)
ax2.plot(sout)



    