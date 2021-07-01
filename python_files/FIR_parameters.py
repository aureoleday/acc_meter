#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:43:13 2020

@author: aureoleday
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

ntaps = 48
fl,fh=0.082, 0.4
# b = signal.firwin(ntaps,fl,pass_zero=True)
b = signal.firwin(ntaps,fl)
w,h = signal.freqz(b)

# plt.plot(w, 20*np.log10(np.abs(h)), 'b')

def costas_param(N,nita,lb,dg):
    theta = (lb/N)/(nita+0.25/nita)
    k1 = (-4*nita*theta)/((1+2*nita*theta+theta**2)*dg)
    k2 = (-4*theta**2)/((1+2*nita*theta+theta**2)*dg)
    return k1,k2

print(np.array(costas_param(20, 0.707, 0.1, 1)))
# print(np.array(costas_param(16, 0.707, 0.1, 10))*(1<<20))


