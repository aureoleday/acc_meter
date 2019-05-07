# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:37:17 2019

@author: Administrator
"""

import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

x = np.arange(0,32*np.pi,2*np.pi/64)

inp = np.sin(x)

yy = fft(inp)

for i in range(100):
    print("%d: r:%f,i:%f\n" % (i, yy.real[i],yy.imag[i]))
#print("real\n")
#print(yy.real)
#
#print("\nimag\n")
#print(yy.imag)

