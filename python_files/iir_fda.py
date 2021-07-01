#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:32:32 2021

@author: aureoleday
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

b, a = signal.iirfilter(3, 50, rs=60,fs=4000,
                        btype='highpass', analog=False, ftype='cheby2')

w, h = signal.freqz(b, a, fs=4000)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
ax.set_title('Chebyshev Type II bandpass frequency response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude [dB]')
ax.axis((1, 2000, -100, 10))
ax.grid(which='both', axis='both')
plt.show()