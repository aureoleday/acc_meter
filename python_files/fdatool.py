#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:20:31 2020

@author: aureoleday
"""
import matplotlib.pyplot as plt
# from scipy import signal
import scipy.signal as signal
import numpy as np

FS = 4e3
nyq = FS/2
T = 1/FS
filt_ord = 31

FC = 0.06*nyq

b = signal.firwin(filt_ord,cutoff=FC/nyq,window=("kaiser",8),pass_zero='highpass')
w, h = signal.freqz(b)

fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid()
ax2.axis('tight')
plt.show()
c = np.array(list(map(int,b*(2**17))))
print(c)

