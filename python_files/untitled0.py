#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:20:31 2020

@author: aureoleday
"""

# from scipy import signal
import scipy.signal as signal
import numpy as np

FS = 4e3
nyq = FS/2
T = 1/FS
filt_ord = 50

FC = 1e3

a = signal.firwin(filt_ord,FC/nyq,"hann")

print(a)

