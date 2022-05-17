#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:33:19 2022

@author: aureoleday
"""

import numpy as np
from scipy import signal
from fxpmath import Fxp
import matplotlib.pyplot as plt
from functools import reduce

fs = 1000
fc = fs/64
ft = fs/8

N = 80

def tri(fs,fc,d):
    mod = fs/fc
    return (np.array(list(map(lambda x: x%mod if x%mod < mod/2 else mod-x%mod,d))) - mod/4)*4/mod

sin = np.sin(2*np.pi*np.arange(N)*fc/fs)
tri = tri(fs,ft,np.arange(N))
spwm = for (a,b) in zip(sin,tri)

# spwm = 

plt.figure()
plt.plot(sin)
plt.plot(tri)