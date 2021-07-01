#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:51:28 2020

@author: aureoleday
"""
from functools import reduce
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

cf = 200
spr = 4000
rep = 8
off = 20


def sig_gen(carrier_freq,spr,rep,off):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    sig_bits = np.zeros(rep+2*off)
    sig_bits[off:rep+off] = 1
    print(sig_bits)
    return np.kron(sig_bits,osc_cos)


def matched_filter(din,carrier_freq,spr,rep):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    
    lc = np.kron(np.ones(rep),osc_cos)
    print(lc.size)
    a = np.array(0)
    for x in range(din.size):
        if(x<lc.size):
            temp = reduce(lambda x,y:x+y,np.append(din[:x],np.zeros(lc.size-x))*lc)
        else:
            temp = reduce(lambda x,y:x+y,din[(x-lc.size):x]*lc)
        a = np.append(a,temp)
    return a

data = sig_gen(cf,spr,rep,off)
out = matched_filter(data,cf,spr,rep)
plt.plot(out)