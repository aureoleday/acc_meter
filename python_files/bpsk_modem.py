#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:28:40 2020

@author: aureoleday
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 


def diff(din,mode = 'encode'):
    din_t = din.astype(int)
    d = np.zeros(din.shape[0]).astype(int)
    d[0]=din_t[0]
    if mode == 'encode':
        for i in range(din.shape[0]):
            if i>0:
                d[i] = d[i-1]^din_t[i]
    else:
        for i in range(din.shape[0]):
            if i>0:
                d[i] = din_t[i-1]^din_t[i]
    return d

def B2IQ(din, mode = 'bpsk', dif = 'off', repeat = 1):        
    I = np.kron(din[::2],np.ones(repeat)).astype(int)
    if mode == 'bpsk':
        Q = np.zeros(I.shape)
    elif mode == 'qpsk':
        Q = np.kron(din[1::2],np.ones(repeat)).astype(int)
    
    if(dif == 'on'):
        I = diff(I,'encode')
        Q = diff(Q,'encode')

    return np.vstack((I,Q))


def mod_iq(IQ, carrier_freq = 1000, spr = 4000, offset = 'off'):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    osc_sin = np.sin(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    mod_I = np.kron(IQ[0],osc_cos)
    mod_Q = np.kron(IQ[1],osc_sin)
    if (offset == 'on'):
        mod_Q = np.roll(mod_Q,int(spr/(2*carrier_freq)))

    return mod_I + mod_Q

mode = 'bpsk'
dif = 'off'
s2p = B2IQ(np.random.randint(0,2,40),mode,dif,1)
