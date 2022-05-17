#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:28:40 2020

@author: aureoleday
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from functools import reduce


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
    print("sizeof din:",np.size(din))
    if mode == 'bpsk':
        Q = np.zeros(I.shape)
    elif mode == 'qpsk':
        Q = np.kron(din[1::2],np.ones(repeat)).astype(int)
    
    if(dif == 'on'):
        I = diff(I,'encode')
        Q = diff(Q,'encode')

    return np.vstack((I,Q))


def mod_iq(IQ, carrier_freq = 1000, spr = 16000, offset = 'off'):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    osc_sin = np.sin(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    # print(osc_cos)
    mod_I = np.kron(IQ[0],osc_cos)
    mod_Q = np.kron(IQ[1],osc_sin)
    # print(mod_I)
    if (offset == 'on'):
        mod_Q = np.roll(mod_Q,int(spr/(2*carrier_freq)))

    return mod_I
    # return mod_I + mod_Q

def matched_filter_re(din,pattern,carrier_freq,spr,rep):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    
    lc = np.kron(pattern,osc_cos)
    print(pattern)
    a = np.array(0)
    for x in range(din.size):
        if(x<lc.size):
            temp = reduce(lambda x,y:x+y,np.append(din[:x],np.zeros(lc.size-x))*lc)
        else:
            temp = reduce(lambda x,y:x+y,din[(x-lc.size):x]*lc)
        a = np.append(a,temp)
    return a

mode = 'bpsk'
dif = 'off'
fs = 16000
fc = 1000
N=400

pattern = np.array([1,0])
# s2p = B2IQ(np.random.randint(0,2,40),mode,dif,1)
# ms = mod_iq((s2p-0.5)*2,fc,fs)

mod = np.ones(N)
# s2p = B2IQ(mod,mode,dif,1)
work_seq = mod_iq((mod-0.5)*2,fc,fs)

POS=4
mod = np.ones(N)
mod[POS:(POS+np.size(pattern))] = pattern
print(mod)
# s2p = B2IQ(mod,mode,dif,1)
sim_seq = mod_iq((mod-0.5)*2,fc,fs)

out_reg = matched_filter_re(sim_seq,(pattern-0.5)*2,fc,fs,N)

plt.close('all')
plt.figure()
# plt.plot(s2p[0])
plt.plot(sim_seq,label='din')
plt.plot(out_reg,label='corr')
plt.legend()
