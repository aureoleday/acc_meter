#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:30:51 2020

@author: aureoleday
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

SIG_FREQ = 100
SAMPLE_FREQ = 2000

def dpll(din,upsample):
    osc = np.kron([1,0],np.ones(upsample))
    

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

def constellation(IQ, mode = 'bpsk'):
    if mode == 'bpsk':
        tmp = (IQ-0.5)*2
        tmp[1] = 0
        return tmp
    else:
        return (IQ-0.5)*2

def mod_iq(IQ, carrier_freq = 1000, spr = 4000, offset = 'off'):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    osc_sin = np.sin(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    mod_I = np.kron(IQ[0],osc_cos)
    mod_Q = np.kron(IQ[1],osc_sin)
    if (offset == 'on'):
        mod_Q = np.roll(mod_Q,int(spr/(2*carrier_freq)))

    return mod_I + mod_Q

    # return np.vstack((mod_I,mod_Q))

def demod_iq(IQ, carrier_freq = 1000, spr = 4000, offset = 'off'):
    slen = int(IQ.shape[0]/(spr/carrier_freq))
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    osc_sin = np.sin(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)
    demod_I = IQ*np.kron(np.ones(slen),osc_cos)
    # demod_Q = IQ*np.kron(np.ones(slen),osc_sin)
    if (offset == 'on'):
        demod_Q = IQ*np.kron(np.ones(slen),np.roll(osc_sin,int(spr/(2*carrier_freq))))
    else:
        demod_Q = IQ*np.kron(np.ones(slen),osc_sin)

    return demod_I,demod_Q


mode = 'qpsk'
dif = 'on'
s2p = B2IQ(np.random.randint(0,2,50),mode,dif,1)

iq = constellation(s2p,mode)

offset = 'off'
m_iq= mod_iq(iq,SIG_FREQ,SAMPLE_FREQ,offset)

d_i,d_q = demod_iq(m_iq,SIG_FREQ,SAMPLE_FREQ,offset)

b, a = signal.butter(3, [0.1], 'lowpass')
zi = signal.filtfilt(b,a,d_i)
zq = signal.filtfilt(b,a,d_q)

fig = plt.figure()
ax = fig.add_subplot(411)
bx = fig.add_subplot(412)
cx = fig.add_subplot(413)
dx = fig.add_subplot(414)


xa = np.arange(m_iq.shape[0])/SAMPLE_FREQ
# ax.plot(xa,m_iq[0],'g',label='i')
# ax.plot(xa,m_iq[1],'r',label='q')
ax.plot(xa,m_iq,'b',label='iq')
ax.legend()

xb = np.arange(s2p[0].shape[0])/SAMPLE_FREQ
bx.plot(xb,s2p[0],'g',label='bi')
bx.plot(xb,s2p[1],'r',label='bq')
bx.legend()

xc = np.arange(zi.shape[0])/SAMPLE_FREQ
cx.plot(xc,zi,'g',label='I_filt')
cx.plot(xc,zq,'r',label='Q_filt')
cx.legend()

# xd = np.arange(mod_I.shape[0])/SAMPLE_FREQ
# dx.plot(xb,mod_IQ,'g',label='mod_i')
# dx.legend()