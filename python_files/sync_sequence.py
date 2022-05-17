#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:27:54 2022

@author: aureoleday
"""

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


def mod_iq(din, carrier_freq = 1000, spr = 16000, offset = 'off'):
    osc_cos = np.cos(2*np.pi*carrier_freq*np.arange(spr/carrier_freq)/spr)

    return np.kron(din,osc_cos)


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
fs = 4000
fc = 1000
N=40

pattern = np.array([1,1,1,0,0,1,0])
# pattern = np.array([1,1,1,0,1)

mod = np.ones(N)
work_seq = mod_iq((mod-0.5)*2,fc,fs)

POS=4
# mod = np.ones(N)
# mod[POS:(POS+np.size(pattern))] = pattern
mod = np.kron(np.ones(int(N/2)),pattern)
mod = mod[:N]
print(mod)
sim_seq = mod_iq((mod-0.5)*2,fc,fs)

out_reg = matched_filter_re(sim_seq,(pattern-0.5)*2,fc,fs,N)

out_bits = matched_filter_re((mod-0.5)*2,(pattern-0.5)*2,1,1,N)

plt.close('all')
plt.figure()
# plt.plot(s2p[0])
plt.plot(sim_seq,label='din')
plt.plot(out_reg,label='corr')
plt.legend()

plt.figure()
plt.plot(out_bits)