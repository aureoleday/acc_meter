# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:56:12 2019

@author: Administrator
"""

import numpy as np
import rcos
from scipy import signal
import matplotlib.pyplot as plt 

SPS = 4
#PN_CODE = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1])#BARK CODE[1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
#PN_CODE = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
#PN_CODE = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
#PN_CODE = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
PN_CODE = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
PN_CODE = (PN_CODE-0.5)*2

#generate baseband signal
base_sig = (np.random.randint(0,2,2)-0.5)*2
base_up_sig = np.kron(base_sig,np.append(1,np.zeros(SPS-1)))
base_up_sig = np.kron(base_up_sig,np.ones_like(PN_CODE))*8

#spread signal
spread_sig = np.kron(base_sig,PN_CODE)

#chip filter
rcc_inst = rcos.my_rcc(0.4,SPS,8)
upsampel_sig = np.kron(spread_sig,np.append(1,np.zeros(SPS-1)))
shape_filtered = rcc_inst.rcc_filt(upsampel_sig)

#additive channel
noise = np.random.randn(shape_filtered.shape[0])/4
signal_wnoise = shape_filtered + noise*0

#match filter
shape_filtered2 = rcc_inst.rcc_filt(signal_wnoise)

#despread
def despread(din,code):
    dlen = din.shape[0]
#    upcode = np.kron(code,np.ones(SPS))
    upcode = np.kron(code,np.append(1,np.zeros(SPS-1)))
    clen = upcode.shape[0]
    out = []
    for i in range(dlen):
        if i == 0:
            out = np.append(out,0)
        else:
            out = np.append(out, np.dot(din[max(0,i-clen):i],upcode[:min(i,clen)]))
    return out  

despread_d = despread(shape_filtered2,PN_CODE)



#figure show
fig = plt.figure()

ax = fig.add_subplot(311)
ax.plot(spread_sig)
bx = fig.add_subplot(312)
bx.plot(shape_filtered,label="srcc")
bx.plot(shape_filtered2,label="rcc")
bx.legend()
cx = fig.add_subplot(313)
cx.plot(shape_filtered2*10,label='rx')
cx.plot(despread_d,label='dspread')
#cx.plot(base_up_sig,label='upsampel_sig')
cx.legend()
