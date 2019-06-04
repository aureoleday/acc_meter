# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:26:33 2019

@author: Administrator
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

class my_rcc:
    def __init__(self,beta, sps, span=None):
        """Generates a raised cosine FIR filter.
        :param beta: shape of the raised cosine filter (0-1)
        :param sps: number of samples per symbol
        :param span: length of the filter in symbols (None => automatic selection)
    
        >>> import arlpy
        >>> rc = arlpy.comms.rcosfir(0.25, 6)
        >>> bb = arlpy.comms.modulate(arlpy.comms.random_data(100), arlpy.comms.psk())
        >>> pb = arlpy.comms.upconvert(bb, 6, 27000, 18000, rc)
        """
        if beta < 0 or beta > 1:
            raise ValueError('Beta must be between 0 and 1')
        if span is None:
            # from http://www.commsys.isy.liu.se/TSKS04/lectures/3/MichaelZoltowski_SquareRootRaisedCosine.pdf
            # since this recommendation is for root raised cosine filter, it is conservative for a raised cosine filter
            span = 33-int(44*beta) if beta < 0.68 else 4
        delay = int(span*sps/2)
        t = np.arange(-delay, delay+1, dtype=np.float)/sps
        denom = 1 - (2*beta*t)**2
        eps = np.finfo(float).eps
        idx1 = np.nonzero(np.abs(denom) > np.sqrt(eps))
        b = np.full_like(t, beta*np.sin(np.pi/(2*beta))/(2*sps))
        b[idx1] = np.sinc(t[idx1]) * np.cos(np.pi*beta*t[idx1])/denom[idx1] / sps
        b /= np.sqrt(np.sum(b**2))
        self.b = b
        self.z = np.zeros(len(self.b)-1,dtype=np.float)
        self.sps = sps
    
    def rcc_coef(self):
        return self.b
    
    def rcc_lfilt(self,din):
        dout, self.z = signal.lfilter(self.b, np.sqrt(self.sps), din, zi=self.z)
        return dout
    
    def rcc_filt(self,din):
        dout = signal.filtfilt(self.b, np.sqrt(self.sps), din)
        return dout

if __name__ == "__main__":
    beta = 0.3
    sps = 8
    span = 8
    rc_inst = my_rcc(beta, sps, span)
    b = rc_inst.rcc_coef()
    
    din = np.kron((np.random.randint(0,2,32)-0.5)*2,np.append([1],np.zeros(sps-1)))
    
#    dout = signal.filtfilt(b,np.sqrt(sps),din)
    dout = rc_inst.rcc_filt(din)
    
    dout_rx = rc_inst.rcc_filt(dout)
    
    din2 = din.reshape(-1,int((len(b)-1)/2))
    
    dout2 = []
    for tx in din2:
        ty  = np.array(rc_inst.rcc_lfilt(tx))
        dout2 = np.append(dout2,ty)
        
    fig = plt.figure()
    ax = fig.add_subplot(311)
    bx = fig.add_subplot(312)
    cx = fig.add_subplot(313)
    ax.plot(dout,label='filter')
    ax.plot(dout2,label='lfilter')
    ax.plot(din,label='source')
    ax.legend()
    
    bx.plot(dout_rx,label="out")
    bx.legend()
    
    