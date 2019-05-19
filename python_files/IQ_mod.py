# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:35:41 2019

@author: Administrator
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 

SIG_FREQ = 500
SAMPLE_FREQ = 4000

#PN_CODE = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
PN_CODE = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1])

class iq_mod:
    def __init__(self,sig_freq=1000,sample_freq=32000,rep_N=8):
        self.i_wave = np.kron(np.ones(rep_N),np.cos(2*np.pi*sig_freq*np.arange(sample_freq/sig_freq)/sample_freq))
        self.q_wave = np.kron(np.ones(rep_N),np.sin(2*np.pi*sig_freq*np.arange(sample_freq/sig_freq)/sample_freq))
        self.period = int(sample_freq*rep_N/sig_freq)
    
    def apl_mod(self,i_d,q_d):
        i_data = np.kron((i_d*2 - 1),self.i_wave)
        q_data = np.kron((q_d*2 - 1),self.q_wave)
        return i_data,q_data    

    def mix(self,din):
        out = np.zeros(din.shape[0])
        for i in range(din.shape[0]):
            out[i] = din[i]*self.q_wave[(i+2)%self.period]
        return out
    
    def despread(self,din,code):
        out = np.zeros(din.shape[0])
        code_p = code*2 -1
        intp_code = np.kron(code_p,np.ones(self.period))
        print("cor len:%d\n" % intp_code.shape[0])
        for i in range(intp_code.shape[0],din.shape[0]):
            out[i] = np.dot(din[i-intp_code.shape[0]:i],intp_code)
        return out  

def rrc(beta, filter_width, Ts):
    """
    https://en.wikipedia.org/wiki/Root-raised-cosine_filter 
    :param beta: roll-off factor
    :param filter_width: The width of the filter, samples
    :param Ts: The width of a symbol, samples
    :return: impulse response of the filter, the tuple of filter_width float numbers coefficients
    """
    rrc_out = []
    for i in range(0, filter_width):
        rrc_out.append(0.0)
    if beta != 0.0:
        t1 = Ts/(4*beta)
    else:
        t1 = Ts

    for p in range(0, filter_width):
        t = (p - filter_width / 2)
        if t == 0.0:
            rrc_out[p] = (1 + beta*(4/np.pi - 1))
        elif t == t1 or t == -t1:
            if beta != 0.0:
                arg = np.pi/(4*beta)
                s = (1 + 2/np.pi)*np.sin(arg)
                c = (1 - 2/np.pi)*np.cos(arg)
                rrc_out[p] = (s + c) * (beta/np.sqrt(2))
            else:
                rrc_out[p] = 0
        else:
            pts = np.pi*t/Ts
            bt = 4*beta*t/Ts
            s = np.sin(pts*(1-beta))
            c = np.cos(pts*(1+beta))
            div = pts*(1 - bt*bt)
            rrc_out[p] = (s + bt*c)/div
    return tuple(rrc_out)

def my_fft(din):
    fftx = np.fft.rfft(din)/din.shape[0]
    xfp = np.abs(fftx)*2
    return xfp

iq_mod_inst = iq_mod(SIG_FREQ,SAMPLE_FREQ,rep_N=16)
#idi = np.random.randint(0,2,32)
#idi = np.zeros(32)
idi = np.tile(PN_CODE,5)
#idi = np.array([0,0,0,0,0,1,1,0,0,1,0,1,0])
iqi = np.random.randint(0,2,idi.shape[0])
idm,qdm = iq_mod_inst.apl_mod(idi,iqi)

noise = np.random.randn(idm.shape[0])*8

idm = idm*2 + noise

di = iq_mod_inst.mix(idm)
print("di len:%d\n" % di.shape[0])

b, a = signal.butter(3, 0.15, 'lowpass')
zt = signal.filtfilt(b,a,di)

z = np.zeros(max(len(a),len(b))-1,dtype=np.float)

z1, z = signal.lfilter(b, a, di[0:20], zi=z)
z2, z = signal.lfilter(b, a, di[20:40], zi=z)
z3, z = signal.lfilter(b, a, di[40:60], zi=z)
z4, z = signal.lfilter(b, a, di[60:80], zi=z)
z5, z = signal.lfilter(b, a, di[80:], zi=z)

zo = np.concatenate((z1,z2,z3,z4,z5))
cor = iq_mod_inst.despread(zo,PN_CODE)

print("zi len:%d\n" % zo.shape[0])

fig = plt.figure()
ax = fig.add_subplot(311)
bx = fig.add_subplot(312)
cx = fig.add_subplot(313)

x = np.arange(idm.shape[0])/SAMPLE_FREQ
xh = np.arange(idm.shape[0]/2 + 1)*SAMPLE_FREQ/idm.shape[0]

ax.plot(x,idm,'y',label='i')
ax.plot(x,qdm,'g',label='q')
#idff = my_fft(idm)
ax.legend()
bx.plot(x,cor)
bx.grid(True, linestyle='-.')
cx.plot(x,di,label='di')
cx.plot(x,zo,label='zo')
cx.plot(x,zt,'r',label='zt')
cx.legend()

plt.show()
    
        
        
        
        


