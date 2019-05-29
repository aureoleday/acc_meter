# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:35:41 2019

@author: Administrator
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
import gold
import wave
import struct

SIG_FREQ = 500
SAMPLE_FREQ = 4000

#PN_CODE = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1])
#PN_CODE = np.array([1,1,1,1,1,0,0,1,1,0,1,0,1])#BARK CODE
#PN_CODE = np.ones(127)
#PN_CODE = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0])#M CODE
#PN_CODE = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
#PN_CODE = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
PN_CODE = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0])
PN_CODE1 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
#PN_CODEG = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1])
#PN_CODE = np.random.randint(0,2,16)#RANDOM CODE


class iq_mod:
    def __init__(self,sig_freq=1000,sample_freq=32000,rep_N=8):
        i_wave = np.kron(np.ones(rep_N),np.cos(2*np.pi*sig_freq*np.arange(sample_freq/sig_freq)/sample_freq))
        q_wave = np.kron(np.ones(rep_N),np.sin(2*np.pi*sig_freq*np.arange(sample_freq/sig_freq)/sample_freq))
        self.wave = np.vstack((i_wave,q_wave))
        self.period = int(sample_freq*rep_N/sig_freq)
    
    def apl_mod(self,d_iq,mod=0):
        if mod==1:
            din = d_iq*2 - 1
        return np.vstack((np.kron(din[0],self.wave[0]),np.kron(din[1],self.wave[1])))

    def mix(self,d_iq,phase=0):
        return d_iq*np.tile(np.roll(self.wave,phase,axis=1),int(np.ceil(d_iq.shape[1]/self.wave.shape[1])))
    
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

class my_filter:
    def __init__(self,N,filt_zone=[0.2],filt_type='lowpass'):
        self.b,self.a = signal.butter(N, filt_zone, filt_type)
        self.z = np.zeros(max(len(self.a),len(self.b))-1,dtype=np.float)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout
        

def my_fft(din):
    fftx = np.fft.rfft(din)/din.shape[0]
    xfp = np.abs(fftx)*2
    return xfp

iq_mod_inst = iq_mod(SIG_FREQ,SAMPLE_FREQ,rep_N=1)
lpf_inst_i = my_filter(3,[0.15],'lowpass')
lpf_inst_q = my_filter(3,0.15,'lowpass')

din = np.tile(np.vstack((PN_CODE,PN_CODE)),4)
din2 = np.tile(np.vstack((PN_CODE1,PN_CODE1)),4)

din = din + din2

dm = iq_mod_inst.apl_mod(din,mod=1)

noise = np.random.randn(dm.shape[0],dm.shape[1])

dmn = dm + noise*2
dmn[1]=dmn[0]

dmm = iq_mod_inst.mix(dmn,1)

print("di len:%d\n" % din.shape[0])

b, a = signal.butter(3, [0.15], 'lowpass')

df = dmm[0]

zt = signal.filtfilt(b,a,df)

z1 = lpf_inst_i.filt(df[0:20])
z2 = lpf_inst_i.filt(df[20:40])
z3 = lpf_inst_i.filt(df[40:60])
z4 = lpf_inst_i.filt(df[60:80])
z5 = lpf_inst_i.filt(df[80:])

zo = np.concatenate((z1,z2,z3,z4,z5))

cor_i = iq_mod_inst.despread(zo,PN_CODE)

df = dmm[1]

zt = signal.filtfilt(b,a,df)

z1 = lpf_inst_q.filt(df[0:20])
z2 = lpf_inst_q.filt(df[20:40])
z3 = lpf_inst_q.filt(df[40:60])
z4 = lpf_inst_q.filt(df[60:80])
z5 = lpf_inst_q.filt(df[80:])

zo = np.concatenate((z1,z2,z3,z4,z5))

cor_q = iq_mod_inst.despread(zo,PN_CODE)

cor = np.vstack((cor_i,cor_q))

print("zi len:%d\n" % zo.shape[0])

fig = plt.figure()
ax = fig.add_subplot(411)
bx = fig.add_subplot(412)
cx = fig.add_subplot(413)
dx = fig.add_subplot(414)

x = np.arange(dm.shape[1])/SAMPLE_FREQ
xh = np.arange(dm.shape[1]/2 + 1)*SAMPLE_FREQ/dm.shape[1]

ax.plot(x,dmn[1],'g',label='qdm')
ax.plot(x,dm[0],'r',label='dm')


ax.legend()
bx.plot(x,cor[0],label='cor_i')
bx.plot(x,cor[1],label='cor_q')
bx.plot(x,np.linalg.norm(cor,axis=0),label='norm')
bx.grid(True, linestyle='-.')
bx.legend()
cx.plot(x,dmm[1],label='di')
cx.plot(x,zo,label='zo')
cx.plot(x,zt,'r',label='zt')
cx.legend()
#dx.plot(x,dm[0],label="di")
#dx.plot(x,dm[1],label="dq")
idff = my_fft(dmn[0])
dx.plot(xh,idff,label="i_freq/amp")
dx.legend()

plt.show()
    
        
        
        
        


