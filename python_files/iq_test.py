#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:30:51 2020

@author: aureoleday
"""

import numpy as np
from scipy.fftpack import fft
from scipy import signal
import matplotlib.pyplot as plt 
from functools import reduce

SIG_FREQ = 100
SAMPLE_FREQ = 800

fs = SIG_FREQ
spr = SAMPLE_FREQ

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


def demod_tt(IQ, carrier_freq = 1000, spr = 4000, offset = 'off',os = 0):
    slen = int(IQ.shape[0]/(spr/carrier_freq))
    osc_cos = np.cos(2*np.pi*(carrier_freq*np.arange(spr/carrier_freq)/spr + os))
    osc_sin = np.sin(2*np.pi*(carrier_freq*np.arange(spr/carrier_freq)/spr + os))
    demod_I = IQ*np.kron(np.ones(slen),osc_cos)
    # demod_Q = IQ*np.kron(np.ones(slen),osc_sin)
    if (offset == 'on'):
        demod_Q = IQ*np.kron(np.ones(slen),np.roll(osc_sin,int(spr/(2*carrier_freq))))
    else:
        demod_Q = IQ*np.kron(np.ones(slen),osc_sin)

    return demod_I,demod_Q

def loopfilter2(din,k1,k2,phi):
    phase = np.array([0])
    oscI = np.array([0])
    oscQ = np.array([0])
    err = np.array([0])
    k2_sum = 0
    
    for i in range(din.size):
        oscI = np.append(oscI,np.cos(2*np.pi*SIG_FREQ*i/SAMPLE_FREQ + phase[i]+phi))
        oscQ = np.append(oscQ,np.sin(2*np.pi*SIG_FREQ*i/SAMPLE_FREQ + phase[i]+phi))
        local_I = din[i]*oscI[i]
        local_Q = din[i]*oscQ[i]
        
        pd = local_I*local_Q
        
        k1_prod = k1*pd
        k2_prod = k2*pd        
        err = np.append(err, k2_sum + k1_prod)
        k2_sum = k2_sum+k2_prod
        phase = np.append(phase, phase[i]+err[i])
    
    return phase,err,oscI
        

# mode = 'bpsk'
# dif = 'off'
# s2p = B2IQ(np.random.randint(0,2,100),mode,dif,1)

# iq = constellation(s2p,mode)
# # iq = np.vstack((iq[1],iq[0]))

# offset = 'off'
# m_iq= mod_iq(iq,SIG_FREQ,SAMPLE_FREQ,offset)

def costas_param(N,nita,lb,dg):
    theta = (lb/N)/(nita+0.25/nita)
    k1 = (-4*nita*theta)/((1+2*nita*theta+theta**2)*dg)
    k2 = (-4*theta**2)/((1+2*nita*theta+theta**2)*dg)
    return k1,k2

def AWGN(sin,snr):
    SNR = 10**(snr/10)
    print(SNR)
    Ps = reduce(lambda x,y:x+y,map(lambda x:x**2,sin))/sin.size
    print(Ps)
    Pn = Ps/SNR
    print(Pn)
    s_wn = sin + np.random.randn(sin.size)*Pn**0.5
    return s_wn


# k1,k2 = costas_param(SAMPLE_FREQ/SIG_FREQ, 0.707, 0.2,0.2)
# print("N:%d,k1:%f,k2:%f" %(SAMPLE_FREQ/SIG_FREQ,k1,k2))

# swn = AWGN(m_iq,30)

# ph,er,oscI = loopfilter2(swn,k1 ,k2,np.random.randn()*3)
# # plt.plot(er)
# plt.plot(ph)
# plt.plot(oscI/2)
# plt.plot(swn/2)

def smod(N,fs,spr):
    return np.kron((np.random.randint(0,2,N)-0.5)*2,np.kron([1,-1],np.ones(int(spr/fs))))

def lfsr(N=3,cf=1,phase = 0):
    out = np.zeros(0)
    regs = np.zeros(N)
    regs[-1] = 1
    ph = phase%(2**N)
    
    # print(bin(cf))
    for x in range(2**(N+1)-1):
        c0 = 0
        # print(regs)
        n = reduce(lambda x,y:(int(x)<<1)+int(y),regs)&int(cf)
        # print(n)
        while(n>0):
            c0 = c0^(n&1)
            n = n>>1            
        # print(c0)
        out = np.append(out,regs[-1])
        regs[1:]=regs[:-1]
        regs[0] = c0
    # print(out)
    return 1-2*out[ph:(ph+2**N-1)]
    
pn = lfsr(5,0o45,0)
si = 1- 2*np.random.randint(0,2,24)
2
sp_si = np.kron(si,pn)
# sp_si = np.kron(np.ones(si.size),pn)
# sic = 1-2*np.ones(sp_si.size)
msp_si = np.kron(sp_si,np.cos(2*np.pi*fs*np.arange(spr/fs)/spr))
# mc_si = np.kron(sic,np.cos(2*np.pi*fs*np.arange(spr/fs)/spr))

snr = 10
s_wn = AWGN(msp_si,snr)
# mc_si= AWGN(mc_si,snr)

phi = np.pi*3/16
cos = np.kron(np.ones(sp_si.size),np.cos(2*np.pi*fs*np.arange(spr/fs)/spr+phi))
sin = np.kron(np.ones(sp_si.size),np.sin(2*np.pi*fs*np.arange(spr/fs)/spr+phi))

mix_i = s_wn*cos
mix_q = s_wn*sin

# corr_si = np.kron(pn[::-1],np.cos(2*np.pi*fs*np.arange(spr/fs)/spr))
corr_sq = np.kron(pn[::-1],np.sin(2*np.pi*fs*np.arange(spr/fs)/spr))
corr_si = np.kron(pn[::-1],np.ones(int(spr/fs)))

ds_ = np.convolve(s_wn,corr_sq)
ds_i = np.convolve(mix_i,corr_si)
ds_q = np.convolve(mix_q,corr_si)
# mcs_fft = fft(mc_si)/mc_si.size
# mf = abs(mcs_fft)


plt.figure()
plt.subplot(221)
plt.plot(np.arange(ds_i.size),ds_q,'g',ds_i,'r')
plt.subplot(222)
plt.plot(np.arange(msp_si.size),s_wn,'r',msp_si,'g')
plt.subplot(223)
plt.plot(np.arange(ds_i.size),(ds_i**2+ds_q**2))
plt.subplot(224)
plt.plot(ds_**2)
plt.show()


# def pos(N,esp):
#     cnt = 0
#     for x in range(N):
#         a = np.random.rand()
#         b = np.random.rand()
#         if(abs(a-b) < esp):
#             cnt = cnt + 1            
#     return cnt/N

# import math
# def birth(N,Days):
#     return 1 - math.factorial(Days)/(Days**N*math.factorial(Days-N))

# print(a)

# fi,fq = costas(m_iq[0],SIG_FREQ,SAMPLE_FREQ,offset)

# d_i,d_q = demod_iq(m_iq,SIG_FREQ,SAMPLE_FREQ,offset)
# d_i,d_q = demod_tt(m_iq,SIG_FREQ,SAMPLE_FREQ,offset,1)


# b, a = signal.butter(3, [0.1], 'lowpass')
# zi = signal.filtfilt(b,a,d_i)
# zq = signal.filtfilt(b,a,d_q)

# zz = (zi)**2+(zq)**2

# z_ = zi*zq
# err = loopfilter(z_, 0.1889, 0.00049377)
# plt.plot(z_)
# plt.plot(err)

# fig = plt.figure()
# ax = fig.add_subplot(411)
# bx = fig.add_subplot(412)
# cx = fig.add_subplot(413)
# dx = fig.add_subplot(414)


# xa = np.arange(m_iq.shape[0])/SAMPLE_FREQ
# # ax.plot(xa,m_iq[0],'g',label='i')
# # ax.plot(xa,m_iq[1],'r',label='q')
# ax.plot(xa,m_iq,'b',label='iq')
# ax.legend()

# xb = np.arange(s2p[0].shape[0])/SAMPLE_FREQ
# bx.plot(xb,s2p[0],'g',label='bi')
# bx.plot(xb,s2p[1],'r',label='bq')
# bx.legend()

# xc = np.arange(zi.shape[0])/SAMPLE_FREQ
# cx.plot(xc,zi,'g',label='I_filt')
# cx.plot(xc,zq,'r',label='Q_filt')
# cx.legend()

# xd = np.arange(zi.shape[0])/SAMPLE_FREQ
# dx.plot(xd,zz,'r',label='res')
# dx.legend()

# # xd = np.arange(mod_I.shape[0])/SAMPLE_FREQ
# # dx.plot(xb,mod_IQ,'g',label='mod_i')
# # dx.legend()