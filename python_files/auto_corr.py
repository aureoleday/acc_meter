# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:08:28 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt 

FREQ = 200
FREQ1 = 400
SAMPLE_RATE = 4000
P_CNT = 10

def auto_corr(din):
    n = din.shape[0]
    buf = np.zeros(n)
    for i in range(n):
        buf[i] = np.dot(din,np.roll(din,i))
    return buf

def my_fft(din):
    fftx = np.fft.rfft(din)/din.shape[0]
    xfp = np.abs(fftx)*2
    return xfp

x = np.kron(np.ones(P_CNT),np.sin(2*np.pi*FREQ*np.arange(SAMPLE_RATE/FREQ)/SAMPLE_RATE))
x1 = np.kron(np.ones(P_CNT),np.cos(2*np.pi*FREQ*np.arange(SAMPLE_RATE/FREQ)/SAMPLE_RATE))
#x1 = np.kron(np.ones(P_CNT*2),np.sin(2*np.pi*FREQ1*np.arange(SAMPLE_RATE/FREQ1)/SAMPLE_RATE))
x = x + x1

noise = np.random.randn(x.shape[0])
x = (x + noise*0)/16

fig = plt.figure()
ax = fig.add_subplot(421)
ax.plot(x)

y1 = auto_corr(x)
bx = fig.add_subplot(422)
bx.plot(y1,label='y1')

y2 = auto_corr(y1)
cx = fig.add_subplot(423)
cx.plot(y2,label='y2')

y3 = auto_corr(y2)
dx = fig.add_subplot(424)
dx.plot(y3,label='y3')

y4 = auto_corr(y3)/100
ex = fig.add_subplot(425)
ex.plot(y4,label='y4')

fx = fig.add_subplot(426)
xh = np.arange(x.shape[0]/2 + 1)*SAMPLE_RATE/x.shape[0]
idff1 = my_fft(x)
fx.plot(xh,idff1,label="source")

gx = fig.add_subplot(427)
idff2 = my_fft(y4)
gx.plot(xh,idff2,label="y4")
