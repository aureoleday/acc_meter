#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:37:16 2020

@author: aureoleday
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

REF = 5.0
fs = 1e6
fc = 1e3
N = 1000
file_dir='/home/aureoleday/share/dc_1v.log'

def my_fft(din,fft_size):
    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

plt.close('all')
# df = pd.read_csv('/home/aureoleday/share/sadc_1k.txt',sep=':',skiprows=3)
df = pd.read_csv(file_dir,sep=':',skiprows=3)
di = df.iloc[:,-1].astype(str)
dd = np.array(list(map(lambda x:int(x,16),di)))
# dd = dd*REF/4096
data_len = dd.shape[0]
wave_gen = np.sin(2*np.pi*1000/fs*np.arange(data_len))

hd_adc = my_fft(dd-dd.mean(),data_len)
print("mean:%f,std:%f,range:%f\n" %(np.mean(dd),np.std(dd),np.ptp(dd)))
hd_sim = my_fft(wave_gen,data_len)
# fspan = np.arange(data_len/2+1)*fs/2
fspan = np.arange(0,data_len/2+1)*fs/(data_len)

plt.figure()
ax = plt.subplot(411)
ax.plot(dd,label='adc data')
plt.legend()
ax = plt.subplot(412)
ax.plot(wave_gen,label='sim_gen wave')
plt.legend()
ax = plt.subplot(413)
ax.plot(fspan,hd_adc,label='adc data fft')
plt.legend()
ax = plt.subplot(414)
ax.plot(fspan,hd_sim,label='sim_gen data fft')
plt.legend()
