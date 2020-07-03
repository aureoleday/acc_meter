#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:55:46 2019

@author: aureoleday
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

FS=4000
WINDOW_SIZE = 2**12

fft_size = WINDOW_SIZE

df=pd.read_excel('/home/aureoleday/h29.xlsx')#这个会直接默认读取到这个Excel的第一个表单

data=df.head()#默认读取前5行的数据
print("获取到所有的值:\n{0}".format(data))#格式化输出

arr = np.array(df['ch1'][4:(4+WINDOW_SIZE)])/1
def choose_windows(name='Hanning', N=20): # Rect/Hanning/Hamming 
    if name == 'Hamming':
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Hanning':
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)]) 
    elif name == 'Rect':
        window = np.ones(N) 
    return window

def my_fft(din):
    temp = din[:fft_size]*choose_windows(name='Rect',N=fft_size)
    fftx = np.fft.rfft(temp)*2/fft_size
    ampl = np.abs(fftx)
    ph = np.angle(fftx)
    return ampl,ph


xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(arr)
x = np.arange(0,WINDOW_SIZE)/FS
plt.plot(xh,habx_t)