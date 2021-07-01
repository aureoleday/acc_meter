#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:33:41 2021

@author: aureoleday
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:52:42 2020

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
# 生成观测模拟数据
np.random.seed(0)
n_samples = 2048
time = np.linspace(0, 8, n_samples)
s1 = np.sin(20 * time) # 信号源 1 : 正弦信号


s2 = np.random.normal(1,1,n_samples)
# s2 = np.zeros(n_samples)
# s2[200:240] = 1

S = np.c_[s1, s2]

S += 0.2 * np.random.normal(size=S.shape) # 增加噪音数据
S /= S.std(axis=0) # 标准化
# 混合数据
A = np.array([[0.01, 80], [0.1, 100]]) # 混合矩阵
# plt.scatter(x, y, kwargs)A
X = np.dot(S, A.T) # 生成观测信号源
# ICA模型
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X) # 重构信号
A_ = ica.mixing_ # 获得估计混合后的矩阵
# PCA模型
pca = PCA(n_components=2)
H = pca.fit_transform(X) # 基于PCA的成分正交重构信号源
# 图形展示
plt.figure()
models = [X, S, S_, H]
names = ['Observations (mixed signal)',
         'True Sources',
         'ICA recovered signals',
         'PCA recovered signals']
colors = ['red', 'steelblue', 'orange']
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()


FS=4000
WINDOW_SIZE = 2**11

fft_size = WINDOW_SIZE

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

plt.figure()
#xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(X.T[0])
#x = np.arange(0,WINDOW_SIZE)/FS
plt.plot(habx_t)

plt.figure()
#xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(S_.T[0])
#x = np.arange(0,WINDOW_SIZE)/FS
plt.plot(habx_t)

plt.figure()
#xh = np.arange(0,WINDOW_SIZE/2+1)*FS/(WINDOW_SIZE)
habx_t,ph = my_fft(S_.T[1])
#x = np.arange(0,WINDOW_SIZE)/FS
plt.plot(habx_t)