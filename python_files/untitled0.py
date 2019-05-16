# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:50:52 2019

@author: Administrator
"""


# -*- coding: utf-8 -*-
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
# 某个均衡滤波器的参数
 
a = np.array([1.0, -1.947463016918843, 0.9555873701383931])
b = np.array([0.9833716591860479, -1.947463016918843, 0.9722157109523452])
 
# 44.1kHz， 1秒的频率扫描波
t = np.arange(0, 0.5, 1 / 44100.0)
x = signal.chirp(t, f0=10, t1=0.5, f1=1000.0)
 
#直接一次计算出所有的y
y = signal.lfilter(b,a,x)
#
# #绘图
# plt.figure(1)
# plt.subplot(211)
# plt.plot(t,x,'r')
# plt.subplot(212)
# plt.plot(t,y,'k')
# plt.show()
 
#分批依次计算y
x2=x.reshape(-1,50)
 
#滤波器的初始状态为0，长度是滤波器系数长度-1
z = np.zeros(max(len(a),len(b))-1,dtype=np.float)
y2=[]
 
for tx in x2:
    #对于每段信号进行滤波，新的初始状态保存在z中
    ty,z=signal.lfilter(b,a,tx,zi=z)
    y2.append(ty)
y2=np.array(y2)
y2=y2.reshape((-1,))
print(np.sum((y-y2)**2))
 
#将三个信号绘图
plt.figure(1)
plt.subplot(311)
plt.plot(t,x,'r')
plt.title('original data')
plt.subplot(312)
plt.plot(y,x,'b')
plt.title('calculate onec time')
plt.subplot(313)
plt.plot(y2,x,'k')
plt.title('calculate uncontinuously')
plt.show()