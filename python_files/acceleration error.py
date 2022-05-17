#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:23:16 2022

@author: aureoleday
"""

import numpy as np
import matplotlib.pyplot as plt

# plt.figure()
# ax = plt.subplot(111)
# lin = np.arange(100000)
# a_10 = 1/100000
# a_12 = 10/180000
# a_14 = 10/45000
# a_16 = 2/500
# r_13 = 5.67/200000

# ax.set_ylim(0,10)
# ax.set_xlim(0,100000)
# ax.set_ylabel('Tracking error(degree)')
# ax.set_xlabel('Acceleration(rps^2)')

# ax.plot(a_10*lin,label='adi_10bit')
# ax.plot(a_12*lin,label='adi_12bit')
# ax.plot(a_14*lin,label='adi_14bit')
# ax.plot(a_16*lin,label='adi_16bit')
# ax.plot(r_13*lin,label='rdc_13bit')

# plt.legend()
# plt.grid()

##dc offset mismatch original
# N=10000
# x = 2*np.pi*np.arange(N)/N
# dc = 0.01*(1+1j)
# iq = np.exp(x*1j)
# iq_bias = iq+dc

# y = np.arctan2(iq.imag,iq.real)
# y_bias = np.arctan2(iq_bias.imag,iq_bias.real)

# dif = (y_bias-y)
# # plt.figure()
# # fig,ax1=plt.subplots()
# # ax1.plot(y,color='r')
# # ax1.plot(y_bias,color='g')
# # ax1.plot(dif,color='b')
# # plt.show()
# print(np.ptp(dif[0:int(N/2-1)]))


#gain error
# lin = np.arange(100)
# err = 1/0.003*lin/1000
# plt.figure()
# ax = plt.subplot(111)
# ax.set_ylabel('Error(LSB)')
# ax.set_xlabel('gain mismatch(%)')
# ax.set_ylim(0,10)
# ax.set_xlim(0,3)
# ax.plot(lin/10,err)
# ax.grid()

##common mode dc error
# lin = np.arange(100)
# err = np.array([[0.0000,14.17],[0.0001,14.0],[0.0003,13.67],[0.0005,13.33],[0.0007,12.95],[0.001,12.5],[0.002,11.6],[0.003,11.06],[0.004,10.67],[0.005,10.35]])
# plt.figure()
# ax = plt.subplot(111)
# # ax.set_xlim(0,0.2)
# # ax.set_ylim(0,2.5)
# ax.set_ylabel('Error(LSB)')
# ax.set_xlabel('DC offset (FS%)')
# ax.plot(err[:,0]*100,14-err[:,1])
# ax.grid()

##differential phase error error
lin = np.arange(100)
err = np.array([[0.0000,13.7],[0.001,12.53],[0.002,11.61],[0.003,11.04],[0.004,10.63],[0.005,10.3],[0.006,10.04],[0.01,9.23]])
plt.figure()
ax = plt.subplot(111)
# ax.set_xlim(0,0.2)
# ax.set_ylim(0,2.5)
ax.set_ylabel('Error(LSB)')
ax.set_xlabel('Differential phase error (FS%)')
ax.plot(err[:,0]*100,14-err[:,1])
ax.grid()
