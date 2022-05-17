#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:20:49 2022

@author: aureoleday
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import sympy as sy


sy.init_printing()

class LoopFilter(object):
    def __init__(self, gain, Bn, zeta):
        self.kp = (1/gain)*(4*zeta/(zeta+1/(4*zeta)))*Bn
        self.ki = (1/gain)*(4/(zeta+1/(4*zeta))**2)*(Bn**2) 
        self.integrater = 0
        self.lf_out = 0
        print("kp:%f, ki:%f" %(self.kp,self.ki))        
        
    def advance_filter(self, phase_difference):
        self.integrater += self.ki*phase_difference
        self.lf_out = self.integrater + self.kp*phase_difference
        # print(self.lf_out)
        
    def ef(self):
        return self.lf_out
    def get_p(self):
        return self.kp,self.ki

def cross(arr, th):
    for i in np.arange(arr.size):
        if arr[i] >= th and arr[i+1]<th:
            return i

kd,kp,ki,z = sy.symbols('kd,kp,ki,z')
g = (kp+ki-kp/z)/(1-1/z)
h = 1/(1-1/z)
oltf = sy.cancel(g*h*z**(-1),z)
cltf = sy.cancel(oltf/(1+oltf),z)

T,theta = sy.symbols('T,theta')
hs = theta/(1-1/z)
hr = T*z/(z-1)**2
hp = (T**2/2)*z*(z+1)/(z-1)**3
e_p = sy.cancel(hs*(1-cltf),z)
e_f = sy.cancel(hr*(1-cltf),z)
e_a = sy.cancel(hp*(1-cltf),z)

fvs = sy.limit((z-1)*e_p,z,1)
fvr = sy.limit((z-1)*e_f,z,1)
fvp = sy.limit((z-1)*e_a,z,1)

print("fvs:",fvs)
print("fvr:",fvr)
print("fvp:",fvp)

tf = 0.001
fs = 200000
# #costas
# gain = 0.1
# bn = 0.01
# zeta = 0.707
#main loop
gain = 1
bn = 0.02
zeta = 0.707
print("enbw:", bn*fs*2*np.pi)

plt.close('all')
#main loop
lp = LoopFilter(gain,bn,zeta)
kp,ki = lp.get_p()

z=sy.Symbol('z')
G = (kp+ki-kp/z)/(1-1/z)
H = 1/(1-1/z)
OLTF = sy.cancel(G*H*(z**(-1)))
CLTF = sy.cancel(OLTF/sy.cancel(1+OLTF))

# num_o,den_o = map(lambda x: sy.Poly(x,z).all_coeffs(),sy.fraction(OLTF))
# num_o = np.array(num_o,dtype='float')
# den_o = np.array(den_o,dtype='float')
# sys_o = signal.TransferFunction(num_o,den_o,dt=1/fs)
# w1,H = signal.dlti.freqresp(sys_o)

num,den = map(lambda x: sy.Poly(x,z).all_coeffs(),sy.fraction(CLTF))
num = np.array(num,dtype='float')
den = np.array(den,dtype='float')
print("CLTF:",CLTF)
print("num:",num)
print("den:",den)
print("ts:",-np.log(tf*np.sqrt(1-zeta**2))/(zeta*bn*fs*np.pi))

# num = np.array([kp+ki,-kp])
# den = [1,kp+ki-2,1-kp]
sys = signal.TransferFunction(num,den,dt=1/fs)
# w1,H = signal.dlti.freqresp(sys)
w,mag,phase = signal.dbode(sys,n=10000)
ts,ys = signal.dstep(sys,n=200)
ti,yi = signal.dimpulse(sys,n=200)

po = 10
phase_margin = phase[abs(mag[po:]).argmin()]+180
gain_margin = -mag[abs(phase+180).argmin()]
print("phase_margin:%f,phase_ind:%d\ngain_margin:%f,gain_ind:%d" %(phase_margin,abs(mag[po:]).argmin(),gain_margin,abs(phase+180).argmin()))

plt.close('all')
fig,ax = plt.subplots(2,1,True)
ax[0].semilogx(w,mag,'r')
ax[0].set_ylabel('Gain(dB)')
ax[1].semilogx(w,phase,'b')
ax[1].set_ylabel('Phase(degree)')
ax[1].set_xlabel('Frequency(Hz)')
plt.subplots_adjust(hspace=0)
ax[0].grid(1)
ax[1].grid(1)
plt.show()

fig,ax = plt.subplots(2,1,True)
ax[0].step(ts,np.squeeze(ys),color='r')
ax[0].set_ylabel('Step')
ax[1].step(ti,np.squeeze(yi),color='b')
ax[1].set_ylabel('Impulse')
ax[1].set_xlabel('Time(s)')
plt.subplots_adjust(hspace=0)
ax[0].grid(1)
ax[1].grid(1)
plt.show()

# plt.figure()
# plt.plot(H.real,H.imag,'b')
# plt.plot(H.real,-H.imag,'r')
# plt.scatter(-1,0,color='purple')
# plt.grid(True)
