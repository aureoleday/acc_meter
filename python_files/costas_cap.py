#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 11:02:48 2022

@author: aureoleday
"""

import json
from types import SimpleNamespace
import numpy as np
from scipy import signal
from fxpmath import Fxp
import matplotlib.pyplot as plt
import pandas as pd

RAW = Fxp(None,dtype='S1.15')
DATA = Fxp(None,dtype='S1.15')
ANGLE = Fxp(None,dtype='S2.16')
PANGLE = Fxp(None,dtype='S1.17',overflow='wrap')
CONST = Fxp(None,dtype='U0.18')
ADDER = Fxp(None,dtype='S1.24')
PADDER = Fxp(None,dtype='S1.23',overflow='wrap')
MULTER = Fxp(None,dtype='S1.15')
COMEGA = Fxp(None,dtype='U0.3',overflow='wrap')
CANGLE = Fxp(None,dtype='S4.6',overflow='wrap')
UANGLE = Fxp(None,dtype='S1.9',overflow='wrap')

INV_PI_CONST = Fxp(1/(2*np.pi)).like(CONST)
def AGWN(Ps,snr):
    SNR = 10**(snr/10)
    Pn = Ps/SNR
    # np.random.seed(9)
    agwn = np.random.randn(1)[0]*(Pn**0.5)
    return agwn

def my_fft(din,fft_size):
    temp = din[:fft_size]
    fftx = np.fft.rfft(temp)/fft_size
    xfp = np.abs(fftx)*2
    return xfp

class my_filter:
    def __init__(self,N_ord,filt_zone=[0.1],filt_type='lowpass'):
        self.b,self.a = signal.butter(N_ord, filt_zone, filt_type)
        self.z = np.zeros(max(len(self.a),len(self.b))-1)
        
    def filt(self,din):
        dout, self.z = signal.lfilter(self.b, self.a, din, zi=self.z)
        return dout
    
class iir_filter(object):
    def __init__(self,k,opf='R'):
        self.k = k
        self.x = np.exp(0j)
        self.y = np.exp(0j)
        self.opf = opf
        self.out = np.exp(0j)
        if opf=='Q':
            self.k = Fxp(self.k).like(DATA)
            self.x = Fxp(self.x).like(DATA)
            self.y = Fxp(self.y).like(DATA)
            self.out = Fxp(self.out).like(DATA)
    
    def filt(self,d_iq):
        if self.opf=='R':
            self.out = d_iq - self.x+ self.k * self.y
            self.x = d_iq
            self.y = self.out
        else:
            self.out(Fxp(d_iq).like(DATA) - self.x+ self.k * self.y)
            self.x(d_iq)
            self.y(self.out.get_val())
        return self.y

class LoopFilter(object):
    def __init__(self, gain, Bn, zeta,opf='R'):
        self.kp = (1/gain)*(4*zeta/(zeta+1/(4*zeta)))*Bn
        self.ki = (1/gain)*(4/(zeta+1/(4*zeta))**2)*(Bn**2) 
        self.integrater = 0
        self.lf_out = 0
        self.opf = opf
        if opf=='Q':
            self.kp = Fxp(self.kp).like(CONST)
            self.ki = Fxp(self.ki).like(CONST)
            self.integrater = Fxp(self.integrater).like(ADDER)
            self.lf_out = Fxp(self.lf_out).like(ADDER)
        print("kp:%f, ki:%f" %(self.kp,self.ki))
        
    def advance_filter(self, phase_difference):
        if self.opf=='R':
            self.integrater += self.ki*phase_difference
            self.lf_out = self.integrater + self.kp*phase_difference
        else:
            self.integrater(self.integrater + self.ki*phase_difference)
            self.lf_out(self.integrater + self.kp*phase_difference)
        return self.lf_out

class PhaseDetector(object):
    def __init__(self,mode = 'pll',opf='R',dpsc=1):
        self.phase_difference = 0
        self.temp = 0
        if opf=='Q':
            self.phase_difference = Fxp(self.phase_difference).like(DATA)
            self.temp = Fxp(self.temp).like(MULTER)
        self.mode = mode
        self.opf = opf
        self.dpsc=dpsc

    def phd(self, d_iq, vco):
        if(self.mode == 'costas'):
            if self.opf=='R':
                self.temp = vco*d_iq.real
                self.phase_difference = self.temp.real * self.temp.imag
            else:
                self.temp(vco*Fxp(d_iq.real).like(RAW))
                self.phase_difference(self.temp.real * self.temp.imag)

        elif(self.mode == 'pll'):
            if self.opf=='R':
                self.phase_difference = d_iq.real*vco.imag - self.dpsc*d_iq.imag*vco.real
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco).imag)
            
        else:
            if self.opf=='R':
                self.phase_difference = (np.conjugate(d_iq)*vco)
            else:
                self.phase_difference((Fxp(np.conjugate(d_iq)).like(RAW)*vco))

        return self.phase_difference

class PLL(object):
    def __init__(self,fs,fc, lf_gain, lf_bandwidth, lf_damping,lf_delay = 1,pd_type='pll',opf='R'):
        self.delay = lf_delay 
        self.fs = fs
        self.fc = fc
        self.phase_estimate = 0.0    
        self.phase_difference = 0.0
        self.vco = np.exp(0j)
        self.omega_const = self.fc/self.fs 
        self.omega = self.fc/self.fs 
        self.loop_reg = np.zeros(32)
        if opf == 'Q':
            self.phase_difference = Fxp(self.phase_difference).like(ANGLE)
            self.phase_estimate = Fxp(self.phase_estimate).like(PANGLE)
            self.vco = Fxp(self.vco).like(DATA)
            self.omega_const = Fxp(self.omega).like(COMEGA)
            self.omega = Fxp(self.omega).like(COMEGA)
            self.ph_sum = Fxp(self.omega).like(UANGLE)
        self.loop_filter = LoopFilter(lf_gain, lf_bandwidth, lf_damping,opf=opf)
        self.phase_detector = PhaseDetector(pd_type,opf=opf)      
        self.pd_type = pd_type
        self.opf = opf
        print("%s omega:%f" %(pd_type,self.omega))
        
    def update_phase_estimate(self):
        if self.pd_type == 'costas':
            if self.opf == 'R':
                self.vco = np.exp(1j*(2*np.pi*self.omega + self.phase_estimate))
                self.phase_estimate += self.loop_reg[self.delay]
                self.omega += self.omega_const
            else:
                self.vco(np.exp(1j*(2*np.pi*(self.ph_sum.get_val()))))  
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.omega(self.omega+self.omega_const)
                self.ph_sum(self.omega+self.phase_estimate)                          
        else:
            if self.opf == 'R':
                self.vco = np.exp(1j*self.phase_estimate)
                self.phase_estimate += self.loop_reg[self.delay]
                self.rad = self.phase_estimate
                
            else:
                self.vco(np.exp(1j*2*np.pi*self.phase_estimate.get_val()))
                self.phase_estimate(self.phase_estimate + self.loop_reg[self.delay]*INV_PI_CONST)
                self.rad = 2*np.pi*self.phase_estimate.get_val()                
        self.loop_reg[1:] = self.loop_reg[:-1]
        self.loop_reg[0] = self.loop_filter.lf_out

    def step(self, d_iq):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        if self.opf == 'R':
            self.phase_difference = self.phase_detector.phd(d_iq,self.vco)
        else:
            self.phase_difference(self.phase_detector.phd(d_iq,self.vco))
        self.loop_filter.advance_filter(self.phase_difference)
        self.update_phase_estimate()
        
fs = 80000
fc = 10000
N = 40000
SNR = -10

def sig_gen(fs,fc):
    i=0
    while True:
       yield  np.exp(1j*2*np.pi*fc/fs*i)
       i += 1

def add_kv(dic,k,v):
    if k not in dic.keys():
        dic[k] = []
    dic[k].append(v)

def costas_tb():
    ts = np.arange(N-1)/fs
    dic ={}
    costas = PLL(fs,fc,1, 0.005, 0.707,lf_delay = 1,pd_type='costas')
    sig_in = sig_gen(fs,fc)
    for i in range(N - 1):
        sig = next(sig_in)
        add_kv(dic,'raw',sig)
        add_kv(dic,'swnoise',sig+complex(AGWN(0.5,SNR),AGWN(0.5,SNR)))
        costas.step(dic['swnoise'][-1])
        add_kv(dic,'vco_costas',costas.vco)
        
    plt.close('all')
    plt.figure()
    ax = plt.subplot(311)    
    ax.plot(ts,list(map(lambda x:x.imag,dic['swnoise'])),label='sig_w_noise')
    ax.set_ylabel('Amplitude')
    ax.set_title('input signal')
    plt.legend()
    
    ax = plt.subplot(312)
    ax.plot(ts,list(map(lambda x:x.real,dic['raw'])),label='sig')
    ax.plot(ts,list(map(lambda x:x.real,dic['vco_costas'])),label='recovered')
    ax.set_ylabel('Amplitude')
    ax.set_title('recoverd signal')
    plt.legend()

costas_tb()
        

        
        
        
        